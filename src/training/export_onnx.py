import argparse
import hashlib
import json
import os
import time
from datetime import date
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config import CLASSES
from src.training.evaluate import load_run

TOLERANCE = 1e-3
PREPROCESS = Path(__file__).resolve().parents[1] / "data" / "preprocess.py"


def text_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def check_parity(model, size, path):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    worst = 0.0
    for batch in (1, 4):
        x = torch.rand(batch, 3, size, size, generator=torch.Generator().manual_seed(batch))
        with torch.no_grad():
            expected = model(x).numpy()
        got = session.run(["logits"], {"input": x.numpy()})[0]
        worst = max(worst, float(np.abs(expected - got).max()))
    if worst > TOLERANCE:
        raise SystemExit(f"{Path(path).name}: PyTorch y ONNX Runtime difieren en {worst:.2e} (> {TOLERANCE})")
    return worst


def export_to_onnx(model, onnx_path: str | Path, image_size: int = 32):
    """Exporta un modelo PyTorch a formato ONNX."""
    onnx_path = Path(onnx_path)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = onnx_path.with_name(onnx_path.stem + ".tmp.onnx")

    device = next(model.parameters()).device
    model.cpu().eval()
    dummy_input = torch.randn(2, 3, image_size, image_size)

    try:
        torch.onnx.export(
            model,
            dummy_input,
            str(tmp_path),
            dynamo=True,
            external_data=False,
            input_names=["input"],
            output_names=["logits"],
            dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=1024)}},
        )
        diff = check_parity(model, image_size, tmp_path)
        os.replace(tmp_path, onnx_path)
    finally:
        tmp_path.unlink(missing_ok=True)
        model.to(device)
    print(f"Modelo ONNX guardado en: {onnx_path}")
    return diff


def ort_latency(path, size, repeats=30):
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).random((1, 3, size, size), dtype=np.float32)
    for _ in range(3):
        session.run(["logits"], {"input": x})
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        session.run(["logits"], {"input": x})
        times.append(time.perf_counter() - started)
    return {"latency_ms_p50": round(float(np.median(times) * 1000), 2),
            "latency_ms_p95": round(float(np.percentile(times, 95) * 1000), 2)}


def split_counts(folder):
    return {split.name: sum(1 for _ in split.rglob("*.jpg")) for split in sorted(Path(folder).iterdir()) if split.is_dir()}


def card_section(name, entry):
    val, test = entry["val_metrics"] or {}, entry.get("test_metrics") or {}
    lines = [f"## {name}", "",
             f"- Versión: `{entry['version']}` — archivo `{entry['file']}` ({entry['size_mb']} MB, sha256 `{entry['sha256'][:16]}...`)",
             f"- Dataset: {entry['dataset']} (imágenes por split: {entry['splits']})",
             f"- Entrada: {entry['input_size']} x {entry['input_size']}, JPEG calidad {entry['jpeg_quality']}, recorte cuadrado central",
             f"- Hiperparámetros: {json.dumps(entry['hyperparameters'])}",
             f"- Validación (época {entry['best_epoch']}): ROC-AUC {val.get('roc_auc')}, F1 {val.get('f1')}, accuracy {val.get('accuracy')}",
             f"- Test: {json.dumps(test.get('test')) if test else 'sin evaluar (correr src.training.evaluate)'}"]
    if test.get("unseen"):
        lines.append(f"- Generadores no vistos: {json.dumps(test['unseen'])}")
    lines += [f"- ONNX Runtime en CPU, 1 imagen: p50 {entry['ort_cpu']['latency_ms_p50']} ms "
              f"(medido en la PC de entrenamiento; en Lambda se mide aparte)", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="exports")
    parser.add_argument("--default", default="resnet18", help="modelo que usa la API si el pedido no elige")
    parser.add_argument("--uncertain-band", type=float, default=0.1,
                        help="prob_ai a menos de esta distancia de 0,5 se informa como 'uncertain'")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {"schema": 1, "classes": list(CLASSES), "default_model": args.default,
                "uncertain_band": args.uncertain_band, "exported_at": date.today().isoformat(),
                "preprocess_sha256": text_sha256(PREPROCESS), "models": {}}
    cards = []
    for run_dir in map(Path, args.runs):
        cfg, model = load_run(run_dir, torch.device("cpu"))
        name, size = cfg["model"], cfg["image_size"]
        if name in manifest["models"]:
            raise SystemExit(f"Dos corridas del mismo modelo: {name}")
        path = out / f"{name}.onnx"
        diff = export_to_onnx(model, path, size)
        done = json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
        test_path = run_dir / "test_metrics.json"
        entry = {
            "file": path.name, "version": f"{run_dir.name}@{manifest['exported_at']}",
            "dataset": cfg["dataset"], "input_size": size, "jpeg_quality": cfg["jpeg_quality"],
            "input_name": "input", "output_name": "logits", "opset": onnx.load(str(path)).opset_import[0].version,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size_mb": round(path.stat().st_size / 2**20, 1), "parity_max_abs_diff": diff,
            "best_epoch": done["best_epoch"], "val_metrics": done["val"],
            "test_metrics": json.loads(test_path.read_text(encoding="utf-8")) if test_path.exists() else None,
            "hyperparameters": {**cfg["model_args"], **{k: cfg["train"][k] for k in
                                ("batch_size", "learning_rate", "weight_decay", "epochs", "label_smoothing")}},
            "splits": split_counts(cfg["processed_dir"]), "ort_cpu": ort_latency(path, size),
        }
        manifest["models"][name] = entry
        cards.append(card_section(name, entry))
        print(f"{name}: {path} ({entry['size_mb']} MB, diferencia máxima {diff:.1e})")
    if args.default not in manifest["models"]:
        raise SystemExit(f"--default {args.default} no está entre los modelos exportados")

    slim = {k: v for k, v in manifest.items() if k != "models"}
    slim["models"] = {n: {k: e[k] for k in ("file", "version", "input_size", "jpeg_quality", "input_name",
                                            "output_name", "sha256")} for n, e in manifest["models"].items()}
    (out / "models.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (out / "export_report.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    header = ["# Ficha de los modelos", "",
              "Detectores de imágenes generadas por IA. Salida: probabilidad de que la imagen sea generada "
              f"(prob_ai). Entre {0.5 - args.uncertain_band:.2f} y {0.5 + args.uncertain_band:.2f} la API "
              "responde 'uncertain'.", "",
              "Uso previsto: demo académica. No usar para decidir sobre personas ni como prueba de autenticidad. "
              "Solo vio fotos y generadores de CIFAKE y ArtiFact recortados a cuadrado; con otros generadores, "
              "capturas de pantalla o imágenes muy editadas puede fallar sin aviso. Varias fuentes de ArtiFact "
              "tienen licencias no comerciales.", ""]
    (out / "MODEL_CARD.md").write_text("\n".join(header + cards), encoding="utf-8")


if __name__ == "__main__":
    main()
