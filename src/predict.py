import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.config import BASE_DIR, CLASSES
from src.data.preprocess import to_model_input

DEFAULT_MODELS_DIR = BASE_DIR / "exports" / "final"
SESSIONS = {}


def get_session(onnx_path) -> ort.InferenceSession:
    key = str(onnx_path)
    if key not in SESSIONS:
        SESSIONS[key] = ort.InferenceSession(key, providers=["CPUExecutionProvider"])
    return SESSIONS[key]


def load_manifest(models_dir) -> dict:
    return json.loads((Path(models_dir) / "models.json").read_text(encoding="utf-8"))


def softmax(logits: np.ndarray) -> np.ndarray:
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)


def label_for(prob_ai: float, uncertain_band: float) -> str:
    if abs(prob_ai - 0.5) < uncertain_band:
        return "uncertain"
    return "ai" if prob_ai >= 0.5 else "real"


def predict(manifest: dict, models_dir, model_name: str, data: bytes) -> dict:
    entry = manifest["models"][model_name]
    x = to_model_input(data, entry["input_size"], entry["jpeg_quality"])
    session = get_session(Path(models_dir) / entry["file"])
    logits = session.run([entry["output_name"]], {entry["input_name"]: x})[0][0]
    prob_ai = float(softmax(logits)[CLASSES.index("FAKE")])
    return {
        "model": model_name,
        "prob_ai": round(prob_ai, 4),
        "label": label_for(prob_ai, manifest["uncertain_band"]),
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--model", help="mlp, cnn o resnet18 (por defecto, todos los exportados)")
    parser.add_argument("--models-dir", default=str(DEFAULT_MODELS_DIR))
    args = parser.parse_args(argv)

    manifest = load_manifest(args.models_dir)
    names = [args.model] if args.model else list(manifest["models"])
    unknown = [n for n in names if n not in manifest["models"]]
    if unknown:
        raise SystemExit(f"Modelo desconocido: {unknown}. Disponibles: {list(manifest['models'])}")
    data = Path(args.image).read_bytes()
    for name in names:
        result = predict(manifest, args.models_dir, name, data)
        print(f"{result['model']:9} prob_ai {result['prob_ai']:.4f}  label {result['label']}")


if __name__ == "__main__":
    main()
