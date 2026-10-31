import ast
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
import torch
import yaml
from torch.utils.data import DataLoader

import src.config
from conftest import synthetic
from src import predict
from src.config import CLASSES, run_config
from src.data import prepare_artifact, prepare_cifake
from src.data.dataset import create_dataloaders, create_dataset
from src.data.preprocess import decode, harmonize, load_rgb, preprocess_image, to_model_input
from src.training import evaluate, export_onnx, sweep, train
from src.training.metrics import compute_metrics
from src.training.trainer import Trainer

FAST = {"epochs": 1, "num_workers": 0, "batch_size": 16, "mixed_precision": False, "device": "cpu"}
CPU = torch.device("cpu")


def run(model, dataset, sets, run_dir):
    args = ["--model", model, "--dataset", dataset, "--name", str(run_dir)]
    for key, value in sets.items():
        args += ["--set", f"{key}={json.dumps(value)}"]
    return train.main(args)


VALID = [("mlp", "cifake", {}), ("cnn", "cifake", {}),
         ("mlp", "artifact", {}), ("cnn", "artifact", {}), ("resnet18", "artifact", {"pretrained": False})]


def count(folder, split):
    return len(list((Path(folder) / split).rglob("*.jpg")))


@pytest.fixture(scope="module")
def prepared(tmp_path_factory, cifake_raw, artifact_zip):
    base = tmp_path_factory.mktemp("base")
    prepare_cifake.main(["--src", str(cifake_raw), "--out", str(base / "data/processed/cifake")])
    prepare_artifact.main(["extract", "--zip", str(artifact_zip), "--out", str(base / "data/processed/artifact"),
                           "--per-class", "40", "--holdout", "ddpm", "--holdout-per-source", "10"])
    return base


@pytest.fixture
def base_dir(prepared, monkeypatch):
    monkeypatch.setattr(src.config, "BASE_DIR", prepared)
    return prepared


def test_preprocess_only_depends_on_pillow_and_numpy():
    tree = ast.parse(Path(src.data.preprocess.__file__).read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            roots.add(node.module.split(".")[0])
    assert roots <= {"base64", "io", "numpy", "PIL"}


def test_preprocess_parity_with_prepared_data():
    original = synthetic("FAKE", (220, 180), 7)
    stored = harmonize(load_rgb(original), 200, 90)
    expected = decode(stored).astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    got = to_model_input(original, 200, 90)
    assert got.shape == (1, 3, 200, 200) and got.dtype == np.float32
    assert np.array_equal(got, expected)


def test_preprocess_image_accepts_base64_and_data_uri():
    import base64
    original = synthetic("REAL", 40, 3)
    b64 = base64.b64encode(original).decode()
    assert np.array_equal(preprocess_image(b64, 32, 90), to_model_input(original, 32, 90))
    assert np.array_equal(preprocess_image("data:image/png;base64," + b64, 32, 90), to_model_input(original, 32, 90))


def test_prepared_file_loaded_by_dataset_equals_to_model_input(prepared, cifake_raw):
    original = (cifake_raw / "test" / "FAKE" / "3.png").read_bytes()
    digest = hashlib.sha256(original).hexdigest()[:16]
    folder = prepared / "data/processed/cifake"
    assert (folder / "test" / "FAKE" / f"cifake-fake__{digest}.jpg").exists()
    ds = create_dataset(folder, "test", 32)
    idx = [i for i, (p, _) in enumerate(ds.samples) if p.endswith(f"cifake-fake__{digest}.jpg")][0]
    tensor, label = ds[idx]
    assert label == 1
    assert np.array_equal(tensor.numpy()[None], to_model_input(original, 32, 90))


def test_prepare_cifake_splits_labels_and_duplicates(prepared):
    folder = prepared / "data/processed/cifake"
    counts = {s: count(folder, s) for s in ("train", "val", "test")}
    assert counts["test"] == 20
    assert counts["train"] + counts["val"] == 80
    assert counts["val"] == 8
    names = {p.name.split("__")[0] + "/" + p.parent.name for p in (folder / "train").rglob("*.jpg")}
    assert names == {"cifake-real/REAL", "cifake-fake/FAKE"}


def test_prepare_cifake_val_is_stable_and_needs_force(prepared, cifake_raw, tmp_path):
    folder = prepared / "data/processed/cifake"
    first = sorted(p.name for p in (folder / "val").rglob("*.jpg"))
    out = tmp_path / "cifake"
    prepare_cifake.main(["--src", str(cifake_raw), "--out", str(out)])
    assert sorted(p.name for p in (out / "val").rglob("*.jpg")) == first
    with pytest.raises(SystemExit):
        prepare_cifake.main(["--src", str(cifake_raw), "--out", str(out)])
    prepare_cifake.main(["--src", str(cifake_raw), "--out", str(out), "--force"])


def test_artifact_peek(artifact_zip, capsys):
    prepare_artifact.main(["peek", "--zip", str(artifact_zip)])
    out = capsys.readouterr().out
    assert out.startswith("5 archivos metadata.csv") and "image_path,target,category" in out


def test_artifact_list(artifact_zip, capsys):
    prepare_artifact.main(["list", "--zip", str(artifact_zip)])
    out = capsys.readouterr().out
    assert "imagenet" in out and "TOTAL" in out
    assert out.split("TOTAL")[1].split()[:3] == ["60", "90", "0"]


def test_artifact_extract_holdout_and_stratification(prepared):
    folder = prepared / "data/processed/artifact"
    unseen = list((folder / "unseen").rglob("*.jpg"))
    assert {p.name.split("__")[0] for p in unseen} == {"ddpm"} and len(unseen) == 10
    assert not (folder / "unseen" / "REAL").exists()
    train = {p.name.split("__")[0] for p in (folder / "train").rglob("*.jpg")}
    assert "ddpm" not in train
    assert sum(count(folder, s) for s in ("train", "val", "test")) == 80
    assert {p.name.split("__")[0] for p in (folder / "test").rglob("*.jpg")} == {"coco", "imagenet", "glide", "stylegan2"}
    x, y = create_dataset(folder, "train", 200)[0]
    assert x.shape == (3, 200, 200)


def test_unseen_split_with_only_fake_loads_with_fixed_mapping(prepared):
    ds = create_dataset(prepared / "data/processed/artifact", "unseen", 200)
    assert ds.class_to_idx == {"REAL": 0, "FAKE": 1} and len(ds) == 10
    assert {label for _, label in ds.samples} == {1}
    assert set(ds.sources) == {"ddpm"}


def test_class_mapping_is_real_0_fake_1(prepared):
    assert CLASSES == ["REAL", "FAKE"]
    ds = create_dataset(prepared / "data/processed/cifake", "train", 32)
    assert ds.class_to_idx == {"REAL": 0, "FAKE": 1}


def test_create_dataloaders_uses_val_folder_not_test(prepared):
    train_loader, val_loader = create_dataloaders(prepared / "data/processed/cifake", 32, 16, num_workers=0)
    assert len(val_loader.dataset) == 8 and len(train_loader.dataset) == 72
    with pytest.raises(FileNotFoundError):
        create_dataloaders(prepared / "data/processed/nope", 32, 16, num_workers=0)


def test_spawned_workers_can_read_the_dataset(prepared):
    ds = create_dataset(prepared / "data/processed/cifake", "val", 32)
    loader = DataLoader(ds, batch_size=4, num_workers=2, multiprocessing_context="spawn")
    assert sum(len(y) for _, y in loader) == len(ds)


def test_metrics_use_fake_as_positive_class():
    y_true = np.array([0, 0, 1, 1])
    prob_ai = np.array([0.1, 0.2, 0.8, 0.9])
    m = compute_metrics(y_true, (prob_ai >= 0.5).astype(int), prob_ai)
    assert m["roc_auc"] == 1.0 and m["f1"] == 1.0 and m["precision"] == 1.0 and m["recall"] == 1.0
    inverted = compute_metrics(y_true, (prob_ai < 0.5).astype(int), 1 - prob_ai)
    assert inverted["f1"] == 0.0


def test_run_config_reads_config_yaml_and_overrides():
    cfg = run_config("mlp", "artifact", {"lr": "3e-4", "dropout": 0.5, "seed": 7})
    assert cfg["image_size"] == 200 and cfg["model_args"]["pool"] == 4
    assert cfg["train"]["learning_rate"] == 0.0003 and cfg["model_args"]["dropout"] == 0.5 and cfg["seed"] == 7
    assert cfg["train"]["early_stopping_patience"] == 5 and cfg["train"]["mixed_precision"] is True
    assert run_config("mlp", "cifake")["model_args"]["pool"] == 1
    with pytest.raises(ValueError):
        run_config("cnn", "cifake", {"no_existe": 1})


def test_resnet18_is_rejected_with_cifake(base_dir, capsys):
    with pytest.raises(ValueError, match="solo con ArtiFact"):
        run_config("resnet18", "cifake")
    with pytest.raises(SystemExit):
        train.main(["--model", "resnet18", "--dataset", "cifake"])
    assert "solo con ArtiFact" in capsys.readouterr().err


def test_sweep_yaml_is_valid_and_has_no_cifake_resnet18():
    entries = sweep.load_runs(Path(src.config.CONFIG_PATH).with_name("sweep.yaml"))
    assert len(entries) == 38
    for e in entries:
        run_config(e["model"], e["dataset"], e.get("set"))
    assert not [e for e in entries if e["model"] == "resnet18" and e["dataset"] == "cifake"]


def test_hyperparameters_reach_the_models():
    from src.models import build_model
    mlp = build_model("mlp", 200, hidden_dims=[64, 32], dropout=0.1, pool=4)
    assert [m.out_features for m in mlp.modules() if isinstance(m, torch.nn.Linear)] == [64, 32, 2]
    assert next(m for m in mlp.modules() if isinstance(m, torch.nn.Linear)).in_features == 7500
    cnn = build_model("cnn", 32, base_channels=8, dropout=0.2)
    assert next(m for m in cnn.modules() if isinstance(m, torch.nn.Conv2d)).out_channels == 8
    assert cnn(torch.rand(2, 3, 32, 32)).shape == (2, 2)


@pytest.mark.parametrize("model,dataset,extra", VALID)
def test_one_epoch_each_model(base_dir, tmp_path, model, dataset, extra):
    summary = run(model, dataset, {**FAST, **extra}, tmp_path / "run")
    assert summary["epochs_run"] == 1 and 0.0 <= summary["val"]["roc_auc"] <= 1.0
    run_dir = tmp_path / "run"
    assert (run_dir / "best.pt").exists() and (run_dir / "config.json").exists() and (run_dir / "done.json").exists()
    assert not (run_dir / "last.pt").exists()
    assert len((run_dir / "metrics.csv").read_text(encoding="utf-8").splitlines()) == 2


def test_run_refuses_changed_config(base_dir, tmp_path):
    cfg = run_config("cnn", "cifake", FAST)
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "config.json").write_text(json.dumps({**cfg, "seed": 99}), encoding="utf-8")
    with pytest.raises(SystemExit, match="otra configuración"):
        run("cnn", "cifake", FAST, tmp_path / "run")


def test_early_stopping_uses_early_stopping_patience(base_dir, tmp_path, monkeypatch):
    flat = {"accuracy": 0.5, "f1": 0.5, "precision": 0.5, "recall": 0.5, "roc_auc": 0.5, "n": 8, "loss": 0.7}
    monkeypatch.setattr(Trainer, "evaluate", lambda self, loader, return_probs=False: dict(flat))
    summary = run("cnn", "cifake", {**FAST, "epochs": 10, "early_stopping_patience": 2}, tmp_path / "run")
    assert summary["epochs_run"] == 3 and summary["stopped_early"] and summary["best_epoch"] == 1


def test_resume_after_crash_keeps_batch_order(base_dir, tmp_path, monkeypatch):
    sets = {**FAST, "epochs": 3}
    reference = run("cnn", "cifake", sets, tmp_path / "reference")

    original = Trainer.train_epoch
    calls = {"n": 0}

    def crash_on_second(self, loader):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("corte de luz")
        return original(self, loader)

    monkeypatch.setattr(Trainer, "train_epoch", crash_on_second)
    with pytest.raises(RuntimeError):
        run("cnn", "cifake", sets, tmp_path / "run")
    assert (tmp_path / "run" / "last.pt").exists()
    monkeypatch.setattr(Trainer, "train_epoch", original)
    summary = run("cnn", "cifake", sets, tmp_path / "run")

    rows = (tmp_path / "run" / "metrics.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert [int(r.split(",")[0]) for r in rows] == [1, 2, 3] and summary["epochs_run"] == 3

    assert summary["val"] == reference["val"]
    ref_rows = (tmp_path / "reference" / "metrics.csv").read_text(encoding="utf-8").splitlines()[1:]
    assert [r.split(",")[2] for r in rows] == [r.split(",")[2] for r in ref_rows]


def test_sweep_skips_finished_runs_and_records_failures(base_dir, tmp_path, capsys):
    spec = {"runs": [
        {"name": "a", "model": "mlp", "dataset": "cifake", "set": FAST},
        {"name": "b", "model": "cnn", "dataset": "cifake", "set": {**FAST, "lr": "3e-4"}},
    ]}
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    runs = tmp_path / "runs"
    sweep.main(["--file", str(path), "--runs-dir", str(runs)])
    assert json.loads((runs / "b/config.json").read_text(encoding="utf-8"))["train"]["learning_rate"] == 0.0003
    capsys.readouterr()
    sweep.main(["--file", str(path), "--runs-dir", str(runs), "--dry-run"])
    assert "correría" not in capsys.readouterr().out
    assert len((runs / "summary.csv").read_text(encoding="utf-8").splitlines()) == 3


    (runs / "b" / "done.json").unlink()
    (runs / "b" / "config.json").write_text("{}", encoding="utf-8")
    spec["runs"].append({"name": "c", "model": "mlp", "dataset": "cifake", "set": {**FAST, "seed": 5}})
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    sweep.main(["--file", str(path), "--runs-dir", str(runs)])
    assert (runs / "b" / "failed.txt").exists() and (runs / "c" / "done.json").exists()


@pytest.fixture(scope="module")
def artifact_runs(prepared, tmp_path_factory):
    original = src.config.BASE_DIR
    src.config.BASE_DIR = prepared
    try:
        runs = tmp_path_factory.mktemp("runs")
        dirs = []
        for model, extra in (("mlp", {}), ("cnn", {}), ("resnet18", {"pretrained": False})):
            run(model, "artifact", {**FAST, **extra}, runs / f"artifact-{model}")
            dirs.append(runs / f"artifact-{model}")
        return dirs
    finally:
        src.config.BASE_DIR = original


def test_evaluate_writes_test_and_unseen(artifact_runs):
    results = evaluate.evaluate_run(artifact_runs[1], device=CPU, num_workers=0)
    assert set(results) == {"test", "unseen", "speed"}
    assert results["unseen"]["n"] == 10 + results["test"]["n"] // 2
    rows = (artifact_runs[1] / "per_source.csv").read_text(encoding="utf-8")
    assert rows.count("unseen,ddpm") == 1 and "test,coco,REAL" in rows
    assert results["speed"]["throughput_imgs_per_sec"] > 0 and results["speed"]["latency_ms_p50"] > 0


def test_export_onnx_manifest_and_end_to_end_parity(artifact_runs, tmp_path, artifact_zip, capsys):
    out = tmp_path / "exports"
    export_onnx.main([*map(str, artifact_runs), "--out", str(out)])
    manifest = json.loads((out / "models.json").read_text(encoding="utf-8"))
    assert list(manifest) == ["schema", "classes", "default_model", "uncertain_band", "exported_at",
                              "preprocess_sha256", "models"]
    assert manifest["schema"] == 1 and manifest["classes"] == ["REAL", "FAKE"]
    assert manifest["default_model"] == "resnet18" and manifest["uncertain_band"] == 0.1
    assert set(manifest["models"]) == {"mlp", "cnn", "resnet18"}
    source = Path(src.data.preprocess.__file__).read_bytes().replace(b"\r\n", b"\n")
    assert manifest["preprocess_sha256"] == hashlib.sha256(source).hexdigest()
    for name, entry in manifest["models"].items():
        assert list(entry) == ["file", "version", "input_size", "jpeg_quality", "input_name", "output_name", "sha256"]
        assert entry["input_size"] == 200 and entry["jpeg_quality"] == 90 and entry["file"] == f"{name}.onnx"
        assert entry["sha256"] == hashlib.sha256((out / entry["file"]).read_bytes()).hexdigest()
    assert not list(out.glob("*.data")) and not list(out.glob("*.tmp*"))
    card = (out / "MODEL_CARD.md").read_text(encoding="utf-8")
    assert "## cnn" in card and "ONNX Runtime en CPU" in card and "MB" in card


    with zipfile.ZipFile(artifact_zip) as zf:
        original = zf.read("glide/fake/img_0.jpg")
    cfg, model = evaluate.load_run(artifact_runs[2], CPU)
    session = ort.InferenceSession(str(out / "resnet18.onnx"), providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "input" and session.get_outputs()[0].name == "logits"
    x = to_model_input(original, 200, 90)
    got = session.run(["logits"], {"input": x})[0]
    with torch.no_grad():
        stored = harmonize(load_rgb(original), 200, 90)
        expected = model(torch.from_numpy(decode(stored)).permute(2, 0, 1)[None].float() / 255).numpy()
    assert got.shape == (1, 2) and np.abs(got - expected).max() < 1e-3

    assert session.run(["logits"], {"input": np.repeat(x, 3, axis=0)})[0].shape == (3, 2)


    image = tmp_path / "foto.jpg"
    image.write_bytes(original)
    capsys.readouterr()
    predict.main([str(image), "--models-dir", str(out)])
    lines = capsys.readouterr().out.strip().splitlines()
    assert [l.split()[0] for l in lines] == ["mlp", "cnn", "resnet18"] and all("prob_ai" in l for l in lines)
    result = predict.predict(manifest, out, "resnet18", original)
    p = float(np.exp(got[0] - got[0].max())[1] / np.exp(got[0] - got[0].max()).sum())
    assert abs(result["prob_ai"] - p) < 1e-3


def test_export_to_onnx_rejects_mismatch(tmp_path, monkeypatch):
    from src.models import build_model
    model = build_model("cnn", 32, base_channels=4)
    monkeypatch.setattr(export_onnx, "TOLERANCE", -1.0)
    with pytest.raises(SystemExit, match="difieren"):
        export_onnx.export_to_onnx(model, tmp_path / "cnn.onnx", 32)
    assert not list(tmp_path.glob("*.onnx"))
