import copy
from pathlib import Path
import yaml

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "configs" / "config.yaml"

CLASSES = ["REAL", "FAKE"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)

ALIASES = {"lr": "learning_rate", "amp": "mixed_precision", "patience": "early_stopping_patience"}

INVALID_COMBINATIONS = {
    ("resnet18", "cifake"): "ResNet-18 se usa solo con ArtiFact (200x200); con CIFAKE (32x32) llega a 1 píxel.",
}


def load_config(path=CONFIG_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_config(model: str, dataset: str, overrides: dict | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    if model not in cfg["models"]:
        raise ValueError(f"Modelo desconocido: {model}")
    if dataset not in cfg["datasets"]:
        raise ValueError(f"Dataset desconocido: {dataset}")
    if (model, dataset) in INVALID_COMBINATIONS:
        raise ValueError(f"Combinación no válida {model} + {dataset}: {INVALID_COMBINATIONS[(model, dataset)]}")

    model_args = copy.deepcopy(cfg["models"][model])
    by_dataset = model_args.pop("by_dataset", None) or {}
    model_args.update(by_dataset.get(dataset, {}))
    for path_key in ("weights_path", "onnx_path"):
        model_args.pop(path_key, None)
    train_args = copy.deepcopy(cfg["training"])
    seed = cfg["project"]["seed"]

    for key, value in (overrides or {}).items():
        key = ALIASES.get(key, key)
        if isinstance(value, str):
            try:
                value = float(value)
            except ValueError:
                pass
        if key == "seed":
            seed = int(value)
        elif key in train_args:
            train_args[key] = value
        elif key in model_args:
            model_args[key] = value
        else:
            raise ValueError(f"Parámetro desconocido para {model}: {key}")

    ds = cfg["datasets"][dataset]
    return {
        "model": model,
        "dataset": dataset,
        "image_size": ds["image_size"],
        "jpeg_quality": ds["jpeg_quality"],
        "processed_dir": str(BASE_DIR / ds["processed_dir"]),
        "seed": seed,
        "model_args": model_args,
        "train": train_args,
    }
