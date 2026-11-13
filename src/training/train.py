import argparse
import json
import random
import numpy as np
import torch
import yaml
from src.config import BASE_DIR, run_config
from src.data.dataset import create_dataloaders
from src.models import SUPPORTED_MODELS, build_model
from src.training.trainer import Trainer

RUNS_DIR = BASE_DIR / "runs"


def parse_overrides(pairs):
    overrides = {}
    for pair in pairs or []:
        key, _, value = pair.partition("=")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def main(argv=None):

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset", default="cifake", choices=["cifake", "artifact"])
    parser.add_argument("--set", action="append", metavar="CLAVE=VALOR")
    parser.add_argument("--name")
    args = parser.parse_args(argv)

    try:
        cfg = run_config(args.model, args.dataset, parse_overrides(args.set))
    except ValueError as error:
        parser.error(str(error))
    ta = cfg["train"]
    image_size = cfg["image_size"]

    run_dir = RUNS_DIR / (args.name or f"{args.dataset}-{args.model}-manual")
    if (run_dir / "done.json").exists():
        print(f"{run_dir.name}: ya terminada, se omite")
        return json.loads((run_dir / "done.json").read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    if cfg_path.exists() and json.loads(cfg_path.read_text(encoding="utf-8")) != json.loads(json.dumps(cfg)):
        raise SystemExit(f"{run_dir} tiene otra configuración. Borrá la carpeta o usá otro nombre.")
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])

    device = torch.device(ta["device"] if torch.cuda.is_available() or ta["device"] != "cuda" else "cpu")
    print(f"Entrenando {args.model} en {device} con dataset {args.dataset}...")

    model = build_model(args.model, image_size=image_size, **cfg["model_args"])

    train_loader, val_loader = create_dataloaders(cfg["processed_dir"], image_size=image_size, batch_size=ta["batch_size"],
                                                  num_workers=ta["num_workers"], limit=ta["limit"], seed=cfg["seed"])
    trainer = Trainer(model, args.model, device, lr=ta["learning_rate"], image_size=image_size,
                      weight_decay=ta["weight_decay"], mixed_precision=ta["mixed_precision"],
                      label_smoothing=ta["label_smoothing"], warmup_epochs=ta["warmup_epochs"],
                      early_stopping_patience=ta["early_stopping_patience"], seed=cfg["seed"])
    summary = trainer.fit(train_loader, val_loader, epochs=ta["epochs"], weights_dir=run_dir)

    summary.update(
        name=run_dir.name, model=args.model, dataset=args.dataset,
        params=sum(p.numel() for p in model.parameters()),
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        device=torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        torch=torch.__version__,
    )
    (run_dir / "done.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "last.pt").unlink(missing_ok=True)
    return summary


if __name__ == "__main__":
    main()
