import argparse
import torch
from src.config import BASE_DIR, load_config
from src.data.dataset import create_dataloaders
from src.models import SUPPORTED_MODELS, build_model
from src.training.trainer import Trainer


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--dataset", default="cifake", choices=["cifake", "artifact"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config()
    ds_cfg = cfg["datasets"][args.dataset]
    image_size = ds_cfg["image_size"]
    dataset_root = BASE_DIR / ds_cfg["root_dir"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando {args.model} en {device} con dataset {args.dataset}...")

    model = build_model(args.model, image_size=image_size, freeze_backbone=args.freeze_backbone)
    
    if args.resume:
        model.load_state_dict(torch.load(args.resume, map_location=device, weights_only=True))

    train_loader, val_loader = create_dataloaders(dataset_root, image_size=image_size)
    trainer = Trainer(model, args.model, device, lr=args.lr, image_size=image_size)
    trainer.fit(train_loader, val_loader, epochs=args.epochs, weights_dir=BASE_DIR / "weights")


if __name__ == "__main__":
    main()
