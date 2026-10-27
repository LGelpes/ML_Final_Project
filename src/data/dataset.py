from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.data.preprocessing import get_transforms


def create_dataloaders(dataset_root: str | Path, image_size: int = 32, batch_size: int = 64):
    
    root = Path(dataset_root)
    val_folder = "val" if (root / "val").exists() else "test"

    train_ds = ImageFolder(root / "train", transform=get_transforms(image_size, is_train=True))
    val_ds = ImageFolder(root / val_folder, transform=get_transforms(image_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader
