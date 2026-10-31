from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from src.config import CLASS_TO_IDX, CLASSES
from src.data.preprocessing import get_transforms
from src.data.prepared import source_of


class PreparedFolder(ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform, allow_empty=True)

    def find_classes(self, directory):
        return list(CLASSES), dict(CLASS_TO_IDX)

    @property
    def sources(self):
        return [source_of(path) for path, _ in self.samples]


def create_dataset(dataset_root: str | Path, split: str, image_size: int = 32, is_train: bool = False,
                   limit: int | None = None, seed: int = 0):
    folder = Path(dataset_root) / split
    if not folder.exists():
        raise FileNotFoundError(f"No existe {folder}: preparar los datos con src.data.prepare_cifake o prepare_artifact")
    ds = PreparedFolder(folder, transform=get_transforms(image_size, is_train=is_train))
    if limit and limit < len(ds):
        idx = sorted(np.random.default_rng(seed).choice(len(ds), limit, replace=False).tolist())
        sub = Subset(ds, idx)
        sub.sources = [ds.sources[i] for i in idx]
        return sub
    return ds


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0, seed: int = 0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=shuffle and len(dataset) > batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )


def create_dataloaders(dataset_root: str | Path, image_size: int = 32, batch_size: int = 64,
                       num_workers: int = 2, limit: int | None = None, seed: int = 0):
    train_ds = create_dataset(dataset_root, "train", image_size, True, limit, seed)
    val_ds = create_dataset(dataset_root, "val", image_size, False, limit, seed)
    train_loader = make_loader(train_ds, batch_size, True, num_workers, seed)
    val_loader = make_loader(val_ds, batch_size * 2, False, num_workers, seed)
    return train_loader, val_loader
