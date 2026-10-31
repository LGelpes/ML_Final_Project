from torchvision import transforms
from src.data.preprocess import harmonize_jpeg  # noqa: F401


def get_transforms(image_size: int = 32, is_train: bool = False):
    steps = []
    if is_train:
        steps.append(transforms.RandomHorizontalFlip())
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)
