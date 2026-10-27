import io
from PIL import Image
from torchvision import transforms
from src.config import IMAGE_MEAN, IMAGE_STD


def harmonize_jpeg(img: Image.Image, quality: int = 90) -> Image.Image:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def get_transforms(image_size: int = 32, is_train: bool = False):
    steps = [
        transforms.Lambda(harmonize_jpeg),
        transforms.Resize((image_size, image_size)),
    ]
    if is_train:
        steps.append(transforms.RandomHorizontalFlip())

    steps.extend([
        transforms.ToTensor(),
        transforms.Normalize(IMAGE_MEAN, IMAGE_STD),
    ])
    return transforms.Compose(steps)
