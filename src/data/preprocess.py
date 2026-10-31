# Se copia tal cual a la Lambda: solo Pillow y NumPy. La normalización va dentro del modelo (Detector).
import base64
import io

import numpy as np
from PIL import Image, ImageOps


def load_rgb(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as img:
        return ImageOps.exif_transpose(img).convert("RGB")


def center_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def encode_jpeg(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, subsampling=2)
    return buf.getvalue()


def harmonize_jpeg(img: Image.Image, quality: int = 90) -> Image.Image:
    with Image.open(io.BytesIO(encode_jpeg(img, quality))) as out:
        return out.convert("RGB")


def harmonize(img: Image.Image, size: int, quality: int) -> bytes:
    img = center_square(img)
    if img.size != (size, size):
        img = img.resize((size, size), Image.Resampling.BICUBIC)
    return encode_jpeg(img, quality)


def decode(jpeg_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(jpeg_bytes)) as img:
        return np.array(img.convert("RGB"))


def to_model_input(data: bytes, size: int, quality: int) -> np.ndarray:
    pixels = decode(harmonize(load_rgb(data), size, quality))
    return (pixels.astype(np.float32) / 255.0).transpose(2, 0, 1)[None].copy()


def preprocess_image(b64_str: str, size: int, quality: int) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    return to_model_input(base64.b64decode(b64_str), size, quality)
