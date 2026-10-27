import base64
import io
import numpy as np
import onnxruntime as ort
from PIL import Image
from src.config import BASE_DIR, CLASSES, IMAGE_MEAN, IMAGE_STD

SESSIONS = {}


def get_session(model_name: str) -> ort.InferenceSession:
    if model_name not in SESSIONS:
        onnx_path = BASE_DIR / "weights" / f"{model_name}.onnx"
        SESSIONS[model_name] = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return SESSIONS[model_name]


def preprocess_image(b64_str: str, size: int = 32) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    img = img.resize((size, size))

    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGE_MEAN) / IMAGE_STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def predict(model_name: str, b64_str: str) -> dict:
    session = get_session(model_name)
    size = session.get_inputs()[0].shape[2]
    x = preprocess_image(b64_str, size=size if isinstance(size, int) else 32)

    logits = session.run(None, {"input": x})[0][0]
    exp = np.exp(logits - np.max(logits))
    probs = exp / np.sum(exp)
    idx = int(np.argmax(probs))

    return {
        "model": model_name,
        "prediction": CLASSES[idx],
        "confidence": round(float(probs[idx]), 4),
    }
