from fastapi import FastAPI
from mangum import Mangum
from src.api.inference import predict
from src.api.schemas import ImageRequest

app = FastAPI(title="Synthetic Image Detection API (ONNX)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/mlp")
def predict_mlp(req: ImageRequest):
    return predict("mlp", req.image_base64)


@app.post("/predict/cnn")
def predict_cnn(req: ImageRequest):
    return predict("cnn", req.image_base64)


@app.post("/predict/resnet18")
def predict_resnet18(req: ImageRequest):
    return predict("resnet18", req.image_base64)


lambda_handler = Mangum(app, lifespan="off")
