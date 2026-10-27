# Clasificación de Imágenes Sintéticas vs. Reales (PyTorch GPU → ONNX → AWS Lambda)

Proyecto Final de **Machine Learning (Universidad de Montevideo)** — Detección de imágenes generadas por IA frente a fotografías reales empleando **CIFAKE** y **ArtiFact**.

---

## Estructura del Proyecto

```text
ML/
├── configs/
│   └── config.yaml              # Configuración de datasets y rutas de modelos
├── data/
│   ├── raw/                     # Datasets: data/raw/cifake y data/raw/artifact
│   └── processed/
├── weights/                     # Pesos entrenados (.pt locales y .onnx para Docker/Lambda)
├── src/
│   ├── config.py                # Carga de configuración y constantes
│   ├── data/
│   │   ├── preprocessing.py     # Armonización JPEG y transformaciones
│   │   └── dataset.py           # Carga de datos con ImageFolder y DataLoader
│   ├── models/
│   │   ├── __init__.py          # Factoría build_model("mlp" | "cnn" | "resnet18")
│   │   ├── mlp.py               # Modelo 1: MLPClassifier
│   │   ├── cnn.py               # Modelo 2: CustomCNN
│   │   └── resnet.py            # Modelo 3: ResNet18Classifier
│   ├── training/
│   │   ├── metrics.py           # Accuracy, F1, ROC-AUC y memoria GPU
│   │   ├── export_onnx.py       # Exportación de PyTorch (.pt) a ONNX (.onnx)
│   │   ├── trainer.py           # Entrenamiento en GPU y guardado automático de .pt y .onnx
│   │   └── train.py             # Script CLI de entrenamiento
│   └── api/
│       ├── schemas.py           # Esquema de petición Base64
│       ├── inference.py         # Inferencia con ONNX Runtime (sin PyTorch)
│       └── app.py               # Endpoints FastAPI (/predict/mlp, /predict/cnn, /predict/resnet18) + Mangum
├── Dockerfile.train             # Imagen con PyTorch + CUDA para entrenar en GPU
├── Dockerfile                   # Imagen ultraligera para AWS Lambda (ONNX Runtime)
├── docker-compose.yml           # Ejecución de entrenamiento en GPU y prueba local de API
├── .dockerignore                # Ignora data/ y archivos .pt al construir la imagen final
├── requirements.txt             # Dependencias de entrenamiento GPU y exportación ONNX
└── requirements-lambda.txt      # Dependencias de la imagen final para AWS Lambda
```

---

## 1. Entrenar con `Dockerfile.train` (usando GPU)

```powershell
# Construir la imagen de entrenamiento con soporte CUDA
docker build -f Dockerfile.train -t ml-trainer .

# Entrenar cualquiera de los 3 modelos montando data/ y weights/ (guarda .pt y .onnx en weights/)
docker run --rm --gpus all -v ${PWD}/data:/app/data -v ${PWD}/weights:/app/weights ml-trainer python -m src.training.train --model mlp --dataset cifake --epochs 20

docker run --rm --gpus all -v ${PWD}/data:/app/data -v ${PWD}/weights:/app/weights ml-trainer python -m src.training.train --model cnn --dataset cifake --epochs 20

docker run --rm --gpus all -v ${PWD}/data:/app/data -v ${PWD}/weights:/app/weights ml-trainer python -m src.training.train --model resnet18 --dataset cifake --epochs 20
```
*(Opcionalmente también puedes usar `docker compose run --rm trainer python -m src.training.train --model mlp --dataset cifake`).*

---

## 2. Construir y Subir Manualmente `Dockerfile` a Amazon ECR

Una vez que `weights/` tenga los archivos `mlp.onnx`, `cnn.onnx` y `resnet18.onnx`:

```powershell
# 1. Autenticar Docker con AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

# 2. Construir la imagen final de AWS Lambda (usa Dockerfile por defecto)
docker build --platform linux/amd64 --provenance=false -t ml-lambda .

# 3. Etiquetar la imagen
docker tag ml-lambda:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<REPO_NAME>:latest

# 4. Subir la imagen a ECR
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/<REPO_NAME>:latest
```

---

## 3. Endpoints Expuestos en AWS Lambda + API Gateway

* `GET /health`
* `POST /predict/mlp` — Body JSON: `{"image_base64": "..."}`
* `POST /predict/cnn` — Body JSON: `{"image_base64": "..."}`
* `POST /predict/resnet18` — Body JSON: `{"image_base64": "..."}`
