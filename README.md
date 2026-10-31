# Clasificación de Imágenes Sintéticas vs. Reales (PyTorch GPU → ONNX → AWS Lambda)

Proyecto Final de **Machine Learning (Universidad de Montevideo)** — Detección de imágenes generadas por IA frente a fotografías reales empleando **CIFAKE** y **ArtiFact**.

Clases: `REAL = 0`, `FAKE = 1`. `prob_ai = P(FAKE)` y todas las métricas (F1, precisión, recall, ROC-AUC) toman FAKE como clase positiva.

---

## Estructura del Proyecto

```text
ML/
├── configs/
│   ├── config.yaml              # Datos, entrenamiento y modelos (todo se lee de acá)
│   └── sweep.yaml               # Barrido de hiperparámetros (38 corridas, en orden de prioridad)
├── data/
│   ├── raw/                     # Datasets originales (ignorado por Git)
│   └── processed/<dataset>/<split>/<REAL|FAKE>/<fuente>__<hash>.jpg   # imágenes ya armonizadas
├── runs/<corrida>/              # config.json, metrics.csv, best.pt, last.pt, done.json, test_metrics.json
├── exports/                     # ONNX, models.json y MODEL_CARD.md para la Lambda
├── src/
│   ├── config.py                # Configuración, clases y armado de la configuración de cada corrida
│   ├── predict.py               # python -m src.predict <imagen>: prob_ai y label con los ONNX
│   ├── data/
│   │   ├── preprocess.py        # ÚNICO preprocesamiento (Pillow + NumPy); la Lambda lo copia tal cual
│   │   ├── preprocessing.py     # harmonize_jpeg y get_transforms (volteo + tensor)
│   │   ├── prepare_cifake.py    # CIFAKE: splits train / val (10 % de train) / test
│   │   ├── prepare_artifact.py  # ArtiFact: lee directo del zip; peek | list | extract
│   │   └── dataset.py           # ImageFolder con REAL = 0, FAKE = 1 y DataLoaders
│   ├── models/                  # MLPClassifier, CustomCNN, ResNet18Classifier y Detector (normaliza dentro)
│   └── training/
│       ├── metrics.py           # Accuracy, F1, precisión, recall y ROC-AUC (FAKE positiva)
│       ├── trainer.py           # Entrenamiento, early stopping por ROC-AUC y reanudación
│       ├── train.py             # Una corrida
│       ├── sweep.py             # Barrido reanudable
│       ├── evaluate.py          # Test, generadores no vistos, aciertos por fuente y velocidad
│       └── export_onnx.py       # ONNX (un solo archivo), models.json y MODEL_CARD.md
├── tests/                       # pytest en CPU con datos sintéticos
└── requirements.txt             # Sin PyTorch: se instala aparte desde pytorch.org
```

La API (`/v1/predict`) vive en el repo de AWS; acá solo se entrena y se exporta.

---

## Uso (PowerShell, Python 3.14, desde la raíz del repo)

PyTorch 2.15 se instala aparte desde [pytorch.org](https://pytorch.org/get-started/locally/); después:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest

.venv\Scripts\python.exe -m src.data.prepare_cifake --src D:\datasets\cifake-raw
.venv\Scripts\python.exe -m src.data.prepare_artifact peek --zip D:\datasets\artifact-dataset.zip
.venv\Scripts\python.exe -m src.data.prepare_artifact list --zip D:\datasets\artifact-dataset.zip
.venv\Scripts\python.exe -m src.data.prepare_artifact extract --zip D:\datasets\artifact-dataset.zip --holdout <generador-1> <generador-2>

.venv\Scripts\python.exe -m src.training.train --model cnn --dataset cifake --set epochs=1 --name humo-cifake-cnn
.venv\Scripts\python.exe -m src.training.sweep --dry-run
.venv\Scripts\python.exe -m src.training.sweep
.venv\Scripts\python.exe -m src.training.sweep --summary

.venv\Scripts\python.exe -m src.training.evaluate runs\<corrida-elegida>
.venv\Scripts\python.exe -m src.training.export_onnx runs\<mlp> runs\<cnn> runs\<resnet18> --out exports\final
.venv\Scripts\python.exe -m src.predict foto.jpg --models-dir exports\final
```

ResNet-18 se usa solo con ArtiFact. Si una preparación ya existe, `prepare_*` pide `--force` para rehacerla.
