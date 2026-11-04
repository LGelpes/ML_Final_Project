# Ficha de los modelos

Detectores de imágenes generadas por IA. Salida: probabilidad de que la imagen sea generada (prob_ai). Entre 0.40 y 0.60 la API responde 'uncertain'.

Uso previsto: demo académica. No usar para decidir sobre personas ni como prueba de autenticidad. Solo vio fotos y generadores de CIFAKE y ArtiFact recortados a cuadrado; con otros generadores, capturas de pantalla o imágenes muy editadas puede fallar sin aviso. Varias fuentes de ArtiFact tienen licencias no comerciales.

## mlp

- Versión: `artifact-mlp-wd1e-2@2026-11-03` — archivo `mlp.onnx` (15.3 MB, sha256 `ae92f335a37e8616...`)
- Dataset: artifact (imágenes por split: {'test': 99995, 'train': 799933, 'unseen': 4000, 'val': 99995})
- Entrada: 200 x 200, JPEG calidad 90, recorte cuadrado central
- Hiperparámetros: {"hidden_dims": [512, 256, 128], "dropout": 0.3, "pool": 4, "batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.01, "epochs": 30, "label_smoothing": 0.0}
- Validación (época 27): ROC-AUC 0.6637, F1 0.5876, accuracy 0.6186
- Test: {"accuracy": 0.6179, "f1": 0.5876, "precision": 0.6381, "recall": 0.5445, "roc_auc": 0.6629, "n": 99995, "gpu_mb": 369.3}
- Generadores no vistos: {"accuracy": 0.6641, "f1": 0.1256, "precision": 0.0778, "recall": 0.3257, "roc_auc": 0.5461, "n": 54000, "gpu_mb": 369.3}
- ONNX Runtime en CPU, 1 imagen: p50 0.15 ms (medido en la PC de entrenamiento; en Lambda se mide aparte)

## cnn

- Versión: `artifact-cnn-c64@2026-11-03` — archivo `cnn.onnx` (5.9 MB, sha256 `68309bf27c1cbcc0...`)
- Dataset: artifact (imágenes por split: {'test': 99995, 'train': 799933, 'unseen': 4000, 'val': 99995})
- Entrada: 200 x 200, JPEG calidad 90, recorte cuadrado central
- Hiperparámetros: {"base_channels": 64, "dropout": 0.4, "batch_size": 128, "learning_rate": 0.001, "weight_decay": 0.0001, "epochs": 30, "label_smoothing": 0.0}
- Validación (época 30): ROC-AUC 0.9257, F1 0.846, accuracy 0.8444
- Test: {"accuracy": 0.8439, "f1": 0.8456, "precision": 0.8368, "recall": 0.8545, "roc_auc": 0.926, "n": 99995, "gpu_mb": 3054.5}
- Generadores no vistos: {"accuracy": 0.8403, "f1": 0.4625, "precision": 0.3081, "recall": 0.9275, "roc_auc": 0.9524, "n": 54000, "gpu_mb": 3054.5}
- ONNX Runtime en CPU, 1 imagen: p50 4.08 ms (medido en la PC de entrenamiento; en Lambda se mide aparte)

## resnet18

- Versión: `artifact-resnet18-lr1e-4@2026-11-03` — archivo `resnet18.onnx` (42.7 MB, sha256 `e20629d2973a1e46...`)
- Dataset: artifact (imágenes por split: {'test': 99995, 'train': 799933, 'unseen': 4000, 'val': 99995})
- Entrada: 200 x 200, JPEG calidad 90, recorte cuadrado central
- Hiperparámetros: {"pretrained": true, "freeze_backbone": false, "dropout": 0.3, "batch_size": 128, "learning_rate": 0.0001, "weight_decay": 0.0001, "epochs": 30, "label_smoothing": 0.0}
- Validación (época 30): ROC-AUC 0.9775, F1 0.9245, accuracy 0.925
- Test: {"accuracy": 0.9233, "f1": 0.9228, "precision": 0.929, "recall": 0.9167, "roc_auc": 0.9775, "n": 99995, "gpu_mb": 3057.4}
- Generadores no vistos: {"accuracy": 0.928, "f1": 0.6504, "precision": 0.5079, "recall": 0.9038, "roc_auc": 0.9751, "n": 54000, "gpu_mb": 3057.4}
- ONNX Runtime en CPU, 1 imagen: p50 3.08 ms (medido en la PC de entrenamiento; en Lambda se mide aparte)
