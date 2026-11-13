# Guía para el LLM asistente — proyecto ML (detector de imágenes generadas por IA)

Leé este archivo entero antes de hacer nada. Está escrito para ponerte al día desde cero: contexto, forma de trabajo del usuario, estado actual, arquitectura, restricciones duras, trampas conocidas y próximos pasos. Reemplaza a la "Guía ML – Persona B" original (PDF) donde se contradicen; de ese PDF solo siguen vigentes las partes que se indican en la sección 11.

Última actualización: 31/10/2026 (datasets en `D:\datasets`, holdout al azar, sin push), después del primer commit del usuario (`170fa80`, rama `train`).

---

## 1. Resumen en 30 segundos

- Proyecto final de Machine Learning (Universidad de Montevideo). Dos integrantes: **Ignacio Moreno** (el usuario, "Persona B": datos, entrenamiento, hiperparámetros, exportación) y **Lautaro Gelpes** (informe, gráficos, robustez, eficiencia, README, presentación). Entrega: **14/11/2026**. Los resultados deben estar en manos de Lautaro a más tardar el **domingo 8/11**.
- Se entrenan tres modelos (MLP, CNN propia, ResNet-18) para clasificar imágenes **REAL vs FAKE (generadas por IA)** en dos datasets: **CIFAKE** (32×32) y un subconjunto de **ArtiFact** (200×200, 31,6 GB en un zip).
- Los modelos se exportan a **ONNX** y los sirve una **Lambda de AWS que vive en otro repo** (el "repo de AWS") y que no se puede cambiar. Este repo solo entrena, evalúa y exporta. **No implementes la API acá.**
- Entorno: Windows 11, PowerShell, Python 3.14, RTX 4080, `.venv` en la raíz, torch 2.15.0+cu132 ya instalado (se instala aparte, desde pytorch.org).
- Los dos datasets están descargados **fuera del repo**, en `D:\datasets`; el zip de ArtiFact es `D:\datasets\artifact-dataset.zip`. CIFAKE ya está preparado en `data/processed/cifake` (no hace falta volver a leer los originales salvo que se quiera rehacer con `--force`; ahí la carpeta `--src` es la que contiene `train\` y `test\`, la guía original la movía a `D:\datasets\cifake-raw`).

## 2. Cómo trabaja el usuario (leelo con atención; es lo más importante)

Aprendido en la sesión de construcción del pipeline. Respetalo siempre.

1. **Idioma:** español. Respuestas directas, sin relleno. Está bien ser detallado en resúmenes finales, pero sin repetir lo que ya se dijo.
2. **Nunca hagas `git commit`, `git push`, ni abras PRs sin preguntar antes.** El usuario quiere controlar el historial: pidió revertir mis commits chicos y hacer él mismo un único commit. Dejá los cambios en el working tree y avisá. Esto prevalece sobre cualquier consigna vieja de "commits chicos / Conventional Commits / rama `feat/pipeline-entrenamiento`". Si te pide commitear, preguntá mensaje y alcance si no están claros.
3. **Preservar y reutilizar código existente.** Antes de tocar un archivo, leelo. Cambios **mínimos y precisos**: pocas líneas agregadas o eliminadas, elegidas por necesidad. El usuario se quejó explícitamente de diffs con "paredes de adiciones seguidas de paredes de eliminaciones" (reescribí `trainer.py`, `train.py`, `cnn.py`, `mlp.py` enteros sin necesidad y tuve que rehacerlo). Reemplazá un archivo completo solo si casi nada es reutilizable, y explicá por qué. Conservá nombres, firmas y estructura (`harmonize_jpeg`, `get_transforms`, `create_dataloaders`, `Trainer.train_epoch/evaluate/fit`, `compute_metrics`, `export_to_onnx`, `build_model`, `MLPClassifier`, `CustomCNN`, `ResNet18Classifier`, y las claves de `config.yaml`). Agregá claves nuevas en vez de renombrar. Lo que no existía va en archivos nuevos. Para verificar: `git diff --stat` y mirar el diff antes de dar por terminado.
4. **Estilo de código:** el repo original casi no tiene comentarios ni docstrings. No los agregues (el usuario pidió quitar casi todos). Comentá solo un "por qué" no obvio (hoy hay 2). Mismo estilo de los archivos vecinos: type hints simples, líneas en blanco como en el original.
5. **Si algo choca con el código, con la guía o no está claro, preguntá antes de decidir** (herramienta de preguntas, con una recomendación). No inventes.
6. **Entregables de cada tarea grande:** resumen con (a) por archivo existente qué se preservó y qué cambió, (b) archivos nuevos y borrados, (c) comandos o pasos de la guía que ya no funcionan igual y su reemplazo, (d) decisiones tomadas y por qué.
7. **Pruebas:** `pytest` en CPU con datos sintéticos; **no bajes datasets**; ResNet-18 con `pretrained=False` en tests. Corré `.venv\Scripts\python.exe -m pytest -p no:logging` después de cada cambio de código (tarda ~20 s; sin `-p no:logging` el exportador de torch inunda la salida de DEBUG).
8. **Windows:** `if __name__ == "__main__"` en los scripts, `encoding="utf-8"` al abrir archivos, `DataLoader` con `multiprocessing_context="spawn"` debe funcionar, siempre `python -m src...` desde la raíz, usar `.venv\Scripts\python.exe` sin activar el venv.
9. **Herramientas:** para crear/editar archivos grandes usá Write/Edit, no heredocs de Python en Bash con triple comillas (falló una vez). Para el PDF de la guía: está protegido con contraseña de propietario; la herramienta Read falla, pero `pdftotext -layout "D:\Nacho\Downloads\Guía ML - Persona B entrenamiento y API.pdf" salida.txt` funciona (guardá la salida en el scratchpad, no en el repo; el texto sale en cp1252).
10. **Seguridad/alcance:** no hagas acciones externas (push, subir a AWS, mandar mensajes) sin pedirlo. No copies archivos al repo de AWS: dale al usuario los comandos.
11. Hay memoria persistente en `C:\Users\Nacho\.claude\projects\D--Nacho-Desktop-ML-Final-Project\memory\` (por ahora solo la preferencia de no commitear sin consultar). Actualizala si aprendés preferencias nuevas.

## 3. Estado actual (al 31/10/2026)

Hecho y verificado:
- Pipeline completo implementado y commiteado por el usuario (`170fa80`, rama `train`; **todavía sin push**, `main` y `origin/main` tienen solo `7dc2d24`).
- **30 pruebas pytest pasan** (preparación de ambos datasets, una época de cada combinación válida, reanudación tras corte, early stopping, barrido que omite lo terminado y registra fallas, evaluación con `unseen`, exportación con paridad PyTorch/ONNX Runtime, paridad de punta a punta, workers `spawn`, CLI `predict`).
- Corrida corta en la RTX 4080 con datos sintéticos (bf16, workers spawn): funciona.
- **CIFAKE ya preparado** en `data/processed/cifake` (train 88 742, val 9 960, test 19 962 imágenes; menos que las ~90 000/10 000/20 000 esperadas porque se descartan duplicados exactos por SHA-256, incluso dentro del mismo split).
- **Humo de CIFAKE hecho** para `mlp` y `cnn` (`runs/humo-cifake-mlp`, `runs/humo-cifake-cnn`, 1 época): MLP val ROC-AUC 0,866; CNN val ROC-AUC 0,976. Velocidad medida ≈ 3 500 img/s en ambos, ≈ 25 s por época, uso de GPU mínimo (38–64 MB). **Esa cifra era engañosa:** casi todo era el arranque de los 6 workers `spawn` en cada época (train y val), no la lectura de JPEG; con `persistent_workers` (31/10) la época real dura unos pocos segundos (ver sección 10). Las estimaciones de tiempo de ArtiFact hay que rehacerlas con el humo.
- **Barrido de CIFAKE terminado (31/10)**: 15/15 corridas, sin fallas, ~2,8 min por corrida (~4,9 s por época, ~18 000 img/s; la época 1 tarda ~24 s por el arranque único de los workers). Val ROC-AUC: CNN 0,9938–0,9967 (mejor `cifake-cnn-c64`), MLP 0,9388–0,9496 (mejor `cifake-mlp-base`). Variación entre semillas: CNN 0,9951–0,9955, MLP 0,9485–0,9496. Todas las mejores épocas caen en 25–30 (el coseno lleva el LR a 0 en la 30; las curvas ya están planas en el 4.º decimal). La CNN sobreajusta en pérdida (train 0,005–0,02 vs val ~0,10–0,11, que sube desde la época ~15) aunque el ROC-AUC sigue subiendo apenas: probabilidades sobreconfiadas. La MLP no sobreajusta (train 0,24 vs val 0,29) y `drop0.5` empeora, así que está limitada por capacidad. `peek`/`list` de ArtiFact hechos y holdout elegido (sección 12).
- **Elegidas de CIFAKE (31/10, por validación, antes de mirar test):** MLP `cifake-mlp-base` (mejor val ROC-AUC 0,9496, configuración por defecto); CNN `cifake-cnn-lr3e-3` (0,9962): a menos de 0,002 de la mejor (`cifake-cnn-c64`, 0,9967) y con 4 veces menos parámetros (390 mil vs 1,55 M), así que por la regla de "la más simple" queda la de LR 3e-3. **Test (evaluadas 31/10):** MLP ROC-AUC 0,9501, F1 0,8755, acc 0,8759; CNN ROC-AUC 0,9960, F1 0,9717, acc 0,9717 (n = 19 962; test ≈ val, sin señales de sobreajuste a validación). Velocidad en la 4080 (solo el modelo): MLP 652 mil img/s, latencia p50 0,37 ms; CNN 155 mil img/s, p50 1,49 ms. Nota menor: `stopped_early=True` en `cifake-cnn-base` y `cifake-cnn-lr3e-4` es un artefacto (llegaron a 5 épocas sin mejora justo en la 30; corrieron las 30).

- **ArtiFact preparado (31/10, primera versión, la reemplaza la reextracción con 500 000 por clase)** en `data/processed/artifact` con `--holdout vq_diffusion big_gan` y `--per-class 75000`: train 119 982 (60 000 REAL / 59 983 FAKE), val 15 004, test 15 003, unseen 4 000 (solo FAKE, 2 000 por generador). 150 000 es lo esperado (`--per-class 75000` × 2; se perdieron 11 por duplicados). Train: 23 fuentes falsas con 2 694 cada una (`ddpm` 716, entra entera) y 10 reales con 7 056 cada una (`landscape` 3 454, `metfaces` 1 068, `cycle_gan` 6 085, entran enteras).

- **Humo de ArtiFact hecho (31/10)** para los tres modelos (`runs/humo-artifact-<m>`, 1 época, `limit=6000`): val ROC-AUC MLP 0,567, CNN 0,546, ResNet-18 0,695 (esperable con 4 800 imágenes y 1 época); GPU MLP 415 MB, CNN 2 047 MB, ResNet-18 1 472 MB. Su `imgs_per_sec` (~270–290) **no sirve**: la época dura ~21 s y casi todo es el arranque de workers. Medición aparte (31/10, script en el scratchpad, lote 128, bf16): entrenamiento puro MLP ~66 000 img/s, CNN ~2 500, ResNet-18 ~2 600 (la CNN propia es tan lenta como ResNet-18 a 200×200 porque convoluciona a resolución completa); carga de datos ~4 500 img/s con 6 workers (con 12 o 20 workers baja a 3 900 / 3 100: no subir `num_workers`). Con 150 000 imágenes: ~30 s por época (MLP) y ~50 s (CNN/ResNet-18), barrido de 23 corridas ≈ 9 h a 30 épocas. El `extract` de 150 000 tardó ~2,5 min (~1 000 img/s, un solo proceso).

- **Decisión (31/10): ArtiFact con 500 000 por clase** (nuevo valor por defecto de `--per-class` en `prepare_artifact.py`; antes 75 000). Simulado con `select` sobre el zip real: train 400 000 / 400 000, val 50 000 / 49 999, test 50 000 / 49 999, unseen 4 000. Reales: `coco` y `lsun` 23,8 % cada una, `imagenet` 19,4 %, `ffhq` 14 %; falsas: `stylegan2`, `stylegan3`, `taming_transformer` 12,7 % cada una, `mat` 12 %, el resto ≤ 5 %. Se descartó el dataset entero (~965 000 por clase): ~115 h de barrido y mezcla dominada por `lsun` (56 % de las reales) y `stylegan2` (1 M de 1,5 M falsas). Estimado: época ~3,3 min (MLP) y ~6 min (CNN/ResNet-18), barrido de ArtiFact ≈ 60 h a 30 épocas.

- **Barrido de ArtiFact terminado (3/11)**: 23/23 corridas, sin fallas, con datos reextraídos (train 400 000 REAL / 399 933 FAKE, val y test 50 000 / 49 995 cada uno, unseen 4 000). Época: MLP ~3,5 min, CNN ~6,8, CNN c64 ~12, ResNet-18 ~6,7. Val ROC-AUC: ResNet-18 0,9758–0,9775 sin contar `lr1e-3` (0,9674), `ls0.1` (0,9664) y `frozen` (0,7213); CNN 0,8739–0,8919 y `c64` 0,9257; MLP 0,6542–0,6637. Semillas: ResNet-18 0,9758–0,9762, CNN 0,8861–0,8913, MLP 0,6614–0,6627.
- **Elegidas de ArtiFact (3/11, por validación, confirmadas por el usuario antes de mirar test):** `artifact-resnet18-lr1e-4` (0,9775, F1 0,9245), `artifact-cnn-c64` (0,9257, F1 0,846; +0,034 sobre la mejor de 390 mil parámetros, muy por encima de la regla de 0,002), `artifact-mlp-wd1e-2` (0,6637; mismo tamaño que las demás MLP, diferencia dentro del ruido entre semillas). Mismo criterio que en CIFAKE: la mejor por val entre las de la arquitectura más simple que no pierda más de 0,002. Limitaciones a informar: la mejor época de ResNet-18 y CNN c64 es la última (30). ResNet-18 sobreconfiada: pérdida train 0,0009 vs val 0,52 (mínimo ~0,28 en la época 10). `artifact-resnet18-ls0.1` y `artifact-resnet18-frozen` cortaron por early stopping en las épocas 15 y 13 con el LR todavía alto: su ROC-AUC no es comparable con las que llegaron a la 30. La MLP queda cerca del azar: no sobreajusta (0,63 vs 0,647) y las semillas son estables.
- **Test y unseen de ArtiFact (evaluadas 3/11):** ResNet-18 test ROC-AUC 0,9775, F1 0,9228, acc 0,9233; unseen ROC-AUC 0,9751, recall 0,9038. CNN c64 test 0,9260, F1 0,8456, acc 0,8439; unseen 0,9524, recall 0,9275. MLP test 0,6629, F1 0,5876, acc 0,6179; unseen 0,5461, recall 0,3257. Test ≈ val en los tres. Unseen = 4 000 falsas + 50 000 reales de test (7,4 % positivas): **su F1, precisión y accuracy dependen de esa proporción; usar ROC-AUC y recall**. Velocidad en la 4080: ResNet-18 8 052 img/s, p50 3,49 ms; CNN c64 3 380 img/s, p50 0,91 ms; MLP 198 273 img/s, p50 0,91 ms. Por fuente (ResNet-18): generadores no vistos `big_gan` 89,6 %, `vq_diffusion` 91,2 %, en línea con los vistos; débiles `metfaces` REAL 38 % (n = 134), `ddpm` 51 % (n = 90), `cycle_gan` REAL 76 %, `generative_inpainting` 78 %, `diffusion_gan` y `projected_gan` 79 %, `landscape` REAL 82 %; `stable_diffusion` 90 %. La MLP clasifica por contenido más que por huellas de generador: caras reales como falsas (`celebahq` 10 %, `ffhq` 24 %, `metfaces` 8 %) y `sfhq` 99 %.
- **Exportación final hecha (3/11)** en `exports/final` (`--default resnet18 --uncertain-band 0.1`): `mlp.onnx` 15,3 MB, `cnn.onnx` 5,9 MB, `resnet18.onnx` 42,7 MB, paridad PyTorch/ONNX Runtime OK. Comprobación del camino de servicio (2 000 imágenes de test **originales del zip** → `to_model_input` → ONNX, script en el scratchpad): acc ResNet-18 0,9245, CNN 0,8650, MLP 0,6320. Banda `uncertain` (|p − 0,5| < 0,1): ResNet-18 0,55 %, CNN 9 %, MLP 36 %; ResNet-18 da p < 0,01 o > 0,99 en el 92 % de los casos, así que `uncertain` casi no aparece con el modelo por defecto.
- `results/` y `results.zip` regenerados (3/11, sin trackear y no ignorados por Git): 129 archivos, 38 corridas (23 ArtiFact, 15 CIFAKE; el usuario pidió sacar las de humo, también de `summary.csv`; siguen en `runs/`), `test_metrics.json` y `per_source.csv` de las 5 elegidas evaluadas, más `summary.csv`, `sweep.yaml`, `MODEL_CARD.md`, `export_report.json` y `models.json`; sin `.pt` ni `.onnx`. Faltan las latencias de Lambda.

Pendiente (en este orden, ver sección 9): copiar `exports/final` y `preprocess.py` al repo de AWS y medir latencia en Lambda; entrega a Lautaro.

No verificado todavía: los modelos finales en la Lambda real (despliegue, prueba de hashes del repo de AWS, latencia).

## 4. Restricciones duras (contrato con el repo de AWS — no negociables)

La Lambda ya está escrita y consume lo que exporta este repo. Debe cumplirse exactamente:

1. **`src/data/preprocess.py` es el único preprocesamiento.** Lo usan la preparación de datos y la Lambda (que lo copia tal cual). Solo puede depender de **Pillow y NumPy** (nada de torch/torchvision). Expone `to_model_input(data: bytes, size: int, quality: int) -> np.ndarray` float32 `(1, 3, size, size)` en [0, 1], listo para el ONNX. Pasos y orden fijos, iguales al preparar y al servir: **recorte cuadrado central → redimensión bicúbica a `size` → JPEG calidad `quality` (subsampling 4:2:0) → píxeles**. Si cambiás este archivo, hay que **volver a preparar los datos, reentrenar y reexportar**, y copiarlo de nuevo al repo de AWS.
2. **ONNX:** un solo archivo por modelo (`dynamo=True`, `external_data=False` explícitos; en torch 2.15 por defecto los pesos irían a un `.onnx.data`). Batch dinámico, entrada `"input"`, salida `"logits"` `(N, 2)`, **índice 1 = FAKE**. Antes de escribir el archivo definitivo se compara PyTorch vs ONNX Runtime (diferencia máxima < 1e-3; si no, se aborta).
3. **`models.json`** junto a los ONNX, con este formato exacto (la Lambda lo lee y una prueba del repo de AWS, `test_committed_models_match_preprocess`, verifica los hashes):
   `{"schema": 1, "classes": ["REAL","FAKE"], "default_model": "resnet18", "uncertain_band": 0.1, "exported_at": "AAAA-MM-DD", "preprocess_sha256": "<sha256 de preprocess.py con CRLF→LF>", "models": {"<mlp|cnn|resnet18>": {"file", "version": "<corrida>@<fecha>", "input_size", "jpeg_quality", "input_name": "input", "output_name": "logits", "sha256": "<sha256 del .onnx>"}}}`.
   Además `MODEL_CARD.md` (métricas, hiperparámetros, tamaño, latencia ONNX Runtime en CPU) y `export_report.json`. Nunca copiar un `.onnx` sin su `models.json`.
4. **Clases:** `REAL = 0`, `FAKE = 1` en todo el repo. `prob_ai = P(FAKE)`. F1, precisión, recall y ROC-AUC con FAKE como clase positiva.
5. **Versiones:** `Pillow==13.0.0` (idéntica a la capa de la Lambda; la recompresión JPEG depende de ella), NumPy 2.x, Python 3.14, torch 2.15 instalado aparte.
6. Contrato de la API (vive en el repo de AWS, `docs/model-api.yaml` v1.1): `GET /v1/health`, `POST /v1/predict` con `image_b64`, `sha256`, `content_type` y `model` opcional; responde `prob_ai`, `label` (`ai`/`real`/`uncertain`, uncertain si |prob_ai − 0,5| < `uncertain_band`), `model`, `model_version`, `inference_ms`. Lambda: límite 250 MB descomprimidos (capa ~132 MB + modelos ~60 MB, entran).

## 5. Mapa del repo

```
configs/config.yaml        datos, entrenamiento y modelos (todo se lee de acá)
configs/sweep.yaml         barrido: 38 corridas en orden de prioridad (15 cifake + 23 artifact)
src/config.py              CLASSES=["REAL","FAKE"], CLASS_TO_IDX, ALIASES, INVALID_COMBINATIONS, load_config, run_config
src/data/preprocess.py     ÚNICO preprocesamiento (Pillow+NumPy): load_rgb, center_square, encode_jpeg, harmonize_jpeg,
                           harmonize, decode, to_model_input, preprocess_image (base64)
src/data/preprocessing.py  re-exporta harmonize_jpeg; get_transforms(image_size, is_train): flip (train) + ToTensor, SIN normalizar
src/data/prepared.py       sha256, stable_hash, save_prepared (<split>/<REAL|FAKE>/<fuente>__<hash16>.jpg), source_of
src/data/prepare_cifake.py CIFAKE → splits; --src --out --val-fraction --force
src/data/prepare_artifact.py  ArtiFact desde el zip: peek | list | extract (--zip --out --per-class --holdout --holdout-per-source --force)
src/data/dataset.py        PreparedFolder(ImageFolder con REAL=0/FAKE=1 fijo, tolera carpetas faltantes), create_dataset,
                           make_loader, create_dataloaders (train/val; nunca test)
src/models/                MLPClassifier(hidden_dims,pool), CustomCNN(base_channels), ResNet18Classifier(dropout), build_model,
                           Detector (normaliza media/desvío ImageNet DENTRO del modelo; entrada [0,1])
src/training/metrics.py    compute_metrics(y_true, y_pred, y_prob): accuracy,f1,precision,recall,roc_auc,n,gpu_mb
src/training/trainer.py    Trainer: AdamW, warmup+coseno, bf16 en GPU, early stopping por ROC-AUC val, reanudación (last.pt)
src/training/train.py      main(argv): una corrida; guarda en runs/<nombre>/ (config.json, metrics.csv, best.pt, last.pt, done.json)
src/training/sweep.py      barrido reanudable; failed.txt; runs/summary.csv; --only --dry-run --summary
src/training/evaluate.py   test + unseen + per_source.csv + velocidad → test_metrics.json
src/training/export_onnx.py  export_to_onnx + CLI → ONNX, models.json, MODEL_CARD.md, export_report.json
src/predict.py             python -m src.predict <imagen> [--model] [--models-dir exports\final]: prob_ai y label por modelo
tests/                     conftest.py (datos sintéticos) + test_pipeline.py (30 pruebas)
weights/                   ya sin uso (los pesos viven en runs/); config.yaml aún tiene weights_path/onnx_path sin usar
```
Ignorados por Git: `runs/`, `exports/`, `data/raw/*`, `data/processed/*` (salvo `.gitkeep`), `.pytest_cache/`.

## 6. Flujo de datos y modelo (cómo encaja todo)

- **Preparación (una sola vez):** cada imagen original → `preprocess.harmonize` → se guarda como `.jpg` ya recortado/redimensionado/recomprimido en `data/processed/<dataset>/<split>/<REAL|FAKE>/<fuente>__<sha256[:16]>.jpg`. La fuente en el nombre sirve para métricas por generador (`source_of`). En CIFAKE la fuente es `cifake-real`/`cifake-fake`.
- **Splits:** CIFAKE: 10 % de train por clase para val (siempre las mismas, orden por `stable_hash` con la semilla), test oficial solo para evaluación final; duplicados exactos se descartan procesando test primero. ArtiFact: 500 000 imágenes por clase (desde el 31/10; antes 75 000), en partes iguales entre fuentes (una fuente chica entra completa y el resto se reparte), 80/10/10 dentro de cada fuente, lectura en orden de `header_offset` del zip, duplicados exactos por SHA-256 descartados, y los 2–3 generadores de `--holdout` (2 000 imágenes cada uno por defecto) van **solo** a `unseen/FAKE` (puede no existir `unseen/REAL`; la carga lo tolera).
- **Entrenamiento:** `get_transforms` solo hace flip horizontal (train) + `ToTensor` → [0,1]. El `Detector` normaliza dentro del modelo, así que el ONNX recibe exactamente la salida de `to_model_input`. Selección de la mejor época **siempre por ROC-AUC de validación**; `early_stopping_patience` épocas sin mejorar corta la corrida. F1/accuracy usan umbral 0,5.
- **Reanudación determinista:** al empezar cada época se siembra `train_loader.generator` y `torch.manual_seed(seed+epoch)`; el orden de los lotes es determinista aunque los DataLoader usen `persistent_workers` (el flip aleatorio de los workers no es reproducible al reanudar, a propósito). El historial de épocas viaja dentro de `last.pt` (campo `rows`) y `metrics.csv` se reescribe completo en cada época. `last.pt` se borra al terminar (`done.json` marca la corrida terminada).
- **Evaluación (`evaluate`):** métricas de test; si existe `unseen/`, métricas de "unseen" = falsas de los generadores no vistos + reales del test; `per_source.csv` (aciertos y prob_ai media por fuente); velocidad (img/s con lote 256 y latencia de 1 imagen, en el dispositivo disponible). Usar **solo con las corridas ya elegidas por validación**.
- **Exportación:** una vez, al final, desde `best.pt` (no por época).

## 7. Config y overrides

`config.yaml` → claves de `training`: `device`, `mixed_precision`, `batch_size` (128), `epochs` (30), `learning_rate` (1e-3), `weight_decay`, `warmup_epochs`, `label_smoothing`, `num_workers` (6), `early_stopping_patience` (5), `limit` (null). `datasets.<d>`: `root_dir` (raw), `processed_dir`, `image_size` (32 / 200), `jpeg_quality` (90), `val_fraction`. `models.<m>`: `hidden_dims`, `dropout`, `pool` (MLP; `by_dataset.artifact.pool: 4` → 7 500 entradas), `base_channels`, `pretrained`, `freeze_backbone`.
Overrides por corrida: `--set CLAVE=VALOR` (YAML: `3e-4`, `null`, `[256,128]`) y `set:` en `sweep.yaml`. Alias aceptados: `lr`→`learning_rate`, `amp`→`mixed_precision`, `patience`→`early_stopping_patience`; `seed` también. Clave desconocida → error. `run_config` rechaza `resnet18 + cifake`.

## 8. Comandos (PowerShell, desde la raíz del repo)

```powershell
.venv\Scripts\python.exe -m pytest -p no:logging
.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.is_bf16_supported())"

# Datos
.venv\Scripts\python.exe -m src.data.prepare_cifake --src <carpeta-con-train-y-test> [--force]
.venv\Scripts\python.exe -m src.data.prepare_artifact peek --zip D:\datasets\artifact-dataset.zip
.venv\Scripts\python.exe -m src.data.prepare_artifact list --zip D:\datasets\artifact-dataset.zip          # columna "sin archivo" debe dar ~0
.venv\Scripts\python.exe -m src.data.prepare_artifact extract --zip D:\datasets\artifact-dataset.zip --holdout <gen1> <gen2> [--per-class 500000 --holdout-per-source 2000 --force]

# Entrenar
.venv\Scripts\python.exe -m src.training.train --model cnn --dataset cifake --set epochs=1 --name humo-cifake-cnn
.venv\Scripts\python.exe -m src.training.sweep --only cifake --dry-run
.venv\Scripts\python.exe -m src.training.sweep --only cifake
.venv\Scripts\python.exe -m src.training.sweep --summary

# Evaluar y exportar
.venv\Scripts\python.exe -m src.training.evaluate runs\<corrida> [runs\<otra> ...]
.venv\Scripts\python.exe -m src.training.export_onnx runs\<mlp> runs\<cnn> runs\<resnet18> --out exports\final [--default resnet18 --uncertain-band 0.1]
.venv\Scripts\python.exe -m src.predict foto.jpg --models-dir exports\final
```
Reglas: si una preparación ya existe pide `--force`; si cambiás `--per-class`/`--holdout` después de entrenar, borrá `data\processed\artifact` y `runs\artifact-*`; no cambies los valores de una corrida ya empezada (si lo hacés: "tiene otra configuración" → borrá `runs\<nombre>` o renombrá).

## 9. Próximos pasos (plan vigente)

Cronograma de la guía original (sigue valiendo): sáb 31/10 CIFAKE + humo; noche del sábado barrido CIFAKE; dom 1/11 preparar ArtiFact + humo ArtiFact + ONNX de prueba para AWS; dom 1/11 noche → vie 6/11 barrido completo; sáb 7/11 elegir, evaluar, exportar finales; dom 8/11 entrega a Lautaro.

**Objetivo del usuario (31/10): adelantar el cronograma todo lo posible.** Las fechas de arriba son el tope, no el plan: si un paso se puede hacer antes (p. ej. `peek`/`list` de ArtiFact y la elección de holdout mientras corre el barrido de CIFAKE, o lanzar el barrido de ArtiFact apenas esté el humo), proponelo y encadenalo. Cuidá que el paralelismo no ahogue el disco/CPU (la lectura de archivos sueltos puede ser un cuello de botella) y avisá a Lautaro si algo entrega antes de lo previsto.

1. **Preparar la PC para correr sola** (powercfg `standby-timeout-ac 0` y `hibernate-timeout-ac 0`; pausar Windows Update; usar Windows Terminal o desactivar "modo de edición rápida" en la consola clásica; excluir repo y carpeta de datasets del antivirus — ahora importa más: son decenas de miles de archivos chicos).
2. **Barrido de CIFAKE** (humo ya hecho para mlp/cnn; ResNet-18 NO se corre con CIFAKE): `sweep --only cifake --dry-run` y luego `sweep --only cifake` (15 corridas). Si se corta, repetir el mismo comando.
3. **ArtiFact:** `peek` (encabezado debe tener `image_path` y `target`; si los nombres difieren, ajustar `read_metadata` en `prepare_artifact.py` o pedir ayuda con el encabezado real) → `list` (columna "sin archivo" ≈ 0; si es alta, las rutas no son relativas ni a la carpeta de la fuente ni a la raíz del zip: mirar un ejemplo) → **elegir 2 generadores de holdout**: el usuario dijo que no importa cuáles y que se pueden elegir **al azar** entre los generadores que imprime `list` (usar el nombre exacto de la carpeta; evitá los de más imágenes si querés que `unseen` quede completo, y si se reconocen, mejor uno de difusión y uno GAN). Elegilos vos, decí cuáles son, **anotalos en la sección 12 de este archivo** y recordale al usuario avisar a Lautaro, porque van en el informe → `extract` (15–40 min).
4. **Humo de ArtiFact** (los tres modelos, `--set epochs=1 --set limit=6000 --name humo-artifact-<m>`). Medir `imgs_per_sec`, `gpu_mb`, segundos. Estimar si las 23 corridas de ArtiFact caben antes del viernes 6/11 (fórmula: minutos ≈ épocas × (img train / imgs_per_sec + validación) / 60; ArtiFact tiene ~800 000 imágenes de train desde el 31/10 (antes ~120 000); con early stopping la mayoría corta entre la época 10 y 20). Si no entran, recortar corridas del final de `configs/sweep.yaml` (grupos 4 y 5 son las menos importantes) o bajar `epochs`.
5. **ONNX de prueba para AWS** con las corridas de humo de ArtiFact: `export_onnx runs\humo-artifact-mlp runs\humo-artifact-cnn runs\humo-artifact-resnet18 --out exports\humo`; probar localmente con `src.predict`; dar al usuario los comandos de copia al repo de AWS (`exports\humo\*` → `backend\model_api\models\` y `src\data\preprocess.py` → `backend\model_api\preprocess.py`). Nunca copiar un ONNX sin su `models.json`.
6. **Barrido completo** (`sweep --dry-run`, luego `sweep`); no tocar la ventana.
7. **Elegir (sáb 7/11):** `sweep --summary`; abrir el `metrics.csv` de la candidata (si la mejor época es la última, anotar como limitación); si dos corridas están a menos de 0,002 de ROC-AUC, elegir la más simple o rápida; **nunca elegir mirando test**. Anotar las 6 elegidas (3 modelos × 2 datasets) y el motivo — es parte de la entrega.
8. **`evaluate`** solo sobre las elegidas (opcional: también `*-base-s1`/`*-base-s2` para la variación entre semillas; no se usan para elegir, no hay fuga), luego **`export_onnx` final** (`--out exports\final`) con las 3 corridas de ArtiFact elegidas.
9. **Entrega a Lautaro (domingo 8/11):** copiar lo chico de `runs\` a `results\` (`robocopy runs results *.csv *.json /S /XD humo-*`; robocopy devuelve código 1 si copió: no es error; se excluyen `.pt`; `summary.csv` trae filas `humo-*` que hay que quitar), `configs\sweep.yaml`, `MODEL_CARD.md` y `export_report.json` de `exports\final\`, más latencias en Lambda (de la guía de AWS, que este repo no tiene). Y la lista de decisiones: notas N1–N14 (sección 11), generadores no vistos, seis corridas elegidas con motivo, banda de uncertain.

Optimización pendiente (solo si el humo de ArtiFact la justifica; el "cuello de botella" de CIFAKE resultó ser el arranque de workers, ya resuelto con `persistent_workers`, así que medir de nuevo antes de asumir que la lectura limita). Palancas, de menor a mayor invasividad: subir `num_workers` (`--set num_workers=10` y reflejarlo en `config.yaml` antes del barrido), y si no alcanza, el formato `.bin` + `.csv` por split (un archivo grande con los JPEG concatenados + índice; era el diseño de la guía original) o precargar CIFAKE completo en RAM (32×32, muy chico). Consultalo con el usuario antes de implementarlo.

## 10. Trampas y particularidades conocidas

- `config.json` de cada corrida guarda `processed_dir` **absoluto**; si el repo cambia de carpeta, reanudar/relanzar da "tiene otra configuración".
- El hash `preprocess_sha256` normaliza CRLF→LF (Git en Windows tiene `core.autocrlf=true`). Cualquier edición a `preprocess.py` invalida los `models.json` ya exportados y las copias en AWS.
- `evaluate` sobre `unseen` emite un `UndefinedMetricWarning` de ROC-AUC (una sola clase en ese conjunto aislado); es inofensivo: la métrica de "unseen" que se reporta ya incluye las reales de test.
- `Trainer.fit(..., weights_dir=...)` conserva el nombre original del parámetro pero ahora es la **carpeta de la corrida**.
- `train.main` devuelve el resumen y tolera `--name` absoluto (lo usan los tests y el barrido). El barrido atrapa también `SystemExit`, así que una corrida con config cambiada se anota en `failed.txt` y sigue; volver a lanzar el barrido la reintenta.
- Los tests fijan `device=cpu` con `--set device=cpu` (en esta máquina `cuda` está disponible).
- `export_to_onnx` usa `dynamo=True`, `external_data=False`, `dynamic_shapes={"x": {0: Dim("batch", 1..1024)}}` (el argumento del `forward` del `Detector` se llama `x`: no lo renombres). No se pasa `opset_version`.
- Con `--set limit=N` se usan N imágenes por split (submuestreo reproducible); solo para humo. `PreparedFolder.sources` es una propiedad que recorre todas las muestras en cada acceso: no la llames dentro de un bucle (hasta el 31/10 `create_dataset` lo hacía con `limit` y tardaba ~1 h con ArtiFact, con la GPU ociosa; ahora usa `source_of(ds.samples[i][0])`).
- La carga de `unseen` requiere `allow_empty=True` en `ImageFolder` (torchvision ≥ 0.19) y mapeo fijo de clases; no volver a depender del orden alfabético (el original daba FAKE=0, REAL=1 y causó el bug de métricas invertidas).
- `make_loader` usa `persistent_workers=num_workers > 0` (31/10): sin eso, en Windows cada época volvía a arrancar los 6 workers (~30 s muertos por época). Costo aceptado: el flip aleatorio de los workers no es bit a bit reproducible al reanudar (ver sección 6). Las corridas anteriores a este cambio (`humo-*` y las que se cortaron) usaron el modo viejo.
- `weights/` y las claves `weights_path`/`onnx_path` de `config.yaml` quedaron sin uso.

## 11. Diferencias con la guía original (PDF) y qué de ella sigue vigente

Qué cambió respecto al PDF: **(1)** formato de datos: carpetas de `.jpg` (no `.bin`/`.csv`; el Anexo B ya no sirve como copia literal); **(2)** barrido de **38** corridas, no 48 (se eliminaron las 10 `cifake-resnet18-*`; CIFAKE tiene 15 corridas, no 25, ArtiFact 23); **(3)** ResNet-18 solo con ArtiFact, sin `resize_to`, `train.py` lo rechaza con CIFAKE con un mensaje claro; **(4)** claves canónicas `learning_rate`/`mixed_precision`/`early_stopping_patience` (alias `lr`/`amp`/`patience` aceptados); **(5)** rutas de datos `processed_dir`, flag `--force`; **(6)** se conservaron los archivos originales (`preprocessing.py`, `dataset.py` con `create_dataloaders`, etc.) con ediciones mínimas en vez de reemplazarlos; **(7)** `src/predict.py` es nuevo (CLI local); **(8)** el smoke de CIFAKE solo para `mlp` y `cnn`; **(9)** el usuario hace él mismo los commits.

Sigue vigente del PDF: el cronograma; los cuidados de la PC (sección 1.3); la elección de hiperparámetros (sección 5.3); la lista de entregables a Lautaro; la guía de AWS (que no tenemos acá) para desplegar y medir latencia en Lambda; el Anexo A de errores frecuentes (adaptar a los nombres actuales).

Notas de corrección N1–N14 (para el informe de Lautaro, resumidas): **N1** clases invertidas (config.py daba FAKE=0; ahora REAL=0/FAKE=1 y FAKE positiva); **N2** se elegía la mejor época mirando test (ahora validación propia, test solo al final); **N3** entrenamiento y servicio preprocesaban distinto (ahora un único `preprocess.py` + hash en `models.json`); **N4** la armonización JPEG corría en cada época (ahora una vez al preparar, con orden recorte→tamaño→JPEG); **N5** `config.yaml` no se usaba (ahora gobierna cada corrida y se guarda en `config.json`); **N6** ONNX se exportaba en cada época y sin exportador definido (ahora una vez, `dynamo=True`, `external_data=False`, con comprobación de paridad); **N7** ResNet-18 con 32×32 llega a 1 píxel (decisión del usuario: no usarla con CIFAKE); **N8** ArtiFact a 200×200 (no 224), MLP con `pool: 4`; **N9** ArtiFact viene como zip con una carpeta por fuente y 31,6 GB (se lee directo del zip); **N10** se eliminan Docker y la API FastAPI (acordado con Lautaro); **N11** versiones: NumPy 2, Pillow 13.0.0; **N12** el test de CIFAKE no tiene fuentes (la fuente es `cifake-real`/`cifake-fake`); **N13** generadores no vistos para medir robustez (a elegir); **N14** una sola API (`/v1/predict`) en el repo de AWS.

## 12. Preguntas abiertas / decisiones del usuario pendientes

- Generadores de `--holdout` **elegidos (31/10): `vq_diffusion` (difusión) y `big_gan` (GAN)**, 10 000 falsas cada uno en el zip (se usan 2 000 de cada uno). Sorteo con `random.Random(42)`: uno entre `glide`, `latent_diffusion`, `vq_diffusion` y otro entre `big_gan`, `cips`, `gansformer`, `gau_gan`, `projected_gan`, `star_gan`, `stylegan1`. Fuera del sorteo: `stable_diffusion` (es el generador más relevante en la práctica, mejor que la API lo vea en el entrenamiento), los muy grandes (`stylegan2/3`, `taming_transformer`, `mat`), los que también tienen reales (`pro_gan`, `cycle_gan`), `ddpm` (896, muy chico) y los de inpainting o sin clasificar. `peek`/`list` OK: 33 fuentes, "sin archivo" = 0 en todas. **Falta avisar a Lautaro** (va en el informe).
- Cuándo y cómo commitear/pushear (el usuario decide; hoy el commit está en la rama `train`, sin push). El README fue reescrito sin Docker aunque la guía lo asigna a Lautaro: conviene avisarle.
- Modelo por defecto de la API: **`resnet18`** (mejor ROC-AUC de validación; exportado así el 3/11).

## 13. Primer mensaje sugerido al retomar

Verificá el estado (`git status`, `git log --oneline -5`, `dir runs`, `dir data\processed`, `type runs\summary.csv` si existe), leé los últimos `metrics.csv` si hay corridas, y preguntale al usuario en qué paso del plan (sección 9) está antes de proponer nada. No ejecutes entrenamientos largos ni preparaciones de datasets sin que lo pida; sí podés correr `pytest`.
