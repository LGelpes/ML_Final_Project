import csv
import io
import zipfile

import numpy as np
import pytest
from PIL import Image


def synthetic(label, size, seed, fmt="PNG"):
    rng = np.random.default_rng(seed)
    h, w = size if isinstance(size, tuple) else (size, size)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack([xx / w, yy / h, (xx + yy) / (w + h)], -1) * 255 * rng.uniform(0.5, 1.0)
    if label == "FAKE":
        base = base * 0.5 + ((xx + yy) % 2)[..., None] * 120
    pixels = np.clip(base + rng.normal(0, 4, base.shape), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def cifake_raw(tmp_path_factory):
    root = tmp_path_factory.mktemp("cifake_raw")
    seed = 0
    for split, n in (("train", 40), ("test", 10)):
        for label in ("REAL", "FAKE"):
            folder = root / split / label
            folder.mkdir(parents=True)
            for i in range(n):
                seed += 1
                (folder / f"{i}.png").write_bytes(synthetic(label, 32, seed))

    (root / "train" / "REAL" / "dup.png").write_bytes((root / "test" / "REAL" / "0.png").read_bytes())
    return root


SOURCES = {"coco": "REAL", "imagenet": "REAL", "glide": "FAKE", "stylegan2": "FAKE", "ddpm": "FAKE"}


@pytest.fixture(scope="session")
def artifact_zip(tmp_path_factory):
    path = tmp_path_factory.mktemp("artifact_raw") / "artifact-dataset.zip"
    seed = 1000
    with zipfile.ZipFile(path, "w") as zf:
        for source, label in SOURCES.items():
            rows = []
            for i in range(30):
                seed += 1
                size = (220, 180) if i % 5 == 0 else 200
                name = f"{source}/{label.lower()}/img_{i}.jpg"
                zf.writestr(name, synthetic(label, size, seed, fmt="JPEG"))

                rows.append({"image_path": name if source == "imagenet" else name.split("/", 1)[1],
                             "target": 0 if label == "REAL" else 1, "category": "x"})
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["image_path", "target", "category"])
            writer.writeheader()
            writer.writerows(rows)
            zf.writestr(f"{source}/metadata.csv", buf.getvalue())
    return path
