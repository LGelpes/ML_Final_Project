import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from src.config import BASE_DIR, CLASSES, load_config
from src.data.prepared import save_prepared, sha256, stable_hash
from src.data.preprocess import harmonize, load_rgb

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in EXTENSIONS)


def reset_splits(out, splits, force):
    existing = [out / s for s in splits if (out / s).exists()]
    if existing and not force:
        raise SystemExit(f"{out} ya tiene datos preparados ({', '.join(p.name for p in existing)}). "
                         "Usá --force para volver a armarlos.")
    for folder in existing:
        shutil.rmtree(folder)


def plan_splits(src, val_fraction, seed):
    plan = []
    for name in CLASSES:
        plan += [("test", name, p) for p in list_images(Path(src, "test", name))]
        train = list_images(Path(src, "train", name))
        train.sort(key=lambda p: stable_hash(p.relative_to(src).as_posix(), seed))
        n_val = round(len(train) * val_fraction)
        plan += [("val", name, p) for p in train[:n_val]]
        plan += [("train", name, p) for p in train[n_val:]]
    if not plan:
        raise SystemExit(f"No encontré imágenes en {src}\\{{train,test}}\\{{REAL,FAKE}}")

    order = {"test": 0, "val": 1, "train": 2}
    return sorted(plan, key=lambda item: order[item[0]])


def main(argv=None):
    cfg = load_config()
    ds = cfg["datasets"]["cifake"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(BASE_DIR / ds["root_dir"]))
    parser.add_argument("--out", default=str(BASE_DIR / ds["processed_dir"]))
    parser.add_argument("--val-fraction", type=float, default=ds.get("val_fraction", 0.1))
    parser.add_argument("--force", action="store_true", help="reemplaza los splits ya preparados")
    args = parser.parse_args(argv)

    out = Path(args.out)
    reset_splits(out, ("train", "val", "test"), args.force)
    plan = plan_splits(Path(args.src), args.val_fraction, cfg["project"]["seed"])
    counts = {"train": 0, "val": 0, "test": 0}
    seen, duplicates = set(), 0
    for split, name, path in tqdm(plan, desc="CIFAKE", unit="img"):
        data = path.read_bytes()
        digest = sha256(data)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        jpeg = harmonize(load_rgb(data), ds["image_size"], ds["jpeg_quality"])
        save_prepared(out, split, name, "cifake-" + name.lower(), digest, jpeg)
        counts[split] += 1
    for split, n in counts.items():
        print(f"{split}: {n} imágenes")
    print(f"Duplicadas exactas descartadas: {duplicates}")


if __name__ == "__main__":
    main()
