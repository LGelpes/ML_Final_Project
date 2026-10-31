import argparse
import csv
import io
import posixpath
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from src.config import BASE_DIR, CLASSES, load_config
from src.data.prepare_cifake import reset_splits
from src.data.prepared import save_prepared, sha256, stable_hash
from src.data.preprocess import harmonize, load_rgb

SPLITS = ("train", "val", "test", "unseen")


def read_metadata(zf):
    names = set(zf.namelist())
    rows, unresolved = [], Counter()
    for meta in sorted(n for n in names if posixpath.basename(n) == "metadata.csv"):
        folder = posixpath.dirname(meta)
        source = posixpath.basename(folder) or "root"
        with zf.open(meta) as f:
            for row in csv.DictReader(io.TextIOWrapper(f, encoding="utf-8")):
                rel = row["image_path"].replace("\\", "/")
                while rel.startswith("./"):
                    rel = rel[2:]
                rel = rel.lstrip("/")

                member = next((c for c in (posixpath.join(folder, rel), rel) if c in names), None)
                if member is None:
                    unresolved[source] += 1
                    continue
                rows.append((source, member, 0 if int(row["target"]) == 0 else 1))
    if not rows:
        raise SystemExit("No encontré metadata.csv con imágenes en el zip.")
    return rows, unresolved


def allocate(counts, total):
    alloc, remaining, budget = {}, dict(counts), total
    while remaining and budget > 0:
        share = budget // len(remaining)
        small = {s: c for s, c in remaining.items() if c <= share}
        if share == 0 or not small:
            for s in remaining:
                alloc[s] = min(remaining[s], share)
            break
        for s, c in small.items():
            alloc[s] = c
            budget -= c
            del remaining[s]
    return alloc


def select(rows, per_class, holdout, holdout_per_source, seed):
    by_key = defaultdict(list)
    for source, member, label in rows:
        by_key[(source, label)].append(member)
    chosen = {}
    for label in (0, 1):
        pool = {s: len(m) for (s, lab), m in by_key.items() if lab == label and s not in holdout}
        for source, k in allocate(pool, per_class).items():
            members = sorted(by_key[(source, label)], key=lambda m: stable_hash(m, seed))[:k]
            n_val = n_test = round(k * 0.1)
            for i, member in enumerate(members):
                split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
                chosen[member] = (split, label, source)
    for (source, label), members in by_key.items():
        if source in holdout:
            for member in sorted(members, key=lambda m: stable_hash(m, seed))[:holdout_per_source]:
                chosen[member] = ("unseen", label, source)
    return chosen


def cmd_peek(zf, args=None, cfg=None):
    metas = sorted(n for n in zf.namelist() if posixpath.basename(n) == "metadata.csv")
    print(f"{len(metas)} archivos metadata.csv; primeros: {metas[:3]}")
    if metas:
        print(zf.read(metas[0])[:300].decode("utf-8", errors="replace"))


def cmd_list(zf, args=None, cfg=None):
    rows, unresolved = read_metadata(zf)
    table = Counter((s, lab) for s, _, lab in rows)
    sources = sorted({s for s, _, _ in rows})
    print(f"{'fuente':28} {'reales':>9} {'fake':>9} {'sin archivo':>12}")
    for s in sources:
        print(f"{s:28} {table[(s, 0)]:9d} {table[(s, 1)]:9d} {unresolved[s]:12d}")
    print(f"{'TOTAL':28} {sum(v for (s, l), v in table.items() if l == 0):9d} "
          f"{sum(v for (s, l), v in table.items() if l == 1):9d} {sum(unresolved.values()):12d}")


def cmd_extract(zf, args, cfg):
    rows, _ = read_metadata(zf)
    sources = {s for s, _, _ in rows}
    missing = set(args.holdout) - sources
    if missing:
        raise SystemExit(f"Fuentes de --holdout que no existen en el zip: {sorted(missing)}")
    ds = cfg["datasets"]["artifact"]
    reset_splits(args.out_path, SPLITS, args.force)
    chosen = select(rows, args.per_class, set(args.holdout), args.holdout_per_source, cfg["project"]["seed"])
    infos = sorted((zf.getinfo(m) for m in chosen), key=lambda i: i.header_offset)
    counts, seen, duplicates, broken = Counter(), set(), 0, 0
    for info in tqdm(infos, desc="ArtiFact", unit="img"):
        split, label, source = chosen[info.filename]
        data = zf.read(info)
        digest = sha256(data)
        if digest in seen:
            duplicates += 1
            continue
        seen.add(digest)
        try:
            jpeg = harmonize(load_rgb(data), ds["image_size"], ds["jpeg_quality"])
        except Exception:
            broken += 1
            continue
        save_prepared(args.out_path, split, CLASSES[label], source, digest, jpeg)
        counts[split] += 1
    for split in SPLITS:
        print(f"{split}: {counts[split]} imágenes")
    print(f"Duplicadas exactas descartadas: {duplicates}; ilegibles: {broken}")


COMMANDS = {"peek": cmd_peek, "list": cmd_list, "extract": cmd_extract}


def main(argv=None):
    cfg = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=list(COMMANDS))
    parser.add_argument("--zip", required=True)
    parser.add_argument("--out", default=str(BASE_DIR / cfg["datasets"]["artifact"]["processed_dir"]))
    parser.add_argument("--per-class", type=int, default=75000)
    parser.add_argument("--holdout", nargs="*", default=[])
    parser.add_argument("--holdout-per-source", type=int, default=2000)
    parser.add_argument("--force", action="store_true", help="reemplaza los splits ya preparados")
    args = parser.parse_args(argv)
    args.out_path = Path(args.out)
    with zipfile.ZipFile(args.zip) as zf:
        COMMANDS[args.command](zf, args, cfg)


if __name__ == "__main__":
    main()
