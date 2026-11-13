import hashlib
from pathlib import Path

SOURCE_SEP = "__"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stable_hash(text: str, seed: int = 0) -> int:
    return int(hashlib.sha1(f"{seed}:{text}".encode("utf-8")).hexdigest()[:12], 16)


def save_prepared(out_dir, split: str, label_name: str, source: str, digest: str, jpeg: bytes) -> Path:
    folder = Path(out_dir) / split / label_name
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{source}{SOURCE_SEP}{digest[:16]}.jpg"
    path.write_bytes(jpeg)
    return path


def source_of(path) -> str:
    return Path(path).stem.rsplit(SOURCE_SEP, 1)[0]
