import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import create_dataset, make_loader
from src.models import build_model
from src.training.metrics import compute_metrics
from src.training.trainer import Trainer


def load_run(run_dir, device):
    cfg = json.loads((Path(run_dir) / "config.json").read_text(encoding="utf-8"))
    args = {**cfg["model_args"], **({"pretrained": False} if cfg["model"] == "resnet18" else {})}
    model = build_model(cfg["model"], image_size=cfg["image_size"], **args)
    model.load_state_dict(torch.load(Path(run_dir) / "best.pt", map_location="cpu", weights_only=True))
    return cfg, model.to(device).eval()


def per_source(sources, y_true, prob_ai, split):
    groups = defaultdict(list)
    for i, source in enumerate(sources):
        groups[(source, int(y_true[i]))].append(i)
    rows = []
    for (source, label), idx in sorted(groups.items()):
        p = prob_ai[idx]
        correct = (p >= 0.5) == (label == 1)
        rows.append({"split": split, "source": source, "label": "FAKE" if label else "REAL", "n": len(idx),
                     "accuracy": round(float(correct.mean()), 4), "mean_prob_ai": round(float(p.mean()), 4)})
    return rows


@torch.no_grad()
def speed(model, size, device, amp):
    batch = 256 if device.type == "cuda" else 16

    def timed(n, repeats):
        x = torch.rand(n, 3, size, size, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            for _ in range(3):
                model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(repeats):
                started = time.perf_counter()
                model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
        return np.array(times)

    big, one = timed(batch, 10), timed(1, 50)
    return {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "throughput_imgs_per_sec": round(float(batch / np.median(big)), 1),
            "latency_ms_p50": round(float(np.median(one) * 1000), 2),
            "latency_ms_p95": round(float(np.percentile(one, 95) * 1000), 2)}


def evaluate_run(run_dir, device=None, num_workers=4):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    cfg, model = load_run(run_dir, device)
    folder = Path(cfg["processed_dir"])
    trainer = Trainer(model, cfg["model"], device, image_size=cfg["image_size"], mixed_precision=amp)
    results, rows = {}, []

    def scores(split):
        ds = create_dataset(folder, split, cfg["image_size"])
        _, y_true, prob_ai = trainer.evaluate(make_loader(ds, 256, False, num_workers), return_probs=True)
        return ds.sources, y_true, prob_ai

    sources, y_true, prob_ai = scores("test")
    results["test"] = compute_metrics(y_true, (prob_ai >= 0.5).astype(int), prob_ai)
    rows += per_source(sources, y_true, prob_ai, "test")
    if (folder / "unseen").exists():
        u_sources, u_true, u_prob = scores("unseen")
        real = y_true == 0
        all_true, all_prob = np.concatenate([u_true, y_true[real]]), np.concatenate([u_prob, prob_ai[real]])
        results["unseen"] = compute_metrics(all_true, (all_prob >= 0.5).astype(int), all_prob)
        rows += per_source(u_sources, u_true, u_prob, "unseen")
    results["speed"] = speed(model, cfg["image_size"], device, amp)

    (Path(run_dir) / "test_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with open(Path(run_dir) / "per_source.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "source", "label", "n", "accuracy", "mean_prob_ai"])
        writer.writeheader()
        writer.writerows(rows)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args(argv)
    for run_dir in args.runs:
        results = evaluate_run(run_dir, num_workers=args.num_workers)
        print(Path(run_dir).name, json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
