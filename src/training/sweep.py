import argparse
import csv
import gc
import json
import traceback
from pathlib import Path

import torch
import yaml

from src.config import BASE_DIR, run_config
from src.training.train import RUNS_DIR, main as train_main

SUMMARY_FIELDS = ["name", "dataset", "model", "val_roc_auc", "val_f1", "val_accuracy", "best_epoch",
                  "epochs_run", "stopped_early", "train_minutes", "imgs_per_sec", "gpu_mb", "params", "overrides"]


def load_runs(path):
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    names = [r["name"] for r in spec["runs"]]
    duplicated = {n for n in names if names.count(n) > 1}
    if duplicated:
        raise SystemExit(f"Nombres repetidos en {path}: {sorted(duplicated)}")
    return spec["runs"]


def write_summary(runs_dir, entries):
    overrides = {e["name"]: e.get("set") or {} for e in entries}
    rows = []
    for done in sorted(Path(runs_dir).glob("*/done.json")):
        s = json.loads(done.read_text(encoding="utf-8"))
        val = s.get("val") or {}
        rows.append({
            "name": s["name"], "dataset": s["dataset"], "model": s["model"],
            "val_roc_auc": val.get("roc_auc"), "val_f1": val.get("f1"), "val_accuracy": val.get("accuracy"),
            "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"], "stopped_early": s["stopped_early"],
            "train_minutes": s["train_minutes"], "imgs_per_sec": s["imgs_per_sec"], "gpu_mb": s["gpu_mb"],
            "params": s["params"], "overrides": json.dumps(overrides.get(s["name"], {})),
        })
    rows.sort(key=lambda r: (r["dataset"], r["model"], -(r["val_roc_auc"] or 0)))
    Path(runs_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(runs_dir) / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def print_best(rows):
    best = {}
    for r in rows:
        best.setdefault((r["dataset"], r["model"]), r)
    print(f"{'dataset':10} {'modelo':9} {'val_auc':>8} {'val_f1':>7} corrida")
    for (dataset, model), r in sorted(best.items()):
        print(f"{dataset:10} {model:9} {r['val_roc_auc']:8.4f} {r['val_f1']:7.4f} {r['name']}")


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(BASE_DIR / "configs" / "sweep.yaml"))
    parser.add_argument("--runs-dir", default=str(RUNS_DIR))
    parser.add_argument("--only", help="corre solo las corridas cuyo nombre contenga este texto")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    entries = load_runs(args.file)
    if args.summary:
        print_best(write_summary(args.runs_dir, entries))
        return


    for e in entries:
        run_config(e["model"], e["dataset"], e.get("set"))
    for entry in entries:
        name = entry["name"]
        if args.only and args.only not in name:
            continue
        run_dir = Path(args.runs_dir) / name
        if (run_dir / "done.json").exists():
            continue
        if args.dry_run:
            print(f"correría {name}: {entry['model']} en {entry['dataset']} {entry.get('set') or ''}")
            continue
        try:
            train_main(["--model", entry["model"], "--dataset", entry["dataset"], "--name", str(run_dir),
                        *[f"--set={k}={json.dumps(v)}" for k, v in (entry.get("set") or {}).items()]])
            (run_dir / "failed.txt").unlink(missing_ok=True)
        except KeyboardInterrupt:
            raise
        except (Exception, SystemExit):
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "failed.txt").write_text(traceback.format_exc(), encoding="utf-8")
            print(f"{name}: falló, ver {run_dir / 'failed.txt'}", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        write_summary(args.runs_dir, entries)
    if not args.dry_run:
        print_best(write_summary(args.runs_dir, entries))


if __name__ == "__main__":
    main()
