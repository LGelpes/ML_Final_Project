import csv
import math
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.training.metrics import compute_metrics


class Trainer:
    def __init__(self, model, model_name: str, device, lr: float = 1e-3, image_size: int = 32,
                 weight_decay: float = 1e-4, mixed_precision: bool = False, label_smoothing: float = 0.0,
                 warmup_epochs: int = 1, early_stopping_patience: int = 5, seed: int = 0):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.warmup_epochs = warmup_epochs
        self.patience = early_stopping_patience
        self.seed = seed
        self.amp = mixed_precision and device.type == "cuda"
        self.scheduler = None
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr, weight_decay=weight_decay
        )

    def train_epoch(self, loader):

        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(loader, desc=f"Train {self.model_name}", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                loss = self.criterion(self.model(images), labels)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)


    @torch.no_grad()
    def evaluate(self, loader, return_probs: bool = False):

        self.model.eval()
        y_true, y_pred, y_prob = [], [], []
        total_loss = 0.0

        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.amp):
                logits = self.model(images).float()
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())

        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["loss"] = round(total_loss / max(len(y_true), 1), 5)
        if return_probs:
            return metrics, np.array(y_true), np.array(y_prob)
        return metrics
    

    def fit(self, train_loader, val_loader, epochs: int = 10, weights_dir: str | Path = "weights"):
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_path = weights_dir / "best.pt"
        last_path = weights_dir / "last.pt"

        steps = max(len(train_loader), 1)
        warmup = max(1, self.warmup_epochs * steps)

        def lr_factor(step):
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, epochs * steps - warmup)
            return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_factor)

        state = {"best_auc": -1.0, "best_epoch": 0, "best_val": None, "bad_epochs": 0,
                 "seconds": 0.0, "imgs_per_sec": [], "gpu_mb": 0.0, "epochs_run": 0}
        start_epoch, rows = 1, []

        if last_path.exists():
            ckpt = torch.load(last_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            state, start_epoch, rows = ckpt["state"], ckpt["epoch"] + 1, ckpt["rows"]
            print(f"Reanudando desde la época {start_epoch}")

        for epoch in range(start_epoch, epochs + 1):
            if state["bad_epochs"] >= self.patience:
                break

            # mismo orden de lotes si la corrida se reanuda
            train_loader.generator.manual_seed(self.seed + epoch)
            torch.manual_seed(self.seed + epoch)
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            started = time.perf_counter()
            loss = self.train_epoch(train_loader)
            seconds = time.perf_counter() - started
            speed = len(train_loader.dataset) / max(seconds, 1e-9)
            metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | {metrics}")

            gpu_mb = torch.cuda.max_memory_allocated() / 2**20 if self.device.type == "cuda" else 0.0
            rows.append({"epoch": epoch, "lr": self.scheduler.get_last_lr()[0], "train_loss": round(loss, 5),
                         **{f"val_{k}": v for k, v in metrics.items() if k not in ("n", "gpu_mb")},
                         "imgs_per_sec": round(speed, 1), "gpu_mb": round(gpu_mb, 1), "seconds": round(seconds, 1)})
            state["seconds"] += seconds
            state["imgs_per_sec"].append(round(speed, 1))
            state["gpu_mb"] = max(state["gpu_mb"], round(gpu_mb, 1))
            state["epochs_run"] = epoch

            if metrics["roc_auc"] > state["best_auc"]:
                state.update(best_auc=metrics["roc_auc"], best_epoch=epoch, best_val=metrics, bad_epochs=0)
                torch.save(self.model.state_dict(), best_path)
            else:
                state["bad_epochs"] += 1

            with open(weights_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                        "scheduler": self.scheduler.state_dict(), "epoch": epoch, "state": state, "rows": rows},
                       last_path)

        return {
            "best_epoch": state["best_epoch"],
            "val": state["best_val"],
            "epochs_run": state["epochs_run"],
            "stopped_early": state["bad_epochs"] >= self.patience,
            "train_minutes": round(state["seconds"] / 60, 2),
            "imgs_per_sec": round(float(np.median(state["imgs_per_sec"])), 1) if state["imgs_per_sec"] else 0.0,
            "gpu_mb": state["gpu_mb"],
        }
