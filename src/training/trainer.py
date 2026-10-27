from pathlib import Path
import torch
import torch.nn as nn
from tqdm import tqdm
from src.training.export_onnx import export_to_onnx
from src.training.metrics import compute_metrics


class Trainer:
    def __init__(self, model, model_name: str, device, lr: float = 1e-3, image_size: int = 32):
        self.model = model.to(device)
        self.model_name = model_name
        self.device = device
        self.image_size = image_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad], lr=lr
        )

    def train_epoch(self, loader):

        self.model.train()
        total_loss = 0.0

        for images, labels in tqdm(loader, desc=f"Train {self.model_name}", leave=False):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(images), labels)

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / max(len(loader), 1)


    @torch.no_grad()
    def evaluate(self, loader):

        self.model.eval()
        y_true, y_pred, y_prob = [], [], []

        for images, labels in loader:
            images = images.to(self.device)
            probs = torch.softmax(self.model(images), dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs[:, 1].cpu().tolist())

        return compute_metrics(y_true, y_pred, y_prob)
    

    def fit(self, train_loader, val_loader, epochs: int = 10, weights_dir: str | Path = "weights"):
        weights_dir = Path(weights_dir)
        weights_dir.mkdir(parents=True, exist_ok=True)
        pt_path = weights_dir / f"{self.model_name}.pt"
        onnx_path = weights_dir / f"{self.model_name}.onnx"

        best_f1 = -1.0

        for epoch in range(1, epochs + 1):
            
            loss = self.train_epoch(train_loader)
            metrics = self.evaluate(val_loader)
            print(f"Epoch {epoch}/{epochs} | Loss: {loss:.4f} | {metrics}")

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                torch.save(self.model.state_dict(), pt_path)
                export_to_onnx(self.model, onnx_path, self.image_size)
