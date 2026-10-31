import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(y_true, y_pred, y_prob):
    try:
        auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        auc = 0.0

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1": round(float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        "roc_auc": round(auc, 4),
        "n": len(y_true),
        "gpu_mb": round(torch.cuda.max_memory_allocated() / (1024**2), 1) if torch.cuda.is_available() else 0.0,
    }
