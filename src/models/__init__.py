import torch
import torch.nn as nn
from src.config import IMAGE_MEAN, IMAGE_STD
from src.models.mlp import MLPClassifier
from src.models.cnn import CustomCNN
from src.models.resnet import ResNet18Classifier

SUPPORTED_MODELS = ("mlp", "cnn", "resnet18")


class Detector(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor(IMAGE_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGE_STD).view(1, 3, 1, 1))

    def forward(self, x):
        return self.backbone((x - self.mean) / self.std)


def build_model(model_name: str, image_size: int = 32, pretrained: bool = True, freeze_backbone: bool = False, **model_args):
    if model_name == "mlp":
        return Detector(MLPClassifier(image_size=image_size, **model_args))
    if model_name == "cnn":
        return Detector(CustomCNN(**model_args))
    if model_name == "resnet18":
        return Detector(ResNet18Classifier(pretrained=pretrained, freeze_backbone=freeze_backbone, **model_args))
    raise ValueError(f"Modelo no soportado: {model_name}")
