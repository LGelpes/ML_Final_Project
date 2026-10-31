import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained: bool = True, freeze_backbone: bool = False,
                 dropout: float = 0.0):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.backbone(x)
