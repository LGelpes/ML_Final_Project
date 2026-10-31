import torch.nn as nn


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.4, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.features = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=3, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c, c * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c * 2, c * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(c * 4, c * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(c * 8, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))
