import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, image_size: int = 32, num_classes: int = 2, dropout: float = 0.3,
                 hidden_dims=(512, 256, 128), pool: int = 1):
        super().__init__()
        in_features = 3 * (image_size // pool) ** 2
        layers = [nn.AvgPool2d(pool)] if pool > 1 else []
        layers.append(nn.Flatten())
        for width in hidden_dims:
            layers += [nn.Linear(in_features, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            in_features = width
        layers.append(nn.Linear(in_features, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
