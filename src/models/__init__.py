from src.models.mlp import MLPClassifier
from src.models.cnn import CustomCNN
from src.models.resnet import ResNet18Classifier

SUPPORTED_MODELS = ("mlp", "cnn", "resnet18")


def build_model(model_name: str, image_size: int = 32, pretrained: bool = True, freeze_backbone: bool = False):
    if model_name == "mlp":
        return MLPClassifier(image_size=image_size)
    if model_name == "cnn":
        return CustomCNN()
    if model_name == "resnet18":
        return ResNet18Classifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    raise ValueError(f"Modelo no soportado: {model_name}")
