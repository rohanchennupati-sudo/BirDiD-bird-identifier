import torch.nn as nn
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights


def build_model(
    num_classes: int = 200,
    dropout: float = 0.4,
    pretrained: bool = True,
) -> nn.Module:
    """Build EfficientNet-B3 adapted for bird classification."""
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout, inplace=True),
        nn.Linear(in_features, num_classes),
    )

    return model


def freeze_backbone(model: nn.Module) -> None:
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True
