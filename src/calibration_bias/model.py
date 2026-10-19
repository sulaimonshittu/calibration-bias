"""ResNet-50 backbone with a binary classification head (Chapter 3, Section 3.7)."""
import torch
from torch import nn
from transformers import ResNetConfig, ResNetModel


class SkinLesionClassifier(nn.Module):
    def __init__(self, num_classes=2, dropout=0.5,
                 checkpoint="microsoft/resnet-50", pretrained=True):
        super().__init__()
        if pretrained:
            self.backbone = ResNetModel.from_pretrained(checkpoint)
        else:  # random weights, same ResNet-50 architecture; used by offline tests
            self.backbone = ResNetModel(ResNetConfig())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(pixel_values=x).pooler_output  # (B, 2048, 1, 1)
        return self.head(features)  # raw logits, shape (B, num_classes)