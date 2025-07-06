from torchvision.models import resnet34, ResNet34_Weights
import torch.nn as nn
class EmotionModel(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.model = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)