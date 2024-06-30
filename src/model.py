import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ResNet18_Weights

class LeafDiseaseModel(nn.Module):
    def __init__(self, num_classes, backbone='resnet50'):
        super().__init__()
        if backbone == 'resnet50':
            self.base_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone == 'resnet18':
            self.base_model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported backbone specified. Choose either 'resnet50' or 'resnet18'.")

        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

def get_model(num_classes, backbone='resnet50'):
    return LeafDiseaseModel(num_classes, backbone)