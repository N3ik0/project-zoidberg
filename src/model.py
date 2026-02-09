import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes=3):
    # Chargement de resnet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Récup nb caract et fini par une couche en fc (fully connected)
    num_ftrs = model.fc.in_features

    # Remplacement de la dernière couche par une adapté aux 3 classes
    model.fc = nn.Linear(num_ftrs, num_classes)

    return model