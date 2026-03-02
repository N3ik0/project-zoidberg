"""
Factory de modèles pré-entraînés.
Chaque modèle est configuré pour du fine-tuning :
  - Socle gelé (features ImageNet préservées — optimal pour petit dataset)
  - Tête personnalisée avec Dropout (régularisation anti-overfitting)

Note : le dégel partiel du backbone nécessite >50K images pour être bénéfique.
Avec ~5K images, le backbone gelé donne de meilleurs résultats.
"""
import torch.nn as nn
from torchvision import models


def _build_densenet121(num_classes):
    """
    DenseNet121 : standard en imagerie médicale (CheXNet).
    Connexions denses → capture les textures fines (verre dépoli, opacités).
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # Gel du socle
    for param in model.features.parameters():
        param.requires_grad = False

    # Remplacement du classifieur (in_features = 1024 pour DenseNet121)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def _build_efficientnet_b0(num_classes):
    """
    EfficientNet-B0 : excellent ratio performance/taille.
    Bon généralisateur, résistant à l'overfitting.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

    # Gel du socle
    for param in model.features.parameters():
        param.requires_grad = False

    # Remplacement du classifieur (in_features = 1280 pour EfficientNet-B0)
    num_ftrs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


def _build_resnet50(num_classes):
    """
    ResNet50 : architecture éprouvée avec connexions résiduelles.
    Capture les patterns globaux de la radiographie.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Gel du socle (tout sauf la couche fc)
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False

    # Remplacement de la couche fc
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_ftrs, num_classes),
    )
    return model


# ---------------------------------------------------------------
# Registre des modèles disponibles
# Pour ajouter un nouveau modèle : ajouter une entrée ici.
# ---------------------------------------------------------------
MODEL_REGISTRY = {
    "densenet121": _build_densenet121,
    "efficientnet_b0": _build_efficientnet_b0,
    "resnet50": _build_resnet50,
}


def get_model(name, num_classes=3):
    """
    Factory : retourne un modèle prêt pour le fine-tuning.

    Args:
        name: Clé du registre (ex: "densenet121")
        num_classes: Nombre de classes de sortie

    Returns:
        nn.Module avec socle gelé et tête personnalisée
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Modèle '{name}' inconnu. Disponibles : {available}")

    return MODEL_REGISTRY[name](num_classes)
