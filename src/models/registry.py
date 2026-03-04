"""
Factory DenseNet121 avec entraînement en 2 phases (Discriminative Fine-Tuning).

Phase 1 — Linear Probing :
    Backbone 100% gelé, seul le classifieur est entraîné.
    Objectif : calibrer la tête de classification sur nos 3 classes.

Phase 2 — Fine-Tuning partiel :
    Dégèle les blocs profonds (denseblock3, transition3, denseblock4)
    pour adapter les features de haut niveau aux textures pulmonaires.
    Les BatchNorm restent gelées pour stabiliser les petits batches.
"""
import torch.nn as nn
from torchvision import models


def _build_densenet121(num_classes, phase=1):
    """
    Construit un DenseNet121 pré-entraîné configuré selon la phase d'entraînement.

    Args:
        num_classes: Nombre de classes de sortie (3 : Normal, Viral, Bactérien)
        phase: 1 = Linear Probing, 2 = Fine-Tuning partiel

    Returns:
        nn.Module prêt pour l'entraînement
    """
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

    # Geler 100% du backbone
    for param in model.parameters():
        param.requires_grad = False

    if phase == 2:
        # Dégeler les blocs profonds pour le fine-tuning
        for name, param in model.features.named_parameters():
            if any(block in name for block in ["denseblock3", "transition3", "denseblock4"]):
                param.requires_grad = True

        # Re-geler les BatchNorm (stats bruitées avec petit batch)
        for name, module in model.features.named_modules():
            if isinstance(module, nn.BatchNorm2d) and any(
                block in name for block in ["denseblock3", "transition3", "denseblock4"]
            ):
                module.requires_grad_(False)
                module.eval()  # Utilise les stats ImageNet

    # Tête de classification personnalisée (toujours entraînable)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_ftrs, num_classes)
    )

    return model


def get_model(name="densenet121", num_classes=3, phase=1):
    """
    Factory : retourne un DenseNet121 configuré pour la phase spécifiée.

    Args:
        name: Nom du modèle (seul "densenet121" est supporté)
        num_classes: Nombre de classes de sortie
        phase: 1 = Linear Probing, 2 = Fine-Tuning partiel

    Returns:
        nn.Module prêt pour l'entraînement
    """
    if name != "densenet121":
        raise ValueError(f"Modèle '{name}' non supporté. Seul 'densenet121' est disponible.")

    return _build_densenet121(num_classes, phase=phase)
