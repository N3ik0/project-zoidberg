"""
Boucle d'entraînement et validation avec Mixup et gradient accumulation.
Réutilisable avec n'importe quel modèle PyTorch.
"""
import torch
import numpy as np


def mixup_data(images, labels, alpha=0.4):
    """
    Applique le Mixup sur un batch : mélange linéaire de paires d'images et labels.
    Technique prouvée pour améliorer la généralisation.

    Args:
        images: Batch d'images (B, C, H, W)
        labels: Batch de labels (B,)
        alpha: Paramètre de la distribution Beta (plus grand = plus de mélange)

    Returns:
        mixed_images, labels_a, labels_b, lam
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = images.size(0)
    index = torch.randperm(batch_size, device=images.device)

    mixed_images = lam * images + (1 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]

    return mixed_images, labels_a, labels_b, lam


def mixup_criterion(criterion, outputs, labels_a, labels_b, lam):
    """Calcule la loss Mixup : combinaison pondérée des deux losses."""
    return lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)


def train_one_epoch(model, loader, criterion, optimizer, device,
                    use_mixup=True, mixup_alpha=0.4, grad_accum_steps=1):
    """
    Entraîne le modèle sur un epoch complet.

    Args:
        model: Modèle PyTorch
        loader: DataLoader d'entraînement
        criterion: Fonction de loss
        optimizer: Optimiseur
        device: Device (cuda/cpu)
        use_mixup: Activer le Mixup
        mixup_alpha: Paramètre alpha du Mixup
        grad_accum_steps: Nombre de steps d'accumulation de gradients
    """
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for step, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        # Mixup
        if use_mixup:
            images, labels_a, labels_b, lam = mixup_data(images, labels, mixup_alpha)
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Gradient accumulation : diviser la loss par le nombre de steps
        loss = loss / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * grad_accum_steps  # Compenser la division

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """Évalue le modèle sur le jeu de validation. Retourne (loss, accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_loss = running_loss / len(loader)
    accuracy = correct / total
    return val_loss, accuracy


class EarlyStopper:
    """Arrêt précoce si la val_loss ne s'améliore plus pendant `patience` epochs."""

    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
