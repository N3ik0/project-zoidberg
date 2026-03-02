"""
Orchestrateur principal.
Entraîne séquentiellement les modèles de l'ensemble,
puis évalue chaque modèle + l'ensemble combiné.

Usage :
    python main.py train                           # Entraîne les 3 modèles
    python main.py train --models resnet50         # Entraîne un seul modèle
    python main.py train --models resnet50 densenet121  # Sélection multiple
    python main.py evaluate                        # Évalue les modèles sauvegardés
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from src.data import Lungdataset, get_train_transforms, get_val_transforms
from src.models import get_model, MODEL_REGISTRY, EnsemblePredictor
from src.training import train_one_epoch, validate, EarlyStopper, evaluate_model, compute_class_weights

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
NUM_CLASSES = 3
BATCH_SIZE = 16          # Réduit pour plus de mises à jour par epoch
LR = 1e-4
EPOCHS = 50              # Suffisant avec backbone gelé
PATIENCE = 10            # Patience early stopping
LABEL_SMOOTHING = 0.1    # Régularisation anti-surconfiance
GRAD_ACCUM_STEPS = 2     # Simule batch 32 avec batch 16
MIXUP_ALPHA = 0.0        # Désactivé pour l'instant (à activer quand baseline stable)
MODELS_DIR = "models"
TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/test"


def parse_args():
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement et d'évaluation multi-modèles"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # -- train --
    train_parser = subparsers.add_parser("train", help="Entraîner un ou plusieurs modèles")
    train_parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_REGISTRY.keys()) + ["all"],
        default=["all"],
        help="Modèle(s) à entraîner (défaut: all)",
    )

    # -- evaluate --
    subparsers.add_parser("evaluate", help="Évaluer les modèles sauvegardés + ensemble")

    return parser.parse_args()


def resolve_model_names(selection):
    """
    Résout la sélection de modèles.
    Si 'all' est présent, retourne tous les modèles du registre.
    """
    if "all" in selection:
        return list(MODEL_REGISTRY.keys())
    return selection


def train_single_model(name, train_loader, val_loader, class_weights, device):
    """
    Entraîne un seul modèle avec :
    - Label smoothing
    - Mixup data augmentation
    - Gradient accumulation
    - ReduceLROnPlateau scheduler

    Args:
        name: Nom du modèle (clé du registre)
        train_loader: DataLoader d'entraînement
        val_loader: DataLoader de validation
        class_weights: Tensor de poids pour la loss
        device: Device (cuda/cpu)

    Returns:
        Chemin vers le fichier de poids sauvegardé
    """
    print(f"\n{'='*50}")
    print(f"  ENTRAÎNEMENT : {name.upper()}")
    print(f"{'='*50}")
    print(f"  Config : BS={BATCH_SIZE} | LR={LR} | Label Smooth={LABEL_SMOOTHING}")
    print(f"           Mixup α={MIXUP_ALPHA} | Grad Accum={GRAD_ACCUM_STEPS}")

    model = get_model(name, num_classes=NUM_CLASSES).to(device)

    # Comptage des paramètres entraînables
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params : {trainable:,} entraînables / {total:,} total ({trainable/total*100:.1f}%)")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=LABEL_SMOOTHING,
    )
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
    )

    # LR Scheduler : réduit le LR quand la val_loss stagne
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    model_path = os.path.join(MODELS_DIR, f"{name}.pth")
    best_val_loss = float("inf")

    # Si un précédent modèle existe, on récupère son score comme baseline
    if os.path.exists(model_path):
        prev_model = get_model(name, num_classes=NUM_CLASSES).to(device)
        prev_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        prev_val_loss, prev_acc = validate(prev_model, val_loader, criterion, device)
        best_val_loss = prev_val_loss
        print(f"  Score à battre → Val Loss: {best_val_loss:.4f} | Acc: {prev_acc*100:.2f}%")
        del prev_model

    early_stopper = EarlyStopper(patience=PATIENCE)

    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{EPOCHS} (LR: {current_lr:.2e})")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=True, mixup_alpha=MIXUP_ALPHA,
            grad_accum_steps=GRAD_ACCUM_STEPS,
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # LR Scheduler step
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print(f"  🌟 Nouveau record ! Sauvegardé dans {model_path}")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("  ⏹ Early Stopping déclenché.")
            break

    return model_path


def setup():
    """Prépare le device, les datasets et les DataLoaders."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    train_ds = Lungdataset(root_dir=TRAIN_DIR, transform=get_train_transforms())
    val_ds = Lungdataset(root_dir=VAL_DIR, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\nDataset train : {len(train_ds)} images")
    print(f"Dataset val   : {len(val_ds)} images")

    class_weights = compute_class_weights(train_ds)
    print(f"Poids de classe : {class_weights.tolist()}")

    return device, train_loader, val_loader, train_ds, val_ds, class_weights


def cmd_train(model_names, train_loader, val_loader, class_weights, device):
    """Sous-commande train : entraîne les modèles sélectionnés."""
    print(f"\nModèles à entraîner : {', '.join(m.upper() for m in model_names)}")

    saved_paths = {}
    for name in model_names:
        path = train_single_model(name, train_loader, val_loader, class_weights, device)
        saved_paths[name] = path

    # Évaluation individuelle après entraînement
    print(f"\n{'='*50}")
    print(f"  ÉVALUATION INDIVIDUELLE")
    print(f"{'='*50}")

    accuracies = {}
    for name, path in saved_paths.items():
        if os.path.exists(path):
            print(f"\n--- {name.upper()} ---")
            model = get_model(name, num_classes=NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            result = evaluate_model(model, val_loader, device, model_name=name)
            accuracies[name] = result["accuracy"]
            del model

    # Si tous les modèles sont entraînés, évaluer l'ensemble
    all_model_paths = {
        name: os.path.join(MODELS_DIR, f"{name}.pth")
        for name in MODEL_REGISTRY.keys()
    }
    available = {n: p for n, p in all_model_paths.items() if os.path.exists(p)}

    if len(available) > 1:
        _evaluate_ensemble(available, val_loader, device, accuracies)


def cmd_evaluate(val_loader, device):
    """Sous-commande evaluate : évalue les modèles sauvegardés + ensemble."""
    all_model_paths = {
        name: os.path.join(MODELS_DIR, f"{name}.pth")
        for name in MODEL_REGISTRY.keys()
    }
    available = {n: p for n, p in all_model_paths.items() if os.path.exists(p)}

    if not available:
        print("\nAucun modèle sauvegardé trouvé dans le dossier models/.")
        return

    # Évaluation individuelle
    print(f"\n{'='*50}")
    print(f"  ÉVALUATION INDIVIDUELLE")
    print(f"{'='*50}")

    accuracies = {}
    for name, path in available.items():
        print(f"\n--- {name.upper()} ---")
        model = get_model(name, num_classes=NUM_CLASSES).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        result = evaluate_model(model, val_loader, device, model_name=name)
        accuracies[name] = result["accuracy"]
        del model

    # Évaluation ensemble
    if len(available) > 1:
        _evaluate_ensemble(available, val_loader, device, accuracies)


def _evaluate_ensemble(available_models, val_loader, device, accuracies=None):
    """Évalue l'ensemble des modèles disponibles par weighted soft voting."""
    import numpy as np
    from sklearn.metrics import confusion_matrix, classification_report
    from src.training.evaluate import CLASSES, plot_confusion_matrix, RESULTS_DIR

    print(f"\n{'='*50}")
    print(f"  ÉVALUATION DE L'ENSEMBLE (WEIGHTED SOFT VOTING)")
    print(f"  Modèles : {', '.join(n.upper() for n in available_models)}")
    print(f"{'='*50}")

    model_configs = [
        {"name": name, "weights_path": path, "num_classes": NUM_CLASSES}
        for name, path in available_models.items()
    ]

    # Poids basés sur l'accuracy individuelle
    weights = None
    if accuracies:
        weights = [accuracies.get(name, 1.0) for name in available_models]
        total = sum(weights)
        print(f"  Poids : {', '.join(f'{n}={w/total:.3f}' for n, w in zip(available_models, weights))}")

    ensemble = EnsemblePredictor(model_configs, device, weights=weights)
    preds, labels = ensemble.evaluate(val_loader)

    preds = np.array(preds)
    labels = np.array(labels)
    accuracy = (preds == labels).sum() / len(labels) * 100

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, target_names=CLASSES, digits=4)

    print(f"\nPrécision globale de l'ensemble : {accuracy:.2f}%")
    print(f"\nMatrice de confusion :")
    print(f"{'':>12}", end="")
    for c in CLASSES:
        print(f"{c:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{CLASSES[i]:>12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()
    print(f"\n{report}")

    # Heatmap de l'ensemble
    plot_confusion_matrix(
        cm, CLASSES,
        title="Matrice de confusion — ENSEMBLE",
        save_path=os.path.join(RESULTS_DIR, "confusion_ensemble.png"),
    )


def main():
    args = parse_args()

    if args.command is None:
        print("Usage : python main.py {train,evaluate}")
        print("  python main.py train --models resnet50 densenet121")
        print("  python main.py train --models all")
        print("  python main.py evaluate")
        return

    device, train_loader, val_loader, _, _, class_weights = setup()

    if args.command == "train":
        model_names = resolve_model_names(args.models)
        cmd_train(model_names, train_loader, val_loader, class_weights, device)

    elif args.command == "evaluate":
        cmd_evaluate(val_loader, device)


if __name__ == "__main__":
    main()