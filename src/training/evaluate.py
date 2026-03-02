"""
Module d'évaluation clinique.
Fournit les métriques pertinentes pour un outil d'aide au diagnostic :
  - Matrice de confusion (texte + heatmap)
  - Precision / Recall / F1-Score par classe
  - Calcul des poids de classe pour compenser le déséquilibre du dataset
"""
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif (pas besoin de display)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import confusion_matrix, classification_report


CLASSES = ["Normal", "Viral", "Bactérien"]
RESULTS_DIR = "results"


def plot_confusion_matrix(cm, classes=None, title="Matrice de confusion", save_path=None):
    """
    Génère une heatmap de la matrice de confusion avec seaborn.

    Args:
        cm: Matrice de confusion (numpy array)
        classes: Liste des noms de classes
        title: Titre du graphique
        save_path: Chemin de sauvegarde (si None, sauvegarde dans results/)

    Returns:
        Chemin vers l'image sauvegardée
    """
    if classes is None:
        classes = CLASSES

    os.makedirs(RESULTS_DIR, exist_ok=True)

    if save_path is None:
        safe_title = title.lower().replace(" ", "_").replace("'", "")
        save_path = os.path.join(RESULTS_DIR, f"{safe_title}.png")

    # Calcul des pourcentages par ligne (par vraie classe)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    # Annotations : "count\n(xx.x%)"
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"label": "Nombre de prédictions"},
        ax=ax,
    )

    ax.set_xlabel("Prédiction", fontsize=12, fontweight="bold")
    ax.set_ylabel("Vrai label", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"  📊 Heatmap sauvegardée → {save_path}")
    return save_path


def evaluate_model(model, loader, device, classes=None, model_name=None):
    """
    Évalue un modèle et affiche le bilan clinique complet.

    Args:
        model: Le modèle PyTorch à évaluer
        loader: DataLoader du jeu de validation/test
        device: Device (cuda/cpu)
        classes: Liste des noms de classes
        model_name: Nom du modèle (pour le titre de la heatmap)
    """
    if classes is None:
        classes = CLASSES

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Accuracy globale
    accuracy = (all_preds == all_labels).sum() / len(all_labels) * 100

    # Matrice de confusion
    cm = confusion_matrix(all_labels, all_preds)

    # Classification report (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)

    print("\n╔══════════════════════════════════════╗")
    print("║      BILAN D'ÉVALUATION CLINIQUE     ║")
    print("╚══════════════════════════════════════╝")
    print(f"\nPrécision globale : {accuracy:.2f}%")
    print(f"\nMatrice de confusion :")
    print(f"{'':>12}", end="")
    for c in classes:
        print(f"{c:>12}", end="")
    print()
    for i, row in enumerate(cm):
        print(f"{classes[i]:>12}", end="")
        for val in row:
            print(f"{val:>12}", end="")
        print()

    print(f"\nRapport détaillé :")
    print(report)

    # Génération de la heatmap
    if model_name:
        title = f"Matrice de confusion — {model_name.upper()}"
        save_path = os.path.join(RESULTS_DIR, f"confusion_{model_name}.png")
    else:
        title = "Matrice de confusion"
        save_path = None

    heatmap_path = plot_confusion_matrix(cm, classes, title=title, save_path=save_path)

    return {"accuracy": accuracy, "confusion_matrix": cm, "report": report, "heatmap": heatmap_path}


def compute_class_weights(dataset):
    """
    Calcule les poids inversement proportionnels à la fréquence de chaque classe.
    Permet de compenser le déséquilibre du dataset dans la CrossEntropyLoss.

    Args:
        dataset: Un objet Dataset dont chaque sample retourne (image, label)

    Returns:
        torch.Tensor de poids, un par classe
    """
    # Compter les labels sans charger les images
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)

    total = sum(counts.values())
    num_classes = len(counts)

    # Poids = total / (num_classes * count_per_class)
    weights = []
    for cls_idx in sorted(counts.keys()):
        w = total / (num_classes * counts[cls_idx])
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)

