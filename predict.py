"""
Script de prédiction — DenseNet121.
Charge le modèle entraîné et produit un diagnostic.

Usage :
    # Prédiction sur une seule image
    python predict.py --image path/to/xray.png

    # Prédiction sur un dossier entier
    python predict.py --dir data/raw/test/NORMAL

    # Prédiction sur N images d'un dossier
    python predict.py --dir data/raw/test --sample 20

    # Avec Test-Time Augmentation (5 passes)
    python predict.py --image path/to/xray.png --tta 5
"""
import os
import random
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image

from src.data import get_val_transforms, get_tta_transforms
from src.models import get_model
from src.training import CLASSES

MODELS_DIR = "models"
NUM_CLASSES = 3
MODEL_NAME = "densenet121"
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prédiction DenseNet121 sur une ou plusieurs radiographies pulmonaires"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Chemin vers une image unique")
    group.add_argument("--dir", help="Chemin vers un dossier d'images")

    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Nombre d'images à tester (utilisé avec --dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed pour l'échantillonnage reproductible (défaut: 42)",
    )
    parser.add_argument(
        "--tta",
        type=int,
        default=0,
        help="Nombre de passes Test-Time Augmentation (0=désactivé, recommandé: 5)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Seuil de confiance (défaut: 0.7)",
    )
    return parser.parse_args()


def load_image(image_path):
    """Charge et transforme une image pour l'inférence."""
    image = Image.open(image_path).convert("RGB")
    transform = get_val_transforms()
    tensor = transform(image).unsqueeze(0)  # (1, C, H, W)
    return tensor


def collect_images(dir_path, sample_n=None, seed=42):
    """Collecte les images d'un dossier (récursif), avec échantillonnage reproductible."""
    root = Path(dir_path)
    if not root.exists():
        raise FileNotFoundError(f"Dossier introuvable : {dir_path}")

    images = [
        p for p in root.rglob("*")
        if p.suffix.lower() in VALID_EXTENSIONS
    ]

    if not images:
        raise FileNotFoundError(f"Aucune image trouvée dans {dir_path}")

    if sample_n is not None and sample_n < len(images):
        random.seed(seed)
        images = random.sample(images, sample_n)

    return sorted(images)


def predict_single(model, image_tensor, device, threshold=0.7):
    """
    Prédit la classe d'une image avec un seul modèle.

    Returns:
        dict avec classe, probas, confiance, fiable
    """
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probas = F.softmax(output, dim=1)
        confiance, classe = torch.max(probas, dim=1)

    return {
        "classe": classe.item(),
        "probas": probas.squeeze().cpu().tolist(),
        "confiance": confiance.item(),
        "fiable": confiance.item() >= threshold,
    }


def predict_tta(model, image_pil, tta_transforms, device, threshold=0.7):
    """
    Prédit avec Test-Time Augmentation : applique N transforms différentes
    et moyenne les probabilités pour un résultat plus robuste.
    """
    import numpy as np

    model.eval()
    all_probas = []

    with torch.no_grad():
        for transform in tta_transforms:
            tensor = transform(image_pil).unsqueeze(0).to(device)
            output = model(tensor)
            probas = F.softmax(output, dim=1)
            all_probas.append(probas.squeeze().cpu().numpy())

    mean_probas = np.mean(all_probas, axis=0)
    classe = int(np.argmax(mean_probas))
    confiance = float(mean_probas[classe])

    return {
        "classe": classe,
        "probas": mean_probas.tolist(),
        "confiance": confiance,
        "fiable": confiance >= threshold,
        "tta_passes": len(tta_transforms),
    }


def display_single_result(result, image_path=None):
    """Affiche les résultats de la prédiction pour une image."""
    print()
    print("╔══════════════════════════════════════╗")
    print("║        PRÉDICTION DENSENET121        ║")
    print("╚══════════════════════════════════════╝")

    if image_path:
        print(f"  Image : {image_path}")

    print()
    print("━" * 45)

    # Résultat
    classe_finale = CLASSES[result["classe"]]
    confiance_finale = result["confiance"] * 100

    print(f"  DIAGNOSTIC    : {classe_finale}")
    print(f"  Confiance     : {confiance_finale:.2f}%")

    if result["fiable"]:
        print(f"  Fiabilité     : FIABLE ✅")
    else:
        print(f"  Fiabilité     : À REVOIR ⚠️  (confiance < seuil)")

    if "tta_passes" in result:
        print(f"  TTA           : {result['tta_passes']} passes")

    print("━" * 45)

    # Barres de probabilités
    print()
    print("  Probabilités par classe :")
    probas = result["probas"]
    for i, classe_name in enumerate(CLASSES):
        bar_len = int(probas[i] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {classe_name:<12} {bar} {probas[i]*100:.1f}%")

    print()


def display_batch_results(results, threshold=0.7):
    """Affiche un tableau récapitulatif pour un batch de prédictions."""
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                  PRÉDICTION BATCH — DENSENET121                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  {len(results)} images analysées\n")

    # En-tête du tableau
    header = f"  {'#':<4} {'Image':<35} {'Diagnostic':<12} {'Conf.':<8} {'Fiable'}"
    print(header)
    print("  " + "─" * 65)

    # Statistiques globales
    class_counts = {c: 0 for c in CLASSES}
    fiable_count = 0

    for i, entry in enumerate(results, 1):
        result = entry["result"]
        image_name = Path(entry["path"]).name
        if len(image_name) > 33:
            image_name = image_name[:30] + "..."

        classe = CLASSES[result["classe"]]
        confiance = result["confiance"] * 100
        fiable = result["fiable"]

        class_counts[classe] += 1
        if fiable:
            fiable_count += 1

        fiable_str = "✅" if fiable else "⚠️"

        print(f"  {i:<4} {image_name:<35} {classe:<12} {confiance:>5.1f}%  {fiable_str}")

    # Résumé
    total = len(results)

    print()
    print("  " + "═" * 65)
    print(f"  RÉSUMÉ")
    print("  " + "─" * 65)

    for classe_name, count in class_counts.items():
        pct = count / total * 100 if total > 0 else 0
        bar_len = int(pct / 100 * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"    {classe_name:<12} {bar} {count:>3} ({pct:.1f}%)")

    print()
    print(f"  Fiabilité globale  : {fiable_count}/{total} images fiables ({fiable_count/total*100:.1f}%)")
    print()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Chargement du modèle
    model_path = os.path.join(MODELS_DIR, f"{MODEL_NAME}.pth")
    if not os.path.exists(model_path):
        print(f"\n❌ Modèle introuvable : {model_path}")
        print("   Lancez d'abord : python main.py train --phase 1")
        return

    model = get_model(MODEL_NAME, num_classes=NUM_CLASSES, phase=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"Modèle chargé : {MODEL_NAME}")

    # --- Mode image unique ---
    if args.image:
        if not os.path.exists(args.image):
            print(f"\n❌ Image introuvable : {args.image}")
            return
        print(f"Image  : {args.image}")

        if args.tta > 1:
            print(f"TTA    : {args.tta} passes")
            image_pil = Image.open(args.image).convert("RGB")
            tta_transforms = get_tta_transforms(args.tta)
            result = predict_tta(model, image_pil, tta_transforms, device, args.threshold)
        else:
            image_tensor = load_image(args.image)
            result = predict_single(model, image_tensor, device, args.threshold)

        display_single_result(result, args.image)

    # --- Mode batch / dossier ---
    elif args.dir:
        images = collect_images(args.dir, args.sample, seed=args.seed)
        print(f"Dossier : {args.dir}")
        print(f"Images  : {len(images)}" + (f" (seed={args.seed})" if args.sample else ""))
        if args.tta > 1:
            print(f"TTA     : {args.tta} passes par image")

        results = []
        for img_path in images:
            if args.tta > 1:
                image_pil = Image.open(str(img_path)).convert("RGB")
                tta_transforms = get_tta_transforms(args.tta)
                result = predict_tta(model, image_pil, tta_transforms, device, args.threshold)
            else:
                image_tensor = load_image(str(img_path))
                result = predict_single(model, image_tensor, device, args.threshold)
            results.append({"path": str(img_path), "result": result})

        display_batch_results(results, args.threshold)


if __name__ == "__main__":
    main()
