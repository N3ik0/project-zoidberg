"""
Script de prédiction par ensemble.
Charge les 3 modèles entraînés et produit un diagnostic combiné.

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
from PIL import Image

from src.data import get_val_transforms, get_tta_transforms
from src.models import MODEL_REGISTRY, EnsemblePredictor
from src.training import CLASSES

MODELS_DIR = "models"
NUM_CLASSES = 3
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prédiction ensemble sur une ou plusieurs radiographies pulmonaires"
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


def find_available_models():
    """Détecte les modèles entraînés disponibles dans le dossier models/."""
    available = []
    for name in MODEL_REGISTRY.keys():
        path = os.path.join(MODELS_DIR, f"{name}.pth")
        if os.path.exists(path):
            available.append({"name": name, "weights_path": path, "num_classes": NUM_CLASSES})
    return available


def display_single_result(result, image_path=None):
    """Affiche les résultats de la prédiction pour une image."""
    details = result["details"]
    num_models = len(details)

    print()
    print("╔══════════════════════════════════════╗")
    print("║        PRÉDICTION ENSEMBLE           ║")
    print("╚══════════════════════════════════════╝")

    if image_path:
        print(f"  Image : {image_path}")

    print()

    # Résultats individuels
    for detail in details:
        name = detail["name"]
        classe = CLASSES[detail["classe"]]
        confiance = detail["confiance"] * 100
        display_name = name.replace("_", " ").title()
        print(f"  {display_name:<18} → {classe:<12} (confiance: {confiance:.1f}%)")

    print()
    print("━" * 45)

    # Résultat ensemble
    classe_finale = CLASSES[result["classe"]]
    confiance_finale = result["confiance"] * 100

    print(f"  DIAGNOSTIC FINAL : {classe_finale}")
    print(f"  Confiance        : {confiance_finale:.2f}%")

    consensus = sum(1 for d in details if d["classe"] == result["classe"])
    if consensus == num_models:
        print(f"  Consensus        : {consensus}/{num_models} modèles concordent ✅")
    else:
        print(f"  Consensus        : {consensus}/{num_models} modèles concordent ⚠️")

    if result["fiable"]:
        print(f"  Fiabilité        : FIABLE ✅")
    else:
        print(f"  Fiabilité        : À REVOIR ⚠️  (confiance < seuil)")

    print("━" * 45)

    # Barres de probabilités
    print()
    print("  Probabilités moyennes par classe :")
    probas = result["probas"]
    for i, classe_name in enumerate(CLASSES):
        bar_len = int(probas[i] * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {classe_name:<12} {bar} {probas[i]*100:.1f}%")

    print()


def display_batch_results(results):
    """Affiche un tableau récapitulatif pour un batch de prédictions."""
    num_models = len(results[0]["result"]["details"])

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                  PRÉDICTION BATCH — ENSEMBLE                   ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  {len(results)} images analysées\n")

    # En-tête du tableau
    header = f"  {'#':<4} {'Image':<35} {'Diagnostic':<12} {'Conf.':<8} {'Cons.':<7} {'Fiable'}"
    print(header)
    print("  " + "─" * 75)

    # Statistiques globales
    class_counts = {c: 0 for c in CLASSES}
    fiable_count = 0
    consensus_total = 0

    for i, entry in enumerate(results, 1):
        result = entry["result"]
        image_name = Path(entry["path"]).name
        if len(image_name) > 33:
            image_name = image_name[:30] + "..."

        classe = CLASSES[result["classe"]]
        confiance = result["confiance"] * 100
        consensus = sum(1 for d in result["details"] if d["classe"] == result["classe"])
        fiable = result["fiable"]

        class_counts[classe] += 1
        if fiable:
            fiable_count += 1
        consensus_total += consensus

        cons_str = f"{consensus}/{num_models}"
        fiable_str = "✅" if fiable else "⚠️"

        print(f"  {i:<4} {image_name:<35} {classe:<12} {confiance:>5.1f}%  {cons_str:<7} {fiable_str}")

    # Résumé
    total = len(results)
    avg_consensus = consensus_total / total if total > 0 else 0

    print()
    print("  " + "═" * 75)
    print(f"  RÉSUMÉ")
    print(f"  " + "─" * 75)

    for classe_name, count in class_counts.items():
        pct = count / total * 100 if total > 0 else 0
        bar_len = int(pct / 100 * 25)
        bar = "█" * bar_len + "░" * (25 - bar_len)
        print(f"    {classe_name:<12} {bar} {count:>3} ({pct:.1f}%)")

    print()
    print(f"  Fiabilité globale  : {fiable_count}/{total} images fiables ({fiable_count/total*100:.1f}%)")
    print(f"  Consensus moyen    : {avg_consensus:.1f}/{num_models} modèles")
    print()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # Détection des modèles
    model_configs = find_available_models()
    if not model_configs:
        print("\n❌ Aucun modèle entraîné trouvé dans le dossier models/.")
        print("   Lancez d'abord : python main.py train --models all")
        return

    print(f"Modèles chargés : {', '.join(c['name'] for c in model_configs)}")

    ensemble = EnsemblePredictor(model_configs, device, confidence_threshold=args.threshold)

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
            result = ensemble.predict_tta(image_pil, tta_transforms)
        else:
            image_tensor = load_image(args.image)
            result = ensemble.predict(image_tensor)

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
                result = ensemble.predict_tta(image_pil, tta_transforms)
            else:
                image_tensor = load_image(str(img_path))
                result = ensemble.predict(image_tensor)
            results.append({"path": str(img_path), "result": result})

        display_batch_results(results)


if __name__ == "__main__":
    main()
