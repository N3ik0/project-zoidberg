"""
Orchestrateur principal — DenseNet121 avec entraînement 2-phases.

Phase 1 (Linear Probing) : Backbone gelé, seul le classifieur s'entraîne.
Phase 2 (Fine-Tuning)    : Dégel des blocs profonds pour affiner les features.

Usage :
    python main.py train                          # Phase 1 par défaut
    python main.py train --phase 2                # Fine-tuning partiel
    python main.py train --phase 1 --seed 42      # Reproductible
    python main.py train --seed 42 123 456        # Multi-seed → moyenne ± std
    python main.py evaluate                       # Évalue le modèle sauvegardé
"""
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data import Lungdataset, get_train_transforms, get_val_transforms
from src.models import get_model
from src.training import train_one_epoch, validate, EarlyStopper, evaluate_model, compute_class_weights

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
NUM_CLASSES = 3
BATCH_SIZE = 16
EPOCHS = 50
PATIENCE = 15
LABEL_SMOOTHING = 0.05
GRAD_ACCUM_STEPS = 2
MODELS_DIR = "models"
TRAIN_DIR = "data/raw/train"
VAL_DIR = "data/raw/test"
DEFAULT_SEED = 42

# Hyperparamètres par phase
PHASE_CONFIG = {
    1: {"lr_head": 1e-3, "lr_backbone": 0.0,  "description": "Linear Probing (backbone 100% gelé)"},
    2: {"lr_head": 1e-4, "lr_backbone": 1e-5, "description": "Fine-Tuning partiel (denseblock4 + transition3)"},
}

MIXUP_ALPHA = 0.2


def set_seed(seed):
    """
    Fixe toutes les sources d'aléatoire pour une reproductibilité totale.
    Même seed → mêmes batchs, même ordre, mêmes augmentations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed fixée : {seed} (mode déterministe)")


def parse_args():
    """Parse les arguments CLI."""
    parser = argparse.ArgumentParser(
        description="Pipeline d'entraînement DenseNet121 — 2 phases"
    )
    subparsers = parser.add_subparsers(dest="command", help="Commande à exécuter")

    # -- train --
    train_parser = subparsers.add_parser("train", help="Entraîner le modèle")
    train_parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        help="Modèle à entraîner (défaut: densenet121)",
    )
    train_parser.add_argument(
        "--phase",
        type=int,
        choices=[1, 2],
        default=1,
        help="Phase d'entraînement : 1=Linear Probing, 2=Fine-Tuning (défaut: 1)",
    )
    train_parser.add_argument(
        "--seed",
        nargs="+",
        type=int,
        default=[DEFAULT_SEED],
        help="Seed(s) pour la reproductibilité (défaut: 42). Plusieurs seeds → moyenne ± std.",
    )

    # -- evaluate --
    eval_parser = subparsers.add_parser("evaluate", help="Évaluer le modèle sauvegardé")
    eval_parser.add_argument(
        "--model",
        type=str,
        default="densenet121",
        help="Modèle à évaluer (défaut: densenet121)",
    )

    return parser.parse_args()


def train_model(name, phase, train_loader, val_loader, class_weights, device):
    """
    Entraîne le modèle avec la configuration de la phase spécifiée.

    Phase 1 : LR élevé (1e-3), backbone gelé → convergence rapide du classifieur
    Phase 2 : LR bas (1e-5), blocs profonds dégelés → affinage des features
    """
    config = PHASE_CONFIG[phase]
    lr_head = config["lr_head"]
    lr_backbone = config["lr_backbone"]

    print(f"\n{'='*60}")
    print(f"  ENTRAÎNEMENT : {name.upper()} — PHASE {phase}")
    print(f"  {config['description']}")
    print(f"{'='*60}")
    print(f"  Config : BS={BATCH_SIZE} | LR Head={lr_head:.1e} | LR Backbone={lr_backbone:.1e}")
    print(f"           Label Smooth={LABEL_SMOOTHING} | Mixup={MIXUP_ALPHA if phase == 2 else 'Non'}")
    print(f"           Grad Accum={GRAD_ACCUM_STEPS} | Patience={PATIENCE}")

    model = get_model(name, num_classes=NUM_CLASSES, phase=phase).to(device)

    # Comptage des paramètres entraînables
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params : {trainable:,} entraînables / {total:,} total ({trainable/total*100:.1f}%)")

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)

    # Séparation des paramètres pour Discriminative LR
    head_params = []
    backbone_params = []
    for p_name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "classifier" in p_name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.Adam([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head}
    ], weight_decay=1e-4)

    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)

    model_path = os.path.join(MODELS_DIR, f"{name}.pth")
    best_val_acc = 0.0
    best_val_loss = float("inf")

    # Si un précédent modèle existe, on récupère son score comme baseline
    if os.path.exists(model_path):
        prev_model = get_model(name, num_classes=NUM_CLASSES, phase=phase).to(device)
        prev_model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=True)
        )
        prev_val_loss, prev_acc = validate(prev_model, val_loader, criterion, device)
        best_val_acc = prev_acc
        best_val_loss = prev_val_loss
        print(f"  Score à battre → Val Acc: {best_val_acc*100:.2f}% | (Loss: {prev_val_loss:.4f})")
        del prev_model

        # En phase 2, charger les poids de la phase 1 comme point de départ
        if phase == 2:
            model.load_state_dict(
                torch.load(model_path, map_location=device, weights_only=True)
            )
            print(f"  📦 Poids de la Phase 1 chargés comme point de départ")

    early_stopper = EarlyStopper(patience=PATIENCE)

    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\nEpoch {epoch + 1}/{EPOCHS} (LR: {current_lr:.2e})")

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=(phase == 2),
            mixup_alpha=MIXUP_ALPHA if phase == 2 else 0.0,
            grad_accum_steps=GRAD_ACCUM_STEPS,
            scheduler=None  # CosineAnnealingLR step par epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Sauvegarde basée sur l'amélioration de l'Accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  🌟 Nouveau record ! Sauvegardé dans {model_path} (Acc: {best_val_acc*100:.2f}%)")

        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("  ⏹ Early Stopping déclenché.")
            break

    return model_path


def _seed_worker(worker_id):
    """Assure que chaque worker DataLoader utilise un seed déterministe."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def setup(seed=DEFAULT_SEED):
    """Prépare le device, les datasets et les DataLoaders (déterministe)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    if device.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(0)}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    train_ds = Lungdataset(root_dir=TRAIN_DIR, transform=get_train_transforms())
    val_ds = Lungdataset(root_dir=VAL_DIR, transform=get_val_transforms())

    class_weights = compute_class_weights(train_ds)
    print(f"Poids de classe : {class_weights.tolist()}")

    # Générateur déterministe
    g = torch.Generator()
    g.manual_seed(seed)

    sample_weights = [class_weights[label].item() for _, label in train_ds.samples]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
        generator=g
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=4, pin_memory=True,
        worker_init_fn=_seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True
    )

    print(f"\nDataset train : {len(train_ds)} images")
    print(f"Dataset val   : {len(val_ds)} images")

    return device, train_loader, val_loader, class_weights


def cmd_train(model_name, phase, train_loader, val_loader, class_weights, device):
    """Sous-commande train : entraîne puis évalue le modèle."""
    print(f"\n📋 Modèle : {model_name.upper()} | Phase : {phase}")

    model_path = train_model(model_name, phase, train_loader, val_loader, class_weights, device)

    # Évaluation après entraînement
    if os.path.exists(model_path):
        print(f"\n{'='*60}")
        print(f"  ÉVALUATION — {model_name.upper()}")
        print(f"{'='*60}")

        model = get_model(model_name, num_classes=NUM_CLASSES, phase=phase).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        evaluate_model(model, val_loader, device, model_name=model_name)
        del model


def cmd_evaluate(model_name, val_loader, device):
    """Sous-commande evaluate : évalue le modèle sauvegardé."""
    model_path = os.path.join(MODELS_DIR, f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"\nAucun modèle sauvegardé trouvé : {model_path}")
        return

    print(f"\n{'='*60}")
    print(f"  ÉVALUATION — {model_name.upper()}")
    print(f"{'='*60}")

    # Phase n'importe pas pour l'évaluation (architecture identique)
    model = get_model(model_name, num_classes=NUM_CLASSES, phase=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    evaluate_model(model, val_loader, device, model_name=model_name)
    del model


def main():
    args = parse_args()

    if args.command is None:
        print("Usage : python main.py {train,evaluate}")
        print("  python main.py train                    # Phase 1 (Linear Probing)")
        print("  python main.py train --phase 2          # Phase 2 (Fine-Tuning)")
        print("  python main.py train --seed 42 123 456  # Multi-seed")
        print("  python main.py evaluate                 # Évaluation")
        return

    if args.command == "train":
        seeds = args.seed
        model_name = args.model
        phase = args.phase

        if len(seeds) == 1:
            set_seed(seeds[0])
            device, train_loader, val_loader, class_weights = setup(seed=seeds[0])
            cmd_train(model_name, phase, train_loader, val_loader, class_weights, device)
        else:
            # Mode multi-seed : moyenne ± écart-type
            print(f"\n🔬 Mode multi-seed : {len(seeds)} runs avec seeds {seeds}")
            all_accs = []

            for i, seed in enumerate(seeds):
                print(f"\n{'#'*60}")
                print(f"  RUN {i+1}/{len(seeds)} — SEED {seed}")
                print(f"{'#'*60}")

                set_seed(seed)
                device, train_loader, val_loader, class_weights = setup(seed=seed)
                cmd_train(model_name, phase, train_loader, val_loader, class_weights, device)

                # Récupérer l'accuracy pour ce run
                path = os.path.join(MODELS_DIR, f"{model_name}.pth")
                if os.path.exists(path):
                    model = get_model(model_name, num_classes=NUM_CLASSES, phase=phase).to(device)
                    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
                    _, val_acc = validate(model, val_loader, nn.CrossEntropyLoss(), device)
                    all_accs.append(val_acc * 100)
                    del model

            # Rapport final
            print(f"\n{'='*60}")
            print(f"  RAPPORT MULTI-SEED ({len(seeds)} runs)")
            print(f"{'='*60}")
            if all_accs:
                mean = np.mean(all_accs)
                std = np.std(all_accs)
                print(f"  {model_name.upper():20s} : {mean:.2f}% ± {std:.2f}%")
                print(f"  Runs : {[f'{a:.2f}' for a in all_accs]}")

    elif args.command == "evaluate":
        set_seed(DEFAULT_SEED)
        device, train_loader, val_loader, _ = setup()
        cmd_evaluate(args.model, val_loader, device)


if __name__ == "__main__":
    main()