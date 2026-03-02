<div align="center">
  <img src="assets/icons/zoidberg_logo.svg" width="100" height="100" alt="Zoidberg Logo">
  <h1>Project Zoidberg: Pneumonia Classification</h1>
</div>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)

**Project Zoidberg** is a Deep Learning project developed as part of the **Master 1 in Artificial Intelligence** curriculum.

Its main objective is to assist medical diagnosis by automatically detecting and classifying pneumonia from Chest X-Ray images. Unlike standard binary classifiers, this model distinguishes between **three clinical states**:
1.  **Normal** (Healthy)
2.  **Bacterial Pneumonia**
3.  **Viral Pneumonia**

## Context & Privacy

This project utilizes a dataset of pediatric chest X-rays.
> **⚠️ Data Privacy Notice:**
> Due to the sensitive nature of medical data and privacy regulations, the training dataset **is not included** in this repository. The code is provided for educational and architectural demonstration purposes.

To run this project, you would need to structure your own dataset as described in the [Usage](#-usage) section.

## Architecture & Methodology

The project implements a **Transfer Learning** approach combined with **ensemble learning** to achieve high accuracy and reduce false positives.

* **Models:** 3 architectures pré-entraînées sur ImageNet :
  * **DenseNet121** — Standard en imagerie médicale (CheXNet), capture les textures fines
  * **EfficientNet-B0** — Excellent ratio performance/taille, résistant à l'overfitting
  * **ResNet50** — Connexions résiduelles, capture les patterns globaux
* **Strategy:** Fine-tuning avec socle gelé + tête personnalisée (Dropout + Linear)
* **Ensemble:** Soft voting (moyenne des probabilités softmax) pour un diagnostic plus fiable
* **Data Augmentation:** Rotation, flip horizontal, color jitter, affine transforms

## Project Structure

```text
project-zoidberg/
├── main.py                        # CLI : entraînement & évaluation
├── predict.py                     # Prédiction ensemble sur une image
├── data/
│   └── raw/
│       ├── train/                 # Images d'entraînement
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/         # virus* → Viral, bacteria* → Bactérien
│       └── test/                  # Images de validation
├── models/                        # Poids sauvegardés (.pth)
├── src/
│   ├── data/                      # Chargement & transforms
│   │   ├── loader.py              # Lungdataset (scan récursif + labeling)
│   │   └── augmentation.py        # Train/Val transforms (ImageNet norm)
│   ├── models/                    # Architectures & ensemble
│   │   ├── registry.py            # MODEL_REGISTRY + get_model()
│   │   └── ensemble.py            # EnsemblePredictor (soft voting)
│   └── training/                  # Boucle d'entraînement & évaluation
│       ├── trainer.py             # train_one_epoch, validate, EarlyStopper
│       └── evaluate.py            # Métriques cliniques (confusion, F1, etc.)
└── README.md
```

## 🚀 Usage

### Prérequis

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Entraînement

```bash
# Entraîner les 3 modèles (défaut)
python main.py train

# Entraîner un modèle spécifique
python main.py train --models resnet50

# Entraîner une sélection
python main.py train --models resnet50 densenet121
```

Modèles disponibles : `resnet50`, `densenet121`, `efficientnet_b0`, `all`

Chaque modèle est sauvegardé dans `models/` uniquement s'il bat le score précédent. L'entraînement s'arrête automatiquement si la validation loss ne s'améliore plus (Early Stopping, patience = 7).

### Évaluation

```bash
# Évaluer tous les modèles sauvegardés + l'ensemble
python main.py evaluate
```

Affiche pour chaque modèle : accuracy, matrice de confusion, precision/recall/F1 par classe. Puis évalue l'ensemble par soft voting.

### Prédiction

```bash
# Prédiction sur une image unique
python predict.py --image path/to/xray.png

# Prédiction sur un dossier entier
python predict.py --dir data/raw/test/NORMAL

# Prédiction sur N images aléatoires d'un dossier (récursif)
python predict.py --dir data/raw/test --sample 20

# Avec un seuil de confiance personnalisé
python predict.py --dir data/raw/test --sample 10 --threshold 0.8
```

- **Mode image** : affiche le diagnostic de chaque modèle + le diagnostic ensemble avec probabilités
- **Mode batch** : affiche un tableau récapitulatif avec la distribution par classe, le taux de fiabilité et le consensus moyen