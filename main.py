import os
from pathlib import Path
from src.data_loader import Loader
from src.preprocessing import create_generators
from src.model import build_model

print(f"\n Démarrage du projet")
loader = Loader("data")

#Création du dataset
print(f"\n Création du dataset")
train_ds, test_ds = loader.create_dataset()

# vérification des labels (10premiers)
labels = [y for (x,y) in train_ds[:10]]
print(f"\n Aperçu des 10 premiers labels mélangés : {labels}")

# Préparation pour keras
print(f"Générateur pour Keras")
train_gen, test_gen = create_generators(train_ds, test_ds)

#Verif de la taille du batch
images_batch, labels_batch = next(train_gen)
print(f"\n Forme du batch : {images_batch.shape}")

# Architecture du modèle
print(f"\n Construction du modèle")
model = build_model(num_class=3)

# Affichage du tableau récapitulatif
model.summary()