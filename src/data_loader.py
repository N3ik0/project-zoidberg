import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class Lungdataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.samples = []


        # Parcours récursif pour trouver les images
        for file_path in self.root.rglob('*'):
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:

                label = -1
                parent_folder = file_path.parent.name.lower()
                filename = file_path.name.lower()

                # Logique de tri par nom de dossier et fichier
                if parent_folder == "normal":
                    label = 0
                elif parent_folder == "pneumonia":
                    if "virus" in filename:
                        label = 1
                    elif "bacteria" in filename:
                        label = 2

                # On ajoute uniquement si le fichier correspond à nos critères
                if label != -1:
                    self.samples.append((file_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Récupération du chemin et le label à l'index donné
        img_path, label = self.samples[idx]

        # Ouverture de l'image en mode rgb
        img = Image.open(img_path).convert("RGB")

        # Si on a défini les transfo, on les applique
        if self.transform:
            img = self.transform(img)
        
        return img, label