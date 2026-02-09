import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data_loader import Lungdataset

def main():
    # 1. Définition de la "recette"
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Instanciation du Dataset
    train_ds = Lungdataset(root_dir="data/raw/train", transform=my_transforms)

    # 3. Création du DataLoader (Le chariot qui livre les images au GPU)
    # Batchsize (a voir pour augmenter)
    # shuffle (melange a chaque epoch)
    # Utilise plusieurs coeurs CPU pour chargement parallèle
    train_loader = DataLoader(
        train_ds, 
        batch_size=64,
        shuffle=True,
        num_workers=4 
    )

    # Petit test de vérification
    images, labels = next(iter(train_loader))
    print(f"Forme du batch d'images : {images.shape}")
    print(f"Labels du batch : {labels}")

if __name__ == "__main__":
    main()