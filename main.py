import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data_loader import Lungdataset
from src.model import get_model

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Travail sur {device}")

    #Initialisation du model
    model = get_model(num_classes=3)
    model = model.to(device)
    
    # On envoi le batch de test sur le gpu
    images = images.to(device)
    output = model(images)
    print(f"Forme de la sortie : {output.shape}")

if __name__ == "__main__":
    main()