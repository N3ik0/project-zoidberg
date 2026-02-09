import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data_loader import Lungdataset
from src.model import get_model
import torch.optim as optim
import torch.nn as nn
from src.train import train_one_epoch

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

    #Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    for epoch in range(50):
        train_loss = train_one_epoch(model, train_loader, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)

        #Sauvegarde du model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dic(), 'models/best_model.pth')
        
        early_stopper(val_loss)
        if early_stopper.early_stop:
            break

if __name__ == "__main__":
    main()