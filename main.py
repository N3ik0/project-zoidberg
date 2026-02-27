import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
from src.data_loader import Lungdataset
from src.model import get_model
import torch.optim as optim
import torch.nn as nn
from src.train import train_one_epoch, validate, EarlyStopper

def main():
    # 1. Définition de la "recette"
    my_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Instanciation du Dataset
    train_ds = Lungdataset(root_dir="data/raw/train", transform=my_transforms)
    val_ds = Lungdataset(root_dir="data/raw/test", transform=my_transforms)

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

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
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

    # Dossier models
    if not os.path.exists('models'):
        os.makedirs('models')

    best_val_loss = float('inf')
    
    # ---------------------------------------------------------
    # GESTION DES PRECEDENTES SAUVEGARDES
    # ---------------------------------------------------------
    model_path = 'models/best_model.pth'
    if os.path.exists(model_path):
        print(f"\nUn précédent modèle ({model_path}) existe.")
        print("Évaluation du modèle précédent pour établir le score à battre...")
        
        # On crée un modèle temporel juste pour tester
        prev_model = get_model(num_classes=3).to(device)
        prev_model.load_state_dict(torch.load(model_path, weights_only=True))
        
        # On recupere sa performance
        prev_val_loss, prev_acc = validate(prev_model, val_loader, criterion, device)
        best_val_loss = prev_val_loss
        
        print(f"Score à battre -> Val Loss: {best_val_loss:.4f} | Précision: {prev_acc*100:.2f}%\n")
        del prev_model # Libération de la mémoire

    early_stopper = EarlyStopper(patience=5)

    for epoch in range(50):
        print(f"Epoch {epoch+1}/50")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        #Sauvegarde du model uniquement s'il est meilleur que l'historique
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            print("🌟 Nouveau record ! Le modèle (historique) a été sauvegardé.")
        
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print("Arrêt précoce (Early Stopping) déclenché. Le modèle n'apprend plus.")
            break

    # ---------------------------------------------------------
    # BILAN FINAL
    # ---------------------------------------------------------
    if os.path.exists(model_path):
        print("\nChargement du meilleur modèle pour l'évaluation finale...")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        from src.train import evaluate_model
        evaluate_model(model, val_loader, device)
    else:
        print("\nAucun modèle n'a été sauvegardé.")

if __name__ == "__main__":
    main()