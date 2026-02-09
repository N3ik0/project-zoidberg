import torch

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        # envoi sur le gpu
        images, labels = images.to(device), labels.to(device)

        # Remise a zero des gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # backward pass
        loss.backward()

        # update des poids
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)