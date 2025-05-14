import torch

def evaluate_autoencoder(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            imgs, _ = batch
            imgs = imgs.to(device)

            recon = model(imgs)
            loss = criterion(recon, imgs)
            total_loss += loss.item() * imgs.size(0)

    return total_loss / len(dataloader.dataset)

def evaluate_classifier(encoder, mlp, dataloader, criterion, device):
    encoder.eval()
    mlp.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            z = encoder(imgs)
            logits = mlp(z)
            loss = criterion(logits, labels)

            total_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(logits, dim=1)
            correct += (preds == labels).sum().item()
            total += imgs.size(0)

    return total_loss / total, correct / total
