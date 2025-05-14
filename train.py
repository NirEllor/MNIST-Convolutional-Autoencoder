import torch
def train_autoencoder(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        imgs, _ = batch  # we don't need the labels
        imgs = imgs.to(device)

        optimizer.zero_grad()
        recon = model(imgs)
        loss = criterion(recon, imgs)  # compare reconstruction to original
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)

    return total_loss / len(dataloader.dataset)

def train_classifier(encoder, mlp, dataloader, optimizer, criterion, device):
    encoder.train()
    mlp.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        z = encoder(imgs)
        logits = mlp(z)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)

    return total_loss / total, correct / total


