import torch

def evaluate_autoencoder(model, dataloader, criterion, device):
    """
    Evaluates the reconstruction loss of an autoencoder on a given dataset.

    Args:
        model (nn.Module): Autoencoder model (encoder + decoder).
        dataloader (DataLoader): DataLoader for evaluation set.
        criterion (nn.Module): Loss function (e.g., L1 loss).
        device (torch.device): Device to run evaluation on.

    Returns:
        float: Average reconstruction loss over the dataset.
    """
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
    """
    Evaluates classification performance using a frozen or trainable encoder and MLP.

    Args:
        encoder (nn.Module): Encoder network that outputs latent vectors.
        mlp (nn.Module): MLP classifier that maps latent vectors to logits.
        dataloader (DataLoader): DataLoader for evaluation set.
        criterion (nn.Module): Classification loss function (e.g., CrossEntropyLoss).
        device (torch.device): Device to run evaluation on.

    Returns:
        Tuple[float, float]: (average loss, classification accuracy)
    """
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
