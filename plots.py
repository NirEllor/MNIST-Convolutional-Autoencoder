import torch
import matplotlib.pyplot as plt


import math

def show_reconstructions(model, dataloader, device, n=30, n_cols=6):
    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:n].to(device)

    with torch.no_grad():
        recons = model(imgs)

    imgs = denormalize(imgs).cpu().numpy()
    recons = denormalize(recons).cpu().numpy()
    print("Reconstruction range:", recons.min(), recons.max())

    n_rows = math.ceil(n / n_cols)

    plt.figure(figsize=(n_cols * 2, n_rows * 4))  # Wider and taller

    for i in range(n):
        # Original
        plt.subplot(2 * n_rows, n_cols, i + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.axis('off')

        # Reconstructed
        plt.subplot(2 * n_rows, n_cols, i + 1 + n_rows * n_cols)
        plt.imshow(recons[i][0], cmap='gray')
        plt.axis('off')

    plt.suptitle("Top: Original, Bottom: Reconstructed", fontsize=16)
    plt.tight_layout()
    plt.show()

def denormalize(tensor):
    return (tensor + 1) / 2  # for inputs normalized with mean=0.5, std=0.5


def plot_metrics(train_losses, test_losses, train_accs, test_accs, title):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, test_losses, 's-', label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss vs Epochs")
    plt.grid()
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'o-', label='Train Accuracy')
    plt.plot(epochs, test_accs, 's-', label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.grid()
    plt.legend()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def plot_latent_space(latent_dims, train_losses, test_losses, model_name):
    # Plotting loss vs latent_dim for this model
    plt.figure(figsize=(8, 5))
    plt.plot(latent_dims, train_losses, marker='o', label='Train Loss')
    plt.plot(latent_dims, test_losses, marker='s', label='Test Loss')
    plt.title(f'{model_name.capitalize()} Model – Loss vs. Latent Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Loss (L1)')
    plt.grid(True)
    plt.legend()
    plt.show()