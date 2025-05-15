import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalizes a tensor from [-1, 1] to [0, 1].

    Args:
        tensor (Tensor): A normalized image tensor.

    Returns:
        Tensor: Denormalized tensor.
    """
    return (tensor + 1) / 2


def show_reconstructions(model, dataloader, device, n=10):
    """
    Displays original-reconstruction pairs in alternating columns, row-wise.

    Args:
        model (nn.Module): Trained autoencoder.
        dataloader (DataLoader): DataLoader to sample data from.
        device (torch.device): Torch device ("cpu" or "cuda").
        n (int): Number of total samples (must be even).
    """
    assert n % 2 == 0, "n must be even for proper pairing"

    model.eval()
    imgs, _ = next(iter(dataloader))
    imgs = imgs[:n].to(device)

    with torch.no_grad():
        recons = model(imgs)

    imgs = denormalize(imgs).cpu().numpy()
    recons = denormalize(recons).cpu().numpy()
    print("Reconstruction range:", recons.min(), recons.max())

    n_cols = 10  # total columns per row (5 pairs)
    n_rows = n // (n_cols // 2)  # since each pair takes 2 columns

    plt.figure(figsize=(n_cols * 1.2, n_rows * 2.5))
    for i in range(n):
        row = i // (n_cols // 2)
        col = (i % (n_cols // 2)) * 2
        ax_idx = row * n_cols + col

        # Original
        plt.subplot(n_rows, n_cols, ax_idx + 1)
        plt.imshow(imgs[i][0], cmap='gray')
        plt.axis('off')

        # Reconstruction
        plt.subplot(n_rows, n_cols, ax_idx + 2)
        plt.imshow(recons[i][0], cmap='gray')
        plt.axis('off')

    plt.suptitle("Original–Reconstruction Pairs", fontsize=16)
    plt.tight_layout()
    plt.show()


def plot_metrics(train_losses, test_losses, train_accs, test_accs, title="Training Curves"):
    """
    Plots training and testing loss and accuracy over epochs.

    Args:
        train_losses (list): Training loss values.
        test_losses (list): Testing loss values.
        train_accs (list): Training accuracy values.
        test_accs (list): Testing accuracy values.
        title (str): Title for the plot.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, test_losses, 's-', label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Loss vs Epochs")
    plt.grid()
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'o-', label='Train Accuracy')
    plt.plot(epochs, test_accs, 's-', label='Test Accuracy')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.grid()
    plt.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def plot_loss_only(train_losses, test_losses, title="Loss Curve"):
    """
    Plots training and testing loss over epochs.

    Args:
        train_losses (list): Training loss values.
        test_losses (list): Testing loss values.
        title (str): Plot title.
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(6, 5))
    plt.plot(epochs, train_losses, 'o-', label='Train Loss')
    plt.plot(epochs, test_losses, 's-', label='Test Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_latent_space(latent_dims, train_losses, test_losses, model_name="Autoencoder"):
    """
    Plots train/test L1 loss as a function of latent dimension.

    Args:
        latent_dims (list): Latent dimensionality (plural) used.
        train_losses (list): Corresponding training losses.
        test_losses (list): Corresponding testing losses.
        model_name (str): Name of the model (used in plot title).
    """
    if not (len(latent_dims) == len(train_losses) == len(test_losses)):
        print("⚠️ Skipping latent space plot due to mismatched list lengths.")
        return

    plt.figure(figsize=(8, 5))
    plt.plot(latent_dims, train_losses, marker='o', label='Train Loss')
    plt.plot(latent_dims, test_losses, marker='s', label='Test Loss')
    plt.title(f'{model_name.capitalize()} – Loss vs. Latent Dimension')
    plt.xlabel('Latent Dimension')
    plt.ylabel('L1 Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_latent_space_separation(encoder, dataloader, device, method='tsne', num_samples=1000):
    encoder.eval()
    latents = []
    labels = []

    with torch.no_grad():
        for imgs, lbls in dataloader:
            imgs = imgs.to(device)
            z = encoder(imgs)
            latents.append(z.cpu())
            labels.append(lbls)

            if len(latents) * imgs.size(0) >= num_samples:
                break

    latents = torch.cat(latents)[:num_samples]
    labels = torch.cat(labels)[:num_samples]

    if method == 'tsne':
        projected = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(latents)
    else:
        from sklearn.decomposition import PCA
        projected = PCA(n_components=2).fit_transform(latents)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(projected[:, 0], projected[:, 1], c=labels, cmap='tab10', s=15)
    plt.colorbar(scatter, ticks=range(10), label='Digit Class')
    plt.title(f"{method.upper()} projection of latent space")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def show_all_digits_variability(model, dataloader, device, n=10):
    """
    Shows reconstructions for each digit (0–9) to assess in-class variability using a Q4 model.

    For each digit:
    - Shows n original images (top row)
    - Shows n reconstructions (bottom row)

    Args:
        model: The trained Q4 autoencoder (classification encoder + trained decoder).
        dataloader: DataLoader (e.g., test_loader).
        device: torch.device.
        n: Number of samples per digit.
    """
    import matplotlib.pyplot as plt
    import torch

    model.eval()
    digit_images = {d: [] for d in range(10)}

    # Collect n samples per digit
    with torch.no_grad():
        for imgs, labels in dataloader:
            for d in range(10):
                mask = labels == d
                digit_images[d].extend(imgs[mask])
            if all(len(v) >= n for v in digit_images.values()):
                break

    # Plot per digit
    for d in range(10):
        imgs = torch.stack(digit_images[d][:n]).to(device)
        recons = model(imgs)

        imgs = (imgs + 1) / 2  # denormalize
        recons = (recons + 1) / 2

        plt.figure(figsize=(n, 2))
        for i in range(n):
            # Original
            plt.subplot(2, n, i + 1)
            plt.imshow(imgs[i][0].cpu(), cmap='gray')
            plt.axis('off')

            # Reconstruction
            plt.subplot(2, n, i + 1 + n)
            plt.imshow(recons[i][0].detach().cpu().numpy(), cmap='gray')
            plt.axis('off')

        plt.suptitle(f"Digit {d} – Top: Original, Bottom: Reconstruction", fontsize=14)
        plt.tight_layout()
        plt.show()
