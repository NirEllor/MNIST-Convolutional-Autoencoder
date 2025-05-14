import torch
import matplotlib.pyplot as plt


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
