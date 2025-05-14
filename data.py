import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import ConvAutoencoder

# Transformation: convert to tensor and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=128, shuffle=False)


def load_best_encoder(_encoder, latent_dim, channels, pre_trained, pre_trained_encoder_path, device):
    """
    Loads pretrained encoder weights from a saved autoencoder checkpoint.

    Args:
        _encoder (nn.Module): An instance of the Encoder class (architecture only).
        latent_dim (int): Dimension of the latent space.
        channels (tuple): Channel configuration used in the encoder.
        pre_trained (bool): Whether to load pretrained weights (True) or skip loading (False).
        pre_trained_encoder_path (str): Path to the saved autoencoder checkpoint (.pt file).
        device (torch.device): The device to load the model onto.

    Returns:
        nn.Module: The encoder with loaded (or unchanged) weights.
    """
    if not pre_trained:
        return _encoder

    # Create full autoencoder to extract encoder weights from saved state
    _autoencoder = ConvAutoencoder(latent_dim=latent_dim, channels=channels).to(device)
    _autoencoder.load_state_dict(
        torch.load(pre_trained_encoder_path, map_location=device, weights_only=True)
    )

    # Extract only encoder weights from the full state_dict
    encoder_state_dict = {
        k.replace("encoder.", ""): v
        for k, v in _autoencoder.state_dict().items()
        if k.startswith("encoder.")
    }

    _encoder.load_state_dict(encoder_state_dict)
    return _encoder
