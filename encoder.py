import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    Convolutional encoder module for reducing an image to a latent vector.

    Architecture:
        - 3 convolutional layers with BatchNorm and LeakyReLU
        - Final linear projection to latent space

    Args:
        latent_dim (int): Dimension of the latent vector z.
        channels (tuple): Number of channels in each conv layer (default: (16, 32, 64)).
        input_shape (tuple): Shape of the input image (default: (1, 28, 28)).

    Attributes:
        flat_dim (int): Flattened size after conv layers (used for linear layer input).
        unflatten_shape (tuple): Shape to unflatten from decoder.

    Example:
        encoder = Encoder(latent_dim=16, channels=(16, 32, 64))
        z = encoder(x)
    """
    def __init__(self, latent_dim, channels=(16, 32, 64), input_shape=(1, 28, 28)):
        super().__init__()
        c1, c2, c3 = channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            self.flat_dim = out.view(1, -1).shape[1]
            self.unflatten_shape = out.shape[1:]  # (C, H, W)

        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        """
        Encodes input images into latent vectors.

        Args:
            x (Tensor): Input image tensor of shape (B, 1, 28, 28)

        Returns:
            Tensor: Latent representation of shape (B, latent_dim)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
