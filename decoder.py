import torch.nn as nn


class Decoder(nn.Module):
    """
    Convolutional decoder that reconstructs an image from a latent vector.

    Architecture:
        - Linear layer to project latent vector to conv feature map
        - 3 transposed convolution layers with LeakyReLU (last uses Tanh)

    Args:
        latent_dim (int): Dimension of the input latent vector.
        flat_dim (int): Flattened dimension after the encoder's conv layers.
        unflatten_shape (tuple): Shape to reshape into before deconvolution (C, H, W).
        channels (tuple): Tuple of 3 channel sizes in reverse order (decoder side).

    Example:
        decoder = Decoder(latent_dim=16, flat_dim=3136, unflatten_shape=(64, 7, 7), channels=(64, 32, 16))
        x_recon = decoder(z)
    """
    def __init__(self, latent_dim, flat_dim, unflatten_shape, channels):
        super().__init__()
        c3, c2, c1 = channels  # reverse of encoder order

        self.fc = nn.Linear(latent_dim, flat_dim)
        self.unflatten_shape = unflatten_shape

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(c3, c2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(c2, c1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(c1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()  # Assuming input images were normalized to [-1, 1]
        )

    def forward(self, z):
        """
        Reconstructs images from latent vectors.

        Args:
            z (Tensor): Latent tensor of shape (B, latent_dim)

        Returns:
            Tensor: Reconstructed image tensor of shape (B, 1, H, W)
        """
        x = self.fc(z)
        x = x.view(x.size(0), *self.unflatten_shape)
        return self.deconv(x)
