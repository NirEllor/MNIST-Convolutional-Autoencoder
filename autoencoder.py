import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class ConvAutoencoder(nn.Module):
    """
    Convolutional autoencoder combining an encoder and a decoder.

    This module can be used for both:
    - Unsupervised autoencoding (train both encoder and decoder)
    - Task-specific decoding (freeze encoder and train decoder only)

    Args:
        latent_dim (int): Dimensionality of the latent space.
        channels (tuple): Channel sizes for the convolutional layers (encoder).
        encoder (Encoder, optional): Pre-initialized encoder (e.g., pretrained from classification).
                                     If None, a new encoder will be created with given config.

    Example:
        model = ConvAutoencoder(latent_dim=16, channels=(16, 32, 64))
        output = model(input_batch)
    """
    def __init__(self, latent_dim=16, channels=(16, 32, 64), encoder=None):
        super().__init__()

        # Use provided encoder or create a new one
        self.encoder = encoder if encoder is not None else Encoder(latent_dim, channels)

        # Use encoder's output shape to initialize the decoder
        self.decoder = Decoder(
            latent_dim=latent_dim,
            flat_dim=self.encoder.flat_dim,
            unflatten_shape=self.encoder.unflatten_shape,
            channels=channels[::-1]  # reverse channel order for decoder
        )

    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (Tensor): Input tensor of shape (B, 1, H, W)

        Returns:
            Tensor: Reconstructed image tensor of shape (B, 1, H, W)
        """
        z = self.encoder(x)
        return self.decoder(z)
