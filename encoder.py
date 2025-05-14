import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, latent_dim, channels=(16, 32, 64), input_shape=(1, 28, 28)):
        super().__init__()
        c1, c2, c3 = channels

        self.conv = nn.Sequential(
            nn.Conv2d(1, c1, 3, stride=2, padding=1),
            nn.BatchNorm2d(c1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1),
            nn.BatchNorm2d(c2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c2, c3, 3, stride=1, padding=1),
            nn.BatchNorm2d(c3),
            nn.LeakyReLU(0.2)
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out = self.conv(dummy)
            self.flat_dim = out.view(1, -1).shape[1]
            self.unflatten_shape = out.shape[1:]

        self.fc = nn.Linear(self.flat_dim, latent_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
