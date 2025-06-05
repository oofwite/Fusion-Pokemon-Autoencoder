import torch
import torch.nn as nn
import torch.nn.functional as F

# Define Swish activation
class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# Simplified Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):  # Increased reduction for less computation
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction=8)  # Simplified SEBlock
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels)
            )
        self.swish = Swish()

    def forward(self, x):
        out = self.swish(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = self.swish(out)
        return out

class Model(nn.Module):
    def __init__(self, num_classes=170, latent_channels=8, input_channels=3):  # Reduced latent_channels
        super().__init__()
        self.num_classes = num_classes
        self.latent_channels = latent_channels
        self.input_channels = input_channels

        # Simplified Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # Reduced from 128
            nn.BatchNorm2d(64),
            Swish(),
            ResidualBlock(64, 64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Reduced from 256
            nn.BatchNorm2d(128),
            Swish(),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),  # Reduced from 512
            nn.BatchNorm2d(256),
            Swish(),
            ResidualBlock(256, 256)
        )
        self.quant_conv = nn.Conv2d(256, latent_channels, kernel_size=1)  # Matches reduced channels

        # Simplified Decoder
        self.post_quant_conv = nn.Conv2d(latent_channels, 256, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),  # Reduced from 256
            nn.BatchNorm2d(128),
            Swish(),
            ResidualBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # Reduced from 128
            nn.BatchNorm2d(64),
            Swish(),
            ResidualBlock(64, 64),
            nn.ConvTranspose2d(64, input_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        # Simplified Classifier
        self.classifier = nn.Sequential(
            nn.Linear(latent_channels, 128),  # Reduced from 256
            nn.BatchNorm1d(128),
            Swish(),
            nn.Linear(128, num_classes)
        )

    def encode(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] != self.input_channels:
            raise ValueError(f"Expected {self.input_channels} input channels, got {x.shape[1]}")
        h = self.encoder(x)
        z = self.quant_conv(h)
        return z

    def decode(self, z):
        if z.dim() == 3:
            z = z.unsqueeze(0)
        h = self.post_quant_conv(z)
        x_recon = self.decoder(h)
        return x_recon

    def forward(self, x):
        z = self.encode(x)
        x_recon = self.decode(z)
        z_pooled = F.adaptive_avg_pool2d(z, (1, 1)).view(z.size(0), -1)
        logits = self.classifier(z_pooled)
        return x_recon, logits