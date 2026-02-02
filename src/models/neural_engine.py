import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class NoiseLayer(nn.Module):
    """
    Differentiable Noise Layer to simulate attacks during training.
    Includes: JPEG-like blur, Dropout, Gaussian Noise, and Cropping.
    """
    def __init__(self):
        super().__init__()

    def forward(self, stego_image):
        # 1. Add Gaussian Noise
        noise = torch.randn_like(stego_image) * 0.01
        stego_image = stego_image + noise
        
        # 2. Simulated Dropout (pixel-wise erasure)
        mask = torch.bernoulli(torch.full_like(stego_image, 0.95))
        stego_image = stego_image * mask
        
        # 3. Simulated JPEG (Gaussian Blur)
        # kernel_size must be odd
        stego_image = F.avg_pool2d(stego_image, kernel_size=3, stride=1, padding=1)
        
        return stego_image

class Encoder(nn.Module):
    def __init__(self, payload_bits=128):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64 + 1, 64) # Concatenate 1-channel bits map
        self.conv4 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, image, payload):
        B, C, H, W = image.shape
        L = payload.shape[1]
        
        x = self.conv1(image)
        x = self.conv2(x)
        
        # Expand bits to HxW spatial map
        # payload is [B, L], we want [B, 1, H, W]
        payload_map = payload.view(B, L, 1, 1).expand(B, L, H, W)
        payload_avg = torch.mean(payload_map, dim=1, keepdim=True)
        
        x = torch.cat([x, payload_avg], dim=1)
        x = self.conv3(x)
        stego = self.conv4(x)
        
        # Residual connection to keep stego image close to original
        return image + 0.1 * stego

class Decoder(nn.Module):
    def __init__(self, payload_bits=128):
        super().__init__()
        self.convs = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            ConvBlock(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Linear(512, payload_bits)

    def forward(self, image):
        features = self.convs(image)
        bits = self.fc(features) # Return logits for stability
        return bits

class StegaDNAEngine(nn.Module):
    def __init__(self, payload_bits=128, is_training=True):
        super().__init__()
        self.encoder = Encoder(payload_bits)
        self.decoder = Decoder(payload_bits)
        self.noise_layer = NoiseLayer()
        self.is_training = is_training

    def forward(self, image, payload):
        stego = self.encoder(image, payload)
        
        # During training, apply noise to simulate attacks
        if self.is_training:
            noised_stego = self.noise_layer(stego)
            recovered_bits = self.decoder(noised_stego)
        else:
            recovered_bits = self.decoder(stego)
            
        return stego, recovered_bits
