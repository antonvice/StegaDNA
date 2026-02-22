import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import numpy as np
class MultiScaleResBlock(nn.Module):
    """
    Bio-inspired Multi-scale Residual Block (Inception-style).
    Uses 3x3, 5x5, and 7x7 kernels to capture features at different spatial frequencies.
    This is key to surviving analog-hole noise while minimizing visible 'blobs'.
    """
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2
        
        # 3x3 Branch
        self.conv3 = nn.Conv2d(in_channels, mid_channels, 3, padding=1)
        # 5x5 Branch (implemented as two 3x3 for efficiency and receptive field)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        )
        # 7x7 Branch
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1)
        )
        
        # Fusion layer
        self.fuse = nn.Conv2d(mid_channels * 3, in_channels, 1)
        self.bn = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = x
        f3 = self.conv3(x)
        f5 = self.conv5(x)
        f7 = self.conv7(x)
        
        out = torch.cat([f3, f5, f7], dim=1)
        out = self.fuse(out)
        out = self.bn(out)
        return self.relu(out + res)

class ResBlock(nn.Module):
    """
    Standard Residual Block to maintain feature gradients and image quality.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.bn2 = nn.InstanceNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class EncoderV2(nn.Module):
    """
    High-Capacity Encoder: Hides 128 bits across multiple feature channels.
    Uses Skip Connections (U-Net style) to preserve image quality while embedding.
    """
    def __init__(self, payload_bits=128):
        super().__init__()
        self.initial_conv = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.res_layers_pre = nn.Sequential(
            ResBlock(64),
            ResBlock(64)
        )
        
        # We project the bits into a 64-channel spatial map
        self.bit_projector = nn.Sequential(
            nn.Linear(payload_bits, 64 * 16 * 16),
            nn.ReLU(inplace=True)
        )
        
        # After merging bits and image features
        self.res_layers_post = nn.Sequential(
            ResBlock(64 + 64),
            ResBlock(128),
            ResBlock(128)
        )
        
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, padding=1)
        # Learnable strength factor (starts at 0.1, model can increase if bits are hard to read)
        self.stego_strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, image, payload):
        B, C, H, W = image.shape
        
        # 1. Feature extraction from image
        x_init = self.initial_conv(image)
        x = self.res_layers_pre(x_init) # [B, 64, 256, 256]
        
        # 2. Project bits into a spatial map
        bits_feat = self.bit_projector(payload).view(B, 64, 16, 16)
        bits_feat = F.interpolate(bits_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 3. Concatenate and refine
        combined = torch.cat([x, bits_feat], dim=1) # [B, 128, 256, 256]
        combined = self.res_layers_post(combined)
        
        # 4. Final Stego Residual + Global Skip Connection (U-Net style)
        stego_residual = self.final_conv(combined)
        
        # Use the learnable strength factor
        return image + self.stego_strength * stego_residual

class DecoderV2(nn.Module):
    """
    Deep Residual Decoder to recover precise bit signatures.
    """
    def __init__(self, payload_bits=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, payload_bits)
        )

    def forward(self, x):
        features = self.layers(x)
        return self.fc(features)

class EncoderV3(nn.Module):
    """
    U-Net style Encoder with Multi-scale ResBlocks.
    Uses deeper bit projection and feature fusion to prevent 'foggy blobs'.
    """
    def __init__(self, payload_bits=128):
        super().__init__()
        # Initial Downsampling Path
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.down1 = MultiScaleResBlock(64)
        
        # Bit Prep Network (Similar to Baluja's Prep Net)
        self.bit_prep = nn.Sequential(
            nn.Linear(payload_bits, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 64 * 32 * 32),
            nn.ReLU(inplace=True)
        )
        
        # Upsampling / Fusion Path
        self.up_conv = nn.Sequential(
            MultiScaleResBlock(64 + 64),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            MultiScaleResBlock(64)
        )
        
        self.final_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        self.stego_strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, image, payload):
        B, C, H, W = image.shape
        
        # 1. Feature extraction
        x = self.conv1(image)
        x = self.down1(x) # [B, 64, H, W]
        
        # 2. Dense Bit Prep
        # We project to 32x32 then upsample to H,W
        bits_feat = self.bit_prep(payload).view(B, 64, 32, 32)
        bits_feat = F.interpolate(bits_feat, size=(H, W), mode='bilinear', align_corners=False)
        
        # 3. Concatenate and refine using Multi-scale logic
        combined = torch.cat([x, bits_feat], dim=1)
        combined = self.up_conv(combined)
        
        # 4. Final Residual
        stego_residual = self.final_conv(combined)
        return image + self.stego_strength * stego_residual

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Localization network to find affine transform
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 60 * 60, 32), # 60x60 is for 256x256 input
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 60 * 60)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, align_corners=False)

class DecoderV3(nn.Module):
    """
    Multi-scale Decoder with Spatial Transformer for robust signal extraction under Analog warping.
    """
    def __init__(self, payload_bits=128):
        super().__init__()
        self.stn = SpatialTransformer()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            MultiScaleResBlock(64),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            MultiScaleResBlock(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            MultiScaleResBlock(256),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            MultiScaleResBlock(512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, payload_bits)
        )

    def forward(self, x):
        x = self.stn(x) # Align image first
        features = self.layers(x)
        return self.fc(features)

class NoiseLayerV2(nn.Module):
    """
    Intensity-controllable Noise Layer for curriculum learning.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, intensity=1.0, bypass_clamping=True):
        if not self.training or intensity == 0:
            return x
            
        # 1. Add Gaussian Noise
        x = x + torch.randn_like(x) * (0.02 * intensity)
        
        # 2. Random Dropout
        prob = 1.0 - (0.1 * intensity)
        mask = torch.bernoulli(torch.full_like(x, prob))
        mask = mask.to(x.device)
        x = x * mask
        
        # 3. Mild JPEG-like blur
        if intensity > 0.5:
            x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        return x

class NoiseLayerV3(nn.Module):
    """
    Real-World / Analog Hole Simulation (Print-and-Scan).
    Simulates: Geometric Tilt, Color Shift, Compression, and Complex Blur.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, intensity=1.0, bypass_clamping=True): # Allow bypass for normalized data
        if intensity == 0:
            return x
        
        B, C, H, W = x.shape
        device = x.device

        # --- 1. Geometric Perspective / Rotation via Kornia ---
        if intensity > 0.3:
            # We use kornia to apply robust geometric distortions similar to StegaStamp
            import kornia.augmentation as K
            distortion_scale = min(0.5, 0.2 * intensity)
            
            aug_list = K.AugmentationSequential(
                K.RandomPerspective(distortion_scale=distortion_scale, p=1.0),
                K.RandomRotation(degrees=10.0 * intensity, p=1.0),
                K.RandomResizedCrop(size=(H, W), scale=(0.8, 1.0), p=1.0)
            )
            x = aug_list(x)

        # --- 2. Color Jitter (Simulate Camera Sensor Tuning) ---
        if intensity > 0.2:
            brightness = 1.0 + (torch.randn(B, 1, 1, 1, device=device) * 0.1 * intensity)
            contrast = 1.0 + (torch.randn(B, 1, 1, 1, device=device) * 0.1 * intensity)
            x = (x * contrast) * brightness
            if not bypass_clamping:
                x = torch.clamp(x, 0, 1)

        # --- 3. Combined Noise & Blur ---
        # Simulating CMOS sensor noise + low light
        x = x + torch.randn_like(x) * (0.03 * intensity)
        
        # Variable Blur (Simulates out-of-focus or motion blur)
        if intensity > 0.4:
            k = 3 if intensity < 0.8 else 5
            x = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k//2)

        if bypass_clamping:
            return x
        return torch.clamp(x, 0, 1)

class StegaDNAEngine(nn.Module):
    def __init__(self, payload_bits=128, use_v3_noise=False):
        super().__init__()
        self.encoder = EncoderV3(payload_bits)
        self.decoder = DecoderV3(payload_bits)
        self.noise_layer = NoiseLayerV3() if use_v3_noise else NoiseLayerV2()

    def forward(self, image, payload, noise_intensity=1.0, bypass_clamping=True):
        stego = self.encoder(image, payload)
        noised_stego = self.noise_layer(stego, intensity=noise_intensity, bypass_clamping=bypass_clamping)
        recovered_bits = self.decoder(noised_stego)
        return stego, recovered_bits
