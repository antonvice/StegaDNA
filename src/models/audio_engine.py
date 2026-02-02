import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AudioEncoder(nn.Module):
    """
    1D-CNN Encoder to embed DNA bits into raw audio waveforms.
    """
    def __init__(self, payload_bits=128):
        super().__init__()
        self.conv1 = AudioConvBlock(1, 64, kernel_size=15)
        self.conv2 = AudioConvBlock(64, 64, kernel_size=11)
        self.conv3 = AudioConvBlock(64 + 1, 64, kernel_size=7)
        self.conv4 = nn.Conv1d(64, 1, kernel_size=3, padding=1)

    def forward(self, waveform, payload):
        # waveform: [B, 1, T] (B=batch, 1=mono, T=timesteps)
        # payload: [B, L]
        B, C, T = waveform.shape
        L = payload.shape[1]
        
        x = self.conv1(waveform)
        x = self.conv2(x)
        
        # Broadcast bits across time dimension
        # payload is [B, L] -> [B, L, 1] -> [B, L, T]
        payload_map = payload.view(B, L, 1).expand(B, L, T)
        payload_avg = torch.mean(payload_map, dim=1, keepdim=True) # [B, 1, T]
        
        x = torch.cat([x, payload_avg], dim=1)
        x = self.conv3(x)
        
        residual = self.conv4(x)
        
        # Residual connection with small alpha to keep audio identical to human ear
        return waveform + 0.01 * residual

class AudioDecoder(nn.Module):
    """
    1D-CNN Decoder to recover DNA bits from raw audio waveforms.
    """
    def __init__(self, payload_bits=128):
        super().__init__()
        self.conv1 = AudioConvBlock(1, 64, kernel_size=15)
        self.conv2 = AudioConvBlock(64, 64, kernel_size=11)
        self.conv3 = AudioConvBlock(64, 64, kernel_size=7)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, payload_bits)

    def forward(self, waveform):
        x = self.conv1(waveform)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

class AudioDNAEngine(nn.Module):
    def __init__(self, payload_bits=128):
        super().__init__()
        self.encoder = AudioEncoder(payload_bits)
        self.decoder = AudioDecoder(payload_bits)

    def forward(self, waveform, payload):
        stamped_audio = self.encoder(waveform, payload)
        # In actual training, we'd add an AudioNoiseLayer here
        recovered_bits = self.decoder(stamped_audio)
        return stamped_audio, recovered_bits
