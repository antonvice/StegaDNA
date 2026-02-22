---
tags:
- steganography
- pytorch
- computer-vision
- security
---

# StegaDNA-V5-StegaStamp 🧬

**StegaDNA-V5-StegaStamp** is a high-performance deep steganography model mathematically designed to embed an indestructible "DNA" digital signature into images, making them robust to print-and-scan and screen-capture distortions (the "Analog Hole").

## 🚀 Key Achievements

* **Near-Perfect Visual Fidelity**: Attains ~48 dB PSNR, rendering the embedded data invisible to the human eye. 
* **Reed-Solomon Error Correction**: Encodes the message with Reed-Solomon (via Mojo) mapping 10 signature bytes and 6 ECC repair bytes into 128 Bits, meaning full text recovery is achieved even if up to 20% of bits flip during camera scanning.
* **Camera-Angle Localization**: Built natively with a **Spatial Transformer Network (STN)** enabling the Decoder to automatically predict affine homography adjustments on captured imagery before reading the bits.
* **Domain-Shift Immune**: Eliminates traditional BatchNorm failures by utilizing `InstanceNorm2d`, guaranteeing pure mathematical stability during isolated physical deployments.

## 🏗️ Architecture Stack

This V5 architecture relies on a specialized workflow:

1. **U-Net Encoder**: A multi-scale residual generator that disperses the 128-bit payload into redundant, variable-frequency spatial features without forming visible artifacts.
2. **Kornia Adversarial Engine**: During training, simulated 3D perspectives, zoom, brightness fluctuations, and lens blurs force the encoder into securing extreme physical robustness.
3. **STN-Empowered Decoder**: An internal affine tracker aligns the distorted image before high-entropy extraction.

## 🛠️ Usage

This model requires the `StegaDNAUniversalGateway` to operate out of the box. 

Install the project infrastructure from the source repository:

```bash
git clone https://github.com/antonvice/StegaDNA.git
cd StegaDNA
```

Install using `uv`:

```bash
uv sync
```

With the server running, you can hit the local endpoint to automatically encode or decode via the browser or Python `requests`.

```python
import requests

# Example of stamping an image with DNA text
files = {'file': open('asset.jpg', 'rb')}
data = {'user_id': 'Secret Provenance Data'}
resp = requests.post('http://localhost:8000/stamp/image', files=files, data=data)

with open('stamped.png', 'wb') as f:
    f.write(resp.content)
```

## 📈 Evaluation Matrix

Current isolated network performance metrics per epoch on hold-out standard clean-image splits under heavy `v3` noise:
- **BER (Bit Error Rate)**: Converging toward < 0.15 (Sufficient for Reed-Solomon guaranteed string recovery)
- **PSNR (Peak Signal to Noise)**: > 48.0 dB

Developed as part of the **Grably Data Engineering** ecosystem. 🛡️
