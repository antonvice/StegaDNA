# StegaDNA 🧬

**Adversarial Deep-Digital Watermarking with Mojo-Powered Error Correction.**

StegaDNA is a high-performance steganography engine designed to "stamp" any medium (Images, Audio, Text) with indestructible digital signatures (DNA). It leverages a hybrid architecture combining the flexibility of **PyTorch** for neural embedding and the raw speed of **Mojo** for bit-level error correction.

---

## 🚀 Key Features

- **Indestructible DNA**: Uses **Reed-Solomon ECC** implemented in **Mojo** with SIMD optimizations to recover signatures even from partially destroyed media.
- **Adversarial Robustness**: Neural networks trained against a differentiable **Noise Layer** to survive screenshots ("The Print-Scan Attack"), JPEG compression, and cropping.
- **Universal Gateway**: A unified **FastAPI** controller that routes traffic to specialized modality engines (CNNs for images, Transformers for audio).
- **Mac Optimized**: Native **MPS (Metal Performance Shaders)** support for lightning-fast training on Apple Silicon.
- **Production Monitoring**: Real-time metric tracking (PSNR, BER, Loss) via **Weights & Biases**.

---

## 🏗️ Architecture

```text
Media Input ───► [ Universal Controller ] ───► [ Mojo DNA Core ]
                          │                        │ (ECC & Hashing)
                          ▼                        ▼
                 [ Neural Engine ] ◄─────── [ 128-bit DNA ]
                 (PyTorch Encoder)
                          │
                          ▼
                 [ Stamped Output ]
```

---

## 🛠️ Tech Stack

- **Core Logic**: [Mojo](https://www.modular.com/mojo) (SIMD, GF(2^8) Arithmetic)
- **Neural Engine**: [PyTorch](https://pytorch.org/) (Modified HiDDeN Architecture)
- **API Layer**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Infrastructure**: [UV](https://github.com/astral-sh/uv) (Fast Python Package Manager)
- **Monitoring**: [Weights & Biases](https://wandb.ai/) + [Loguru](https://github.com/Delgan/loguru)

---

## 🚦 Getting Started

### 1. Installation

Ensure you have `uv` and `mojo` installed on your Mac.

```bash
uv sync
```

### 2. Build the Mojo Core

Compile the high-performance Reed-Solomon module:

```bash
uv run mojo build mojo_core/ecc.mojo --emit shared-lib -o dna_ecc.dylib
```

### 3. Training

Launch the adversarial training loop:

```bash
uv run python src/train.py
```

### 4. Serve the API

Run the universal gateway:

```bash
uv run python main.py
```

---

## 🗺️ Roadmap

- [x] **Phase 1**: Mojo Reed-Solomon Core (SIMD Optimized).
- [x] **Phase 2**: Universal FastAPI Gateway & Modality Routing.
- [x] **Phase 3**: Adversarial Training Pipeline (MPS Enabled).
- [ ] **Phase 4**: STFT-Transformer Engine for Audio Watermarking.
- [ ] **Phase 5**: LLM Logit Processor for Text DNA.
- [ ] **Phase 6**: ONNX Export for Edge Deployment.

---

Developed as part of the **Grably Data Engineering** ecosystem. 🛡️
