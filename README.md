# StegaDNA 🧬

**Adversarial Deep-Digital Watermarking with Mojo-Powered Error Correction.**

StegaDNA is a high-performance steganography engine designed to "stamp" any medium (Images, Audio, Text) with indestructible digital signatures (DNA). It leverages a hybrid architecture combining the flexibility of **PyTorch** for neural embedding and the raw speed of **Mojo** for bit-level error correction.

---

## 🚀 Key Features

- **Indestructible DNA**: Uses **Reed-Solomon ECC** implemented in **Mojo** with SIMD optimizations to recover signatures even from partially destroyed media.
- **The Analog Hole (v4 Breakthrough)**: Specialized **Multi-Scale Inception Kernels** and a **Dense Bit Projection Network** allow DNA survival under real-world capture conditions (phone cameras, print-and-scan) while maintaining high visual fidelity (20+ dB PSNR).
- **Adversarial Robustness**: Neural networks trained against a differentiable **Noise Layer** (Geometric warping, color jitter, variable blur) to survive digital and physical attacks.
- **Universal Gateway**: A unified **FastAPI** controller that routes traffic to specialized modality engines.
- **Mac Optimized**: Native **MPS (Metal Performance Shaders)** support for lightning-fast training on Apple Silicon.
- **Research-Driven**: Fully documented architectural evolutions and training experiments available in [research.md](./research.md).

---

## 🏗️ Architecture (v4 Multi-Scale)

```text
Media Input ───► [ Multi-Scale Encoder ] ───► [ Noise Layer V3 ] ───► [ High-Entropy Decoder ]
                          ▲                   (Analog Distortion)              │
                          │                                                    │
                 [ 128-bit DNA Bits ] <────────────────────────────────────────┘
                  (Dense Projection)
```

The v4 architecture utilizes a U-Net style Encoder with triple-path kernels (3x3, 5x5, 7x7) to capture and hide data across multiple spatial frequencies, suppressing the "foggy blob" artifacts common in earlier deep steganography models.

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

### 3. Training (Robust V5)

Launch the robust **StegaStamp** adversarial training loop which uses InstanceNorm and STN for camera-angle survival:

```bash
bash train_stegastamp.sh
```

### 4. Serve the API & Interactive Dashboard

Run the universal gateway and access the premium web dashboard:

```bash
uv run python main.py
```

Then navigate to: [http://localhost:8000](http://localhost:8000)

---

## 🗺️ Roadmap

- [x] **Phase 1**: Mojo Reed-Solomon Core (SIMD Optimized).
- [x] **Phase 2**: Universal FastAPI Gateway & Modality Routing.
- [x] **Phase 3**: Adversarial Training Pipeline (MPS Enabled).
- [x] **Phase 4**: **Analog Hole Survival** (Multi-Scale Pivot).
- [ ] **Phase 5**: STFT-Transformer Engine for Audio Watermarking.
- [ ] **Phase 6**: LLM Logit Processor for Text DNA.
- [ ] **Phase 7**: ONNX Export for Edge Deployment.

---

Developed as part of the **Grably Data Engineering** ecosystem. 🛡️
