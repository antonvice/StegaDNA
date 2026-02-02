# StegaDNA: Adversarial Deep-Digital Watermarking Technical Specification

## 1. System Overview

StegaDNA is a high-performance steganography engine designed to "stamp" media with unique, indestructible DNA (digital signatures). It uses a hybrid approach: **Neural Embedding** for visual hiding and **Mojo-powered ECC** for foolproof recovery.

### Core Components

- **Neural Engine (Python/PyTorch)**: An adversarial Encoder-Decoder (CNN-based) trained to hide bits in media.
- **Mojo Core (Mojo)**: A high-performance Reed-Solomon (ECC) module for bit-level tampering resistance and incredibly fast verification.
- **Universal Gateway (FastAPI)**: A unified API to handle Image, Audio, and Text "stamping".

---

## 2. Component Architectures

### A. The Neural Engine (PyTorch)

**Architecture**: Modified HiDDeN (Hiding Data with Deep Networks).

- **Encoder**: 4-layer CNN with residual connections.
- **Decoder**: A spatial transformation network followed by a CNN classifier to recover bits.
- **Noise Layer**: Differentiable simulation of attacks (Gaussian noise, Dropout, Blur) applied during training to ensure robustness.

### B. The Mojo Processing Core

- **ECC Module**: SIMD-optimized implementation of Reed-Solomon (255, 223) encoding/decoding.
- **Bit-Hasher**: Converts UserID + Salt into a fixed 128-bit vector.
- **FFI Layer**: Exported as a shared library (`dna_ecc.dylib`) for $O(1)$ Python integration.

### C. Universal Gateway (FastAPI)

The FastAPI server acts as a **Universal Gateway** routing traffic to specialized modality engines.

- `POST /stamp/{modality}`: Accepts raw file + UserID, returns stamped file.
- `POST /verify/{modality}`: Accepts suspicious file, returns recovery confidence score.

---

## 3. Implementation Roadmap

### Phase 1: Research & Training (Current)

- [x] **Data Loading**: Implement custom DataLoader for technical image datasets.
- [x] **Architecture Design**: Implement 4-layer CNN with Residual Encoder.
- [/] **Adversarial Training**: Train Encoder vs. Decoder with Noise Layer (Running).
- [x] **Metric Tracking**: Integrate Weights & Biases for PSNR, BER, and Loss logging.
- [ ] **Robustness Testing**: Benchmark against "The Print-Scan Attack".

### Phase 2: The Mojo Integration (Completed)

- [x] **Galois Field Math**: Implement GF(2^8) SIMD-optimized arithmetic.
- [x] **RS Decoder**: Implement Berlekamp-Massey Algorithm for recovery.
- [x] **Shared Library**: Compile Mojo core to `.dylib` and bridge to Python via `ctypes`.
- [x] **FastAPI Gateway**: Universal gateway routes Image/Audio/Text to respective engines.

### Phase 3: Deployment & Scale

- [x] **FastAPI Scaffolding**: Create the HTTP layer and bridge integration.
- [ ] **Concurrency**: Implement an asynchronous queue (Redis) for bulk requests.
- [ ] **Storage**: Implement high-speed DB for DNA hash lookup.

---

## 4. Directory Structure

```text
StegaDNA/
├── mojo_core/
│   ├── ecc.mojo           # Reed-Solomon + SIMD logic
│   ├── processing.mojo    # Image normalization
│   └── main.mojo          # Entry point (removed for dylib)
├── src/
│   ├── models/            # Neural Engine (PyTorch)
│   ├── data/              # StegaDataset & DataLoader
│   └── train.py           # Training loop with W&B
├── server/
│   ├── app.py             # FastAPI Universal Gateway
│   └── bridge.py          # Mojo-to-Python bindings
├── tests/
│   └── test_api.py        # API functional tests
└── dna_ecc.dylib          # Compiled Mojo Core
```

---

## 5. Security & Robustness

- **Salt Customization**: The `salt` is customizable in the environment and can be rotated without retraining the model.
- **Global Embedding**: The Encoder spreads signal across the entire spatial domain, ensuring 50% cropping still allows for bit recovery.
- **ECC Redundancy**: Mojo's RS core adds parity bits, allowing the system to tolerate significant bit-flips in the neural decoder's output.

---

## 6. Extensions

- **Audio**: Replacing 2D CNNs with 1D Spectrogram Transformers using the same Mojo core.
- **Text**: LLM Logit Processor to adjust token probabilities to embed DNA strings.
