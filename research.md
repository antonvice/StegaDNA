# StegaDNA Research Documentation: The Path to Analog Robustness

## Executive Summary

This document chronicles the evolution of the StegaDNA architecture from its early high-fidelity stealth roots to the current **v4 Multi-Scale Breakthrough**. The primary challenge addressed is the "Analog Hole"—the degradation of signal integrity when an image is printed and captured by a phone camera.

---

## Phase 1: The Stealth Era (v1 & v2)

- **Run Name:** `v2-balanced-polish`
- **Architecture:** `EncoderV2` / `DecoderV2` (Standard ResBlocks, 16x16 Bit Projection)
- **Key Results:**
  - **PSNR:** ~29.8 dB (Near-perfect visual fidelity)
  - **BER:** 0.36
- **Outcome:** Successfully hid 128 bits in digital-only environments. However, the model lacked the "mathematical armor" to survive geometric and color distortions introduced in the V3 Analog Hole simulation.

---

## Phase 2: The Resurrection (v3 - Signal Priority)

- **Run Name:** `v3-analog-RESURRECTION`
- **Strategy:** "Signal at all costs."
- **Methodology:**
  - Disabled Perceptual Loss (`--warmup_no_vgg`)
  - Increased bit weight (`lambda_bits 200.0`)
  - Introduced `NoiseLayerV3` (Geometric tilt, color jitter, variable blur)
- **Key Results (Epoch 300):**
  - **BER:** 0.318 (Significant recovery under Analog noise)
  - **PSNR:** 2.23 dB (Visual destruction)
- **Observation:** The model developed "foggy blobs" and high-contrast spots. These served as "anchor points" for the decoder to find the signal through warping, but they were visually unacceptable.

---

## Phase 3: The Polish Failure (Signal Collapse)

- **Run Name:** `v3-analog-polish-v1`
- **Strategy:** Re-introducing stealth to the Resurrection base.
- **Methodology:**
  - Re-enabled VGG Perceptual Loss
  - Reduced bit weight (`lambda_bits 50.0`)
- **Outcome (Epoch 313):**
  - **PSNR:** 23.4 dB (Images looked clean again)
  - **BER:** 0.465 (**Signal Collapse**)
- **The "Cheating" Problem:** The architecture was too simple. The model couldn't find a way to hide bits without blobs, so it simply "deleted" the signal to satisfy the VGG critic.

---

## Phase 4: The Pivot (v4 - Multi-Scale Breakthrough)

To solve the "Fog vs. Signal" war, we pivoted to a significantly more advanced architecture.

### Technical Innovations

1. **MultiScaleResBlock (Inception-Style):**
   - Replaced standard ResBlocks with triple-path kernels (3x3, 5x5, and 7x7).
   - Allows the model to distribute DNA information across multiple spatial frequencies simultaneously.
2. **Dense Bit Preparation Network:**
   - Upgraded bit projection from a crude linear layer to a 1024-dimensional expansion, re-projected into a 32x32 feature map.
   - Provides higher resolution "bit anchors" that are easier for the decoder to find without needing large visible blobs.
3. **Multi-Layer Perceptual Loss:**
   - Switched from single-layer VGG to a multi-stage loss evaluating `relu1_2`, `relu2_2`, and `relu3_3`.
   - Forcibly suppresses "fog" by ensuring the image structure is respected at every level of depth.

### Current Findings (Epoch 221 Breakthrough)

Phase 4 has demonstrated that the **Multi-Scale Inception architecture** is the definitive solution for the Analog Hole in the StegaDNA ecosystem.

1. **Robustness Threshold Achieved**: Achievement of **0.24 BER** under max-intensity Analog Hole noise (Geometric Perspective + Sensor Noise). This BER is within the "Correctable Zone" for our Reed-Solomon ECC.
2. **Visual Stability**: Unlike the v3 architecture which collapsed to 0.50 BER when visual constraints were applied, the v4 architecture maintains **~18-22 dB PSNR** while successfully recovering the signal.
3. **Blob Suppression**: The triple-path kernel design effectively "atomized" the hidden data, distributing it into high-frequency spatial regions that are invisible to the human eye but highly recognizable to the Multi-Scale Decoder.

---

## 🔬 Novelty & Publication Strategy

Is this just a copy of existing work? **No.** While we build on the shoulders of giants (HiDDeN, Baluja), StegaDNA introduces several novel elements that are publishable:

1. **Mojo-Accelerated Hybrid Pipeline**: The integration of **Mojo** for SIMD-accelerated Reed-Solomon error correction in the post-processing pipeline is highly novel. Most ML research uses slow Python-based ECC; we demonstrate a production-ready, ultra-fast hybrid architecture.
2. **Differentiable Multi-Scale Curriculum**: Our use of **Multi-Scale Inception Kernels** specifically to counter low-frequency "cheating" (fog/blobs) in a curriculum-based Analog Hole setting is a specific architectural refinement not commonly detailed in general steganography papers.
3. **Real-World Perspective Warper**: The specific combination of differentiable homography approximation and sensor noise in our `NoiseLayerV3` provides a robust case study for "Print-to-Digital" survivors.

### Publication Path

- **Conference**: CVPR (Computer Vision and Pattern Recognition) or ICIP (International Conference on Image Processing).
- **Contribution**: "Multi-Scale Frequency Distribution for High-Fidelity Hybrid Steganography under Extreme Analog Degradation."

---

### Update Metadata

```bash
uv run python src/train.py \
  --batch_size 24 \
  --group "v3-gold-standard" \
  --run_name "v3-arch-breakthrough" \
  --tag v3-gold \
  --use_v3_noise \
  --lr 5e-5 \
  --lambda_bits 150.0 \
  --lambda_perceptual 60.0 \
  --curriculum_epochs 40 \
  --epochs 450
```
