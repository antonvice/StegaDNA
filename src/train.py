import os
# --- macOS Environment Fixes (Silence Malloc Noise) ---
for key in ["MallocStackLogging", "MallocStackLoggingDirectory"]:
    os.environ.pop(key, None)
os.environ["MallocNanoZone"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import argparse
import re

# Set MPS Fallback for missing ops (like grid_sample backward)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from loguru import logger
from tqdm import tqdm
import wandb
import numpy as np
from pydantic import BaseModel
from typing import Optional

from torchvision.models import vgg16, VGG16_Weights # Added for VGG perceptual loss

# Load environment variables (WANDB_API_KEY, etc.)
load_dotenv()

from src.data.loader import get_stega_dataloader
from src.models.neural_engine import StegaDNAEngine

class TrainConfig(BaseModel):
    project_name: str = "StegaDNA-Alpha"
    epochs: int = 50
    batch_size: int = 24
    lr: float = 1e-4
    lambda_bits: float = 15.0 # Weight for bit recovery loss
    payload_bits: int = 128
    image_size: int = 256
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    tsv_path: str = "data/text/150k_msgs_sample_hashed_pii.tsv"
    image_dir: str = "data/images"
    resume_path: Optional[str] = None
    mixed_precision: bool = False
    tag: str = "default"
    wandb_id: Optional[str] = None # For resuming specific W&B runs
    run_name: Optional[str] = None # Custom display name for W&B
    curriculum_epochs: int = 10    # Epochs to reach maximum noise
    lambda_perceptual: float = 1.0 # Weight for perceptual loss (VGG)
    group: str = "experiment"      # W&B Group name
    use_v3_noise: bool = False     # Use Real-World Noise (Print-and-Scan)
    warmup_no_vgg: bool = False    # Disable VGG loss to prioritize signal recovery

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

def calculate_ber(pred_bits, true_bits):
    """Bit Error Rate"""
    pred_bits = (torch.sigmoid(pred_bits) > 0.5).float()
    errors = torch.sum(torch.abs(pred_bits - true_bits))
    return (errors / true_bits.numel()).item()

class StegaTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 1. Initialize W&B
        wandb.init(
            project=config.project_name,
            name=config.run_name,
            group=config.group,
            job_type="training",
            config=config.model_dump(),
            id=config.wandb_id,
            resume="allow" if config.wandb_id else None,
            reinit=True
        )
        
        # 2. Data
        self.loader = get_stega_dataloader(
            tsv_path=config.tsv_path,
            image_dir=config.image_dir,
            batch_size=config.batch_size,
            image_size=config.image_size,
            payload_bits=config.payload_bits,
            num_workers=4
        )
        
        # 3. Model, Optimizer, Loss
        self.model = StegaDNAEngine(
            payload_bits=config.payload_bits,
            use_v3_noise=config.use_v3_noise
        ).to(self.device)
        
        self.start_epoch = 1
        resume_file = config.resume_path
        
        # 1A. Auto-Detect Latest Checkpoint if none provided
        self.checkpoint_dir = os.path.join("model/checkpoints", config.tag)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if not resume_file:
            import glob
            checkpoints = glob.glob(os.path.join(self.checkpoint_dir, "*.pth"))
            if checkpoints:
                # Find file with highest epoch number in name stegadna_eXX...
                def get_epoch(f):
                    m = re.search(r"_e(\d+)_", os.path.basename(f))
                    return int(m.group(1)) if m else -1
                
                checkpoints.sort(key=get_epoch, reverse=True)
                resume_file = checkpoints[0]
                logger.info(f"Auto-detected latest checkpoint: {resume_file}")

        # 1B. Load State
        if resume_file:
            if os.path.exists(resume_file):
                logger.info(f"Resuming from checkpoint: {resume_file}")
                state_dict = torch.load(resume_file, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Parse start_epoch from filename
                match = re.search(r"_e(\d+)_", os.path.basename(resume_file))
                if match:
                    self.start_epoch = int(match.group(1)) + 1
                    logger.info(f"Continuing from Epoch {self.start_epoch}")
            else:
                raise FileNotFoundError(f"CRITICAL: Resume file NOT FOUND: {resume_file}. Stopping to prevent uninitialized training.")
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        
        # 4. Losses
        self.criterion_img = nn.MSELoss()
        self.criterion_bits = nn.BCEWithLogitsLoss()
        
        # Perceptual Loss (Multi-Layer VGG for better 'Blob' suppression)
        full_vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.to(self.device).eval()
        # We take relu1_2, relu2_2, relu3_3 (standard for perceptual loss)
        self.vgg_layers = nn.ModuleList([
            full_vgg[:4],   # relu1_2
            full_vgg[4:9],  # relu2_2
            full_vgg[9:16]  # relu3_3
        ])
        for layer in self.vgg_layers:
            for param in layer.parameters():
                param.requires_grad = False
        
        # Scalar for mixed precision if enabled (Only for CUDA, MPS doesn't use GradScaler same way)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and config.device == "cuda" else None
        
        # Determine actual autocast device
        self.autocast_device = "cuda" if "cuda" in config.device else "cpu"
        if config.device == "mps" and config.mixed_precision:
             logger.warning("Mixed precision (FP16) on MPS can sometimes be slower. Disable if BER doesn't improve.")
             self.autocast_device = "mps"
             
        # Done in earlier step
        
        # Track global step for seamless resuming
        self.global_step = (self.start_epoch - 1) * len(self.loader)
        
        # Sync with W&B if resuming to avoid step collisions
        if wandb.run and wandb.run.step > self.global_step:
            logger.info(f"Syncing global_step with W&B: {self.global_step} -> {wandb.run.step}")
            self.global_step = wandb.run.step
            
        msg = f"StegaTrainer initialized on {self.device} | Tag: {config.tag} | Step: {self.global_step}"
        if config.use_v3_noise:
            msg += " | NOISE: V3 (Analog Hole)"
        logger.info(msg)

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        epoch_bers = []
        epoch_psnrs = []
        
        pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
        for step, (images, bits) in enumerate(pbar):
            self.global_step += 1
            images, bits = images.to(self.device), bits.to(self.device)
            
            # Curriculum Learning: Gradually increase noise intensity from 0 starting at current run.
            # This allows the model to re-align with the signal before the new noise type takes full effect.
            noise_intensity = min(1.0, (epoch - (self.start_epoch - 1)) / max(1, self.config.curriculum_epochs))
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            if self.config.mixed_precision:
                with torch.autocast(device_type=self.autocast_device, enabled=True):
                    stego, recovered_bits = self.model(images, bits, noise_intensity=noise_intensity)
                    loss_img = self.criterion_img(stego, images)
                    loss_bits = self.criterion_bits(recovered_bits, bits)
                    
                    # Perceptual loss
                    if self.config.warmup_no_vgg:
                        loss_perceptual = torch.tensor(0.0, device=self.device)
                    else:
                        loss_perceptual = 0
                        stego_feat = stego
                        orig_feat = images
                        for layer in self.vgg_layers:
                            stego_feat = layer(stego_feat)
                            orig_feat = layer(orig_feat)
                            loss_perceptual += F.mse_loss(stego_feat, orig_feat)
                    
                    total_loss = loss_img + (self.config.lambda_bits * loss_bits) + (self.config.lambda_perceptual * loss_perceptual)
            else:
                stego, recovered_bits = self.model(images, bits, noise_intensity=noise_intensity)
                loss_img = self.criterion_img(stego, images)
                loss_bits = self.criterion_bits(recovered_bits, bits)
                
                # Perceptual loss
                if self.config.warmup_no_vgg:
                    loss_perceptual = torch.tensor(0.0, device=self.device)
                else:
                    loss_perceptual = 0
                    stego_feat = stego
                    orig_feat = images
                    for layer in self.vgg_layers:
                        stego_feat = layer(stego_feat)
                        orig_feat = layer(orig_feat)
                        loss_perceptual += F.mse_loss(stego_feat, orig_feat)
                
                total_loss = loss_img + (self.config.lambda_bits * loss_bits) + (self.config.lambda_perceptual * loss_perceptual)
            
            # Monitoring Metrics
            ber = calculate_ber(recovered_bits, bits.detach())
            psnr = calculate_psnr(images, stego.detach())
            
            # Backward
            if self.scaler:
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                self.optimizer.step()
            
            # Stats update
            epoch_losses.append(total_loss.item())
            epoch_bers.append(ber) # ber is now float
            epoch_psnrs.append(psnr) # psnr is now float
            
            # Periodic Master ML Logging
            # Log every step as requested, but keep calculations efficient
            grad_norm = 0.0
            if step % 5 == 0: # Only calc grad_norm every 5 to save CPU
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        grad_norm += param_norm.item() ** 2
                grad_norm = grad_norm ** 0.5

            residual = torch.abs(images - stego).mean().item()
            
            # Get learnable strength for logging
            stego_strength = 0.0
            if hasattr(self.model.encoder, 'stego_strength'):
                stego_strength = self.model.encoder.stego_strength.item()

            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/loss_image": loss_img.item(),
                "train/loss_bits": loss_bits.item(),
                "train/loss_perceptual": loss_perceptual.item(),
                "train/bit_error_rate": ber,
                "train/psnr": psnr,
                "train/grad_norm": grad_norm,
                "train/noise_intensity": noise_intensity,
                "train/stego_strength": stego_strength,
                "train/pixel_residual_mean": residual,
                "train/lr": self.optimizer.param_groups[0]['lr']
            }, step=self.global_step)

            # Force MPS Cache Clear (Every 50 steps is enough for e6+)
            if self.device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()
            
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "BER": f"{ber:.4f}",
                "PSNR": f"{psnr:.2f}"
            })
            
        return np.mean(epoch_losses), np.mean(epoch_bers), np.mean(epoch_psnrs)

    def run(self):
        logger.info(f"Starting training loop from epoch {self.start_epoch}...")
        best_ber = 1.0
        
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            avg_loss, avg_ber, avg_psnr = self.train_epoch(epoch)
            
            # Epoch Level Logging
            logger.info(f"Epoch {epoch} Summary | Loss: {avg_loss:.4f} | BER: {avg_ber:.4f} | PSNR: {avg_psnr:.2f}")
            wandb.log({
                "epoch/avg_total_loss": avg_loss,
                "epoch/avg_bit_error_rate": avg_ber,
                "epoch/avg_psnr": avg_psnr,
                "epoch/learning_rate": self.optimizer.param_groups[0]['lr'],
                "epoch/num": epoch
            }, step=self.global_step)
            
            self.scheduler.step(avg_loss)
            
            # Save Checkpoints
            checkpoint_path = os.path.join(self.checkpoint_dir, f"stegadna_e{epoch}_ber{avg_ber:.4f}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            
            if avg_ber < best_ber:
                best_ber = avg_ber
                # Global Best
                torch.save(self.model.state_dict(), "model/stegadna_best.pth")
                # Tag-specific Best (for publication/run tracking)
                tag_best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
                torch.save(self.model.state_dict(), tag_best_path)
                logger.info(f"New Best Model Saved (BER: {best_ber:.4f}) at {tag_best_path}")
                
            # Log sample images
            self.log_samples(epoch)

    def log_samples(self, epoch):
        self.model.eval()
        # Get a smaller sample for logging to save CPU memory
        images, bits = next(iter(self.loader))
        images, bits = images[:4].to(self.device), bits[:4].to(self.device)
        
        with torch.no_grad():
            stego, _ = self.model(images, bits)
            
        # Move to CPU for logging
        images_cpu = images.detach().cpu()
        stego_cpu = stego.detach().cpu()
            
        # Un-normalize for visualization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        images_unnorm = (images_cpu * std + mean).clamp(0, 1)
        stego_unnorm = (stego_cpu * std + mean).clamp(0, 1)
        # Log pairs
        wandb_images = []
        for i in range(images_cpu.shape[0]):
            wandb_images.append(wandb.Image(images_unnorm[i], caption=f"Original_E{epoch}"))
            wandb_images.append(wandb.Image(stego_unnorm[i], caption=f"Stego_E{epoch}"))
        
        # Add residual heat map (where we are hiding data)
        diff = torch.abs(images_unnorm - stego_unnorm).mean(dim=1, keepdim=True)
        max_diff = diff.view(diff.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1) + 1e-8
        diff_norm = (diff / max_diff).clamp(0, 1)
        # Expand 1-channel diff to 3-channel for concatenation
        diff_norm = diff_norm.repeat(1, 3, 1, 1)

        # Create side-by-side strips: [Original | Stego | Heatmap]
        strips = []
        for i in range(images_cpu.shape[0]):
            strip = torch.cat([images_unnorm[i], stego_unnorm[i], diff_norm[i]], dim=2)
            strips.append(wandb.Image(strip, caption=f"Comparison_E{epoch}_S{i} [Orig | Stego | Resid]"))
            
        wandb.log({"visuals/comparison": strips}, step=self.global_step)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StegaDNA Training Script")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_bits", type=float, default=15.0)
    parser.add_argument("--resume_path", type=str, default=None, help="Path to checkpoint .pth to resume")
    parser.add_argument("--wandb_id", type=str, default=None, help="W&B run ID to resume (e.g. h3bxws76)")
    parser.add_argument("--run_name", type=str, default=None, help="Custom name for W&B run")
    parser.add_argument("--group", type=str, default="v2-dev", help="W&B experiment group")
    parser.add_argument("--lambda_perceptual", type=float, default=2.0, help="Weight for VGG perceptual loss")
    parser.add_argument("--curriculum_epochs", type=int, default=10, help="Epochs to ramp up noise")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable Mixed Precision training")
    parser.add_argument("--use_v3_noise", action="store_true", help="Enable Print-and-Scan (Analog) noise")
    parser.add_argument("--warmup_no_vgg", action="store_true", help="Disable VGG loss for signal recovery")
    parser.add_argument("--tag", type=str, default="baseline", help="Tag for this run (for grouping checkpoints)")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        lambda_bits=args.lambda_bits,
        resume_path=args.resume_path,
        wandb_id=args.wandb_id,
        run_name=args.run_name,
        group=args.group,
        lambda_perceptual=args.lambda_perceptual,
        curriculum_epochs=args.curriculum_epochs,
        mixed_precision=args.mixed_precision,
        use_v3_noise=args.use_v3_noise,
        warmup_no_vgg=args.warmup_no_vgg,
        tag=args.tag
    )
    
    trainer = StegaTrainer(config)
    trainer.run()
