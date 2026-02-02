import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dotenv import load_dotenv
import argparse

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from loguru import logger
from tqdm import tqdm
import wandb
import numpy as np
from pydantic import BaseModel
from typing import Optional

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
    tsv_path: str = "/Users/antonvice/Documents/programming/StegaDNA/data/text/150k_msgs_sample_hashed_pii.tsv"
    image_dir: str = "/Users/antonvice/Documents/programming/StegaDNA/data/images"
    resume_path: Optional[str] = None
    mixed_precision: bool = False
    tag: str = "default"

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ber(pred_bits, true_bits):
    """Bit Error Rate"""
    pred_bits = (torch.sigmoid(pred_bits) > 0.5).float()
    errors = torch.sum(torch.abs(pred_bits - true_bits))
    return errors / true_bits.numel()

class StegaTrainer:
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 1. Initialize W&B
        wandb.init(
            project=config.project_name,
            config=config.dict(),
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
        self.model = StegaDNAEngine(payload_bits=config.payload_bits).to(self.device)
        
        # Resume Checkpoint if available
        if config.resume_path and os.path.exists(config.resume_path):
            logger.info(f"Resuming from checkpoint: {config.resume_path}")
            state_dict = torch.load(config.resume_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        
        self.criterion_img = nn.MSELoss()
        self.criterion_bits = nn.BCEWithLogitsLoss()
        
        # Scalar for mixed precision if enabled (Only for CUDA, MPS doesn't use GradScaler same way)
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision and config.device == "cuda" else None
        
        # Determine actual autocast device
        self.autocast_device = "cuda" if "cuda" in config.device else "cpu"
        if config.device == "mps" and config.mixed_precision:
             logger.warning("Mixed precision (FP16) on MPS can sometimes be slower. Disable if BER doesn't improve.")
             self.autocast_device = "mps"
        
        self.checkpoint_dir = os.path.join("model/checkpoints", config.tag)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"StegaTrainer initialized on {self.device} | Tag: {config.tag}")

    def train_epoch(self, epoch):
        self.model.train()
        epoch_losses = []
        epoch_bers = []
        epoch_psnrs = []
        
        pbar = tqdm(self.loader, desc=f"Epoch {epoch}")
        for images, bits in pbar:
            images, bits = images.to(self.device), bits.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (with noise layer enabled)
            if self.config.mixed_precision:
                with torch.autocast(device_type=self.autocast_device, enabled=True):
                    stego, recovered_bits = self.model(images, bits)
                    loss_img = self.criterion_img(stego, images)
                    loss_bits = self.criterion_bits(recovered_bits, bits)
                    total_loss = loss_img + self.config.lambda_bits * loss_bits
            else:
                stego, recovered_bits = self.model(images, bits)
                loss_img = self.criterion_img(stego, images)
                loss_bits = self.criterion_bits(recovered_bits, bits)
                total_loss = loss_img + self.config.lambda_bits * loss_bits
            
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
            epoch_bers.append(ber.item())
            epoch_psnrs.append(psnr.item())
            
            # Master ML Logging
            grad_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    grad_norm += param_norm.item() ** 2
            grad_norm = grad_norm ** 0.5

            # Calculate residual for visualization (how much are we changing pixels?)
            residual = torch.abs(images - stego).mean().item()
            
            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}", 
                "BER": f"{ber.item():.4f}",
                "PSNR": f"{psnr.item():.2f}"
            })
            
            # Step logging to W&B
            wandb.log({
                "train/total_loss": total_loss.item(),
                "train/loss_image": loss_img.item(),
                "train/loss_bits": loss_bits.item(),
                "train/bit_error_rate": ber.item(),
                "train/psnr": psnr.item(),
                "train/grad_norm": grad_norm,
                "train/pixel_residual_mean": residual,
                "train/lr": self.optimizer.param_groups[0]['lr']
            })
            
        return np.mean(epoch_losses), np.mean(epoch_bers), np.mean(epoch_psnrs)

    def run(self):
        logger.info("Starting training loop...")
        best_ber = 1.0
        
        for epoch in range(1, self.config.epochs + 1):
            avg_loss, avg_ber, avg_psnr = self.train_epoch(epoch)
            
            # Epoch Level Logging
            logger.info(f"Epoch {epoch} Summary | Loss: {avg_loss:.4f} | BER: {avg_ber:.4f} | PSNR: {avg_psnr:.2f}")
            wandb.log({
                "epoch/avg_total_loss": avg_loss,
                "epoch/avg_bit_error_rate": avg_ber,
                "epoch/avg_psnr": avg_psnr,
                "epoch/learning_rate": self.optimizer.param_groups[0]['lr'],
                "epoch/num": epoch
            })
            
            self.scheduler.step(avg_loss)
            
            # Save Checkpoints
            checkpoint_path = os.path.join(self.checkpoint_dir, f"stegadna_e{epoch}_ber{avg_ber:.4f}.pth")
            torch.save(self.model.state_dict(), checkpoint_path)
            
            if avg_ber < best_ber:
                best_ber = avg_ber
                torch.save(self.model.state_dict(), "model/stegadna_best.pth")
                logger.info(f"New Best Model Saved (BER: {best_ber:.4f})")
                
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
        diff = (diff / (diff.max() + 1e-8)) # Normalize for visibility
        for i in range(diff.shape[0]):
            wandb_images.append(wandb.Image(diff[i], caption=f"Residual_Heatmap_E{epoch}"))
            
        wandb.log({"visuals/samples": wandb_images})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="StegaDNA Training Script")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lambda_bits", type=float, default=15.0)
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    parser.add_argument("--mixed_precision", action="store_true", help="Enable mixed precision training")
    parser.add_argument("--tag", type=str, default="baseline", help="Tag for this run (for grouping checkpoints)")
    
    args = parser.parse_args()
    
    config = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        lambda_bits=args.lambda_bits,
        resume_path=args.resume,
        mixed_precision=args.mixed_precision,
        tag=args.tag
    )
    
    trainer = StegaTrainer(config)
    trainer.run()
