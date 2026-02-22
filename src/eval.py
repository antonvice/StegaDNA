import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.progress import track
from rich.panel import Panel

# Ensure the project root is in sys.path
sys.path.append(os.getcwd())

from src.data.loader import get_stega_dataloader
from src.models.neural_engine import StegaDNAEngine

console = Console()

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return (20 * torch.log10(1.0 / torch.sqrt(mse))).item()

def calculate_ber(pred_bits, true_bits):
    pred_bits = (torch.sigmoid(pred_bits) > 0.5).float()
    errors = torch.sum(torch.abs(pred_bits - true_bits))
    return (errors / true_bits.numel()).item()

def recalibrate_bn(model, loader, device, num_batches=20):
    """
    Forces BatchNorm layers to re-calculate running statistics 
    based on CLEAN images only. This solves the 'Eval Mode' failure.
    """
    console.print(f"[bold yellow]Recalibrating BatchNorm stats on {num_batches} clean batches...[/]")
    model.train() # Set to train to update BN stats
    with torch.no_grad():
        for i, (images, bits) in enumerate(loader):
            if i >= num_batches: break
            images = images.to(device)
            bits = bits.to(device)
            # Forward pass with intensity 0 (Digital Clean)
            _ = model(images, bits, noise_intensity=0.0)
    model.eval() # Return to eval mode for final testing
    console.print(f"✅ BatchNorm recalibrated.")

def evaluate_model(model_path, loader, device, use_v3_noise=False, intensity=0.0, name="Model", recalibrate=True):
    if not os.path.exists(model_path):
        return None
    
    # Load model
    model = StegaDNAEngine(payload_bits=128, use_v3_noise=use_v3_noise).to(device)
    state_dict = torch.load(model_path, map_location=device)
    
    # Filter state_dict in case of mismatch (noise layer names etc)
    model.load_state_dict(state_dict, strict=False)
    
    # We test in BOTH modes to see if Batch Norm is the culprit
    # But primarily we use eval() as per standard practice
    if recalibrate:
        recalibrate_bn(model, loader, device)
    else:
        model.eval()
    
    total_ber = 0.0
    total_psnr = 0.0
    count = 0
    
    with torch.no_grad():
        for images, bits in track(loader, description=f"Evaluating {name} (Intensity: {intensity})..."):
            images, bits = images.to(device), bits.to(device)
            # Forward pass with explicit intensity
            stego, recovered_bits = model(images, bits, noise_intensity=intensity)
            
            total_ber += calculate_ber(recovered_bits, bits)
            total_psnr += calculate_psnr(images, stego)
            count += 1
            
    return {
        "name": name,
        "intensity": intensity,
        "ber": total_ber / count,
        "psnr": total_psnr / count
    }

def main():
    parser = argparse.ArgumentParser(description="StegaDNA Evaluation Suite v4")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--use_v3_noise", action="store_true", help="Use Analog Hole noise layer")
    parser.add_argument("--intensity", type=float, default=1.0, help="Noise intensity for robustness check")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup Data (using relative paths for portability)
    tsv_path = "data/text/150k_msgs_sample_hashed_pii.tsv"
    image_dir = "data/images"
    
    if not os.path.exists(tsv_path):
        logger.error(f"Data not found at {tsv_path}. Please check your current directory.")
        return

    loader = get_stega_dataloader(
        tsv_path=tsv_path,
        image_dir=image_dir,
        batch_size=args.batch_size,
        payload_bits=128,
        shuffle=False,
        num_workers=2
    )
    
    results = []
    
    # 1. Clean Channel Evaluation (Digital-Only)
    results.append(evaluate_model(
        args.checkpoint, loader, args.device, 
        use_v3_noise=args.use_v3_noise, intensity=0.0, name="Digital Clean"
    ))
    
    # 2. Robustness Evaluation (Analog Hole Simulation)
    results.append(evaluate_model(
        args.checkpoint, loader, args.device, 
        use_v3_noise=args.use_v3_noise, intensity=args.intensity, name=f"Analog Hole (Int:{args.intensity})"
    ))
            
    # Print Pretty Table
    table = Table(title=f"StegaDNA Benchmark: {os.path.basename(args.checkpoint)}", header_style="bold cyan")
    table.add_column("Channel Type", style="white")
    table.add_column("Noise Intensity", justify="center")
    table.add_column("Bit Error Rate (BER) ↓", justify="right", style="bold yellow")
    table.add_column("PSNR (Quality) ↑", justify="right", style="bold green")
    table.add_column("Status", justify="center")

    for res in results:
        # Determine Status
        if res['intensity'] == 0:
            status = "✅ PERFECT" if res['ber'] < 0.01 else "⚠️ LEAKY"
        else:
            status = "✅ ROBUST" if res['ber'] < 0.30 else "❌ FAILED"
        
        if res['ber'] > 0.45: status = "💀 DEAD"
        
        table.add_row(
            res['name'],
            f"{res['intensity']:.1f}",
            f"{res['ber']:.4%}",
            f"{res['psnr']:.2f} dB",
            status
        )

    console.print(Panel("[bold magenta]StegaDNA Evaluation Suite v4[/]", subtitle="Analog Hole Breakthrough Analysis", expand=False))
    console.print(table)
    
    console.print(f"\n[bold cyan]Research Note:[/]")
    console.print("If 'Digital Clean' is > 40%, ensure your weights are compatible with the V3 (MultiScale) architecture.")
    console.print("Common cause: Normalization mismatch or BN stats drift.")

if __name__ == "__main__":
    main()
