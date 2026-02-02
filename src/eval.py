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

def evaluate_model(model_path, loader, device, name="Model"):
    if not os.path.exists(model_path):
        return None
    
    # Load model
    model = StegaDNAEngine(payload_bits=128).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_ber = 0.0
    total_psnr = 0.0
    count = 0
    
    with torch.no_grad():
        for images, bits in track(loader, description=f"Evaluating {name}..."):
            images, bits = images.to(device), bits.to(device)
            stego, recovered_bits = model(images, bits)
            
            total_ber += calculate_ber(recovered_bits, bits)
            total_psnr += calculate_psnr(images, stego)
            count += 1
            
    return {
        "name": name,
        "ber": total_ber / count,
        "psnr": total_psnr / count
    }

def main():
    parser = argparse.ArgumentParser(description="StegaDNA Evaluation Suite")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--compare_best", action="store_true", help="Compare against the current best model")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="mps" if torch.backends.mps.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup Data
    tsv_path = "/Users/antonvice/Documents/programming/StegaDNA/data/text/150k_msgs_sample_hashed_pii.tsv"
    image_dir = "/Users/antonvice/Documents/programming/StegaDNA/data/images"
    
    loader = get_stega_dataloader(
        tsv_path=tsv_path,
        image_dir=image_dir,
        batch_size=args.batch_size,
        payload_bits=128,
        shuffle=False,
        num_workers=2
    )
    
    results = []
    
    # 1. Eval Target Checkpoint
    target_res = evaluate_model(args.checkpoint, loader, args.device, name=os.path.basename(args.checkpoint))
    if target_res:
        results.append(target_res)
    else:
        console.print(f"[bold red]Error:[/] Checkpoint not found at {args.checkpoint}")
        return

    # 2. Eval Best Model (Optional)
    if args.compare_best:
        best_path = "model/stegadna_best.pth"
        best_res = evaluate_model(best_path, loader, args.device, name="Global Best")
        if best_res:
            results.append(best_res)
            
    # Print Pretty Table
    table = Table(title="StegaDNA Performance Benchmark", header_style="bold cyan")
    table.add_column("Model Name", style="white")
    table.add_column("Bit Error Rate (BER) ↓", justify="right", style="bold yellow")
    table.add_column("PSNR (Quality) ↑", justify="right", style="bold green")
    table.add_column("Status", justify="center")

    for res in results:
        # Determine Status
        status = "✅ PASS" if res['ber'] < 0.1 else "⚠️ TRAINING"
        if res['ber'] > 0.4: status = "❌ POOR"
        
        table.add_row(
            res['name'],
            f"{res['ber']:.4%}",
            f"{res['psnr']:.2f} dB",
            status
        )

    console.print(Panel("[bold magenta]StegaDNA Evaluation Suite[/]", expand=False))
    console.print(table)
    
    if len(results) > 1:
        diff_ber = results[1]['ber'] - results[0]['ber']
        diff_psnr = results[0]['psnr'] - results[1]['psnr']
        console.print(f"\n[bold]Comparison Result:[/]")
        color = "green" if diff_ber > 0 else "red"
        console.print(f"Target is [bold {color}]{abs(diff_ber):.2%}[/] {'better' if diff_ber < 0 else 'worse'} in BER than Best.")

if __name__ == "__main__":
    main()
