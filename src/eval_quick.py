import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from loguru import logger

sys.path.append(os.getcwd())
try:
    from src.data.loader import get_stega_dataloader
    from src.models.neural_engine import StegaDNAEngine
except Exception as e:
    print(f"Import error: {e}")
    sys.exit(1)

def calculate_ber(pred_bits, true_bits):
    pred_bits = (torch.sigmoid(pred_bits) > 0.5).float()
    errors = torch.sum(torch.abs(pred_bits - true_bits))
    return (errors / true_bits.numel()).item()

def recalibrate_bn(model, loader, device, num_batches=20):
    model.train()
    with torch.no_grad():
        for i, (images, bits) in enumerate(loader):
            if i >= num_batches: break
            images, bits = images.to(device), bits.to(device)
            _ = model(images, bits, noise_intensity=0.0)
    model.eval()

def main():
    print("Starting eval_quick.py [CPU MODE]...", flush=True)
    device = "cpu"
    tsv_path = "data/text/150k_msgs_sample_hashed_pii.tsv"
    image_dir = "data/images"
    
    loader = get_stega_dataloader(
        tsv_path=tsv_path, image_dir=image_dir, batch_size=8, payload_bits=128, shuffle=False, num_workers=0
    )
    
    for ckpt in ["v4-universal/best_model.pth", "v3-gold/best_model.pth"]:
        path = f"model/checkpoints/{ckpt}"
        if not os.path.exists(path):
            print(f"Skipping {path}, does not exist.", flush=True)
            continue
        try:
            print(f"Loading {ckpt}...", flush=True)
            model = StegaDNAEngine(payload_bits=128, use_v3_noise=True).to(device)
            state_dict = torch.load(path, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            
            print(f"Recalibrating BN on 5 batches...", flush=True)
            recalibrate_bn(model, loader, device, num_batches=5)
            
            model.eval()
            total_ber_clean = 0
            total_ber_analog = 0
            for i, (images, bits) in enumerate(loader):
                if i >= 5: break
                images, bits = images.to(device), bits.to(device)
                
                # Digital Clean
                _, rec_bits_clean = model(images, bits, noise_intensity=0.0)
                total_ber_clean += calculate_ber(rec_bits_clean, bits)
                
                # Analog Hole (1.0)
                _, rec_bits_analog = model(images, bits, noise_intensity=1.0)
                total_ber_analog += calculate_ber(rec_bits_analog, bits)
                
            print(f"[{ckpt}] CLEAN BER: {total_ber_clean/5:.4f} | ANALOG BER: {total_ber_analog/5:.4f}", flush=True)
        except Exception as e:
            print(f"Error loading {ckpt}: {e}", flush=True)

if __name__ == "__main__":
    main()
