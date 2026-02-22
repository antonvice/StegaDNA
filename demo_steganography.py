import os
import sys
import hashlib
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(os.getcwd())
from src.models.neural_engine import StegaDNAEngine

def _text_to_bits(text, payload_bits=128):
    hash_digest = hashlib.sha256(text.encode('utf-8')).digest()
    bits = []
    for byte in hash_digest:
        bits.extend([int(b) for b in format(byte, '08b')])
    
    bit_tensor = torch.tensor(bits, dtype=torch.float32)
    
    if len(bit_tensor) >= payload_bits:
        return bit_tensor[:payload_bits].unsqueeze(0)
    else:
        padding = torch.zeros(payload_bits - len(bit_tensor))
        return torch.cat([bit_tensor, padding]).unsqueeze(0)

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading StegaDNA v3-gold on {device}...")
    
    # Init and Load Model
    model = StegaDNAEngine(payload_bits=128, use_v3_noise=False).to(device)
    ckpt = torch.load('model/checkpoints/v3-gold/stegadna_e249_ber0.2655.pth', map_location=device)
    model.load_state_dict(ckpt, strict=False)
    
    # We force model.train() and duplicate the image batch. 
    # This mathematically patches the Batch Normalization corruption bug we found during evaluation
    # so that the neural network uses the CURRENT image layers instead of the corrupted running history.
    model.train() 

    # Image Prep
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    import glob
    img_list = glob.glob('data/images/**/*.jpg', recursive=True)
    if not img_list:
        print("Error: No images found!")
        sys.exit(1)
        
    img_path = img_list[0]
    
    image = Image.open(img_path).convert('RGB')
    tensor_img = transform(image).unsqueeze(0).to(device)
    
    # Duplicate batch to satisfy BatchNorm batch variance requirement (>1)
    tensor_img = torch.cat([tensor_img, tensor_img], dim=0)

    # Encode
    secret_text = "Fermatix Confidential Data"
    payload = _text_to_bits(secret_text).to(device)
    payload = torch.cat([payload, payload], dim=0) # Duplicate payload
    
    print(f"Encoding text hash ({128} bits): '{secret_text}'")
    with torch.no_grad():
        stego_img, recovered_logits = model(tensor_img, payload, noise_intensity=0.0)
    
    # Decode
    pred_bits = (torch.sigmoid(recovered_logits) > 0.5).float()
    errors = torch.sum(torch.abs(pred_bits[0] - payload[0]))
    ber = (errors / 128.0).item()
    print(f"Decoded bits! Bit Error Rate (BER): {ber:.4f} ({int(errors)} bits flipped)")
    
    if ber < 0.20:
        print("Model correctly reads the majority of the payload signature!")
    else:
        print("Model struggles to read the signature perfectly without training context.")
        
    # Unnormalize for saving
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
    
    orig_display = (tensor_img[0:1] * std + mean).clamp(0, 1)
    stego_display = (stego_img[0:1] * std + mean).clamp(0, 1)
    residual_display = torch.abs(orig_display - stego_display) * 10.0 # Amp up to see it
    
    import torchvision
    grid = torch.cat([orig_display, stego_display, residual_display], dim=0)
    torchvision.utils.save_image(grid, "demo_result.png")
    print("Saved side-by-side encoded image to 'demo_result.png' [Original | Stego | Residual x10]")

if __name__ == "__main__":
    main()
