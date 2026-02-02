import torch
from PIL import Image
from torchvision import transforms
from src.models.neural_engine import StegaDNAEngine
from loguru import logger
import io

# Load model (Pre-trained)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "model/stegadna_best.pth"

# Load the engine once
try:
    _engine = StegaDNAEngine(payload_bits=128).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        _engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _engine.eval()
    logger.info("Image Engine: Neural model loaded successfully.")
except Exception as e:
    logger.warning(f"Image Engine: Model not ready yet. Running in dummy mode. Error: {e}")
    _engine = None

# Transforms
_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def _unnormalize(img_tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(DEVICE)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(DEVICE)
    return (img_tensor * std + mean).clamp(0, 1)

async def embed(file, dna_bits):
    logger.info(f"Embedding DNA into image: {file.filename}")
    
    if _engine is None:
        return {"status": "error", "message": "Model not loaded"}

    # Load image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    orig_size = image.size
    
    img_tensor = _transform(image).unsqueeze(0).to(DEVICE)
    bits_tensor = torch.tensor(dna_bits, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        stamped_tensor, _ = _engine(img_tensor, bits_tensor)
        
    # Convert back to PIL
    stamped_img = _unnormalize(stamped_tensor.squeeze(0))
    stamped_pil = transforms.ToPILImage()(stamped_img.cpu())
    stamped_pil = stamped_pil.resize(orig_size) # Restore size
    
    # Save to buffer
    img_byte_arr = io.BytesIO()
    stamped_pil.save(img_byte_arr, format='PNG')
    
    return {"status": "success", "modality": "image", "data": img_byte_arr.getvalue().hex()[:100] + "..." }

async def extract(file):
    if _engine is None: return {"status": "error", "message": "Model not loaded"}
    
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    img_tensor = _transform(image).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        _, recovered_bits = _engine(img_tensor, torch.zeros((1, 128)).to(DEVICE))
    
    # Convert logits to binary string
    bits = (torch.sigmoid(recovered_bits) > 0.5).int().tolist()[0]
    return {"status": "success", "modality": "image", "dna_vector": bits}
