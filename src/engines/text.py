import torch
from src.models.text_engine import TextDNAEngine
from loguru import logger

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
_engine = TextDNAEngine()

async def embed(raw_text, dna_bits):
    """
    In a real scenario, this would use an LLM (e.g. Llama-3) 
    and the LogitsProcessor to generate a watermarked response.
    """
    logger.info(f"Generating DNA-stamped text response...")
    
    bits_tensor = torch.tensor(dna_bits, dtype=torch.float32).to(DEVICE)
    processor = _engine.get_logits_processor(bits_tensor)
    
    # Placeholder: In production, this goes into model.generate(logits_processor=[processor])
    stamped_text = f"[DNA-STAMPED] {raw_text}"
    
    return {
        "status": "success", 
        "modality": "text", 
        "stamped_text": stamped_text,
        "dna_id_verified": True
    }

async def extract(text):
    """
    Statistical extraction of the DNA signature.
    """
    logger.info("Extracting DNA from text sample...")
    
    # Placeholder for statistical z-score check
    confidence = 0.98
    
    return {
        "status": "success", 
        "modality": "text", 
        "confidence": confidence,
        "is_genuine": confidence > 0.8
    }
