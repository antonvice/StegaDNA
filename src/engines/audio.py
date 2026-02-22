import torch
import os
import torchaudio
from src.models.audio_engine import AudioDNAEngine
from loguru import logger
import io

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
MODEL_PATH = "model/audio_stegadna_best.pth"

try:
    _engine = AudioDNAEngine(payload_bits=128).to(DEVICE)
    if os.path.exists(MODEL_PATH):
        _engine.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _engine.eval()
    logger.info("Audio Engine: Neural model loaded successfully.")
except Exception as e:
    logger.warning(f"Audio Engine: Using dummy mode (Model not trained). Error: {e}")
    _engine = None

async def embed(file, dna_bits):
    logger.info(f"Embedding DNA into audio: {file.filename}")
    
    if _engine is None:
        return {"status": "success", "modality": "audio", "note": "Dummy mode"}

    # Load audio
    contents = await file.read()
    waveform, sample_rate = torchaudio.load(io.BytesIO(contents))
    waveform = waveform.to(DEVICE).unsqueeze(0) # [B, C, T]

    bits_tensor = torch.tensor(dna_bits, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        stamped_audio, _ = _engine(waveform, bits_tensor)

    # Save to buffer
    buffer = io.BytesIO()
    torchaudio.save(buffer, stamped_audio.squeeze(0).cpu(), sample_rate, format="wav")

    return buffer.getvalue()

async def extract(file):
    if _engine is None: return {"status": "error", "message": "Model not trained"}
    
    contents = await file.read()
    waveform, _ = torchaudio.load(io.BytesIO(contents))
    waveform = waveform.to(DEVICE).unsqueeze(0)

    with torch.no_grad():
        _, recovered_bits = _engine(waveform, torch.zeros((1, 128)).to(DEVICE))
    
    bits = (torch.sigmoid(recovered_bits) > 0.5).int().tolist()[0]
    return {"status": "success", "modality": "audio", "dna_vector": bits}
