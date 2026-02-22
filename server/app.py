"""
Universal Controller: Project DNA-Stamp API Gateway.
Routes requests to specialized modality engines while leveraging a shared Mojo DNA Core.
"""
import sys
import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from loguru import logger

# Ensure src/ is in the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from fastapi.responses import HTMLResponse, FileResponse
from server import bridge
from src.engines import image, audio, text

app = FastAPI(title="DNA-Stamp Universal Gateway")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("server/static/index.html")

@app.post("/stamp/{modality}")
async def stamp_media(
    modality: str,
    user_id: str = Form(...),
    file: UploadFile = File(None),
    raw_text: str = Form(None)
):
    """
    Universal Endpoint to embed DNA into any supported modality.
    """
    logger.info(f"Request: modality={modality}, user_id={user_id}")
    
    # 1. SHARED STEP: Generate DNA bits via Mojo Bridge
    dna_bits = bridge.generate_robust_bits(user_id)
    
    # 2. MODALITY STEP: Route to specialized engine
    if modality == "image":
        if not file: raise HTTPException(400, "Image file required")
        img_bytes = await image.embed(file, dna_bits)
        from fastapi import Response
        return Response(content=img_bytes, media_type="image/png")
        
    elif modality == "audio":
        if not file: raise HTTPException(400, "Audio file required")
        audio_data = await audio.embed(file, dna_bits)
        from fastapi import Response
        return Response(content=audio_data, media_type="audio/wav")
        
    elif modality == "text":
        if not raw_text: raise HTTPException(400, "raw_text required for text modality")
        return await text.embed(raw_text, dna_bits)
    
    else:
        raise HTTPException(400, f"Unsupported modality: {modality}")

@app.post("/extract/{modality}")
async def extract_media(
    modality: str,
    file: UploadFile,
    raw_text: str = Form(None)
):
    """
    Universal Endpoint to extract DNA from any supported modality.
    """
    if modality == "image":
        return await image.extract(file)
    elif modality == "audio":
        return await audio.extract(file)
    elif modality == "text":
        return await text.extract(raw_text)
    else:
        raise HTTPException(400, f"Unsupported modality: {modality}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "core": "mojo_bridge_active"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
