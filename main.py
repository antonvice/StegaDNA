import uvicorn
from server.app import app
from loguru import logger

if __name__ == "__main__":
    logger.info("Starting StegaDNA Universal Gateway...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
