# main.py

"""
Main FastAPI application entry point.
Orchestrates the server, WebSocket handling, and module interactions.
"""
import asyncio
import base64
import logging
from typing import Optional

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import our custom modules
import config
from transcriber import KyutaiTranscriber


transcriber = KyutaiTranscriber()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Kyutai STT Server")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Load the model on server startup."""
    transcriber.ensure_ready()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "Kyutai STT Server",
        "model": config.MODEL_ID,
        "ready": transcriber.is_ready
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time transcription."""
    await websocket.accept()
    logger.info("🔌 WebSocket connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            message_type = data.get("type")
            
            if message_type == "audio":
                audio_base64 = data.get("data")
                if not audio_base64:
                    await websocket.send_json({"type": "error", "message": "No audio data"})
                    continue
                
                # 1. Decode base64
                audio_bytes = base64.b64decode(audio_base64)
                
                # 2. Transcribe (runs in a thread pool)
                transcription = await asyncio.to_thread(transcriber.transcribe_bytes, audio_bytes)

                if transcription:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": transcription
                    })
                    logger.info(f"✅ Transcription: {transcription}")
                else:
                    await websocket.send_json({"type": "error", "message": "Transcription failed"})
                    
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
                
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        logger.info("🔌 WebSocket disconnected")

if __name__ == "__main__":
    logger.info("🚀 Starting Simple STT Server...")
    logger.info(f"📍 Server will run on: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
    logger.info(f"🔌 WebSocket endpoint: ws://{config.SERVER_HOST}:{config.SERVER_PORT}/ws")
    
    uvicorn.run(
        app,
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        log_level="info"
    )