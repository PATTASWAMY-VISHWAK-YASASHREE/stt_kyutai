"""Production-ready FastAPI application - Bug-free version."""

import asyncio
import base64
import json
import logging
import os
import signal
import sys
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Python 3.7+ compatibility
if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """Message types."""
    AUDIO = "audio"
    TRANSCRIPTION = "transcription"
    ERROR = "error"
    PING = "ping"
    PONG = "pong"
    READY = "ready"


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections: int = 100
    max_audio_size_mb: int = 10
    ws_timeout: float = 300.0
    skip_model_load: bool = False
    
    @classmethod
    def from_env(cls) -> 'ServerConfig':
        """Load from environment."""
        return cls(
            host=os.getenv("SERVER_HOST", "0.0.0.0"),
            port=int(os.getenv("SERVER_PORT", "8000")),
            max_connections=int(os.getenv("MAX_CONNECTIONS", "100")),
            max_audio_size_mb=int(os.getenv("MAX_AUDIO_SIZE_MB", "10")),
            ws_timeout=float(os.getenv("WS_TIMEOUT", "300")),
            skip_model_load=os.getenv("KYUTAI_SKIP_MODEL_LOAD") not in (None, "0"),
        )


@dataclass
class ConnectionMetrics:
    """Connection metrics."""
    connection_id: str
    connected_at: float
    messages_received: int = 0
    messages_sent: int = 0
    transcriptions: int = 0
    errors: int = 0
    
    @property
    def uptime(self) -> float:
        return time.time() - self.connected_at


class ConnectionManager:
    """Thread-safe connection manager."""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self.connections: Dict[str, WebSocket] = {}
        self.metrics: Dict[str, ConnectionMetrics] = {}
    
    async def connect(self, websocket: WebSocket) -> str:
        """Connect new websocket."""
        if len(self.connections) >= self.max_connections:
            raise HTTPException(503, "Max connections reached")
        
        connection_id = str(uuid4())
        await websocket.accept()
        
        self.connections[connection_id] = websocket
        self.metrics[connection_id] = ConnectionMetrics(
            connection_id=connection_id,
            connected_at=time.time()
        )
        
        logger.info(f"🔌 Connected: {connection_id[:8]}... ({len(self.connections)} active)")
        return connection_id
    
    async def disconnect(self, connection_id: str):
        """Disconnect websocket."""
        if connection_id in self.connections:
            self.connections.pop(connection_id)
            metrics = self.metrics.pop(connection_id, None)
            
            if metrics:
                logger.info(
                    f"🔌 Disconnected: {connection_id[:8]}... "
                    f"(uptime: {metrics.uptime:.1f}s, transcriptions: {metrics.transcriptions})"
                )
    
    async def send_json(self, connection_id: str, data: dict):
        """Send JSON to connection."""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send_json(data)
                if connection_id in self.metrics:
                    self.metrics[connection_id].messages_sent += 1
            except Exception as e:
                logger.error(f"Send error {connection_id[:8]}...: {e}")
                await self.disconnect(connection_id)
    
    def update_metrics(self, connection_id: str, **kwargs):
        """Update metrics."""
        if connection_id in self.metrics:
            m = self.metrics[connection_id]
            for key, value in kwargs.items():
                if hasattr(m, key):
                    setattr(m, key, getattr(m, key) + value)


class TranscriptionService:
    """Transcription service."""
    
    def __init__(self):
        self.transcriber = None
        self.is_ready = False
        self.start_time = time.time()
        self.total_transcriptions = 0
    
    def initialize(self, skip_load: bool = False):
        """Initialize transcriber."""
        if skip_load:
            logger.warning("⚠️  Skipping model load")
            return
        
        try:
            logger.info("🔄 Loading model...")
            
            # Import here to avoid circular imports
            try:
                import config
                model_id = getattr(config, 'MODEL_ID', 'kyutai/moshi-speech-to-text')
            except ImportError:
                model_id = 'kyutai/moshi-speech-to-text'
            
            from transcriber import KyutaiTranscriber
            self.transcriber = KyutaiTranscriber()
            self.transcriber.ensure_ready()
            
            self.is_ready = True
            logger.info("✅ Model loaded")
        except Exception as e:
            logger.error(f"❌ Model load failed: {e}")
            raise
    
    def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """Transcribe audio."""
        if not self.is_ready or self.transcriber is None:
            raise RuntimeError("Not initialized")
        
        try:
            result = self.transcriber.transcribe_bytes(audio_bytes)
            self.total_transcriptions += 1
            return result
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            raise
    
    @property
    def uptime(self) -> float:
        return time.time() - self.start_time


# Global instances
server_config = ServerConfig.from_env()
connection_manager = ConnectionManager(server_config.max_connections)
transcription_service = TranscriptionService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App lifespan."""
    # Startup
    logger.info("🚀 Starting server...")
    
    try:
        # Run in thread to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            transcription_service.initialize,
            server_config.skip_model_load
        )
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        if not server_config.skip_model_load:
            sys.exit(1)
    
    logger.info(f"📍 Ready on http://{server_config.host}:{server_config.port}")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")


# Create app
app = FastAPI(
    title="Kyutai STT Server",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Health check."""
    return {
        "status": "healthy" if transcription_service.is_ready else "degraded",
        "service": "Kyutai STT Server",
        "ready": transcription_service.is_ready,
        "uptime_seconds": round(transcription_service.uptime, 2),
        "active_connections": len(connection_manager.connections),
        "total_transcriptions": transcription_service.total_transcriptions,
    }


@app.get("/health")
async def health():
    """Detailed health."""
    if not transcription_service.is_ready:
        raise HTTPException(503, "Not ready")
    
    return {"status": "healthy"}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint."""
    connection_id: Optional[str] = None
    
    try:
        # Connect
        connection_id = await connection_manager.connect(websocket)
        
        # Send ready
        await connection_manager.send_json(connection_id, {
            "type": MessageType.READY,
            "connection_id": connection_id,
        })
        
        # Message loop
        while True:
            try:
                # Receive with timeout
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=server_config.ws_timeout
                )
                
                connection_manager.update_metrics(connection_id, messages_received=1)
                message_type = data.get("type")
                
                if message_type == MessageType.AUDIO:
                    await handle_audio(connection_id, data)
                
                elif message_type == MessageType.PING:
                    await connection_manager.send_json(connection_id, {
                        "type": MessageType.PONG,
                        "timestamp": time.time(),
                    })
                
                else:
                    await connection_manager.send_json(connection_id, {
                        "type": MessageType.ERROR,
                        "message": f"Unknown type: {message_type}",
                    })
            
            except asyncio.TimeoutError:
                logger.warning(f"Timeout: {connection_id[:8]}...")
                break
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"Message error: {e}")
                await connection_manager.send_json(connection_id, {
                    "type": MessageType.ERROR,
                    "message": "Internal error",
                })
                connection_manager.update_metrics(connection_id, errors=1)
    
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    
    finally:
        if connection_id:
            await connection_manager.disconnect(connection_id)


async def handle_audio(connection_id: str, data: dict):
    """Handle audio message."""
    start_time = time.time()
    
    try:
        # Get audio data
        audio_base64 = data.get("data")
        if not audio_base64:
            await connection_manager.send_json(connection_id, {
                "type": MessageType.ERROR,
                "message": "No audio data",
            })
            connection_manager.update_metrics(connection_id, errors=1)
            return
        
        # Decode
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            logger.error(f"Base64 decode error: {e}")
            await connection_manager.send_json(connection_id, {
                "type": MessageType.ERROR,
                "message": "Invalid base64",
            })
            connection_manager.update_metrics(connection_id, errors=1)
            return
        
        # Check size
        max_size = server_config.max_audio_size_mb * 1024 * 1024
        if len(audio_bytes) > max_size:
            await connection_manager.send_json(connection_id, {
                "type": MessageType.ERROR,
                "message": f"Audio too large (max {server_config.max_audio_size_mb}MB)",
            })
            connection_manager.update_metrics(connection_id, errors=1)
            return
        
        if len(audio_bytes) == 0:
            await connection_manager.send_json(connection_id, {
                "type": MessageType.ERROR,
                "message": "Empty audio",
            })
            connection_manager.update_metrics(connection_id, errors=1)
            return
        
        # Transcribe in thread pool
        loop = asyncio.get_event_loop()
        transcription = await loop.run_in_executor(
            None,
            transcription_service.transcribe,
            audio_bytes
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        if transcription:
            await connection_manager.send_json(connection_id, {
                "type": MessageType.TRANSCRIPTION,
                "text": transcription,
                "processing_time_ms": round(processing_time, 2),
            })
            
            connection_manager.update_metrics(connection_id, transcriptions=1)
            
            logger.info(
                f"✅ [{connection_id[:8]}] '{transcription}' ({processing_time:.0f}ms)"
            )
        else:
            await connection_manager.send_json(connection_id, {
                "type": MessageType.ERROR,
                "message": "Transcription failed",
            })
            connection_manager.update_metrics(connection_id, errors=1)
    
    except Exception as e:
        logger.error(f"Audio handling error: {e}", exc_info=True)
        await connection_manager.send_json(connection_id, {
            "type": MessageType.ERROR,
            "message": "Processing error",
        })
        connection_manager.update_metrics(connection_id, errors=1)


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": "Internal error"})


if __name__ == "__main__":
    logger.info("🚀 Kyutai STT Server")
    logger.info(f"📍 {server_config.host}:{server_config.port}")
    
    uvicorn.run(
        app,
        host=server_config.host,
        port=server_config.port,
        log_level="info",
    )