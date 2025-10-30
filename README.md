# 🎤 Kyutai Speech-to-Text Server

<div align="center">

[![License](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green.svg)](https://fastapi.tiangolo.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-yellow.svg)](https://huggingface.co/kyutai)

A high-performance, production-ready streaming speech recognition server powered by [Kyutai's STT models](https://huggingface.co/kyutai).

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [API Documentation](#-api-documentation) • [Configuration](#%EF%B8%8F-configuration) • [Examples](#-examples)

</div>

---

## 🌟 Features

### Core Capabilities
- 🚀 **Streaming Speech Recognition** - Real-time transcription with WebSocket support
- 🌍 **Multilingual** - Supports English and French (stt-1b-en_fr model)
- ⚡ **High Performance** - Optimized with torch.compile for fast inference
- 🎯 **Production Ready** - Robust error handling, logging, and monitoring
- 🔄 **Real-time Processing** - Low-latency streaming transcription
- 📊 **Performance Metrics** - Built-in RTF (Real-Time Factor) tracking

### Technical Features
- **WebSocket API** - Full-duplex communication for streaming audio
- **REST API** - Traditional HTTP endpoints for batch processing
- **Audio Format Support** - Automatic format detection (WAV, WebM, MP3, FLAC, OGG)
- **Smart Caching** - LRU cache for improved performance
- **Audio Preprocessing** - Automatic resampling, normalization, and enhancement
- **Configurable Models** - Support for different Kyutai model variants
- **CORS Enabled** - Ready for web application integration
- **Health Checks** - Built-in monitoring endpoints

---

## 📋 Requirements

### System Requirements
- **OS**: Windows, Linux, or macOS
- **Python**: 3.8 or higher
- **RAM**: Minimum 8GB (16GB recommended for optimal performance)
- **Storage**: ~5GB for models and dependencies

### Python Dependencies
- FastAPI >= 0.104.0
- Uvicorn >= 0.24.0
- Transformers >= 4.53.0 (for native Kyutai support)
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
- Soundfile >= 0.12.0
- Python-multipart >= 0.0.6

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/stt_kyutai.git
cd stt_kyutai
```

### 2. Create Virtual Environment
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\activate

# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your configuration
```

---

## ⚡ Quick Start

### Start the Server
```bash
python main.py
```

The server will start on `http://0.0.0.0:8000` by default.

### Test with Web UI
Open `examples/test_ui.html` in your browser and:
1. Click "Connect to Server"
2. Click "Start Recording"
3. Speak into your microphone
4. Click "Stop Recording"
5. See the transcription appear!

### Test with cURL
```bash
# Upload an audio file
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_audio.wav"
```

---

## � Docker

### Build the Image

```powershell
docker build --tag stt-kyutai .
```

### Run the Container

```powershell
docker run --rm -p 8000:8000 --env-file .env stt-kyutai
```

This command mounts the API on `http://localhost:8000`. Update `.env` before running to customize the model or device.

### Docker Compose

```powershell
docker compose up --build
```

The provided `docker-compose.yml`:

- Builds the image from the local source
- Binds port `8000`
- Loads environment variables from `.env`
- Persists Hugging Face caches in a named `huggingface-cache` volume

To stop the stack, press `Ctrl+C` and run `docker compose down` when finished.

---

## �📡 API Documentation

### WebSocket API

#### Connect
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
```

#### Send Audio
```javascript
// Send audio data as base64
ws.send(JSON.stringify({
    type: 'audio',
    data: base64AudioData
}));
```

#### Receive Transcription
```javascript
ws.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'transcription') {
        console.log('Transcription:', data.text);
        console.log('Processing time:', data.processing_time_ms, 'ms');
    }
};
```

### REST API

#### POST /transcribe
Upload audio file for transcription.

**Request:**
```bash
curl -X POST "http://localhost:8000/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav"
```

**Response:**
```json
{
  "text": "The transcribed text appears here.",
  "audio_duration": 5.2,
  "processing_time": 1.8,
  "real_time_factor": 0.34,
  "model": "kyutai/stt-1b-en_fr-trfs"
}
```

#### GET /health
Check server health.

**Response:**
```json
{
  "status": "healthy",
  "uptime": 3600.5,
  "model_loaded": true,
  "version": "1.0.0"
}
```

#### GET /stats
Get server statistics.

**Response:**
```json
{
  "total_transcriptions": 42,
  "total_time_ms": 125430.2,
  "avg_time_ms": 2986.4,
  "cache_hit_rate": 15.5,
  "active_connections": 3
}
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file (see `.env.example`):

```env
# Server Configuration
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
MAX_CONNECTIONS=100
MAX_AUDIO_SIZE_MB=10
WS_TIMEOUT=300
LOG_LEVEL=INFO
KYUTAI_SKIP_MODEL_LOAD=0

# Model Configuration
MODEL_ID=kyutai/stt-1b-en_fr-trfs
DEVICE=cpu
DTYPE=float16

# Performance Settings
ENABLE_COMPILE=true
CACHE_SIZE=1000

# Audio Processing
TARGET_SAMPLE_RATE=24000
ENABLE_NORMALIZATION=true
ENABLE_VAD=false
```

### Configuration Options

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Server host address |
| `SERVER_PORT` | `8000` | Server port |
| `MAX_CONNECTIONS` | `100` | Maximum simultaneous WebSocket clients |
| `MAX_AUDIO_SIZE_MB` | `10` | Maximum allowed upload size |
| `WS_TIMEOUT` | `300` | WebSocket idle timeout (seconds) |
| `KYUTAI_SKIP_MODEL_LOAD` | `0` | Skip model load on startup (set `1` for dry-run) |
| `MODEL_ID` | `kyutai/stt-1b-en_fr-trfs` | HuggingFace model ID |
| `DEVICE` | `cpu` | Device for inference (`cpu`, `cuda`) |
| `DTYPE` | `float16` | Model dtype (`float32`, `float16`) |
| `ENABLE_COMPILE` | `true` | Enable torch.compile optimization |
| `CACHE_SIZE` | `1000` | Result cache size |
| `TARGET_SAMPLE_RATE` | `24000` | Audio resampling rate (Hz) |
| `ENABLE_VAD` | `false` | Enable Voice Activity Detection |

---

## 💡 Examples

### Python Client

```python
import asyncio
import json
import base64
import websockets

async def transcribe_audio(audio_file_path):
    uri = "ws://localhost:8000/ws"
    
    async with websockets.connect(uri) as websocket:
        # Wait for ready message
        ready_msg = await websocket.recv()
        print("Connected:", json.loads(ready_msg))
        
        # Read audio file
        with open(audio_file_path, 'rb') as f:
            audio_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Send audio
        await websocket.send(json.dumps({
            'type': 'audio',
            'data': audio_data
        }))
        
        # Receive transcription
        response = await websocket.recv()
        result = json.loads(response)
        print("Transcription:", result['text'])
        print("Processing time:", result['processing_time_ms'], "ms")

# Run
asyncio.run(transcribe_audio('speech.wav'))
```

### JavaScript Client

```javascript
class STTClient {
    constructor(url = 'ws://localhost:8000/ws') {
        this.ws = new WebSocket(url);
        this.setupHandlers();
    }
    
    setupHandlers() {
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            
            if (data.type === 'ready') {
                console.log('Server ready');
            } else if (data.type === 'transcription') {
                this.onTranscription(data.text, data.processing_time_ms);
            } else if (data.type === 'error') {
                console.error('Error:', data.message);
            }
        };
    }
    
    async sendAudio(audioBlob) {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64 = reader.result.split(',')[1];
            this.ws.send(JSON.stringify({
                type: 'audio',
                data: base64
            }));
        };
        reader.readAsDataURL(audioBlob);
    }
    
    onTranscription(text, processingTime) {
        console.log('Transcription:', text);
        console.log('Processing time:', processingTime, 'ms');
    }
}

// Usage
const client = new STTClient();
```

### CLI Tool

```bash
# Transcribe a file
python scripts/transcribe_cli.py --input audio.wav --output transcription.txt

# Transcribe with custom model
python scripts/transcribe_cli.py --input audio.wav --model kyutai/stt-2.6b-en-trfs

# Batch transcription
python scripts/transcribe_cli.py --input-dir ./audio_files/ --output-dir ./transcriptions/
```

---

## 🏗️ Project Structure

```
stt_kyutai/
├── src/                        # Source code
│   ├── __init__.py
│   ├── audio_processor.py      # Audio processing pipeline
│   ├── config.py               # Configuration management
│   ├── encoding.py             # Feature extraction & encoding
│   ├── model_handler.py        # Model management
│   ├── model_loader.py         # Model loading utilities
│   └── transcription_engine.py # Core transcription engine
├── tests/                      # Test files
│   ├── test_audio_diagnostic.py
│   ├── test_soundfile_fix.py
│   └── test_websocket.py
├── scripts/                    # Utility scripts
│   └── transcribe_cli.py       # Command-line interface
├── examples/                   # Example code
│   └── test_ui.html            # Web UI example
├── docs/                       # Documentation
├── main.py                     # Main server application
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment config
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

---

## 🔧 Development

### Running Tests
```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_audio_diagnostic.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Style
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

---

## 📊 Performance

### Benchmarks

| Audio Duration | Processing Time | RTF | Model |
|---------------|-----------------|-----|-------|
| 5s | 1.5s | 0.30x | stt-1b-en_fr (CPU) |
| 10s | 3.2s | 0.32x | stt-1b-en_fr (CPU) |
| 30s | 9.8s | 0.33x | stt-1b-en_fr (CPU) |
| 5s | 0.8s | 0.16x | stt-1b-en_fr (GPU) |

*RTF (Real-Time Factor): Processing time / Audio duration. Lower is better. <1.0 means faster than real-time.*

### Optimization Tips
1. **Use GPU** - Set `DEVICE=cuda` for 2-3x speedup
2. **Enable torch.compile** - Already enabled by default
3. **Adjust cache size** - Increase `CACHE_SIZE` for repeated audio
4. **Use float16** - Already enabled for CPU inference
5. **Disable VAD** - If not needed, keeps `ENABLE_VAD=false`

---

## 🐛 Troubleshooting

### Common Issues

#### Empty Transcriptions
**Problem**: Getting `'...'` or empty results  
**Solution**: 
- Ensure audio contains actual speech (English or French)
- Check audio is not silent or too quiet
- Verify sample rate is 24000 Hz or will be resampled
- Speak clearly for at least 1-2 seconds

#### Model Loading Fails
**Problem**: Model fails to load  
**Solution**:
- Check internet connection (models download from HuggingFace)
- Verify `transformers >= 4.53.0` is installed
- Try clearing HuggingFace cache: `rm -rf ~/.cache/huggingface/`
- Check disk space (models are ~2-5GB)

#### WebSocket Connection Issues
**Problem**: WebSocket fails to connect  
**Solution**:
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure port 8000 is not in use
- Try different browser (Chrome/Firefox recommended)

#### Slow Performance
**Problem**: Transcription takes too long  
**Solution**:
- Enable GPU if available: `DEVICE=cuda`
- Reduce audio length (<30s recommended)
- Check CPU usage (close other applications)
- Verify torch.compile is enabled

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## 📄 License

This project is licensed under the **CC BY 4.0** License.

The Kyutai STT models are released under CC BY 4.0 by [Kyutai Labs](https://kyutai.org/).

---

## 🙏 Acknowledgments

- **[Kyutai Labs](https://kyutai.org/)** - For developing and releasing the STT models
- **[HuggingFace](https://huggingface.co/)** - For model hosting and transformers library
- **[FastAPI](https://fastapi.tiangolo.com/)** - For the excellent web framework
- **LibriSpeech** - For training data and benchmarks

---

## 📮 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/stt_kyutai/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/stt_kyutai/discussions)

---

## 🗺️ Roadmap

- [ ] Add support for more languages
- [ ] GPU optimization
- [ ] Docker containerization
- [ ] Real-time streaming improvements
- [ ] Model quantization (4-bit/8-bit)
- [ ] Speech diarization (speaker identification)
- [ ] Punctuation and capitalization improvements
- [ ] Custom vocabulary support
- [ ] Batch processing optimizations

---

<div align="center">

Made with ❤️ 
</div>
