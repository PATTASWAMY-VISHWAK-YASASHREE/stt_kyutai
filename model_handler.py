# model_handler.py

"""
Handles model loading, compilation, and inference.
"""
import logging
from typing import Optional

import torch
from transformers import AutoModelForCTC, AutoTokenizer

import config

logger = logging.getLogger(__name__)

# Global objects to hold the model and tokenizer
stt_model = None
tokenizer = None

def load_model():
    """
    Loads the Kyutai Mimic-1 model and tokenizer into memory.
    """
    global stt_model, tokenizer
    
    logger.info(f"Loading model: {config.MODEL_NAME}...")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        stt_model = AutoModelForCTC.from_pretrained(config.MODEL_NAME)

        # Set padding token if it's not already set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            stt_model.config.pad_token_id = stt_model.config.eos_token_id
        
        # --- PERFORMANCE BOOST: Compile the model ---
        # This uses PyTorch 2.0's `torch.compile` to optimize the model.
        # It has a one-time cost at startup but makes inference faster.
        # 'reduce-overhead' is a good mode for models that are called frequently
        # with small inputs, which is typical for STT.
        logger.info("Compiling model with torch.compile()... (This may take a moment)")
        stt_model = torch.compile(stt_model, mode="reduce-overhead")
        
        logger.info("✅ Model and tokenizer loaded and compiled successfully!")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        stt_model = None
        tokenizer = None

def transcribe(audio_array: np.ndarray) -> Optional[str]:
    """
    Performs speech-to-text transcription on a NumPy audio array.

    Args:
        audio_array: A NumPy array of audio data.

    Returns:
        The transcribed text, or None if an error occurs.
    """
    if not stt_model or not tokenizer:
        logger.error("Model or tokenizer is not loaded.")
        return None

    try:
        # Convert NumPy array to a PyTorch tensor
        input_tensor = torch.from_numpy(audio_array).unsqueeze(0)

        # Perform inference
        with torch.no_grad():
            logits = stt_model(input_tensor).logits

        # Decode the logits to get the transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        return transcription.strip()

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return None