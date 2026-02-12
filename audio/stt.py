from faster_whisper import WhisperModel
from core.logger import logger
from core.config import (
    WHISPER_MODEL,
    WHISPER_DEVICE,
    WHISPER_COMPUTE_TYPE
)

model = WhisperModel(
    WHISPER_MODEL,
    device=WHISPER_DEVICE,
    compute_type=WHISPER_COMPUTE_TYPE
)

def transcribe(audio_file):
    try:
        segments, _ = model.transcribe(audio_file)
        return " ".join(seg.text for seg in segments).strip()
    except Exception as e:
        logger.error(f"STT failed: {e}")
        return ""
