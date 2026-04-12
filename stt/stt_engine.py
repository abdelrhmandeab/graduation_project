import logging
import threading
from typing import Dict

from faster_whisper import WhisperModel

from utils.language_detector import detect_language

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL = "small"
DEFAULT_DEVICE = "cpu"
DEFAULT_COMPUTE_TYPE = "int8"
DEFAULT_BEAM_SIZE = 4
DEFAULT_TEMPERATURE = 0.0
DEFAULT_VAD_FILTER = True

_model = None
_model_lock = threading.Lock()


def get_model() -> WhisperModel:
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is None:
            LOGGER.info(
                "Loading STT model=%s device=%s compute_type=%s",
                DEFAULT_MODEL,
                DEFAULT_DEVICE,
                DEFAULT_COMPUTE_TYPE,
            )
            _model = WhisperModel(
                DEFAULT_MODEL,
                device=DEFAULT_DEVICE,
                compute_type=DEFAULT_COMPUTE_TYPE,
            )
    return _model


def _collect_text(segments) -> str:
    pieces = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            pieces.append(text)
    return " ".join(pieces).strip()


def _post_process(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _is_weak_result(text: str) -> bool:
    normalized = _post_process(text)
    return (not normalized) or len(normalized) < 5


def _transcribe_auto(audio_path: str) -> str:
    model = get_model()
    segments, _info = model.transcribe(
        audio_path,
        language=None,
        beam_size=DEFAULT_BEAM_SIZE,
        temperature=DEFAULT_TEMPERATURE,
        vad_filter=DEFAULT_VAD_FILTER,
    )
    return _collect_text(segments)


def transcribe(audio_path: str, *, language_hint=None) -> Dict[str, str]:
    _ = language_hint
    text = _post_process(_transcribe_auto(audio_path))
    language = detect_language(text)
    method = "auto"

    if _is_weak_result(text):
        from stt.dual_transcriber import dual_transcribe

        text, language = dual_transcribe(audio_path)
        text = _post_process(text)
        method = "dual"

    if language not in {"ar", "en", "mixed"}:
        language = "unknown"

    LOGGER.info("STT method=%s language=%s text=%s", method, language, text)
    return {
        "text": text,
        "language": language,
        "method": method,
    }


# Example usage:
# from stt.stt_engine import transcribe
# result = transcribe("audio.wav")
# print(result)
