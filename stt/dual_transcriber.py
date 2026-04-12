from utils.language_detector import count_valid_chars, detect_language
from stt.stt_engine import (
    DEFAULT_BEAM_SIZE,
    DEFAULT_TEMPERATURE,
    DEFAULT_VAD_FILTER,
    get_model,
)


def _collect_text(segments) -> str:
    pieces = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            pieces.append(text)
    return " ".join(pieces).strip()


def _transcribe_with_language(audio_path: str, language: str) -> str:
    model = get_model()
    segments, _info = model.transcribe(
        audio_path,
        language=language,
        beam_size=DEFAULT_BEAM_SIZE,
        temperature=DEFAULT_TEMPERATURE,
        vad_filter=DEFAULT_VAD_FILTER,
    )
    return _collect_text(segments)


def _quality_key(text: str) -> tuple[int, int]:
    cleaned = " ".join(str(text or "").split()).strip()
    return (count_valid_chars(cleaned), len(cleaned))


def dual_transcribe(audio_path: str) -> tuple[str, str]:
    arabic_text = _transcribe_with_language(audio_path, "ar")
    english_text = _transcribe_with_language(audio_path, "en")

    ar_quality = _quality_key(arabic_text)
    en_quality = _quality_key(english_text)

    # Compare by valid characters first, then full text length.
    if ar_quality[0] > en_quality[0] or (ar_quality[0] == en_quality[0] and ar_quality[1] >= en_quality[1]):
        chosen_text = arabic_text
        chosen_language = "ar"
    else:
        chosen_text = english_text
        chosen_language = "en"

    cleaned = " ".join(str(chosen_text or "").split()).strip()
    detected = detect_language(cleaned)
    if detected == "unknown":
        detected = chosen_language

    return cleaned, detected
