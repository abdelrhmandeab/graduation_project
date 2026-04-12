from typing import Dict

from core.config import (
    STT_BACKEND,
    STT_EGYPTIAN_DIALECT_ONLY,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_LANGUAGE,
    WHISPER_VAD_FILTER,
)
from core.logger import logger
from stt.dual_transcriber import dual_transcribe
from stt.stt_engine import get_model as _engine_get_model
from stt.stt_engine import transcribe as _engine_transcribe
from utils.language_detector import detect_language

_RUNTIME_DEFAULT_SETTINGS = {
    "beam_size": max(1, int(WHISPER_BEAM_SIZE)),
    "best_of": max(1, int(WHISPER_BEST_OF)),
    "vad_filter": bool(WHISPER_VAD_FILTER),
    "condition_on_previous_text": bool(WHISPER_CONDITION_ON_PREVIOUS_TEXT),
    "language_hint": str(WHISPER_LANGUAGE or "auto").strip().lower(),
    # Compatibility keys kept for runtime profile/status interfaces.
    "quality_retry_threshold": 0.0,
    "quality_retry_beam_size": max(1, int(WHISPER_BEAM_SIZE)),
    "egyptalk_fallback_threshold": 0.0,
    "egyptalk_fallback_low_quality_score": 0.0,
    "egyptalk_fallback_min_text_chars": 5,
    "no_speech_threshold": 0.0,
    "log_prob_threshold": 0.0,
    "egyptalk_chunk_seconds": 0.0,
    "egyptalk_stride_seconds": 0.0,
}

_runtime_stt_settings = dict(_RUNTIME_DEFAULT_SETTINGS)
_runtime_stt_backend = str(STT_BACKEND or "faster_whisper").strip().lower()
_last_transcription_meta: Dict[str, object] = {
    "text": "",
    "language": "unknown",
    "backend": "faster_whisper",
    "language_confidence": 0.0,
    "method": "auto",
}


def _normalize_stt_backend(value) -> str:
    raw = str(value or "faster_whisper").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "fw": "faster_whisper",
        "faster": "faster_whisper",
        "whisper": "faster_whisper",
        "nemo": "egyptalk_transformers",
        "nemo_egyptalk": "egyptalk_transformers",
        "egyptalk": "egyptalk_transformers",
        "egypt_talk": "egyptalk_transformers",
        "egyptian": "egyptalk_transformers",
        "masri": "egyptalk_transformers",
    }
    normalized = aliases.get(raw, raw)
    if normalized in {"faster_whisper", "egyptalk_transformers"}:
        return normalized
    return "faster_whisper"


def _normalize_language_hint(value) -> str:
    raw = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "arabic": "ar",
        "english": "en",
        "none": "auto",
        "": "auto",
    }
    normalized = aliases.get(raw, raw)
    if normalized in {"ar", "en", "auto"}:
        return normalized
    return "auto"


def _coerce_language(language: str, text: str, fallback: str = "unknown") -> str:
    candidate = str(language or "").strip().lower()
    if candidate in {"ar", "en", "mixed", "unknown"}:
        return candidate

    detected = detect_language(text)
    if detected in {"ar", "en", "mixed"}:
        return detected

    fb = str(fallback or "unknown").strip().lower()
    if fb in {"ar", "en", "mixed", "unknown"}:
        return fb
    return "unknown"


def _language_confidence(language: str, text: str) -> float:
    cleaned = " ".join(str(text or "").split()).strip()
    if not cleaned:
        return 0.0

    label = str(language or "").strip().lower()
    if label == "unknown":
        return 0.0
    if label == "mixed":
        return 0.60
    if label in {"ar", "en"}:
        return 0.95
    return 0.0


def _is_weak_result(text: str) -> bool:
    cleaned = " ".join(str(text or "").split()).strip()
    return (not cleaned) or len(cleaned) < 5


def _update_last_transcription_meta(*, text: str, language: str, backend: str, method: str, confidence: float):
    _last_transcription_meta["text"] = str(text or "")
    _last_transcription_meta["language"] = _coerce_language(language, text)
    _last_transcription_meta["backend"] = _normalize_stt_backend(backend)
    _last_transcription_meta["method"] = str(method or "auto").strip().lower() or "auto"
    _last_transcription_meta["language_confidence"] = max(0.0, min(1.0, float(confidence or 0.0)))


def get_runtime_stt_backend() -> str:
    return _normalize_stt_backend(_runtime_stt_backend)


def set_runtime_stt_backend(backend: str) -> str:
    global _runtime_stt_backend
    _runtime_stt_backend = _normalize_stt_backend(backend)
    return _runtime_stt_backend


def get_runtime_stt_backend_info():
    return {
        "backend": get_runtime_stt_backend(),
        "egyptalk_enabled": True,
        "egyptalk_model": "dual_faster_whisper",
    }


def get_runtime_stt_settings():
    settings = dict(_runtime_stt_settings)
    settings["language_hint"] = _normalize_language_hint(settings.get("language_hint"))
    return settings


def set_runtime_stt_settings(**kwargs):
    for key, value in kwargs.items():
        if value is None:
            continue

        if key in {"beam_size", "best_of", "quality_retry_beam_size", "egyptalk_fallback_min_text_chars"}:
            _runtime_stt_settings[key] = max(1, int(value))
            continue

        if key in {
            "quality_retry_threshold",
            "egyptalk_fallback_threshold",
            "egyptalk_fallback_low_quality_score",
            "no_speech_threshold",
            "log_prob_threshold",
            "egyptalk_chunk_seconds",
            "egyptalk_stride_seconds",
        }:
            _runtime_stt_settings[key] = float(value)
            continue

        if key in {"vad_filter", "condition_on_previous_text"}:
            _runtime_stt_settings[key] = bool(value)
            continue

        if key == "language_hint":
            _runtime_stt_settings[key] = _normalize_language_hint(value)
            continue

        _runtime_stt_settings[key] = value

    return get_runtime_stt_settings()


def get_last_transcription_meta():
    return dict(_last_transcription_meta)


def _get_whisper_model():
    return _engine_get_model()


def _collect_text(segments, on_partial=None) -> str:
    pieces = []
    for segment in segments:
        text = (getattr(segment, "text", "") or "").strip()
        if not text:
            continue
        pieces.append(text)
        if on_partial:
            try:
                on_partial(" ".join(pieces))
            except Exception:
                pass
    return " ".join(pieces).strip()


def _transcribe_once_with_meta(
    model,
    audio_file: str,
    *,
    language,
    beam_size: int,
    best_of: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
    no_speech_threshold: float,
    log_prob_threshold: float,
    on_partial=None,
):
    _ = best_of
    _ = condition_on_previous_text
    _ = no_speech_threshold
    _ = log_prob_threshold

    segments, _info = model.transcribe(
        audio_file,
        language=language,
        beam_size=max(1, int(beam_size)),
        temperature=0.0,
        vad_filter=bool(vad_filter),
    )
    text = _collect_text(segments, on_partial=on_partial)
    detected_language = detect_language(text)
    confidence = _language_confidence(detected_language, text)
    return text, detected_language, confidence


def normalize_arabic_post_transcript(text: str) -> str:
    # Minimal post-processing by design.
    cleaned = " ".join(str(text or "").split()).strip()
    if STT_EGYPTIAN_DIALECT_ONLY:
        return cleaned
    return cleaned


def _transcribe_faster_whisper_with_meta(audio_file: str, on_partial=None, language_hint=None):
    model = _get_whisper_model()
    runtime = get_runtime_stt_settings()
    _ = runtime
    _ = language_hint
    whisper_language = None

    text, detected_language, confidence = _transcribe_once_with_meta(
        model,
        audio_file,
        language=whisper_language,
        beam_size=int(runtime.get("beam_size", WHISPER_BEAM_SIZE)),
        best_of=int(runtime.get("best_of", WHISPER_BEST_OF)),
        vad_filter=bool(runtime.get("vad_filter", WHISPER_VAD_FILTER)),
        condition_on_previous_text=bool(runtime.get("condition_on_previous_text", WHISPER_CONDITION_ON_PREVIOUS_TEXT)),
        no_speech_threshold=float(runtime.get("no_speech_threshold", 0.0)),
        log_prob_threshold=float(runtime.get("log_prob_threshold", 0.0)),
        on_partial=on_partial,
    )

    text = normalize_arabic_post_transcript(text)
    detected_language = _coerce_language(detected_language, text, fallback="unknown")
    return text, detected_language, confidence


def _transcribe_egyptalk_transformers_with_meta(audio_file: str, on_partial=None, language_hint=None):
    _ = language_hint
    text, detected_language = dual_transcribe(audio_file)
    text = normalize_arabic_post_transcript(text)

    if on_partial and text:
        try:
            on_partial(text)
        except Exception:
            pass

    confidence = _language_confidence(detected_language, text)
    return text, detected_language, confidence


def _should_try_egyptalk_fallback(
    detected_language: str,
    language_confidence: float,
    *,
    text: str,
    language_hint=None,
) -> bool:
    _ = language_confidence
    _ = language_hint
    language = _coerce_language(detected_language, text, fallback="unknown")
    return _is_weak_result(text) and language in {"ar", "mixed", "unknown"}


def transcribe_backend_direct_with_meta(audio_file: str, *, backend: str, on_partial=None, language_hint=None):
    requested_backend = _normalize_stt_backend(backend)

    if requested_backend == "egyptalk_transformers":
        try:
            text, language, confidence = _transcribe_egyptalk_transformers_with_meta(
                audio_file,
                on_partial=on_partial,
                language_hint=language_hint,
            )
            method = "dual"
            effective_backend = "egyptalk_transformers"
        except Exception as exc:
            logger.warning(
                "Dual transcription path failed (%s). Falling back to faster-whisper auto.",
                exc,
            )
            text, language, confidence = _transcribe_faster_whisper_with_meta(
                audio_file,
                on_partial=on_partial,
                language_hint=language_hint,
            )
            method = "auto"
            effective_backend = "faster_whisper"
    else:
        auto_result = _engine_transcribe(audio_file)
        text = normalize_arabic_post_transcript(auto_result.get("text", ""))
        language = _coerce_language(auto_result.get("language", "unknown"), text)
        method = str(auto_result.get("method", "auto") or "auto").strip().lower()
        confidence = _language_confidence(language, text)
        effective_backend = "faster_whisper"

        if on_partial and text:
            try:
                on_partial(text)
            except Exception:
                pass

    logger.info("STT method=%s detected_language=%s final_text=%s", method, language, text)
    return {
        "text": str(text or ""),
        "language": _coerce_language(language, text),
        "language_confidence": max(0.0, min(1.0, float(confidence or 0.0))),
        "backend": effective_backend,
        "method": method,
    }


def transcribe_streaming_with_meta(audio_file: str, on_partial=None, language_hint=None):
    try:
        active_backend = get_runtime_stt_backend()
        result = transcribe_backend_direct_with_meta(
            audio_file,
            backend=active_backend,
            on_partial=on_partial,
            language_hint=language_hint,
        )
        _update_last_transcription_meta(
            text=str(result.get("text", "")),
            language=str(result.get("language", "unknown")),
            backend=str(result.get("backend", active_backend)),
            method=str(result.get("method", "auto")),
            confidence=float(result.get("language_confidence", 0.0) or 0.0),
        )
        return result
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        fallback = {
            "text": "",
            "language": "unknown",
            "language_confidence": 0.0,
            "backend": get_runtime_stt_backend(),
            "method": "auto",
        }
        _update_last_transcription_meta(
            text="",
            language="unknown",
            backend=fallback["backend"],
            method="auto",
            confidence=0.0,
        )
        return fallback


def transcribe_streaming(audio_file: str, on_partial=None, language_hint=None) -> str:
    result = transcribe_streaming_with_meta(
        audio_file,
        on_partial=on_partial,
        language_hint=language_hint,
    )
    return str(result.get("text", ""))


def transcribe(audio_file: str, language_hint=None) -> str:
    return transcribe_streaming(audio_file, language_hint=language_hint)
