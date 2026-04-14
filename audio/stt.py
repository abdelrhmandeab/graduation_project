from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from core import metrics
from core.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_BASE_URL,
    STT_BACKEND,
    STT_ELEVENLABS_ARABIC_LANGUAGE,
    STT_ELEVENLABS_ENABLED,
    STT_ELEVENLABS_STT_MODEL,
    STT_ELEVENLABS_TIMEOUT_SECONDS,
    STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS,
    STT_LANGUAGE_DETECT_MODEL,
    STT_MIXED_TREAT_AS_ARABIC,
    WHISPER_MODEL,
)
from core.logger import logger
from utils.language_detector import detect_language

_LOCAL_BACKEND = "faster_whisper"
_HYBRID_BACKEND = "hybrid_elevenlabs"
_ELEVENLABS_METHOD = "elevenlabs_stt"

_BACKEND_ALIASES = {
    _HYBRID_BACKEND: _HYBRID_BACKEND,
    "hybrid": _HYBRID_BACKEND,
    "elevenlabs": _HYBRID_BACKEND,
    "elevenlabs_stt": _HYBRID_BACKEND,
    "elevenlabs_hybrid": _HYBRID_BACKEND,
    _LOCAL_BACKEND: _LOCAL_BACKEND,
    "whisper": _LOCAL_BACKEND,
    "local": _LOCAL_BACKEND,
    "faster-whisper": _LOCAL_BACKEND,
    "faster whisper": _LOCAL_BACKEND,
}


def _normalize_backend_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    return _BACKEND_ALIASES.get(raw, _HYBRID_BACKEND)


_RUNTIME_STT_BACKEND = _normalize_backend_name(STT_BACKEND)
_RUNTIME_STT_SETTINGS: Dict[str, Any] = {"language_hint": "auto"}
_LAST_TRANSCRIPTION_META: Dict[str, Any] = {}

_LOCAL_MODEL_LOCK = threading.Lock()
_LOCAL_MODEL: Any = None

_LANG_DETECT_MODEL_LOCK = threading.Lock()
_LANG_DETECT_MODEL: Any = None
_LANG_DETECT_MODEL_NAME = ""


def _runtime_language_hint() -> str:
    hint = str(_RUNTIME_STT_SETTINGS.get("language_hint", "auto") or "auto").strip().lower()
    if hint in {"ar", "arabic", "ar-eg", "ar_eg"}:
        return "ar"
    if hint in {"en", "english", "en-us", "en_us"}:
        return "en"
    return "auto"


def _normalize_detected_language(code: str) -> str:
    value = str(code or "").strip().lower()
    if not value:
        return ""
    if value.startswith("ar"):
        return "ar"
    if value.startswith("en"):
        return "en"
    return value


def _normalize_elevenlabs_language_code(value: Optional[str]) -> str:
    raw = str(value or "").strip().lower().replace("_", "-")
    if not raw:
        return ""

    aliases = {
        "ar": "ara",
        "ara": "ara",
        "arabic": "ara",
        "ar-eg": "ara",
        "ar-sa": "ara",
        "en": "eng",
        "eng": "eng",
        "english": "eng",
        "en-us": "eng",
        "en-gb": "eng",
    }
    if raw in aliases:
        return aliases[raw]

    base = raw.split("-", 1)[0]
    if base in aliases:
        return aliases[base]

    if len(raw) == 3 and raw.isalpha():
        return raw

    return ""


def _classify_language_by_script(text: str) -> str:
    arabic_letters = 0
    latin_letters = 0
    for ch in str(text or ""):
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            arabic_letters += 1
        elif "a" <= ch.lower() <= "z":
            latin_letters += 1

    if arabic_letters and latin_letters:
        if bool(STT_MIXED_TREAT_AS_ARABIC):
            return "ar"
        return "ar" if arabic_letters >= latin_letters else "en"
    if arabic_letters:
        return "ar"
    if latin_letters:
        return "en"
    return ""


def _language_codes_from_hint(language_hint: Optional[str]) -> List[str]:
    hint = _normalize_detected_language(str(language_hint or "auto"))
    if hint == "ar":
        return ["ar-EG", "en-US"]
    if hint == "en":
        return ["en-US", "ar-EG"]
    return ["ar-EG", "en-US"]


def _record_stt_metric(backend: str, latency_ms: float, text: str) -> None:
    stage_name = f"stt_{backend}"
    try:
        metrics.record_stage(stage_name, float(latency_ms), success=bool(text.strip()))
    except Exception:
        pass


def get_runtime_stt_backend() -> str:
    return _RUNTIME_STT_BACKEND


def set_runtime_stt_backend(backend: str) -> str:
    global _RUNTIME_STT_BACKEND
    _RUNTIME_STT_BACKEND = _normalize_backend_name(backend)
    logger.info("Runtime STT backend set to '%s'", _RUNTIME_STT_BACKEND)
    return _RUNTIME_STT_BACKEND


def get_runtime_stt_settings() -> Dict[str, Any]:
    return dict(_RUNTIME_STT_SETTINGS)


def set_runtime_stt_settings(**kwargs: Any) -> Dict[str, Any]:
    if not kwargs:
        return get_runtime_stt_settings()
    for key, value in kwargs.items():
        if key == "language_hint":
            _RUNTIME_STT_SETTINGS[key] = _runtime_language_hint() if value is None else str(value).strip().lower()
        else:
            _RUNTIME_STT_SETTINGS[key] = value
    return get_runtime_stt_settings()


def get_runtime_stt_backend_info() -> Dict[str, Any]:
    elevenlabs_key = bool(str(ELEVENLABS_API_KEY or "").strip())
    return {
        "backend": _RUNTIME_STT_BACKEND,
        "whisper_model": str(WHISPER_MODEL),
        "language_detector_model": str(STT_LANGUAGE_DETECT_MODEL),
        "elevenlabs_enabled": bool(STT_ELEVENLABS_ENABLED),
        "elevenlabs_key_configured": elevenlabs_key,
        "elevenlabs_stt_model": str(STT_ELEVENLABS_STT_MODEL),
    }


def get_last_transcription_meta() -> Dict[str, Any]:
    return dict(_LAST_TRANSCRIPTION_META)


def _read_audio_bytes(audio_file: str) -> bytes:
    path = Path(audio_file)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    return path.read_bytes()


def _get_local_whisper_model() -> Any:
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL

    with _LOCAL_MODEL_LOCK:
        if _LOCAL_MODEL is not None:
            return _LOCAL_MODEL
        from faster_whisper import WhisperModel

        _LOCAL_MODEL = WhisperModel(str(WHISPER_MODEL), device="cpu", compute_type="int8")
        logger.info("Loaded local faster-whisper model '%s'", WHISPER_MODEL)
        return _LOCAL_MODEL


def _get_language_detector_whisper_model() -> Any:
    global _LANG_DETECT_MODEL
    global _LANG_DETECT_MODEL_NAME

    detector_name = str(STT_LANGUAGE_DETECT_MODEL or "tiny").strip() or "tiny"
    if detector_name == str(WHISPER_MODEL):
        return _get_local_whisper_model()

    if _LANG_DETECT_MODEL is not None and _LANG_DETECT_MODEL_NAME == detector_name:
        return _LANG_DETECT_MODEL

    with _LANG_DETECT_MODEL_LOCK:
        if _LANG_DETECT_MODEL is not None and _LANG_DETECT_MODEL_NAME == detector_name:
            return _LANG_DETECT_MODEL

        from faster_whisper import WhisperModel

        _LANG_DETECT_MODEL = WhisperModel(detector_name, device="cpu", compute_type="int8")
        _LANG_DETECT_MODEL_NAME = detector_name
        logger.info("Loaded STT language detector whisper model '%s'", detector_name)
        return _LANG_DETECT_MODEL


def preload_runtime_models() -> Dict[str, Any]:
    backend = get_runtime_stt_backend()
    local_loaded = bool(_LOCAL_MODEL)
    detector_loaded = bool(_LANG_DETECT_MODEL)

    if backend in {_LOCAL_BACKEND, _HYBRID_BACKEND}:
        _get_local_whisper_model()
        local_loaded = True
    if backend == _HYBRID_BACKEND:
        _get_language_detector_whisper_model()
        detector_loaded = True

    return {
        "backend": backend,
        "local_model_loaded": local_loaded,
        "language_detector_model_loaded": detector_loaded,
    }


def _safe_partial_emit(on_partial: Optional[Callable[[str], None]], text: str) -> None:
    if on_partial is None or not text:
        return
    try:
        on_partial(text)
    except Exception:
        pass


def _transcribe_with_faster_whisper(
    audio_file: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    model = _get_local_whisper_model()

    hint = _normalize_detected_language(language_hint or _runtime_language_hint())
    whisper_language = None
    # Keep Arabic pinning for better dialect stability, but do not hard-pin
    # English: allowing auto language detection prevents Arabic speech from
    # being forced into incorrect English transcripts.
    if hint == "ar":
        whisper_language = hint

    segments, info = model.transcribe(
        str(audio_file),
        beam_size=5,
        vad_filter=True,
        language=whisper_language,
        condition_on_previous_text=False,
    )

    parts: List[str] = []
    for segment in segments:
        piece = str(getattr(segment, "text", "") or "").strip()
        if not piece:
            continue
        parts.append(piece)
        _safe_partial_emit(on_partial, " ".join(parts).strip())

    text = " ".join(parts).strip()
    language = _normalize_detected_language(str(getattr(info, "language", "") or ""))
    if not language and text:
        language = _normalize_detected_language(detect_language(text))

    confidence = getattr(info, "language_probability", None)
    confidence_value = float(confidence) if isinstance(confidence, (float, int)) else 0.0

    return {
        "text": text,
        "confidence": confidence_value,
        "language": language,
        "backend": _LOCAL_BACKEND,
        "method": _LOCAL_BACKEND,
        "fallback_used": False,
    }


def _detect_audio_language_with_whisper(audio_file: str, language_hint: Optional[str] = None) -> str:
    hint = _normalize_detected_language(language_hint or _runtime_language_hint())
    # Arabic hint remains a hard preference. English hint is soft and should
    # not bypass detection, otherwise bilingual users can get Arabic turns
    # misrouted to local English STT.
    if hint == "ar":
        return hint

    model = _get_language_detector_whisper_model()
    preview_parts: List[str] = []
    detected = ""

    try:
        segments, info = model.transcribe(
            str(audio_file),
            beam_size=1,
            vad_filter=True,
            language=None,
            condition_on_previous_text=False,
        )
        detected = _normalize_detected_language(str(getattr(info, "language", "") or ""))
        for segment in segments:
            piece = str(getattr(segment, "text", "") or "").strip()
            if not piece:
                continue
            preview_parts.append(piece)
            if len(" ".join(preview_parts)) >= 120:
                break
    except Exception as exc:
        logger.warning("Whisper language detection failed: %s", exc)

    if detected in {"ar", "en"}:
        return detected

    preview_text = " ".join(preview_parts).strip()
    script_guess = _classify_language_by_script(preview_text)
    if script_guess in {"ar", "en"}:
        return script_guess

    if preview_text:
        detected_text_language = _normalize_detected_language(detect_language(preview_text))
        if detected_text_language in {"ar", "en"}:
            return detected_text_language

    if hint in {"ar", "en"}:
        return hint

    return "en"


def _transcribe_with_elevenlabs(
    audio_file: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if not bool(STT_ELEVENLABS_ENABLED):
        raise RuntimeError("ElevenLabs STT is disabled")

    api_key = str(ELEVENLABS_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    language = _normalize_detected_language(language_hint or "")
    language_code = _normalize_elevenlabs_language_code(STT_ELEVENLABS_ARABIC_LANGUAGE) or "ara"
    if language == "en":
        language_code = "eng"

    endpoint = f"{str(ELEVENLABS_BASE_URL or 'https://api.elevenlabs.io').rstrip('/')}/v1/speech-to-text"
    audio_bytes = _read_audio_bytes(audio_file)
    data = {
        "model_id": str(STT_ELEVENLABS_STT_MODEL or "scribe_v1"),
    }
    if language_code:
        data["language_code"] = language_code

    response = httpx.post(
        endpoint,
        headers={"xi-api-key": api_key},
        data=data,
        files={"file": (Path(audio_file).name or "audio.wav", audio_bytes, "audio/wav")},
        timeout=float(STT_ELEVENLABS_TIMEOUT_SECONDS),
    )

    if response.status_code >= 400:
        error_preview = (response.text or "").strip().replace("\n", " ")
        if len(error_preview) > 220:
            error_preview = error_preview[:217] + "..."
        raise RuntimeError(f"ElevenLabs STT HTTP {response.status_code}: {error_preview}")

    payload = response.json() if response.content else {}
    text = str(
        payload.get("text")
        or payload.get("transcript")
        or payload.get("result")
        or ""
    ).strip()

    confidence_raw = payload.get("confidence")
    if not isinstance(confidence_raw, (float, int)):
        confidence_raw = payload.get("average_confidence")
    confidence_value = float(confidence_raw) if isinstance(confidence_raw, (float, int)) else 0.0

    detected_language = _normalize_detected_language(
        str(payload.get("language_code") or payload.get("language") or language or "")
    )
    if not detected_language and text:
        detected_language = _normalize_detected_language(detect_language(text))

    _safe_partial_emit(on_partial, text)
    return {
        "text": text,
        "confidence": confidence_value,
        "language": detected_language,
        "backend": _HYBRID_BACKEND,
        "method": _ELEVENLABS_METHOD,
        "fallback_used": False,
    }


def _is_weak_transcript(text: str) -> bool:
    normalized = " ".join(str(text or "").split()).strip()
    return len(normalized) < int(STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS)


def _transcribe_with_hybrid_elevenlabs(
    audio_file: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    detected_language = _detect_audio_language_with_whisper(audio_file, language_hint=language_hint)
    errors: List[str] = []

    if detected_language == "ar" and bool(STT_ELEVENLABS_ENABLED):
        try:
            primary = _transcribe_with_elevenlabs(
                audio_file,
                language_hint="ar",
                on_partial=on_partial,
            )
            if not _is_weak_transcript(str(primary.get("text", ""))):
                primary["language"] = _normalize_detected_language(primary.get("language") or "ar") or "ar"
                primary["backend"] = _HYBRID_BACKEND
                primary["method"] = _ELEVENLABS_METHOD
                return primary
            errors.append("elevenlabs:weak_transcript")
            logger.warning("ElevenLabs STT returned a weak transcript; falling back to local whisper")
        except Exception as exc:
            errors.append(f"elevenlabs:{exc}")
            logger.warning("ElevenLabs STT failed: %s", exc)

    local = _transcribe_with_faster_whisper(
        audio_file,
        language_hint=detected_language,
        on_partial=on_partial,
    )
    local["backend"] = _HYBRID_BACKEND
    local["method"] = _LOCAL_BACKEND
    local["language"] = _normalize_detected_language(local.get("language") or detected_language)
    if errors:
        local["fallback_used"] = True
        local["errors"] = errors
    return local


def transcribe_backend_direct_with_meta(
    audio_file: str,
    backend: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    normalized_backend = _normalize_backend_name(backend)
    start = time.perf_counter()

    if normalized_backend == _LOCAL_BACKEND:
        result = _transcribe_with_faster_whisper(audio_file, language_hint=language_hint, on_partial=on_partial)
    else:
        result = _transcribe_with_hybrid_elevenlabs(audio_file, language_hint=language_hint, on_partial=on_partial)

    latency_ms = (time.perf_counter() - start) * 1000.0
    result["latency_ms"] = latency_ms
    _record_stt_metric(normalized_backend, latency_ms, str(result.get("text", "")))
    return result


def transcribe_streaming_with_meta(
    audio_file: str,
    on_partial: Optional[Callable[[str], None]] = None,
    language_hint: Optional[str] = None,
) -> Dict[str, Any]:
    global _LAST_TRANSCRIPTION_META

    preferred_backend = get_runtime_stt_backend()
    attempted: List[str] = []
    errors: List[str] = []

    for backend in [preferred_backend, _LOCAL_BACKEND]:
        if backend in attempted:
            continue
        attempted.append(backend)
        try:
            result = transcribe_backend_direct_with_meta(
                audio_file,
                backend,
                language_hint=language_hint,
                on_partial=on_partial,
            )
            if backend != preferred_backend:
                result["fallback_used"] = True
                result["fallback_from"] = preferred_backend
                logger.warning(
                    "STT backend '%s' failed; fallback backend '%s' succeeded",
                    preferred_backend,
                    backend,
                )
            if errors:
                result["errors"] = list(errors)
            _LAST_TRANSCRIPTION_META = dict(result)
            return result
        except Exception as exc:
            message = f"{backend}: {exc}"
            errors.append(message)
            logger.warning("STT backend '%s' failed: %s", backend, exc)

    failed = {
        "text": "",
        "confidence": 0.0,
        "language": _normalize_detected_language(language_hint or ""),
        "backend": preferred_backend,
        "method": preferred_backend,
        "fallback_used": False,
        "errors": errors,
        "latency_ms": 0.0,
    }
    _LAST_TRANSCRIPTION_META = dict(failed)
    return failed


def transcribe_streaming(
    audio_file: str,
    on_partial: Optional[Callable[[str], None]] = None,
    language_hint: Optional[str] = None,
) -> str:
    return str(
        transcribe_streaming_with_meta(
            audio_file,
            on_partial=on_partial,
            language_hint=language_hint,
        ).get("text", "")
    )


def transcribe(audio_file: str, language_hint: Optional[str] = None) -> str:
    return transcribe_streaming(audio_file, on_partial=None, language_hint=language_hint)


def normalize_arabic_post_transcript(text: str) -> str:
    return " ".join(str(text or "").split()).strip()
