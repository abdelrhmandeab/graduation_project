from __future__ import annotations

import threading
import time
import tempfile
import wave
import math
import sys
from array import array
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import httpx

from core.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_BASE_URL,
    STT_BACKEND,
    STT_CLOUD_RACE_LANGUAGES,
    STT_ELEVENLABS_CONNECT_TIMEOUT_SECONDS,
    STT_ELEVENLABS_COOLDOWN_SECONDS,
    STT_ELEVENLABS_ENABLED,
    STT_ELEVENLABS_HTTP2,
    STT_ELEVENLABS_READ_TIMEOUT_SECONDS,
    STT_ELEVENLABS_STT_MODEL,
    STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS,
    STT_AR_INITIAL_PROMPT,
    STT_BEAM_SIZE_LONG,
    STT_BEAM_SIZE_SHORT,
    STT_BEAM_SIZE_SHORT_THRESHOLD_SECONDS,
    STT_EN_INITIAL_PROMPT,
    STT_FORBID_OTHER_LANGUAGES,
    STT_LANGUAGE_HINT,
    STT_LANGUAGE_LOCK,
    STT_MAX_AUDIO_SECONDS,
    STT_MIN_AUDIO_RMS,
    STT_NO_SPEECH_THRESHOLD,
    STT_PARTIAL_WHISPER_MODEL,
    STT_RETRY_OPPOSITE_LANGUAGE,
    STT_VALIDATION_DOMINANT_SCRIPT_MIN,
    WHISPER_COMPUTE_TYPE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)
from core import hardware_detect
from core.logger import get_logger, logger
from core.metrics import get_thread_stage_timing, stage_timer
from core.shutdown import is_shutdown_requested
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

_ELEVENLABS_COOLDOWN_UNTIL = 0.0
_STT_LOG = get_logger("stt")
_ARABIC_PROMPT = "\u0628\u0627\u0644\u0644\u0647\u062c\u0629 \u0627\u0644\u0645\u0635\u0631\u064a\u0629:"
_CLOUD_HTTP_CLIENT: Optional[httpx.Client] = None
_CLOUD_HTTP_CLIENT_LOCK = threading.Lock()


def _elevenlabs_on_cooldown() -> bool:
    return time.time() < _ELEVENLABS_COOLDOWN_UNTIL


def _set_elevenlabs_cooldown(reason: str, *, seconds: Optional[float] = None) -> None:
    global _ELEVENLABS_COOLDOWN_UNTIL
    duration = max(60.0, float(seconds if seconds is not None else STT_ELEVENLABS_COOLDOWN_SECONDS))
    _ELEVENLABS_COOLDOWN_UNTIL = max(_ELEVENLABS_COOLDOWN_UNTIL, time.time() + duration)
    logger.warning("ElevenLabs STT cooldown enabled for %.0fs: %s", duration, reason)


def get_cloud_http_client() -> httpx.Client:
    global _CLOUD_HTTP_CLIENT
    if _CLOUD_HTTP_CLIENT is not None:
        return _CLOUD_HTTP_CLIENT
    with _CLOUD_HTTP_CLIENT_LOCK:
        if _CLOUD_HTTP_CLIENT is not None:
            return _CLOUD_HTTP_CLIENT
        timeout = httpx.Timeout(
            connect=float(STT_ELEVENLABS_CONNECT_TIMEOUT_SECONDS),
            read=float(STT_ELEVENLABS_READ_TIMEOUT_SECONDS),
            write=float(STT_ELEVENLABS_READ_TIMEOUT_SECONDS),
            pool=float(STT_ELEVENLABS_READ_TIMEOUT_SECONDS),
        )
        try:
            _CLOUD_HTTP_CLIENT = httpx.Client(http2=bool(STT_ELEVENLABS_HTTP2), timeout=timeout)
        except ImportError as exc:
            if not bool(STT_ELEVENLABS_HTTP2):
                raise
            logger.debug("HTTP/2 unavailable for ElevenLabs STT client; using HTTP/1.1: %s", exc)
            _CLOUD_HTTP_CLIENT = httpx.Client(http2=False, timeout=timeout)
        return _CLOUD_HTTP_CLIENT


def close_cloud_http_client() -> None:
    global _CLOUD_HTTP_CLIENT
    with _CLOUD_HTTP_CLIENT_LOCK:
        client = _CLOUD_HTTP_CLIENT
        _CLOUD_HTTP_CLIENT = None
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def _normalize_backend_name(name: str) -> str:
    raw = str(name or "").strip().lower()
    return _BACKEND_ALIASES.get(raw, _HYBRID_BACKEND)


_RUNTIME_STT_BACKEND = _normalize_backend_name(STT_BACKEND)
_RUNTIME_STT_SETTINGS: Dict[str, Any] = {"language_hint": STT_LANGUAGE_HINT}
_LAST_TRANSCRIPTION_META: Dict[str, Any] = {}
_RECENT_TRANSCRIPTION_META = deque(maxlen=10)

_LOCAL_MODEL_LOCK = threading.Lock()
_LOCAL_MODEL: Any = None
_LOCAL_MODEL_RUNTIME: Dict[str, Any] = {}

_PARTIAL_MODEL_LOCK = threading.Lock()
_PARTIAL_MODEL: Any = None
_PARTIAL_MODEL_NAME = ""
_PARTIAL_MODEL_RUNTIME: Dict[str, Any] = {}


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
        return "ar" if arabic_letters >= latin_letters else "en"
    if arabic_letters:
        return "ar"
    if latin_letters:
        return "en"
    return ""


def _language_counts(text: str) -> Dict[str, int]:
    arabic = 0
    latin = 0
    other_alpha = 0
    for ch in str(text or ""):
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            arabic += 1
        elif "a" <= ch.lower() <= "z":
            latin += 1
        elif ch.isalpha():
            other_alpha += 1
    return {"arabic": arabic, "latin": latin, "other_alpha": other_alpha}


def _is_allowed_transcript_char(ch: str, *, allow_arabic: bool) -> bool:
    if not ch:
        return True
    code = ord(ch)
    if code < 128:
        return True
    if allow_arabic and (
        0x0600 <= code <= 0x06FF
        or 0x0750 <= code <= 0x077F
        or 0x08A0 <= code <= 0x08FF
        or 0xFB50 <= code <= 0xFDFF
        or 0xFE70 <= code <= 0xFEFF
    ):
        return True
    return not ch.isalpha()


def _make_one_second_probe(audio_file: str) -> str:
    path = Path(audio_file)
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames_to_read = min(source.getnframes(), int(source.getframerate()))
        audio = source.readframes(max(1, frames_to_read))

    probe = tempfile.NamedTemporaryFile(prefix="jarvis_stt_probe_", suffix=".wav", delete=False)
    probe_path = probe.name
    probe.close()
    with wave.open(probe_path, "wb") as target:
        target.setparams(params)
        target.writeframes(audio)
    return probe_path


def _pick_locked_language(
    audio_file: str,
    streaming_text: str = "",
    language_hint: str = "auto",
    probe_model: Any = None,
    allow_ambiguous: bool = False,
) -> str:
    picked_started = time.perf_counter()
    with stage_timer("stt_lang_pick"):
        source = "default"
        lang = _classify_language_by_script(streaming_text)
        if lang not in {"ar", "en"}:
            hint = _normalize_detected_language(language_hint or _runtime_language_hint())
            if hint in {"ar", "en"}:
                lang = hint
                source = "hint"
            else:
                preview = ""
                probe_path = ""
                try:
                    probe_path = _make_one_second_probe(audio_file)
                    model = probe_model if probe_model is not None else _get_partial_whisper_model()
                    segments, _info = model.transcribe(
                        str(probe_path),
                        beam_size=1,
                        vad_filter=True,
                        language=None,
                        task="transcribe",
                        condition_on_previous_text=False,
                    )
                    preview_parts: List[str] = []
                    for segment in segments:
                        piece = str(getattr(segment, "text", "") or "").strip()
                        if piece:
                            preview_parts.append(piece)
                        if len(" ".join(preview_parts)) >= 80:
                            break
                    preview = " ".join(preview_parts).strip()
                except Exception as exc:
                    _STT_LOG.warning("lang_pick probe failed: %s", exc)
                finally:
                    if probe_path:
                        try:
                            Path(probe_path).unlink(missing_ok=True)
                        except Exception:
                            pass
                lang = _classify_language_by_script(preview)
                source = "probe" if lang in {"ar", "en"} else "default"
        else:
            source = "partial"

        if lang not in {"ar", "en"} and allow_ambiguous:
            lang = "ambiguous"
        if lang not in {"ar", "en", "ambiguous"}:
            lang = "en"
        _LAST_TRANSCRIPTION_META.update(
            {
                "lang_pick_source": source,
                "lang_pick_lang": lang,
                "lang_pick_seconds": max(0.0, time.perf_counter() - picked_started),
            }
        )
        _STT_LOG.info("stt_lang_pick source=%s lang=%s", source, lang)
        return lang


def _is_weak_transcript_for_language(text: str, locked_lang: str) -> bool:
    normalized = " ".join(str(text or "").split()).strip()
    if len(normalized) < int(STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS):
        return True
    counts = _language_counts(normalized)
    if locked_lang == "ar" and counts["arabic"] < 3:
        return True
    if locked_lang == "en" and counts["latin"] < 3:
        return True
    return any(
        not _is_allowed_transcript_char(ch, allow_arabic=(locked_lang == "ar"))
        for ch in normalized
    )


def _validate_transcript_language(text: str, locked_lang: str) -> bool:
    with stage_timer("stt_validate", lang=locked_lang):
        normalized = " ".join(str(text or "").split()).strip()
        if _is_weak_transcript_for_language(normalized, locked_lang):
            return False
        counts = _language_counts(normalized)
        total_alpha = counts["arabic"] + counts["latin"] + counts["other_alpha"]
        if total_alpha <= 0:
            return False
        if bool(STT_FORBID_OTHER_LANGUAGES) and counts["other_alpha"] / float(total_alpha) > 0.05:
            return False
        dominant = counts["arabic"] if locked_lang == "ar" else counts["latin"]
        return dominant / float(total_alpha) >= float(STT_VALIDATION_DOMINANT_SCRIPT_MIN)


def _finalize_stt_result(result: Dict[str, Any]) -> Dict[str, Any]:
    global _LAST_TRANSCRIPTION_META
    finalized = dict(result or {})
    text = str(finalized.get("text", "") or "").strip()
    errors = [str(item) for item in list(finalized.get("errors") or [])]
    validation_ok = bool(finalized.get("validation_ok", bool(text)))
    if "stt:silence" in errors:
        status = "silence"
    elif errors and not text:
        status = "error" if "stt:invalid_language" not in errors else "rejected"
    elif text and validation_ok:
        status = "ok"
    elif text:
        status = "rejected"
    else:
        status = "error"

    call_seconds = max(
        get_thread_stage_timing("stt_cloud_call"),
        get_thread_stage_timing("stt_local_call"),
    )
    pick_seconds = float(finalized.get("lang_pick_seconds") or get_thread_stage_timing("stt_lang_pick"))
    validate_seconds = get_thread_stage_timing("stt_validate")
    audio_duration = float(finalized.get("audio_duration") or finalized.get("duration_seconds") or 0.0)
    audio_rms = float(finalized.get("audio_rms") or 0.0)
    retry_used = bool(finalized.get("retry_used") or get_thread_stage_timing("stt_retry") > 0.0)
    lang = str(finalized.get("language") or finalized.get("lang_pick_lang") or "").strip().lower()

    finalized.update(
        {
            "result": status,
            "lang_pick_source": finalized.get("lang_pick_source") or "",
            "lang_pick_lang": finalized.get("lang_pick_lang") or lang,
            "validation_ok": validation_ok,
            "retry_used": retry_used,
            "audio_duration": audio_duration,
            "audio_rms": audio_rms,
            "call_seconds": call_seconds,
            "pick_seconds": pick_seconds,
            "validate_seconds": validate_seconds,
        }
    )
    _LAST_TRANSCRIPTION_META = finalized
    _RECENT_TRANSCRIPTION_META.append(dict(finalized))
    _STT_LOG.info(
        "result=%s backend=%s lang=%s dur=%.2fs call=%.2fs pick=%.2fs validate=%.2fs chars=%d conf=%.2f",
        status,
        finalized.get("method") or finalized.get("backend") or "unknown",
        lang or "unknown",
        audio_duration,
        call_seconds,
        pick_seconds,
        validate_seconds,
        len(text),
        float(finalized.get("confidence") or 0.0),
    )
    return finalized


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
    runtime = dict(_LOCAL_MODEL_RUNTIME)
    return {
        "backend": _RUNTIME_STT_BACKEND,
        "whisper_model": str(runtime.get("model") or WHISPER_MODEL),
        "whisper_device": str(runtime.get("device") or WHISPER_DEVICE),
        "whisper_compute_type": str(runtime.get("compute_type") or WHISPER_COMPUTE_TYPE),
        "elevenlabs_enabled": bool(STT_ELEVENLABS_ENABLED),
        "elevenlabs_key_configured": elevenlabs_key,
        "elevenlabs_stt_model": str(STT_ELEVENLABS_STT_MODEL),
    }


def get_last_transcription_meta() -> Dict[str, Any]:
    return dict(_LAST_TRANSCRIPTION_META)


def get_recent_transcription_meta() -> List[Dict[str, Any]]:
    return [dict(row) for row in list(_RECENT_TRANSCRIPTION_META)]


def _audio_duration_seconds(audio_file: str) -> float:
    try:
        with wave.open(str(audio_file), "rb") as handle:
            rate = float(handle.getframerate() or 0)
            if rate <= 0:
                return 0.0
            return float(handle.getnframes()) / rate
    except Exception:
        return 0.0


def _audio_rms(audio_file: str) -> float:
    try:
        with wave.open(str(audio_file), "rb") as handle:
            sample_width = int(handle.getsampwidth() or 2)
            frame_count = int(handle.getnframes() or 0)
            if sample_width <= 0 or frame_count <= 0:
                return 0.0
            frames = handle.readframes(frame_count)
        if not frames:
            return 0.0
        if sample_width == 2:
            samples = array("h")
            samples.frombytes(frames)
            if sys.byteorder != "little":
                samples.byteswap()
            if not samples:
                return 0.0
            square_mean = sum(float(sample) * float(sample) for sample in samples) / float(len(samples))
            return math.sqrt(square_mean) / 32768.0

        signed = sample_width > 1
        peak = float(1 << (8 * sample_width - 1))
        total = 0.0
        count = 0
        for offset in range(0, len(frames) - sample_width + 1, sample_width):
            sample = int.from_bytes(frames[offset : offset + sample_width], "little", signed=signed)
            if sample_width == 1:
                sample -= 128
            total += float(sample) * float(sample)
            count += 1
        if count <= 0:
            return 0.0
        return math.sqrt(total / float(count)) / peak
    except Exception:
        return 0.0


def _silence_result(*, locked_lang: str, rms: float, duration_seconds: float) -> Dict[str, Any]:
    return {
        "text": "",
        "confidence": 0.0,
        "language": locked_lang if locked_lang in {"ar", "en"} else "",
        "backend": _LOCAL_BACKEND,
        "method": _LOCAL_BACKEND,
        "fallback_used": False,
        "errors": ["stt:silence"],
        "audio_rms": rms,
        "audio_duration": duration_seconds,
    }


def _shutdown_result(*, backend: str = _LOCAL_BACKEND, method: str = _LOCAL_BACKEND) -> Dict[str, Any]:
    return {
        "text": "",
        "confidence": 0.0,
        "language": "",
        "backend": backend,
        "method": method,
        "fallback_used": False,
        "errors": ["stt:shutdown"],
        "validation_ok": False,
    }


def _manual_whisper_runtime() -> Dict[str, Any]:
    model = str(WHISPER_MODEL or "auto").strip() or "auto"
    device = str(WHISPER_DEVICE or "auto").strip().lower() or "auto"
    compute_type = str(WHISPER_COMPUTE_TYPE or "auto").strip().lower() or "auto"
    if model == "auto":
        runtime = dict(hardware_detect.recommend_whisper_runtime())
    else:
        runtime = {
            "model": model,
            "device": "cpu" if device == "auto" else device,
            "compute_type": "int8" if compute_type == "auto" else compute_type,
        }
    if device != "auto":
        runtime["device"] = device
    if compute_type != "auto":
        runtime["compute_type"] = compute_type
    return runtime


def _partial_whisper_runtime() -> Dict[str, Any]:
    main_runtime = _manual_whisper_runtime()
    device = str(main_runtime.get("device") or "cpu")
    compute_type = str(main_runtime.get("compute_type") or "int8")
    configured = str(STT_PARTIAL_WHISPER_MODEL or "auto").strip() or "auto"
    if configured.lower() in {"auto", ""}:
        model = "base" if device == "cuda" else "tiny"
    else:
        model = configured
    return {"model": model, "device": device, "compute_type": compute_type}


def _get_local_whisper_model() -> Any:
    global _LOCAL_MODEL_RUNTIME
    global _LOCAL_MODEL
    if _LOCAL_MODEL is not None:
        return _LOCAL_MODEL

    with _LOCAL_MODEL_LOCK:
        if _LOCAL_MODEL is not None:
            return _LOCAL_MODEL
        from faster_whisper import WhisperModel

        runtime = _manual_whisper_runtime()
        try:
            _LOCAL_MODEL = WhisperModel(
                str(runtime["model"]),
                device=str(runtime["device"]),
                compute_type=str(runtime["compute_type"]),
            )
        except Exception as exc:
            fallback = {"model": "base", "device": "cpu", "compute_type": "int8"}
            logger.warning(
                "Local faster-whisper runtime failed (%s/%s/%s): %s; falling back to base/cpu/int8",
                runtime.get("model"),
                runtime.get("device"),
                runtime.get("compute_type"),
                exc,
            )
            _LOCAL_MODEL = WhisperModel(
                fallback["model"],
                device=fallback["device"],
                compute_type=fallback["compute_type"],
            )
            runtime = fallback
        _LOCAL_MODEL_RUNTIME = dict(runtime)
        _STT_LOG.info(
            "model=%s device=%s compute=%s",
            runtime.get("model"),
            runtime.get("device"),
            runtime.get("compute_type"),
        )
        return _LOCAL_MODEL


def _get_partial_whisper_model() -> Any:
    global _PARTIAL_MODEL
    global _PARTIAL_MODEL_NAME
    global _PARTIAL_MODEL_RUNTIME
    runtime = _partial_whisper_runtime()
    model_name = str(runtime["model"])
    runtime_key = f"{runtime['model']}|{runtime['device']}|{runtime['compute_type']}"
    if _PARTIAL_MODEL is not None and _PARTIAL_MODEL_NAME == runtime_key:
        return _PARTIAL_MODEL

    with _PARTIAL_MODEL_LOCK:
        if _PARTIAL_MODEL is not None and _PARTIAL_MODEL_NAME == runtime_key:
            return _PARTIAL_MODEL
        from faster_whisper import WhisperModel

        try:
            _PARTIAL_MODEL = WhisperModel(
                model_name,
                device=str(runtime["device"]),
                compute_type=str(runtime["compute_type"]),
            )
        except Exception as exc:
            fallback = {"model": "tiny", "device": "cpu", "compute_type": "int8"}
            logger.debug(
                "Partial faster-whisper runtime failed (%s/%s/%s): %s; falling back to tiny/cpu/int8",
                runtime.get("model"),
                runtime.get("device"),
                runtime.get("compute_type"),
                exc,
            )
            _PARTIAL_MODEL = WhisperModel(
                fallback["model"],
                device=fallback["device"],
                compute_type=fallback["compute_type"],
            )
            runtime = fallback
            runtime_key = f"{runtime['model']}|{runtime['device']}|{runtime['compute_type']}"
        _PARTIAL_MODEL_NAME = runtime_key
        _PARTIAL_MODEL_RUNTIME = dict(runtime)
        logger.debug(
            "Loaded partial faster-whisper model '%s' (device=%s compute=%s)",
            runtime.get("model"),
            runtime.get("device"),
            runtime.get("compute_type"),
        )
        return _PARTIAL_MODEL


def preload_critical_model() -> Dict[str, Any]:
    backend = get_runtime_stt_backend()
    local_loaded = bool(_LOCAL_MODEL)

    if backend in {_LOCAL_BACKEND, _HYBRID_BACKEND}:
        _get_local_whisper_model()
        local_loaded = True

    return {
        "backend": backend,
        "local_model_loaded": local_loaded,
    }


def preload_optional_models() -> Dict[str, Any]:
    backend = get_runtime_stt_backend()
    partial_loaded = bool(_PARTIAL_MODEL)

    try:
        _get_partial_whisper_model()
        partial_loaded = True
    except Exception as exc:
        logger.debug("Partial whisper model preload failed: %s", exc)

    return {
        "backend": backend,
        "partial_model_loaded": partial_loaded,
    }


def preload_runtime_models() -> Dict[str, Any]:
    """Compatibility wrapper that preloads both critical and optional STT models."""
    snapshot = preload_critical_model()
    snapshot.update(preload_optional_models())
    return snapshot


def _safe_partial_emit(on_partial: Optional[Callable[[str], None]], text: str) -> None:
    if on_partial is None or not text:
        return
    try:
        on_partial(text)
    except Exception:
        pass


def _transcribe_with_faster_whisper_model(
    model: Any,
    audio_file: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
    *,
    locked_lang: Optional[str] = None,
    whisper_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if is_shutdown_requested():
        return _shutdown_result()

    hint = _normalize_detected_language(language_hint or _runtime_language_hint())
    whisper_language = _normalize_detected_language(locked_lang or hint)
    lock_requested = bool(STT_LANGUAGE_LOCK) and (locked_lang is not None or hint in {"ar", "en"})
    if lock_requested and whisper_language not in {"ar", "en"}:
        whisper_language = "en"

    duration_seconds = _audio_duration_seconds(audio_file)
    rms = _audio_rms(audio_file)
    if rms < float(STT_MIN_AUDIO_RMS):
        return _silence_result(
            locked_lang=whisper_language or "",
            rms=rms,
            duration_seconds=duration_seconds,
        )

    extra: Dict[str, Any] = dict(whisper_kwargs or {})
    if lock_requested:
        extra.pop("language", None)
        extra.pop("task", None)
        extra.pop("initial_prompt", None)
        initial_prompt = (
            str(STT_AR_INITIAL_PROMPT or _ARABIC_PROMPT)
            if whisper_language == "ar"
            else str(STT_EN_INITIAL_PROMPT or "")
        )
    else:
        initial_prompt = extra.pop("initial_prompt", None)
        whisper_language = extra.pop("language", whisper_language)

    parts: List[str] = []
    no_speech_probs: List[float] = []
    beam_size = (
        int(STT_BEAM_SIZE_SHORT)
        if duration_seconds < float(STT_BEAM_SIZE_SHORT_THRESHOLD_SECONDS)
        else int(STT_BEAM_SIZE_LONG)
    )
    with stage_timer("stt_local_call", lang=whisper_language or "auto"):
        segments, info = model.transcribe(
            str(audio_file),
            beam_size=extra.pop("beam_size", beam_size),
            vad_filter=extra.pop("vad_filter", True),
            vad_parameters=extra.pop(
                "vad_parameters",
                {"threshold": 0.5, "min_silence_duration_ms": 200},
            ),
            language=whisper_language,
            task=extra.pop("task", "transcribe"),
            initial_prompt=initial_prompt,
            condition_on_previous_text=False,
            no_speech_threshold=extra.pop("no_speech_threshold", float(STT_NO_SPEECH_THRESHOLD)),
            temperature=extra.pop("temperature", 0.0),
            **extra,
        )
        for segment in segments:
            no_speech_prob = getattr(segment, "no_speech_prob", None)
            if isinstance(no_speech_prob, (float, int)):
                no_speech_probs.append(float(no_speech_prob))
            piece = str(getattr(segment, "text", "") or "").strip()
            if not piece:
                continue
            parts.append(piece)
            _safe_partial_emit(on_partial, " ".join(parts).strip())

    if no_speech_probs and all(prob > 0.85 for prob in no_speech_probs):
        return _silence_result(
            locked_lang=whisper_language or "",
            rms=rms,
            duration_seconds=duration_seconds,
        )

    text = " ".join(parts).strip()
    language = _normalize_detected_language(str(getattr(info, "language", "") or ""))
    if not language and text:
        language = _normalize_detected_language(detect_language(text))

    confidence = getattr(info, "language_probability", None)
    confidence_value = float(confidence) if isinstance(confidence, (float, int)) else 0.0

    return {
        "text": text,
        "confidence": confidence_value,
        "language": whisper_language or language,
        "backend": _LOCAL_BACKEND,
        "method": _LOCAL_BACKEND,
        "fallback_used": False,
        "audio_duration": duration_seconds,
        "audio_rms": rms,
    }


def _transcribe_with_faster_whisper(
    audio_file: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
    *,
    locked_lang: Optional[str] = None,
    streaming_text: str = "",
    whisper_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if locked_lang is None and bool(STT_LANGUAGE_LOCK):
        locked_lang = _pick_locked_language(
            audio_file,
            streaming_text=streaming_text,
            language_hint=language_hint or _runtime_language_hint(),
            probe_model=_get_partial_whisper_model(),
        )
    model = _get_local_whisper_model()
    return _transcribe_with_faster_whisper_model(
        model,
        audio_file,
        language_hint=language_hint,
        on_partial=on_partial,
        locked_lang=locked_lang,
        whisper_kwargs=whisper_kwargs,
    )


def transcribe_partial_with_meta(
    audio_file: str,
    language_hint: Optional[str] = None,
    *,
    whisper_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    model = _get_partial_whisper_model()
    return _transcribe_with_faster_whisper_model(
        model,
        audio_file,
        language_hint=language_hint,
        on_partial=None,
        locked_lang=(
            _normalize_detected_language(language_hint or "")
            if _normalize_detected_language(language_hint or "") in {"ar", "en"}
            else None
        ),
        whisper_kwargs=whisper_kwargs,
    )


def _transcribe_with_elevenlabs(
    audio_file: str,
    locked_lang: str,
    on_partial: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    if is_shutdown_requested():
        raise RuntimeError("STT shutdown requested")

    if not bool(STT_ELEVENLABS_ENABLED):
        raise RuntimeError("ElevenLabs STT is disabled")

    if _elevenlabs_on_cooldown():
        raise RuntimeError("ElevenLabs STT cooldown active")

    api_key = str(ELEVENLABS_API_KEY or "").strip()
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not configured")

    path = Path(audio_file)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")

    duration_seconds = _audio_duration_seconds(str(path))
    if duration_seconds > float(STT_MAX_AUDIO_SECONDS):
        raise RuntimeError(
            f"Audio duration {duration_seconds:.2f}s exceeds cloud STT cap {float(STT_MAX_AUDIO_SECONDS):.2f}s"
        )

    language = "ar" if _normalize_detected_language(locked_lang) == "ar" else "en"
    language_code = "ara" if language == "ar" else "eng"

    endpoint = f"{str(ELEVENLABS_BASE_URL or 'https://api.elevenlabs.io').rstrip('/')}/v1/speech-to-text"
    data = {
        "model_id": str(STT_ELEVENLABS_STT_MODEL or "scribe_v1"),
        "tag_audio_events": "false",
        "diarize": "false",
    }
    if language_code:
        data["language_code"] = language_code

    with stage_timer("stt_cloud_call", lang=language):
        with path.open("rb") as audio_handle:
            response = get_cloud_http_client().post(
                endpoint,
                headers={"xi-api-key": api_key},
                data=data,
                files={"file": (path.name or "audio.wav", audio_handle, "audio/wav")},
            )

    if response.status_code >= 400:
        error_preview = (response.text or "").strip().replace("\n", " ")
        if len(error_preview) > 220:
            error_preview = error_preview[:217] + "..."
        if response.status_code in {401, 429} or "quota_exceeded" in error_preview:
            _set_elevenlabs_cooldown(f"http_{response.status_code}: {error_preview}")
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
        "language": language if detected_language not in {"ar", "en"} else detected_language,
        "backend": _HYBRID_BACKEND,
        "method": _ELEVENLABS_METHOD,
        "fallback_used": False,
        "audio_duration": duration_seconds,
        "audio_rms": _audio_rms(str(path)),
    }


def _opposite_language(lang: str) -> str:
    return "en" if lang == "ar" else "ar"


def _invalid_language_result(
    *,
    backend: str,
    method: str,
    locked_lang: str,
    errors: List[str],
    audio_duration: float = 0.0,
    audio_rms: float = 0.0,
) -> Dict[str, Any]:
    result = {
        "text": "",
        "confidence": 0.0,
        "language": locked_lang,
        "backend": backend,
        "method": method,
        "fallback_used": bool(errors),
        "errors": list(errors) + ["stt:invalid_language"],
        "validation_ok": False,
        "audio_duration": audio_duration,
        "audio_rms": audio_rms,
    }
    for key in ("lang_pick_source", "lang_pick_lang", "lang_pick_seconds"):
        if key in _LAST_TRANSCRIPTION_META:
            result[key] = _LAST_TRANSCRIPTION_META[key]
    return result


def _validated_backend_result(
    result: Dict[str, Any],
    *,
    locked_lang: str,
    backend: str,
    method: str,
) -> Optional[Dict[str, Any]]:
    text = str(result.get("text", "") or "").strip()
    validation_ok = _validate_transcript_language(text, locked_lang)
    result["validation_ok"] = validation_ok
    result["language"] = locked_lang
    result["backend"] = backend
    result["method"] = method
    for key in ("lang_pick_source", "lang_pick_lang", "lang_pick_seconds"):
        if key in _LAST_TRANSCRIPTION_META:
            result[key] = _LAST_TRANSCRIPTION_META[key]
    if validation_ok:
        return result
    return None


def _race_elevenlabs_languages(
    audio_file: str,
    *,
    on_partial: Optional[Callable[[str], None]] = None,
) -> tuple[Optional[Dict[str, Any]], List[str]]:
    errors: List[str] = []
    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="stt-cloud-race") as executor:
        future_to_lang = {
            executor.submit(_transcribe_with_elevenlabs, audio_file, locked_lang=lang, on_partial=on_partial): lang
            for lang in ("ar", "en")
        }
        for future in as_completed(future_to_lang):
            lang = future_to_lang[future]
            try:
                result = future.result()
            except Exception as exc:
                errors.append(f"elevenlabs:{lang}:{exc}")
                continue
            validated = _validated_backend_result(
                result,
                locked_lang=lang,
                backend=_HYBRID_BACKEND,
                method=_ELEVENLABS_METHOD,
            )
            if validated is not None:
                validated["cloud_race_used"] = True
                for pending in future_to_lang:
                    if pending is not future:
                        pending.cancel()
                return validated, errors
            errors.append(f"elevenlabs:{lang}:invalid_language")
    return None, errors


def _transcribe_with_hybrid_elevenlabs(
    audio_file: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
    *,
    streaming_text: str = "",
) -> Dict[str, Any]:
    if is_shutdown_requested():
        return _shutdown_result(backend=_HYBRID_BACKEND, method=_HYBRID_BACKEND)

    duration_seconds = _audio_duration_seconds(audio_file)
    rms = _audio_rms(audio_file)
    if rms < float(STT_MIN_AUDIO_RMS):
        result = _silence_result(locked_lang="", rms=rms, duration_seconds=duration_seconds)
        result["backend"] = _HYBRID_BACKEND
        result["method"] = _LOCAL_BACKEND
        return result

    locked_lang = _pick_locked_language(
        audio_file,
        streaming_text=streaming_text,
        language_hint=language_hint or _runtime_language_hint(),
        probe_model=_get_partial_whisper_model(),
        allow_ambiguous=bool(STT_CLOUD_RACE_LANGUAGES),
    )
    if is_shutdown_requested():
        return _shutdown_result(backend=_HYBRID_BACKEND, method=_HYBRID_BACKEND)

    errors: List[str] = []
    cloud_allowed = duration_seconds <= float(STT_MAX_AUDIO_SECONDS)
    cloud_lang = locked_lang
    local_lang = locked_lang if locked_lang in {"ar", "en"} else "en"

    if bool(STT_ELEVENLABS_ENABLED) and cloud_allowed:
        try:
            if cloud_lang == "ambiguous" and bool(STT_CLOUD_RACE_LANGUAGES):
                raced, race_errors = _race_elevenlabs_languages(audio_file, on_partial=on_partial)
                errors.extend(race_errors)
                if raced is not None:
                    return raced
            elif cloud_lang in {"ar", "en"}:
                primary = _transcribe_with_elevenlabs(
                    audio_file,
                    locked_lang=cloud_lang,
                    on_partial=on_partial,
                )
                validated = _validated_backend_result(
                    primary,
                    locked_lang=cloud_lang,
                    backend=_HYBRID_BACKEND,
                    method=_ELEVENLABS_METHOD,
                )
                if validated is not None:
                    return validated
                errors.append("elevenlabs:invalid_language")
                primary_text = str(primary.get("text", "") or "").strip()
                if primary_text:
                    logger.warning("ElevenLabs STT returned invalid-language text; trying recovery")
                else:
                    logger.debug("ElevenLabs STT returned empty transcript; skipping opposite-language cloud retry")
                if bool(STT_RETRY_OPPOSITE_LANGUAGE) and primary_text and not is_shutdown_requested():
                    retry_lang = _opposite_language(cloud_lang)
                    with stage_timer("stt_retry", lang=retry_lang):
                        retry = _transcribe_with_elevenlabs(
                            audio_file,
                            locked_lang=retry_lang,
                            on_partial=on_partial,
                        )
                    retry_validated = _validated_backend_result(
                        retry,
                        locked_lang=retry_lang,
                        backend=_HYBRID_BACKEND,
                        method=_ELEVENLABS_METHOD,
                    )
                    if retry_validated is not None:
                        retry_validated["retry_used"] = True
                        return retry_validated
                    errors.append("elevenlabs:retry_invalid_language")
        except Exception as exc:
            errors.append(f"elevenlabs:{exc}")
            if "quota_exceeded" in str(exc) or "http 401" in str(exc).lower():
                _set_elevenlabs_cooldown(f"exception:{exc}")
            logger.warning("ElevenLabs STT failed: %s", exc)
    elif bool(STT_ELEVENLABS_ENABLED) and not cloud_allowed:
        errors.append(f"elevenlabs:audio_too_long:{duration_seconds:.2f}s")
        logger.info(
            "Skipping ElevenLabs STT for %.2fs audio over %.2fs cap",
            duration_seconds,
            float(STT_MAX_AUDIO_SECONDS),
        )

    local = _transcribe_with_faster_whisper(
        audio_file,
        language_hint=language_hint,
        on_partial=on_partial,
        locked_lang=local_lang,
    )
    if "stt:silence" in list(local.get("errors") or []):
        local["backend"] = _HYBRID_BACKEND
        local["method"] = _LOCAL_BACKEND
        if errors:
            local["fallback_used"] = True
            local["errors"] = list(errors) + list(local.get("errors") or [])
        return local
    local_text = str(local.get("text", "")).strip()
    validated_local = _validated_backend_result(
        local,
        locked_lang=local_lang,
        backend=_HYBRID_BACKEND,
        method=_LOCAL_BACKEND,
    )
    if validated_local is not None:
        if errors:
            validated_local["fallback_used"] = True
            validated_local["errors"] = errors
        return validated_local

    if local_text:
        logger.warning("Local STT returned invalid-language text (text_len=%d); trying recovery", len(local_text))
    else:
        logger.debug("Local STT returned empty transcript; skipping opposite-language retry")
    errors.append("local:invalid_language")
    if bool(STT_RETRY_OPPOSITE_LANGUAGE) and local_text and not is_shutdown_requested():
        retry_lang = _opposite_language(local_lang)
        with stage_timer("stt_retry", lang=retry_lang):
            retry = _transcribe_with_faster_whisper(
                audio_file,
                language_hint=language_hint,
                on_partial=on_partial,
                locked_lang=retry_lang,
            )
        retry_validated = _validated_backend_result(
            retry,
            locked_lang=retry_lang,
            backend=_HYBRID_BACKEND,
            method=_LOCAL_BACKEND,
        )
        if retry_validated is not None:
            retry_validated["fallback_used"] = True
            retry_validated["retry_used"] = True
            retry_validated["errors"] = errors
            return retry_validated
        errors.append("local:retry_invalid_language")

    return _invalid_language_result(
        backend=_HYBRID_BACKEND,
        method=_LOCAL_BACKEND,
        locked_lang=local_lang,
        errors=errors,
        audio_duration=duration_seconds,
        audio_rms=rms,
    )


def transcribe_backend_direct_with_meta(
    audio_file: str,
    backend: str,
    language_hint: Optional[str] = None,
    on_partial: Optional[Callable[[str], None]] = None,
    *,
    streaming_text: str = "",
    whisper_kwargs: Optional[Dict[str, Any]] = None,
    finalize: bool = True,
) -> Dict[str, Any]:
    if is_shutdown_requested():
        result = _shutdown_result(backend=_normalize_backend_name(backend), method=_normalize_backend_name(backend))
        result["latency_ms"] = 0.0
        return _finalize_stt_result(result) if finalize else result

    normalized_backend = _normalize_backend_name(backend)
    start = time.perf_counter()

    if normalized_backend == _LOCAL_BACKEND:
        locked_lang = None
        if bool(STT_LANGUAGE_LOCK):
            locked_lang = _pick_locked_language(
                audio_file,
                streaming_text=streaming_text,
                language_hint=language_hint or _runtime_language_hint(),
                probe_model=_get_partial_whisper_model(),
            )
        result = _transcribe_with_faster_whisper(
            audio_file,
            language_hint=language_hint,
            on_partial=on_partial,
            locked_lang=locked_lang,
            streaming_text=streaming_text,
            whisper_kwargs=whisper_kwargs,
        )
        if "stt:silence" in list(result.get("errors") or []):
            latency_ms = (time.perf_counter() - start) * 1000.0
            result["latency_ms"] = latency_ms
            return _finalize_stt_result(result) if finalize else result
        if bool(STT_LANGUAGE_LOCK) and locked_lang in {"ar", "en"}:
            validated = _validated_backend_result(
                result,
                locked_lang=locked_lang,
                backend=_LOCAL_BACKEND,
                method=_LOCAL_BACKEND,
            )
            if validated is None and bool(STT_RETRY_OPPOSITE_LANGUAGE):
                result_text = str(result.get("text", "") or "").strip()
                if not result_text or is_shutdown_requested():
                    result = _invalid_language_result(
                        backend=_LOCAL_BACKEND,
                        method=_LOCAL_BACKEND,
                        locked_lang=locked_lang,
                        errors=[],
                        audio_duration=float(result.get("audio_duration") or 0.0),
                        audio_rms=float(result.get("audio_rms") or 0.0),
                    )
                    latency_ms = (time.perf_counter() - start) * 1000.0
                    result["latency_ms"] = latency_ms
                    return _finalize_stt_result(result) if finalize else result
                retry_lang = _opposite_language(locked_lang)
                with stage_timer("stt_retry", lang=retry_lang):
                    retry = _transcribe_with_faster_whisper(
                        audio_file,
                        language_hint=language_hint,
                        on_partial=on_partial,
                        locked_lang=retry_lang,
                        whisper_kwargs=whisper_kwargs,
                    )
                validated = _validated_backend_result(
                    retry,
                    locked_lang=retry_lang,
                    backend=_LOCAL_BACKEND,
                    method=_LOCAL_BACKEND,
                )
                if validated is not None:
                    validated["retry_used"] = True
                    result = validated
            if validated is None:
                result = _invalid_language_result(
                    backend=_LOCAL_BACKEND,
                    method=_LOCAL_BACKEND,
                    locked_lang=locked_lang,
                    errors=[],
                    audio_duration=float(result.get("audio_duration") or 0.0),
                    audio_rms=float(result.get("audio_rms") or 0.0),
                )
    else:
        result = _transcribe_with_hybrid_elevenlabs(
            audio_file,
            language_hint=language_hint,
            on_partial=on_partial,
            streaming_text=streaming_text,
        )

    latency_ms = (time.perf_counter() - start) * 1000.0
    result["latency_ms"] = latency_ms
    return _finalize_stt_result(result) if finalize else result


def transcribe_streaming_with_meta(
    audio_file: str,
    on_partial: Optional[Callable[[str], None]] = None,
    language_hint: Optional[str] = None,
    *,
    streaming_text: str = "",
    whisper_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    preferred_backend = get_runtime_stt_backend()
    attempted: List[str] = []
    errors: List[str] = []

    for backend in [preferred_backend, _LOCAL_BACKEND]:
        if is_shutdown_requested():
            failed = _shutdown_result(backend=preferred_backend, method=preferred_backend)
            failed["errors"] = list(errors) + list(failed.get("errors") or [])
            failed["latency_ms"] = 0.0
            return _finalize_stt_result(failed)
        if backend in attempted:
            continue
        attempted.append(backend)
        try:
            result = transcribe_backend_direct_with_meta(
                audio_file,
                backend,
                language_hint=language_hint,
                on_partial=on_partial,
                streaming_text=streaming_text,
                whisper_kwargs=whisper_kwargs,
                finalize=False,
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
            return _finalize_stt_result(result)
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
    return _finalize_stt_result(failed)


def transcribe_streaming(
    audio_file: str,
    on_partial: Optional[Callable[[str], None]] = None,
    language_hint: Optional[str] = None,
    streaming_text: str = "",
) -> str:
    return str(
        transcribe_streaming_with_meta(
            audio_file,
            on_partial=on_partial,
            language_hint=language_hint,
            streaming_text=streaming_text,
        ).get("text", "")
    )


def transcribe(audio_file: str, language_hint: Optional[str] = None, streaming_text: str = "") -> str:
    return transcribe_streaming(audio_file, on_partial=None, language_hint=language_hint, streaming_text=streaming_text)


import re as _re  # noqa: E402 – module-level import kept at bottom for minimal diff

_ARABIC_DIACRITICS_RE = _re.compile(r"[ً-ٰٟ]")
_ALEF_VARIANTS_RE = _re.compile(r"[أإآٱ]")


def normalize_arabic_post_transcript(text: str) -> str:
    """Strip tashkeel, normalize alef variants, collapse whitespace."""
    t = str(text or "")
    t = _ARABIC_DIACRITICS_RE.sub("", t)   # remove harakat / tashkeel
    t = _ALEF_VARIANTS_RE.sub("ا", t)      # أ إ آ ٱ → ا
    return " ".join(t.split()).strip()
