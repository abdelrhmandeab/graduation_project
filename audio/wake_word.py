import pathlib
import re
import time
import unicodedata
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

from core.config import (
    BARGE_IN_INTERRUPT_ON_WAKE,
    SAMPLE_RATE,
    WAKE_WORD,
    WAKE_WORD_AR_CHECK_INTERVAL_SECONDS,
    WAKE_WORD_AR_CHUNK_SECONDS,
    WAKE_WORD_AR_CONFIRM_WINDOW_SECONDS,
    WAKE_WORD_AR_CONSECUTIVE_HITS_REQUIRED,
    WAKE_WORD_AR_ENABLED,
    WAKE_WORD_AR_STT_MODEL,
    WAKE_WORD_AR_TRIGGERS,
    WAKE_WORD_AUDIO_GAIN,
    WAKE_WORD_CHUNK_SIZE,
    WAKE_WORD_DETECTION_COOLDOWN_SECONDS,
    WAKE_WORD_IGNORE_WHILE_SPEAKING,
    WAKE_WORD_INPUT_DEVICE,
    WAKE_WORD_MODE,
    WAKE_WORD_SCORE_DEBUG,
    WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS,
    WAKE_WORD_THRESHOLD,
)
from core.logger import logger

_model = None
_ar_stt_model = None
_ar_stt_model_name = ""
_last_detection_ts = 0.0
_ar_last_hit_ts = 0.0
_ar_consecutive_hits = 0
_OPENWAKEWORD_RELEASE = "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1"
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_NON_WORD_CHARS_RE = re.compile(r"[^a-z0-9\u0600-\u06ff\s]")
_COLLAPSE_WS_RE = re.compile(r"\s+")
_OPTIONAL_WAKE_CONNECTOR_TOKENS = {"يا"}
_ARABIC_NORMALIZATION_TRANSLATE = str.maketrans(
    {
        "\u0622": "\u0627",
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0624": "\u0648",
        "\u0626": "\u064a",
        "\u0649": "\u064a",
        "\u0629": "\u0647",
        "\u0640": "",
    }
)
_runtime_wake_word_settings = {
    "threshold": float(WAKE_WORD_THRESHOLD),
    "audio_gain": float(WAKE_WORD_AUDIO_GAIN),
    "detection_cooldown_seconds": float(WAKE_WORD_DETECTION_COOLDOWN_SECONDS),
}
_runtime_wake_word_phrase_settings = {
    "mode": str(WAKE_WORD_MODE or "both").strip().lower(),
    "arabic_enabled": bool(WAKE_WORD_AR_ENABLED),
    "arabic_triggers": tuple(str(item).strip() for item in WAKE_WORD_AR_TRIGGERS if str(item).strip()),
    "ar_stt_model": str(WAKE_WORD_AR_STT_MODEL or "tiny").strip() or "tiny",
    "ar_chunk_seconds": float(WAKE_WORD_AR_CHUNK_SECONDS),
    "ar_check_interval_seconds": float(WAKE_WORD_AR_CHECK_INTERVAL_SECONDS),
    "ar_consecutive_hits_required": int(WAKE_WORD_AR_CONSECUTIVE_HITS_REQUIRED),
    "ar_confirm_window_seconds": float(WAKE_WORD_AR_CONFIRM_WINDOW_SECONDS),
}
_runtime_wake_word_behavior = {
    "ignore_while_speaking": bool(WAKE_WORD_IGNORE_WHILE_SPEAKING),
    "barge_in_interrupt_on_wake": bool(BARGE_IN_INTERRUPT_ON_WAKE),
}


def _contains_token_sequence(text_tokens, trigger_tokens) -> bool:
    tokens = list(text_tokens or [])
    trigger = list(trigger_tokens or [])
    if not tokens or not trigger or len(trigger) > len(tokens):
        return False
    width = len(trigger)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == trigger:
            return True
    return False


def _normalize_wake_mode(value) -> str:
    mode = str(value or "both").strip().lower()
    aliases = {
        "en": "english",
        "ar": "arabic",
        "dual": "both",
        "bilingual": "both",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"english", "arabic", "both"}:
        return "both"
    return mode


def _normalize_trigger_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower().strip()
    text = text.translate(_ARABIC_NORMALIZATION_TRANSLATE)
    text = _ARABIC_DIACRITICS_RE.sub("", text)
    text = _NON_WORD_CHARS_RE.sub(" ", text)
    return _COLLAPSE_WS_RE.sub(" ", text).strip()


def _sanitize_trigger_phrase(value: str) -> str:
    return _COLLAPSE_WS_RE.sub(" ", str(value or "")).strip()


def _dedupe_trigger_phrases(values) -> tuple:
    kept = []
    seen = set()
    for item in values or []:
        phrase = _sanitize_trigger_phrase(item)
        if not phrase:
            continue
        key = _normalize_trigger_text(phrase)
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(phrase)
    return tuple(kept)


_runtime_wake_word_phrase_settings["mode"] = _normalize_wake_mode(
    _runtime_wake_word_phrase_settings.get("mode")
)
_runtime_wake_word_phrase_settings["arabic_triggers"] = _dedupe_trigger_phrases(
    _runtime_wake_word_phrase_settings.get("arabic_triggers", ())
)


def get_runtime_wake_word_settings():
    return dict(_runtime_wake_word_settings)


def set_runtime_wake_word_settings(*, threshold=None, audio_gain=None, detection_cooldown_seconds=None):
    if threshold is not None:
        _runtime_wake_word_settings["threshold"] = max(0.05, min(0.95, float(threshold)))
    if audio_gain is not None:
        _runtime_wake_word_settings["audio_gain"] = max(0.5, min(3.0, float(audio_gain)))
    if detection_cooldown_seconds is not None:
        _runtime_wake_word_settings["detection_cooldown_seconds"] = max(
            0.2,
            min(3.0, float(detection_cooldown_seconds)),
        )
    return get_runtime_wake_word_settings()


def get_runtime_wake_word_phrase_settings():
    snapshot = dict(_runtime_wake_word_phrase_settings)
    snapshot["arabic_triggers"] = list(_runtime_wake_word_phrase_settings["arabic_triggers"])
    return snapshot


def set_runtime_wake_word_phrase_settings(
    *,
    mode=None,
    arabic_enabled=None,
    arabic_triggers=None,
    ar_stt_model=None,
    ar_chunk_seconds=None,
    ar_check_interval_seconds=None,
    ar_consecutive_hits_required=None,
    ar_confirm_window_seconds=None,
):
    if mode is not None:
        _runtime_wake_word_phrase_settings["mode"] = _normalize_wake_mode(mode)
    if arabic_enabled is not None:
        _runtime_wake_word_phrase_settings["arabic_enabled"] = bool(arabic_enabled)
    if arabic_triggers is not None:
        _runtime_wake_word_phrase_settings["arabic_triggers"] = _dedupe_trigger_phrases(arabic_triggers)
    if ar_stt_model is not None:
        candidate = str(ar_stt_model or "").strip()
        if candidate:
            _runtime_wake_word_phrase_settings["ar_stt_model"] = candidate
    if ar_chunk_seconds is not None:
        _runtime_wake_word_phrase_settings["ar_chunk_seconds"] = max(0.8, float(ar_chunk_seconds))
    if ar_check_interval_seconds is not None:
        _runtime_wake_word_phrase_settings["ar_check_interval_seconds"] = max(
            0.5,
            float(ar_check_interval_seconds),
        )
    if ar_consecutive_hits_required is not None:
        _runtime_wake_word_phrase_settings["ar_consecutive_hits_required"] = max(
            1,
            int(ar_consecutive_hits_required),
        )
    if ar_confirm_window_seconds is not None:
        _runtime_wake_word_phrase_settings["ar_confirm_window_seconds"] = max(1.0, float(ar_confirm_window_seconds))
    return get_runtime_wake_word_phrase_settings()


def get_runtime_wake_mode() -> str:
    return str(_runtime_wake_word_phrase_settings.get("mode") or "both")


def set_runtime_wake_mode(mode: str) -> str:
    set_runtime_wake_word_phrase_settings(mode=mode)
    return get_runtime_wake_mode()


def list_runtime_wake_triggers():
    return list(_runtime_wake_word_phrase_settings.get("arabic_triggers") or ())


def add_runtime_wake_trigger(trigger_phrase: str):
    phrase = _sanitize_trigger_phrase(trigger_phrase)
    if not phrase:
        return False, list_runtime_wake_triggers()

    current = list_runtime_wake_triggers()
    phrase_key = _normalize_trigger_text(phrase)
    if not phrase_key:
        return False, current
    for existing in current:
        if _normalize_trigger_text(existing) == phrase_key:
            return False, current

    current.append(phrase)
    set_runtime_wake_word_phrase_settings(arabic_triggers=current)
    return True, list_runtime_wake_triggers()


def remove_runtime_wake_trigger(trigger_phrase: str):
    phrase_key = _normalize_trigger_text(trigger_phrase)
    if not phrase_key:
        return False, list_runtime_wake_triggers()

    current = list_runtime_wake_triggers()
    kept = [item for item in current if _normalize_trigger_text(item) != phrase_key]
    removed = len(kept) != len(current)
    if removed:
        set_runtime_wake_word_phrase_settings(arabic_triggers=kept)
    return removed, list_runtime_wake_triggers()


def get_runtime_wake_word_behavior():
    return dict(_runtime_wake_word_behavior)


def set_runtime_wake_word_behavior(*, ignore_while_speaking=None, barge_in_interrupt_on_wake=None):
    if ignore_while_speaking is not None:
        _runtime_wake_word_behavior["ignore_while_speaking"] = bool(ignore_while_speaking)
    if barge_in_interrupt_on_wake is not None:
        _runtime_wake_word_behavior["barge_in_interrupt_on_wake"] = bool(barge_in_interrupt_on_wake)
    return get_runtime_wake_word_behavior()


def _match_arabic_trigger(transcript_text: str, trigger_phrases) -> str:
    normalized_text = _normalize_trigger_text(transcript_text)
    if not normalized_text:
        return ""
    transcript_tokens = [item for item in normalized_text.split(" ") if item]
    if not transcript_tokens:
        return ""

    for trigger in trigger_phrases or ():
        trigger_norm = _normalize_trigger_text(trigger)
        if not trigger_norm:
            continue
        trigger_tokens = [item for item in trigger_norm.split(" ") if item]
        if not trigger_tokens:
            continue

        # Single-word triggers (for example "jarvis" / "جارفيس") are intentionally strict
        # to reduce random wakes from longer background speech.
        if len(trigger_tokens) == 1:
            if len(transcript_tokens) <= 2 and trigger_tokens[0] in transcript_tokens:
                return str(trigger)
            continue

        # Multi-word trigger phrases can match as a short contiguous sequence.
        if len(transcript_tokens) > max(6, len(trigger_tokens) + 2):
            continue
        if _contains_token_sequence(transcript_tokens, trigger_tokens):
            return str(trigger)
        filtered_tokens = [token for token in transcript_tokens if token not in _OPTIONAL_WAKE_CONNECTOR_TOKENS]
        if _contains_token_sequence(filtered_tokens, trigger_tokens):
            return str(trigger)
    return ""


def _register_arabic_hit(*, hit_detected: bool, now_ts: float, required_hits: int, confirm_window_seconds: float) -> bool:
    global _ar_consecutive_hits
    global _ar_last_hit_ts

    required = max(1, int(required_hits))
    window = max(1.0, float(confirm_window_seconds))

    if hit_detected:
        if _ar_last_hit_ts > 0 and (now_ts - _ar_last_hit_ts) <= window:
            _ar_consecutive_hits += 1
        else:
            _ar_consecutive_hits = 1
        _ar_last_hit_ts = now_ts

        if _ar_consecutive_hits >= required:
            _ar_consecutive_hits = 0
            _ar_last_hit_ts = 0.0
            return True
        return False

    if _ar_last_hit_ts > 0 and (now_ts - _ar_last_hit_ts) > window:
        _ar_consecutive_hits = 0
        _ar_last_hit_ts = 0.0
    return False


def _get_ar_stt_model(model_name: str):
    global _ar_stt_model
    global _ar_stt_model_name

    candidate_name = str(model_name or "tiny").strip() or "tiny"
    if _ar_stt_model is not None and _ar_stt_model_name == candidate_name:
        return _ar_stt_model

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        raise RuntimeError(
            "faster-whisper is unavailable for Arabic wake trigger detection."
        ) from exc

    _ar_stt_model = WhisperModel(candidate_name, device="cpu", compute_type="int8")
    _ar_stt_model_name = candidate_name
    return _ar_stt_model


def _transcribe_arabic_window(audio_window: np.ndarray, model_name: str) -> str:
    if audio_window is None or int(getattr(audio_window, "size", 0)) <= 0:
        return ""

    model = _get_ar_stt_model(model_name)
    audio_float = audio_window.astype(np.float32) / 32768.0
    segments, _ = model.transcribe(
        audio_float,
        language=None,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
    )
    pieces = []
    for segment in segments:
        text = (segment.text or "").strip()
        if text:
            pieces.append(text)
    return " ".join(pieces).strip()


def _download_file(url, target_path):
    pathlib.Path(target_path).parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    with open(target_path, "wb") as handle:
        handle.write(data)


def _ensure_onnx_resources():
    try:
        import openwakeword
    except Exception as exc:
        raise RuntimeError(
            "openwakeword is unavailable. Install openwakeword in the active Python environment."
        ) from exc

    package_dir = pathlib.Path(openwakeword.__file__).resolve().parent
    model_dir = package_dir / "resources" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    wakeword_key = WAKE_WORD.replace(" ", "_").strip().lower()
    required = [
        "melspectrogram.onnx",
        "embedding_model.onnx",
        f"{wakeword_key}_v0.1.onnx",
    ]

    for filename in required:
        target = model_dir / filename
        if target.exists():
            continue
        url = f"{_OPENWAKEWORD_RELEASE}/{filename}"
        try:
            _download_file(url, str(target))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download wake-word model resource '{filename}' from {url}."
            ) from exc


def _get_model():
    global _model
    if _model is not None:
        return _model

    try:
        from openwakeword.model import Model
    except Exception as exc:
        raise RuntimeError(
            "openwakeword is unavailable. Install openwakeword in the active environment."
        ) from exc

    try:
        _ensure_onnx_resources()
        _model = Model(wakeword_models=[WAKE_WORD], inference_framework="onnx")
    except Exception as exc:
        raise RuntimeError(
            "Failed to initialize wake-word model with ONNX resources. "
            "Ensure network access for first-time model download."
        ) from exc
    return _model


def preload_runtime_wake_word():
    """Preload wake-word runtime resources so first listen has no cold start."""
    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    phrase_runtime = get_runtime_wake_word_phrase_settings()
    wake_mode = _normalize_wake_mode(phrase_runtime.get("mode"))
    english_layer_enabled = wake_mode in {"english", "both"}
    arabic_layer_enabled = bool(phrase_runtime.get("arabic_enabled")) and wake_mode in {
        "english",
        "arabic",
        "both",
    }
    arabic_triggers = tuple(phrase_runtime.get("arabic_triggers") or ())
    if not arabic_triggers:
        arabic_layer_enabled = False

    if not english_layer_enabled and not arabic_layer_enabled:
        raise RuntimeError("Wake word mode disabled all wake layers. Enable english, arabic, or both.")

    input_device = _resolve_input_device()

    english_model_loaded = False
    arabic_model_loaded = False

    if english_layer_enabled:
        _get_model()
        english_model_loaded = True

    if arabic_layer_enabled:
        ar_model_name = str(phrase_runtime.get("ar_stt_model") or WAKE_WORD_AR_STT_MODEL)
        _get_ar_stt_model(ar_model_name)
        arabic_model_loaded = True

    return {
        "mode": wake_mode,
        "input_device": input_device if input_device is not None else "default",
        "english_layer_enabled": bool(english_layer_enabled),
        "arabic_layer_enabled": bool(arabic_layer_enabled),
        "english_model_loaded": bool(english_model_loaded),
        "arabic_model_loaded": bool(arabic_model_loaded),
    }


def _resolve_input_device():
    cfg = WAKE_WORD_INPUT_DEVICE
    if cfg is None or str(cfg).strip() == "":
        return None
    if isinstance(cfg, int):
        return cfg

    name_query = str(cfg).strip().lower()
    try:
        devices = sd.query_devices()
    except Exception as exc:
        raise RuntimeError(f"Failed to list audio devices: {exc}") from exc

    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        if name_query in str(device.get("name", "")).lower():
            return idx

    available = []
    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) > 0:
            available.append(f"{idx}:{device.get('name')}")
    raise RuntimeError(
        "Configured wake-word input device was not found. "
        f"WAKE_WORD_INPUT_DEVICE={cfg!r}. "
        f"Available input devices: {', '.join(available[:12])}"
    )


def listen_for_wake_word():
    global _last_detection_ts
    global _ar_consecutive_hits
    global _ar_last_hit_ts

    if sd is None:
        raise RuntimeError(
            "sounddevice is unavailable. Install sounddevice in the active Python environment."
        ) from _SOUNDDEVICE_IMPORT_ERROR

    runtime = get_runtime_wake_word_settings()
    phrase_runtime = get_runtime_wake_word_phrase_settings()

    wake_threshold = float(runtime["threshold"])
    wake_audio_gain = float(runtime["audio_gain"])
    wake_cooldown = float(runtime["detection_cooldown_seconds"])

    wake_mode = _normalize_wake_mode(phrase_runtime.get("mode"))
    english_layer_enabled = wake_mode in {"english", "both"}
    arabic_layer_enabled = bool(phrase_runtime.get("arabic_enabled")) and wake_mode in {"english", "arabic", "both"}
    arabic_triggers = tuple(phrase_runtime.get("arabic_triggers") or ())
    if not arabic_triggers:
        arabic_layer_enabled = False

    if not english_layer_enabled and not arabic_layer_enabled:
        raise RuntimeError("Wake word mode disabled all wake layers. Enable english, arabic, or both.")

    model = _get_model() if english_layer_enabled else None
    if arabic_layer_enabled and not english_layer_enabled:
        _get_ar_stt_model(str(phrase_runtime.get("ar_stt_model") or "tiny"))

    input_device = _resolve_input_device()
    print(
        "[WakeWord] Waiting for wake word... "
        f"mode={wake_mode} device={input_device if input_device is not None else 'default'}"
    )

    ar_chunk_seconds = float(phrase_runtime.get("ar_chunk_seconds") or WAKE_WORD_AR_CHUNK_SECONDS)
    ar_interval_seconds = float(
        phrase_runtime.get("ar_check_interval_seconds") or WAKE_WORD_AR_CHECK_INTERVAL_SECONDS
    )
    ar_required_hits = int(
        phrase_runtime.get("ar_consecutive_hits_required") or WAKE_WORD_AR_CONSECUTIVE_HITS_REQUIRED
    )
    ar_confirm_window = float(
        phrase_runtime.get("ar_confirm_window_seconds") or WAKE_WORD_AR_CONFIRM_WINDOW_SECONDS
    )
    ar_model_name = str(phrase_runtime.get("ar_stt_model") or WAKE_WORD_AR_STT_MODEL)

    ar_samples = max(WAKE_WORD_CHUNK_SIZE, int(round(float(ar_chunk_seconds) * float(SAMPLE_RATE))))
    ar_chunks = max(1, int(np.ceil(float(ar_samples) / float(WAKE_WORD_CHUNK_SIZE))))
    ar_ring = deque(maxlen=ar_chunks + 1)
    last_ar_check_ts = 0.0
    ar_layer_warning_emitted = False
    ar_slow_inference_warning_emitted = False
    ar_max_inference_seconds = max(1.0, float(ar_chunk_seconds) * 2.5)
    ar_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="jarvis-wake-ar") if arabic_layer_enabled else None
    ar_future = None
    ar_future_started_ts = 0.0
    _ar_consecutive_hits = 0
    _ar_last_hit_ts = 0.0

    last_debug_ts = 0.0
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype=np.int16,
            device=input_device,
            blocksize=WAKE_WORD_CHUNK_SIZE,
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(WAKE_WORD_CHUNK_SIZE)
                audio_chunk = np.asarray(audio_chunk).reshape(-1).astype(np.int16, copy=False)
                if wake_audio_gain and wake_audio_gain != 1.0:
                    boosted = audio_chunk.astype(np.float32) * wake_audio_gain
                    audio_chunk = np.clip(boosted, -32768, 32767).astype(np.int16, copy=False)

                if english_layer_enabled:
                    prediction = model.predict(audio_chunk)
                    score = prediction.get(WAKE_WORD)
                    if score is None and prediction:
                        score = max(prediction.values())

                    if WAKE_WORD_SCORE_DEBUG and score is not None:
                        now = time.perf_counter()
                        if now - last_debug_ts >= float(WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS):
                            rms = float(np.sqrt(np.mean(np.square(audio_chunk.astype(np.float32) / 32768.0))))
                            print(
                                f"[WakeWord] score={float(score):.6f} threshold={wake_threshold:.3f} "
                                f"rms={rms:.4f}"
                            )
                            last_debug_ts = now

                    if score is not None and score > wake_threshold:
                        now = time.perf_counter()
                        if now - _last_detection_ts < wake_cooldown:
                            continue
                        _last_detection_ts = now
                        print("[WakeWord] Wake word detected (english).")
                        return "english"

                if arabic_layer_enabled:
                    ar_ring.append(audio_chunk.copy())
                    now = time.perf_counter()

                    if ar_future is not None and ar_future.done():
                        try:
                            transcript = ar_future.result()
                        except RuntimeError as exc:
                            if not ar_layer_warning_emitted:
                                logger.warning("Arabic wake layer unavailable: %s", exc)
                                ar_layer_warning_emitted = True
                            if wake_mode == "arabic":
                                raise RuntimeError("Arabic wake layer is unavailable in arabic-only mode.") from exc
                            arabic_layer_enabled = False
                            ar_future = None
                            continue
                        except Exception as exc:
                            logger.warning("Arabic wake STT check failed: %s", exc)
                            _register_arabic_hit(
                                hit_detected=False,
                                now_ts=now,
                                required_hits=ar_required_hits,
                                confirm_window_seconds=ar_confirm_window,
                            )
                            ar_future = None
                            continue

                        matched_trigger = _match_arabic_trigger(transcript, arabic_triggers)
                        triggered = _register_arabic_hit(
                            hit_detected=bool(matched_trigger),
                            now_ts=now,
                            required_hits=ar_required_hits,
                            confirm_window_seconds=ar_confirm_window,
                        )

                        if WAKE_WORD_SCORE_DEBUG and transcript:
                            print(
                                f"[WakeWord][AR] transcript={transcript!r} "
                                f"trigger={matched_trigger or 'none'}"
                            )

                        ar_future = None
                        if matched_trigger and triggered:
                            if now - _last_detection_ts < wake_cooldown:
                                continue
                            _last_detection_ts = now
                            print(f"[WakeWord] Wake word detected (phrase): {matched_trigger}")
                            return "phrase"

                    if ar_future is not None:
                        if (
                            not ar_slow_inference_warning_emitted
                            and (now - ar_future_started_ts) >= ar_max_inference_seconds
                        ):
                            logger.warning(
                                "Arabic wake STT is slower than realtime (%.2fs). "
                                "Set JARVIS_WAKE_WORD_AR_STT_MODEL=tiny for faster wake checks.",
                                now - ar_future_started_ts,
                            )
                            ar_slow_inference_warning_emitted = True
                        continue

                    if now - last_ar_check_ts < float(ar_interval_seconds):
                        continue
                    last_ar_check_ts = now

                    window = np.concatenate(list(ar_ring), axis=0).astype(np.int16, copy=False)
                    if window.shape[0] > ar_samples:
                        window = window[-ar_samples:]

                    if ar_executor is not None:
                        ar_future_started_ts = now
                        ar_future = ar_executor.submit(_transcribe_arabic_window, window, ar_model_name)
    finally:
        if ar_executor is not None:
            ar_executor.shutdown(wait=False, cancel_futures=True)
