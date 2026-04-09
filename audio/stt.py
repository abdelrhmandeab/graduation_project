import wave
import logging
import re

import numpy as np

from core.config import (
    STT_ARABIC_POST_NORMALIZATION,
    STT_ALLOW_CPU_HEAVY_REALTIME,
    STT_BACKEND,
    STT_EGYPTIAN_DIALECT_ONLY,
    STT_HF_BATCH_SIZE,
    STT_HF_CHUNK_LENGTH_S,
    STT_HF_MODEL,
    STT_HF_MODE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE_HINT,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
)
from core.logger import logger
from core.language_gate import detect_supported_language

try:
    from faster_whisper import WhisperModel
except Exception as exc:
    WhisperModel = None
    _WHISPER_IMPORT_ERROR = exc
else:
    _WHISPER_IMPORT_ERROR = None

try:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
except Exception as exc:
    torch = None
    AutoModelForSpeechSeq2Seq = None
    AutoProcessor = None
    pipeline = None
    _HF_IMPORT_ERROR = exc
else:
    _HF_IMPORT_ERROR = None

_model = None
_hf_pipeline = None
_hf_pipeline_model_id = ""
_hf_model = None
_hf_processor = None
_hf_model_id = ""
_hf_device = "cpu"
_hf_pipeline_decode_unavailable = False
_hf_cpu_heavy_warned_models = set()
_hf_generation_warning_filter_attached = False
_MOJIBAKE_CHARS = set("ØÙÃÂÐ")
_HF_CPU_HEAVY_MODEL_MARKERS = (
    "whisper-large",
    "large-v3",
    "large-v2",
)
_WHISPER_LANGUAGE_TOKEN_RE = re.compile(r"<\|([a-z]{2,7})\|>")
_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_ARABIC_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
_ARABIC_POST_WS_RE = re.compile(r"\s+")
_ARABIC_POST_CHAR_TRANSLATE = str.maketrans(
    {
        "\u0623": "\u0627",
        "\u0625": "\u0627",
        "\u0622": "\u0627",
        "\u0624": "\u0648",
        "\u0626": "\u064a",
        "\u0649": "\u064a",
    }
)
_ARABIC_POST_PHRASE_REPLACEMENTS = (
    ("الواى فاى", "الواي فاي"),
    ("الواي فاى", "الواي فاي"),
    ("واى فاى", "واي فاي"),
    ("واى فاي", "واي فاي"),
    ("واي فاى", "واي فاي"),
    ("جوجلكروم", "جوجل كروم"),
    ("سكرينشوت", "سكرين شوت"),
    ("نوتباد", "نوت باد"),
    ("افتحلي", "افتح لي"),
    ("شغللي", "شغل لي"),
    ("دورلي", "دور لي"),
    ("هاتلي", "هات لي"),
    ("اريدك", "عايزك"),
    ("اعزك", "عايزك"),
    ("اريد ان", "عايز"),
    ("تتلاني", "تقولي"),
    ("اخبرني", "قولي"),
    ("اخبار التكس", "اخبار الطقس"),
    ("اسبوري فايل", "سبوتيفاي"),
    ("سبوري فايل", "سبوتيفاي"),
)
_ARABIC_POST_TOKEN_REPLACEMENTS = {
    "عاوز": "عايز",
    "عاوزه": "عايزة",
    "عاوزين": "عايزين",
    "اريد": "عايز",
    "اعز": "عايز",
    "اطفئ": "اطفي",
    "اطفى": "اطفي",
    "واطفئ": "واطفي",
    "واطفى": "واطفي",
    "دلوقتى": "دلوقتي",
    "دلوقت": "دلوقتي",
    "فى": "في",
    "اللى": "اللي",
    "شويه": "شوية",
    "سرعه": "سرعة",
    "قهوه": "قهوة",
    "ساده": "سادة",
    "لاقى": "لاقي",
    "النهارده": "النهاردة",
    "نهارده": "نهاردة",
    "الدونلودز": "الداونلودز",
    "الاشعرات": "الاشعارات",
    "الاعدادت": "الاعدادات",
    "اصعار": "اسعار",
    "الذهب": "الدهب",
    "البورسه": "البورصة",
    "البورسة": "البورصة",
    "بورسان": "بورصة",
    "سبوتفي": "سبوتيفاي",
    "سبوتفى": "سبوتيفاي",
    "سبوتيفي": "سبوتيفاي",
}
_runtime_stt_settings = {
    "beam_size": max(1, int(WHISPER_BEAM_SIZE)),
    "vad_filter": bool(WHISPER_VAD_FILTER),
    "condition_on_previous_text": bool(WHISPER_CONDITION_ON_PREVIOUS_TEXT),
    "language_hint": str(WHISPER_LANGUAGE_HINT or "auto").strip().lower(),
    "quality_retry_threshold": 0.50,
    "quality_retry_beam_size": max(4, int(WHISPER_BEAM_SIZE) + 2),
}
_runtime_hf_settings = {
    "model": str(STT_HF_MODEL or "").strip(),
    "mode": str(STT_HF_MODE or "auto").strip().lower(),
    "chunk_length_s": max(5.0, float(STT_HF_CHUNK_LENGTH_S)),
    "batch_size": max(1, int(STT_HF_BATCH_SIZE)),
}
_runtime_stt_backend = str(STT_BACKEND or "faster_whisper").strip().lower()
_last_transcription_meta = {
    "text": "",
    "language": "en",
    "backend": str(STT_BACKEND or "faster_whisper").strip().lower(),
    "language_confidence": 0.0,
}


def _normalize_stt_backend(value) -> str:
    raw = str(value or "faster_whisper").strip().lower()
    aliases = {
        "hf": "huggingface",
        "transformers": "huggingface",
    }
    backend = aliases.get(raw, raw)
    if backend not in {"faster_whisper", "huggingface"}:
        return "faster_whisper"
    return backend


def get_runtime_stt_backend() -> str:
    return _normalize_stt_backend(_runtime_stt_backend)


def set_runtime_stt_backend(backend: str) -> str:
    global _runtime_stt_backend
    _runtime_stt_backend = _normalize_stt_backend(backend)
    return _runtime_stt_backend


def get_runtime_stt_settings():
    settings = dict(_runtime_stt_settings)
    settings["language_hint"] = _normalize_language_hint(settings.get("language_hint"))
    return settings


def set_runtime_stt_settings(
    *,
    beam_size=None,
    vad_filter=None,
    condition_on_previous_text=None,
    language_hint=None,
    quality_retry_threshold=None,
    quality_retry_beam_size=None,
):
    if beam_size is not None:
        _runtime_stt_settings["beam_size"] = max(1, int(beam_size))
    if vad_filter is not None:
        _runtime_stt_settings["vad_filter"] = bool(vad_filter)
    if condition_on_previous_text is not None:
        _runtime_stt_settings["condition_on_previous_text"] = bool(condition_on_previous_text)
    if language_hint is not None:
        _runtime_stt_settings["language_hint"] = _normalize_language_hint(language_hint)
    if quality_retry_threshold is not None:
        _runtime_stt_settings["quality_retry_threshold"] = max(0.0, min(1.0, float(quality_retry_threshold)))
    if quality_retry_beam_size is not None:
        _runtime_stt_settings["quality_retry_beam_size"] = max(1, int(quality_retry_beam_size))
    return get_runtime_stt_settings()


def _normalize_hf_mode(value) -> str:
    raw = str(value or "auto").strip().lower()
    aliases = {
        "force_manual": "manual",
        "whisper_manual": "manual",
        "force_pipeline": "pipeline",
    }
    mode = aliases.get(raw, raw)
    if mode not in {"auto", "manual", "pipeline"}:
        return "auto"
    return mode


def _normalize_language_hint(value):
    raw = str(value or "").strip().lower()
    aliases = {
        "arabic": "ar",
        "english": "en",
        "none": "auto",
    }
    hint = aliases.get(raw, raw)
    if hint in {"auto", ""}:
        return "auto"
    if hint in {"ar", "en"}:
        return hint
    return "auto"


def _normalize_detected_language(value, fallback=""):
    raw = str(value or "").strip().lower()
    aliases = {
        "arabic": "ar",
        "english": "en",
    }
    language = aliases.get(raw, raw)
    if language in {"ar", "en"}:
        return language
    return str(fallback or "").strip().lower()


def _coerce_supported_language(language, text, fallback="en"):
    direct = _normalize_detected_language(language, fallback="")
    if direct in {"ar", "en"}:
        inferred_from_text = detect_supported_language(
            text,
            previous_language=direct,
        )
        if inferred_from_text.supported and inferred_from_text.language in {"ar", "en"}:
            if inferred_from_text.language != direct:
                return inferred_from_text.language
        return direct

    normalized_fallback = _normalize_detected_language(fallback, fallback="")
    inferred = detect_supported_language(
        text,
        previous_language=normalized_fallback,
    )
    if inferred.supported and inferred.language in {"ar", "en"}:
        return inferred.language
    return normalized_fallback or "en"


def _extract_whisper_language_from_generated_ids(processor, generated_ids):
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return ""
    if not hasattr(tokenizer, "convert_ids_to_tokens"):
        return ""

    sequence = generated_ids
    try:
        if hasattr(sequence, "detach"):
            sequence = sequence.detach().cpu()
        if hasattr(sequence, "tolist"):
            sequence = sequence.tolist()
    except Exception:
        pass

    if isinstance(sequence, list) and sequence and isinstance(sequence[0], list):
        sequence = sequence[0]
    if not isinstance(sequence, list):
        return ""

    try:
        token_text = tokenizer.convert_ids_to_tokens(sequence[:8])
    except Exception:
        return ""

    if not isinstance(token_text, list):
        return ""

    for token in token_text:
        match = _WHISPER_LANGUAGE_TOKEN_RE.search(str(token or ""))
        if not match:
            continue
        candidate = str(match.group(1) or "").strip().lower()
        if candidate in {"ar", "en"}:
            return candidate
        if candidate in {"arabic", "english"}:
            return _normalize_detected_language(candidate, fallback="")
    return ""


def _update_last_transcription_meta(text, language, backend, *, language_confidence=None):
    _last_transcription_meta["text"] = str(text or "")
    _last_transcription_meta["language"] = _coerce_supported_language(language, text, fallback="en")
    _last_transcription_meta["backend"] = _normalize_stt_backend(backend)
    try:
        confidence_value = float(language_confidence)
    except (TypeError, ValueError):
        confidence_value = 0.0
    _last_transcription_meta["language_confidence"] = max(0.0, min(1.0, confidence_value))


def get_last_transcription_meta():
    return dict(_last_transcription_meta)


def _resolve_whisper_language(language_hint=None):
    explicit = _normalize_language_hint(language_hint)
    if explicit in {"ar", "en"}:
        return explicit
    # Keep true per-utterance auto language detection unless an explicit hint is
    # supplied for the same utterance (for example, a targeted retry pass).
    return None


def _is_hf_cpu_heavy_model(model_id) -> bool:
    value = str(model_id or "").strip().lower()
    if not value:
        return False
    return any(marker in value for marker in _HF_CPU_HEAVY_MODEL_MARKERS)


def _has_hf_gpu_acceleration() -> bool:
    if torch is None:
        return False
    try:
        return bool(hasattr(torch, "cuda") and torch.cuda.is_available())
    except Exception:
        return False


def _should_skip_hf_for_realtime(model_id) -> bool:
    if bool(STT_ALLOW_CPU_HEAVY_REALTIME):
        return False
    return (not _has_hf_gpu_acceleration()) and _is_hf_cpu_heavy_model(model_id)


def _warn_hf_cpu_heavy_once(model_id) -> None:
    key = str(model_id or "").strip().lower()
    if not key:
        key = "unknown"
    if key in _hf_cpu_heavy_warned_models:
        return
    _hf_cpu_heavy_warned_models.add(key)
    logger.warning(
        "HF STT model '%s' is CPU-heavy for realtime usage; using faster-whisper fallback.",
        model_id,
    )


def get_runtime_hf_settings():
    settings = dict(_runtime_hf_settings)
    settings["mode"] = _normalize_hf_mode(settings.get("mode"))
    return settings


def set_runtime_hf_settings(
    *,
    model=None,
    mode=None,
    chunk_length_s=None,
    batch_size=None,
):
    global _hf_pipeline
    global _hf_pipeline_model_id
    global _hf_model
    global _hf_processor
    global _hf_model_id
    global _hf_pipeline_decode_unavailable

    old_model = str(_runtime_hf_settings.get("model") or "").strip()

    if model is not None:
        candidate = str(model or "").strip()
        if candidate:
            _runtime_hf_settings["model"] = candidate

    if mode is not None:
        _runtime_hf_settings["mode"] = _normalize_hf_mode(mode)

    if chunk_length_s is not None:
        _runtime_hf_settings["chunk_length_s"] = max(5.0, float(chunk_length_s))

    if batch_size is not None:
        _runtime_hf_settings["batch_size"] = max(1, int(batch_size))

    new_model = str(_runtime_hf_settings.get("model") or "").strip()
    if new_model != old_model:
        _hf_pipeline = None
        _hf_pipeline_model_id = ""
        _hf_model = None
        _hf_processor = None
        _hf_model_id = ""
        _hf_pipeline_decode_unavailable = False

    return get_runtime_hf_settings()


def _transcript_quality_score(text: str) -> float:
    raw = (text or "").strip()
    if not raw:
        return 0.0
    visible_chars = [ch for ch in raw if not ch.isspace()]
    if not visible_chars:
        return 0.0

    alpha_count = sum(1 for ch in visible_chars if ch.isalpha())
    alpha_ratio = float(alpha_count) / float(len(visible_chars))

    words = [token for token in raw.split() if token]
    if not words:
        return 0.0
    unique_ratio = float(len(set(token.lower() for token in words))) / float(len(words))
    short_ratio = float(sum(1 for token in words if len(token) <= 2)) / float(len(words))

    score = alpha_ratio
    score += 0.35 * unique_ratio
    score -= 0.25 * short_ratio
    if len(words) >= 3:
        score += 0.1
    return max(0.0, min(1.0, score))


def _looks_low_quality_transcript(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return True
    return _transcript_quality_score(raw) < 0.50


def _resolve_stt_backend() -> str:
    return _normalize_stt_backend(_runtime_stt_backend)


def _resolve_hf_mode() -> str:
    return _normalize_hf_mode(_runtime_hf_settings.get("mode"))


def _is_hf_local_cache_miss(exc: Exception) -> bool:
    message = str(exc or "").lower()
    markers = (
        "local_files_only",
        "local cache",
        "couldn't find",
        "cannot find the requested files",
        "connection error",
        "offline mode",
    )
    return any(marker in message for marker in markers)


def _load_hf_component_with_local_cache(loader, model_id: str, component_name: str):
    try:
        return loader(local_files_only=True)
    except TypeError:
        # Backward compatibility for loaders that do not accept local_files_only.
        return loader()
    except Exception as exc:
        if not _is_hf_local_cache_miss(exc):
            raise
        logger.info("HF %s not fully cached locally; downloading: %s", component_name, model_id)
        try:
            return loader(local_files_only=False)
        except TypeError:
            return loader()


class _SuppressDuplicateTransformersProcessorWarnings(logging.Filter):
    def filter(self, record):
        message = str(record.getMessage() or "")
        if "A custom logits processor of type" not in message:
            return True
        if "was also created in `.generate()`" not in message:
            return True
        if "SuppressTokens" in message:
            return False
        return True


def _attach_hf_warning_filter_once() -> None:
    global _hf_generation_warning_filter_attached
    if _hf_generation_warning_filter_attached:
        return

    warning_filter = _SuppressDuplicateTransformersProcessorWarnings()
    for logger_name in (
        "transformers.generation.utils",
        "transformers.generation.logits_process",
    ):
        logging.getLogger(logger_name).addFilter(warning_filter)

    _hf_generation_warning_filter_attached = True


def _get_whisper_model():
    global _model
    if _model is not None:
        return _model

    if WhisperModel is None:
        raise RuntimeError(
            "faster-whisper is unavailable in the active Python environment."
        ) from _WHISPER_IMPORT_ERROR

    _model = WhisperModel(
        WHISPER_MODEL,
        device=WHISPER_DEVICE,
        compute_type=WHISPER_COMPUTE_TYPE,
    )
    return _model


def _get_hf_pipeline():
    global _hf_pipeline
    global _hf_pipeline_model_id

    model_id = str(_runtime_hf_settings.get("model") or STT_HF_MODEL).strip()
    if _hf_pipeline is not None and _hf_pipeline_model_id == model_id:
        return _hf_pipeline

    if pipeline is None:
        raise RuntimeError(
            "transformers pipeline is unavailable in the active Python environment."
        ) from _HF_IMPORT_ERROR

    _attach_hf_warning_filter_once()

    device = -1
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = 0

    logger.info("Loading Hugging Face STT model: %s", model_id)
    _hf_pipeline = _load_hf_component_with_local_cache(
        lambda **kwargs: pipeline(
            "automatic-speech-recognition",
            model=model_id,
            device=device,
            **kwargs,
        ),
        model_id,
        "STT pipeline",
    )
    _hf_pipeline_model_id = model_id
    return _hf_pipeline


def _read_wav_mono_float(audio_file: str):
    with wave.open(audio_file, "rb") as handle:
        channels = int(handle.getnchannels())
        sample_width = int(handle.getsampwidth())
        sample_rate = int(handle.getframerate())
        frame_count = int(handle.getnframes())
        raw_bytes = handle.readframes(frame_count)

    if sample_width == 1:
        audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio.astype(np.float32, copy=False), sample_rate


def _resample_audio_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate <= 0 or target_rate <= 0 or audio.size == 0 or source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    if audio.size == 1:
        repeat = max(1, int(round(float(target_rate) / float(source_rate))))
        return np.repeat(audio, repeat).astype(np.float32, copy=False)

    target_len = max(1, int(round(float(audio.shape[0]) * float(target_rate) / float(source_rate))))
    source_positions = np.arange(audio.shape[0], dtype=np.float64)
    target_positions = np.linspace(0.0, float(audio.shape[0] - 1), num=target_len, dtype=np.float64)
    return np.interp(target_positions, source_positions, audio).astype(np.float32, copy=False)


def _get_hf_manual_components():
    global _hf_model
    global _hf_processor
    global _hf_model_id
    global _hf_device

    model_id = str(_runtime_hf_settings.get("model") or STT_HF_MODEL).strip()
    if _hf_model is not None and _hf_processor is not None and _hf_model_id == model_id:
        return _hf_processor, _hf_model, _hf_device

    if AutoProcessor is None or AutoModelForSpeechSeq2Seq is None or torch is None:
        raise RuntimeError(
            "transformers whisper manual components are unavailable in the active Python environment."
        ) from _HF_IMPORT_ERROR

    _attach_hf_warning_filter_once()

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

    logger.info("Loading Hugging Face STT model (manual decode): %s", model_id)
    processor = _load_hf_component_with_local_cache(
        lambda **kwargs: AutoProcessor.from_pretrained(model_id, **kwargs),
        model_id,
        "STT processor",
    )
    model = _load_hf_component_with_local_cache(
        lambda **kwargs: AutoModelForSpeechSeq2Seq.from_pretrained(model_id, **kwargs),
        model_id,
        "STT model",
    ).to(device)
    model.eval()

    _hf_processor = processor
    _hf_model = model
    _hf_model_id = model_id
    _hf_device = device
    return _hf_processor, _hf_model, _hf_device


def _infer_hf_model_dtype(model):
    try:
        for parameter in model.parameters():
            dtype = getattr(parameter, "dtype", None)
            if dtype is not None:
                return dtype
    except Exception:
        return None
    return None


def _to_device(value, device):
    if not hasattr(value, "to"):
        return value
    try:
        return value.to(device)
    except Exception:
        return value


def _to_dtype(value, dtype):
    if dtype is None or not hasattr(value, "to"):
        return value
    try:
        return value.to(dtype=dtype)
    except TypeError:
        try:
            return value.to(dtype)
        except Exception:
            return value
    except Exception:
        return value


def _transcribe_hf_manual_whisper_with_meta(audio_file: str, generate_kwargs: dict):
    processor, model, device = _get_hf_manual_components()
    audio, sample_rate = _read_wav_mono_float(audio_file)
    if audio.size == 0:
        return "", ""

    target_rate = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000) or 16000)
    if sample_rate != target_rate:
        audio = _resample_audio_linear(audio, sample_rate, target_rate)

    inputs = processor(
        audio,
        sampling_rate=target_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    model_dtype = _infer_hf_model_dtype(model)
    tensor_inputs = {}
    for key, value in dict(inputs).items():
        moved = _to_device(value, device)
        if key == "input_features" and model_dtype is not None:
            # Prevent float-vs-half matmul/conv mismatches in manual decode.
            moved = _to_dtype(moved, model_dtype)
        tensor_inputs[key] = moved

    task = str(generate_kwargs.get("task") or "transcribe")
    language = generate_kwargs.get("language")

    modern_kwargs = {}
    if task:
        modern_kwargs["task"] = task
    if language:
        modern_kwargs["language"] = language

    generated_ids = None
    with torch.no_grad():
        try:
            generated_ids = model.generate(**tensor_inputs, **modern_kwargs)
        except TypeError:
            # Compatibility fallback for older transformers builds.
            fallback_kwargs = {}
            forced_decoder_ids = None
            if language and hasattr(processor, "get_decoder_prompt_ids"):
                try:
                    forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
                except Exception as prompt_exc:
                    logger.warning("HF STT decoder prompt ids unavailable for language '%s': %s", language, prompt_exc)
            if forced_decoder_ids is not None:
                fallback_kwargs["forced_decoder_ids"] = forced_decoder_ids
            generated_ids = model.generate(**tensor_inputs, **fallback_kwargs)

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    text = str(decoded[0] if decoded else "").strip()
    detected_language = _extract_whisper_language_from_generated_ids(processor, generated_ids)
    return text, detected_language


def _transcribe_hf_manual_whisper(audio_file: str, generate_kwargs: dict) -> str:
    text, _detected_language = _transcribe_hf_manual_whisper_with_meta(audio_file, generate_kwargs=generate_kwargs)
    return text


def transcribe(audio_file: str, language_hint=None) -> str:
    return transcribe_streaming(audio_file, language_hint=language_hint)


def _collect_text(segments, on_partial=None) -> str:
    partials = []
    for segment in segments:
        piece = (segment.text or "").strip()
        if not piece:
            continue
        partials.append(piece)
        if on_partial:
            try:
                on_partial(" ".join(partials))
            except Exception as callback_exc:
                logger.warning("STT partial callback failed: %s", callback_exc)
    return " ".join(partials).strip()


def _maybe_fix_mojibake(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    mojibake_hits = sum(1 for ch in raw if ch in _MOJIBAKE_CHARS)
    if mojibake_hits < 3:
        return raw
    try:
        repaired = raw.encode("cp1252", errors="strict").decode("utf-8", errors="strict").strip()
    except Exception:
        return raw
    if repaired and repaired != raw:
        logger.info("STT mojibake repair applied")
        return repaired
    return raw


def _replace_whole_token(text: str, source: str, target: str) -> str:
    pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
    return re.sub(pattern, target, text)


def normalize_arabic_post_transcript(text: str) -> str:
    value = " ".join(str(text or "").split()).strip()
    if not value:
        return ""
    if not _ARABIC_CHAR_RE.search(value):
        return value

    value = _ARABIC_DIACRITICS_RE.sub("", value)
    value = value.replace("\u0640", "")
    value = value.translate(_ARABIC_POST_CHAR_TRANSLATE)

    for source, target in _ARABIC_POST_PHRASE_REPLACEMENTS:
        value = _replace_whole_token(value, source, target)

    for source, target in _ARABIC_POST_TOKEN_REPLACEMENTS.items():
        value = _replace_whole_token(value, source, target)

    value = _ARABIC_POST_WS_RE.sub(" ", value).strip()
    return value


def _transcribe_once(
    model,
    audio_file: str,
    *,
    language,
    beam_size: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
    on_partial=None,
) -> str:
    text, _detected_language, _detected_language_confidence = _transcribe_once_with_meta(
        model,
        audio_file,
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=condition_on_previous_text,
        on_partial=on_partial,
    )
    return text


def _extract_faster_whisper_language(info):
    if isinstance(info, dict):
        return _normalize_detected_language(info.get("language"), fallback="")
    return _normalize_detected_language(getattr(info, "language", ""), fallback="")


def _extract_faster_whisper_language_confidence(info):
    if isinstance(info, dict):
        raw_confidence = info.get("language_probability")
    else:
        raw_confidence = getattr(info, "language_probability", None)
    try:
        value = float(raw_confidence)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, value))


def _transcribe_once_with_meta(
    model,
    audio_file: str,
    *,
    language,
    beam_size: int,
    vad_filter: bool,
    condition_on_previous_text: bool,
    on_partial=None,
):
    segments, info = model.transcribe(
        audio_file,
        language=language,
        beam_size=max(1, int(beam_size)),
        vad_filter=bool(vad_filter),
        condition_on_previous_text=bool(condition_on_previous_text),
    )
    text = _maybe_fix_mojibake(_collect_text(segments, on_partial=on_partial))
    detected_language = _extract_faster_whisper_language(info)
    detected_language_confidence = _extract_faster_whisper_language_confidence(info)
    return text, detected_language, detected_language_confidence


def _extract_hf_pipeline_language(result_value):
    if not isinstance(result_value, dict):
        return ""
    for key in ("language", "detected_language", "lang"):
        candidate = _normalize_detected_language(result_value.get(key), fallback="")
        if candidate in {"ar", "en"}:
            return candidate
    return ""


def _transcribe_huggingface(audio_file: str, on_partial=None, language_hint=None, return_meta=False):
    global _hf_pipeline_decode_unavailable

    hf_mode = _resolve_hf_mode()
    hf_runtime = get_runtime_hf_settings()
    hf_model_id = str(hf_runtime.get("model") or STT_HF_MODEL).strip()

    if _should_skip_hf_for_realtime(hf_model_id):
        _warn_hf_cpu_heavy_once(hf_model_id)
        if return_meta:
            return "", _coerce_supported_language("", "", fallback=language_hint or "")
        return ""

    generate_kwargs = {"task": "transcribe"}
    effective_language = _resolve_whisper_language(language_hint=language_hint)
    if effective_language is not None:
        generate_kwargs["language"] = effective_language

    def _run_asr(asr_pipeline, input_value):
        try:
            return asr_pipeline(
                input_value,
                chunk_length_s=max(5.0, float(hf_runtime.get("chunk_length_s") or STT_HF_CHUNK_LENGTH_S)),
                batch_size=max(1, int(hf_runtime.get("batch_size") or STT_HF_BATCH_SIZE)),
                return_timestamps=False,
                generate_kwargs=generate_kwargs,
            )
        except TypeError:
            # Some pipeline/model combinations do not accept these optional kwargs.
            return asr_pipeline(input_value)

    def _extract_text(result_value) -> str:
        if isinstance(result_value, dict):
            return str(result_value.get("text") or "").strip()
        return str(result_value or "").strip()

    if hf_mode == "manual":
        text, detected_language = _transcribe_hf_manual_whisper_with_meta(audio_file, generate_kwargs=generate_kwargs)
        text = _maybe_fix_mojibake(text)
        detected_language = _coerce_supported_language(detected_language, text, fallback=effective_language or "")
        if text and on_partial is not None:
            try:
                on_partial(text)
            except Exception as callback_exc:
                logger.warning("STT partial callback failed: %s", callback_exc)
        if return_meta:
            return text, detected_language
        return text

    if _hf_pipeline_decode_unavailable:
        text, detected_language = _transcribe_hf_manual_whisper_with_meta(audio_file, generate_kwargs=generate_kwargs)
        text = _maybe_fix_mojibake(text)
        detected_language = _coerce_supported_language(detected_language, text, fallback=effective_language or "")
        if text and on_partial is not None:
            try:
                on_partial(text)
            except Exception as callback_exc:
                logger.warning("STT partial callback failed: %s", callback_exc)
        if return_meta:
            return text, detected_language
        return text

    text = ""
    detected_language = ""
    try:
        asr = _get_hf_pipeline()
        result = _run_asr(asr, audio_file)
        text = _extract_text(result)
        detected_language = _extract_hf_pipeline_language(result)
    except Exception as exc:
        message = str(exc).lower()
        if "ffmpeg" in message or "torchcodec" in message or "libtorchcodec" in message:
            _hf_pipeline_decode_unavailable = True
            logger.warning(
                "HF STT pipeline audio decode unavailable; switching to manual whisper decode for this session: %s",
                exc,
            )
            text, detected_language = _transcribe_hf_manual_whisper_with_meta(audio_file, generate_kwargs=generate_kwargs)
        elif hf_mode == "pipeline":
            raise
        else:
            raise

    text = _maybe_fix_mojibake(text)
    detected_language = _coerce_supported_language(detected_language, text, fallback=effective_language or "")

    if text and on_partial is not None:
        try:
            on_partial(text)
        except Exception as callback_exc:
            logger.warning("STT partial callback failed: %s", callback_exc)
    if return_meta:
        return text, detected_language
    return text


def _transcribe_faster_whisper_with_meta(audio_file: str, on_partial=None, language_hint=None):
    model = _get_whisper_model()
    runtime = get_runtime_stt_settings()
    beam_size = int(runtime["beam_size"])
    use_vad_filter = bool(runtime["vad_filter"])
    use_previous_text = bool(runtime["condition_on_previous_text"])
    quality_retry_threshold = float(runtime["quality_retry_threshold"])
    quality_retry_beam_size = int(runtime["quality_retry_beam_size"])
    effective_language = _resolve_whisper_language(language_hint=language_hint)
    text, detected_language, detected_language_confidence = _transcribe_once_with_meta(
        model,
        audio_file,
        language=effective_language,
        beam_size=beam_size,
        vad_filter=use_vad_filter,
        condition_on_previous_text=use_previous_text,
        on_partial=on_partial,
    )

    # Retry strategy for realtime reliability if first pass is empty.
    if not text and use_vad_filter:
        logger.info("Retrying STT without internal VAD after empty transcript")
        retry_text, retry_language, retry_language_confidence = _transcribe_once_with_meta(
            model,
            audio_file,
            language=effective_language,
            beam_size=beam_size,
            vad_filter=False,
            condition_on_previous_text=use_previous_text,
            on_partial=on_partial,
        )
        if retry_text:
            text = retry_text
            detected_language = retry_language
            detected_language_confidence = retry_language_confidence

    if not text and effective_language is not None:
        logger.info("Retrying STT with auto language detection after empty transcript")
        retry_text, retry_language, retry_language_confidence = _transcribe_once_with_meta(
            model,
            audio_file,
            language=None,
            beam_size=beam_size,
            vad_filter=False,
            condition_on_previous_text=use_previous_text,
            on_partial=on_partial,
        )
        if retry_text:
            text = retry_text
            detected_language = retry_language
            detected_language_confidence = retry_language_confidence

    if text and _transcript_quality_score(text) < quality_retry_threshold:
        logger.info("Retrying STT with stronger decoding after low-quality transcript")
        retry_text, retry_language, retry_language_confidence = _transcribe_once_with_meta(
            model,
            audio_file,
            language=effective_language,
            beam_size=quality_retry_beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            on_partial=on_partial,
        )
        if _transcript_quality_score(retry_text) >= _transcript_quality_score(text):
            text = retry_text
            detected_language = retry_language
            detected_language_confidence = retry_language_confidence

    provisional_language = _coerce_supported_language(
        detected_language,
        text,
        fallback=effective_language or language_hint or "en",
    )

    # If Arabic transcript quality is still weak, run one explicit Arabic pass to
    # stabilize dialect transcription before route-level handling.
    if text and provisional_language == "ar":
        base_ar_score = _transcript_quality_score(text)
        low_ar_confidence = 0.0 < float(detected_language_confidence or 0.0) < 0.72
        if base_ar_score < (quality_retry_threshold + 0.12) or low_ar_confidence:
            logger.info("Retrying STT with explicit Arabic hint for dialect refinement")
            ar_text, _ar_detected, ar_confidence = _transcribe_once_with_meta(
                model,
                audio_file,
                language="ar",
                beam_size=max(quality_retry_beam_size, beam_size + 1),
                vad_filter=False,
                condition_on_previous_text=False,
                on_partial=on_partial,
            )
            ar_confidence = max(0.0, min(1.0, float(ar_confidence or 0.0)))
            confidence_improved = ar_confidence > (float(detected_language_confidence or 0.0) + 0.20)
            if ar_text and (
                _transcript_quality_score(ar_text) > (base_ar_score + 0.05)
                or confidence_improved
            ):
                text = ar_text
                provisional_language = "ar"
                detected_language_confidence = max(
                    float(detected_language_confidence or 0.0),
                    ar_confidence,
                )

    # If language confidence is low in auto mode, validate with the opposite hint
    # to avoid cross-turn language lock hallucinations.
    if (
        text
        and effective_language is None
        and provisional_language in {"ar", "en"}
        and 0.0 < float(detected_language_confidence or 0.0) < 0.72
    ):
        opposite_hint = "en" if provisional_language == "ar" else "ar"
        logger.info(
            "Retrying STT with explicit %s hint due low language confidence (%.2f)",
            opposite_hint,
            float(detected_language_confidence or 0.0),
        )
        opposite_text, opposite_detected, opposite_confidence = _transcribe_once_with_meta(
            model,
            audio_file,
            language=opposite_hint,
            beam_size=max(beam_size, quality_retry_beam_size),
            vad_filter=False,
            condition_on_previous_text=False,
            on_partial=on_partial,
        )
        if opposite_text:
            base_score = _transcript_quality_score(text)
            opposite_score = _transcript_quality_score(opposite_text)
            opposite_language = _coerce_supported_language(
                opposite_detected,
                opposite_text,
                fallback=opposite_hint,
            )
            if opposite_language == opposite_hint:
                opposite_score += 0.08
            if opposite_score > (base_score + 0.10):
                text = opposite_text
                provisional_language = opposite_language
                detected_language_confidence = opposite_confidence

    detected_language = _coerce_supported_language(
        provisional_language,
        text,
        fallback=effective_language or language_hint or "en",
    )
    return text, detected_language, max(0.0, min(1.0, float(detected_language_confidence or 0.0)))


def _transcribe_faster_whisper(audio_file: str, on_partial=None, language_hint=None) -> str:
    text, _detected_language, _detected_language_confidence = _transcribe_faster_whisper_with_meta(
        audio_file,
        on_partial=on_partial,
        language_hint=language_hint,
    )
    return text


def transcribe_backend_direct_with_meta(audio_file: str, *, backend: str, on_partial=None, language_hint=None):
    requested_backend = _normalize_stt_backend(backend)

    if requested_backend == "huggingface":
        result = _transcribe_huggingface(
            audio_file,
            on_partial=on_partial,
            language_hint=language_hint,
            return_meta=True,
        )
    else:
        result = _transcribe_faster_whisper_with_meta(
            audio_file,
            on_partial=on_partial,
            language_hint=language_hint,
        )

    if isinstance(result, tuple) and len(result) >= 2:
        text = str(result[0] or "")
        detected_language = str(result[1] or "")
        try:
            detected_language_confidence = float(result[2]) if len(result) >= 3 else 0.0
        except (TypeError, ValueError):
            detected_language_confidence = 0.0
    else:
        text = str(result or "")
        detected_language = ""
        detected_language_confidence = 0.0

    normalized_language = _coerce_supported_language(
        detected_language,
        text,
        fallback=language_hint or "",
    )
    return {
        "text": text,
        "language": normalized_language,
        "language_confidence": max(0.0, min(1.0, detected_language_confidence)),
        "backend": requested_backend,
    }


def transcribe_streaming_with_meta(audio_file: str, on_partial=None, language_hint=None):
    try:
        backend = _resolve_stt_backend()

        text = ""
        detected_language = ""
        language_confidence = 0.0
        active_backend = backend
        if backend == "huggingface":
            try:
                direct_result = transcribe_backend_direct_with_meta(
                    audio_file,
                    backend=backend,
                    on_partial=on_partial,
                    language_hint=language_hint,
                )
                text = str((direct_result or {}).get("text") or "")
                detected_language = str((direct_result or {}).get("language") or "")
                try:
                    language_confidence = float((direct_result or {}).get("language_confidence") or 0.0)
                except (TypeError, ValueError):
                    language_confidence = 0.0
            except Exception as exc:
                logger.error("Hugging Face STT failed: %s", exc)
                if WhisperModel is None:
                    empty_language = _coerce_supported_language(language_hint, "", fallback="en")
                    _update_last_transcription_meta("", empty_language, backend)
                    return {
                        "text": "",
                        "language": empty_language,
                        "backend": backend,
                    }
                logger.warning("Falling back to faster-whisper STT")
                active_backend = "faster_whisper"

        if not text:
            fw_result = transcribe_backend_direct_with_meta(
                audio_file,
                backend="faster_whisper",
                on_partial=on_partial,
                language_hint=language_hint,
            )
            text = str((fw_result or {}).get("text") or "")
            detected_language = str((fw_result or {}).get("language") or "")
            try:
                language_confidence = float((fw_result or {}).get("language_confidence") or 0.0)
            except (TypeError, ValueError):
                language_confidence = 0.0
            active_backend = "faster_whisper"

        if text and (STT_ARABIC_POST_NORMALIZATION or STT_EGYPTIAN_DIALECT_ONLY):
            should_normalize_arabic = (
                bool(STT_EGYPTIAN_DIALECT_ONLY)
                or _normalize_detected_language(detected_language, fallback="") == "ar"
                or bool(_ARABIC_CHAR_RE.search(text))
            )
            if should_normalize_arabic:
                text = normalize_arabic_post_transcript(text)

        if STT_EGYPTIAN_DIALECT_ONLY and text and _ARABIC_CHAR_RE.search(text):
            detected_language = "ar"

        if not text:
            logger.warning("STT produced empty transcript")

        normalized_language = _coerce_supported_language(
            detected_language,
            text,
            fallback=language_hint or "",
        )
        _update_last_transcription_meta(
            text,
            normalized_language,
            active_backend,
            language_confidence=language_confidence,
        )
        return {
            "text": text,
            "language": normalized_language,
            "language_confidence": max(0.0, min(1.0, float(language_confidence or 0.0))),
            "backend": _normalize_stt_backend(active_backend),
        }
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        fallback_language = _coerce_supported_language(language_hint, "", fallback="en")
        _update_last_transcription_meta(
            "",
            fallback_language,
            _resolve_stt_backend(),
            language_confidence=0.0,
        )
        return {
            "text": "",
            "language": fallback_language,
            "language_confidence": 0.0,
            "backend": _resolve_stt_backend(),
        }


def transcribe_streaming(audio_file: str, on_partial=None, language_hint=None) -> str:
    result = transcribe_streaming_with_meta(
        audio_file,
        on_partial=on_partial,
        language_hint=language_hint,
    )
    return str((result or {}).get("text") or "")
