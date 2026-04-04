import wave

import numpy as np

from core.config import (
    STT_BACKEND,
    STT_HF_BATCH_SIZE,
    STT_HF_CHUNK_LENGTH_S,
    STT_HF_MODEL,
    STT_HF_MODE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
)
from core.logger import logger

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
_MOJIBAKE_CHARS = set("ØÙÃÂÐ")
_runtime_stt_settings = {
    "beam_size": max(1, int(WHISPER_BEAM_SIZE)),
    "vad_filter": bool(WHISPER_VAD_FILTER),
    "condition_on_previous_text": bool(WHISPER_CONDITION_ON_PREVIOUS_TEXT),
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
    return dict(_runtime_stt_settings)


def set_runtime_stt_settings(
    *,
    beam_size=None,
    vad_filter=None,
    condition_on_previous_text=None,
    quality_retry_threshold=None,
    quality_retry_beam_size=None,
):
    if beam_size is not None:
        _runtime_stt_settings["beam_size"] = max(1, int(beam_size))
    if vad_filter is not None:
        _runtime_stt_settings["vad_filter"] = bool(vad_filter)
    if condition_on_previous_text is not None:
        _runtime_stt_settings["condition_on_previous_text"] = bool(condition_on_previous_text)
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

    device = -1
    if torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
        device = 0

    logger.info("Loading Hugging Face STT model: %s", model_id)
    _hf_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
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

    device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"

    logger.info("Loading Hugging Face STT model (manual decode): %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
    model.eval()

    _hf_processor = processor
    _hf_model = model
    _hf_model_id = model_id
    _hf_device = device
    return _hf_processor, _hf_model, _hf_device


def _transcribe_hf_manual_whisper(audio_file: str, generate_kwargs: dict) -> str:
    processor, model, device = _get_hf_manual_components()
    audio, sample_rate = _read_wav_mono_float(audio_file)
    if audio.size == 0:
        return ""

    target_rate = int(getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000) or 16000)
    if sample_rate != target_rate:
        audio = _resample_audio_linear(audio, sample_rate, target_rate)

    inputs = processor(audio, sampling_rate=target_rate, return_tensors="pt")
    tensor_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in dict(inputs).items()
    }

    forced_decoder_ids = None
    language = generate_kwargs.get("language")
    task = str(generate_kwargs.get("task") or "transcribe")
    if language and hasattr(processor, "get_decoder_prompt_ids"):
        try:
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=language, task=task)
        except Exception as prompt_exc:
            logger.warning("HF STT decoder prompt ids unavailable for language '%s': %s", language, prompt_exc)

    model_kwargs = {}
    if forced_decoder_ids is not None:
        model_kwargs["forced_decoder_ids"] = forced_decoder_ids

    with torch.no_grad():
        generated_ids = model.generate(**tensor_inputs, **model_kwargs)

    decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)
    return str(decoded[0] if decoded else "").strip()


def transcribe(audio_file: str) -> str:
    return transcribe_streaming(audio_file)


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
    segments, _ = model.transcribe(
        audio_file,
        language=language,
        beam_size=max(1, int(beam_size)),
        vad_filter=bool(vad_filter),
        condition_on_previous_text=bool(condition_on_previous_text),
    )
    return _maybe_fix_mojibake(_collect_text(segments, on_partial=on_partial))


def _transcribe_huggingface(audio_file: str, on_partial=None) -> str:
    global _hf_pipeline_decode_unavailable

    hf_mode = _resolve_hf_mode()
    hf_runtime = get_runtime_hf_settings()
    generate_kwargs = {"task": "transcribe"}
    if WHISPER_LANGUAGE is not None:
        generate_kwargs["language"] = WHISPER_LANGUAGE

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
        text = _transcribe_hf_manual_whisper(audio_file, generate_kwargs=generate_kwargs)
        text = _maybe_fix_mojibake(text)
        if text and on_partial is not None:
            try:
                on_partial(text)
            except Exception as callback_exc:
                logger.warning("STT partial callback failed: %s", callback_exc)
        return text

    if _hf_pipeline_decode_unavailable:
        text = _transcribe_hf_manual_whisper(audio_file, generate_kwargs=generate_kwargs)
        text = _maybe_fix_mojibake(text)
        if text and on_partial is not None:
            try:
                on_partial(text)
            except Exception as callback_exc:
                logger.warning("STT partial callback failed: %s", callback_exc)
        return text

    text = ""
    try:
        asr = _get_hf_pipeline()
        result = _run_asr(asr, audio_file)
        text = _extract_text(result)
    except Exception as exc:
        message = str(exc).lower()
        if "ffmpeg" in message or "torchcodec" in message or "libtorchcodec" in message:
            _hf_pipeline_decode_unavailable = True
            logger.warning(
                "HF STT pipeline audio decode unavailable; switching to manual whisper decode for this session: %s",
                exc,
            )
            text = _transcribe_hf_manual_whisper(audio_file, generate_kwargs=generate_kwargs)
        elif hf_mode == "pipeline":
            raise
        else:
            raise

    text = _maybe_fix_mojibake(text)

    if text and on_partial is not None:
        try:
            on_partial(text)
        except Exception as callback_exc:
            logger.warning("STT partial callback failed: %s", callback_exc)
    return text


def _transcribe_faster_whisper(audio_file: str, on_partial=None) -> str:
    model = _get_whisper_model()
    runtime = get_runtime_stt_settings()
    beam_size = int(runtime["beam_size"])
    use_vad_filter = bool(runtime["vad_filter"])
    use_previous_text = bool(runtime["condition_on_previous_text"])
    quality_retry_threshold = float(runtime["quality_retry_threshold"])
    quality_retry_beam_size = int(runtime["quality_retry_beam_size"])

    text = _transcribe_once(
        model,
        audio_file,
        language=WHISPER_LANGUAGE,
        beam_size=beam_size,
        vad_filter=use_vad_filter,
        condition_on_previous_text=use_previous_text,
        on_partial=on_partial,
    )

    # Retry strategy for realtime reliability if first pass is empty.
    if not text and use_vad_filter:
        logger.info("Retrying STT without internal VAD after empty transcript")
        text = _transcribe_once(
            model,
            audio_file,
            language=WHISPER_LANGUAGE,
            beam_size=beam_size,
            vad_filter=False,
            condition_on_previous_text=use_previous_text,
            on_partial=on_partial,
        )

    if not text and WHISPER_LANGUAGE is not None:
        logger.info("Retrying STT with auto language detection after empty transcript")
        text = _transcribe_once(
            model,
            audio_file,
            language=None,
            beam_size=beam_size,
            vad_filter=False,
            condition_on_previous_text=use_previous_text,
            on_partial=on_partial,
        )

    if text and _transcript_quality_score(text) < quality_retry_threshold:
        logger.info("Retrying STT with stronger decoding after low-quality transcript")
        retry_text = _transcribe_once(
            model,
            audio_file,
            language=WHISPER_LANGUAGE,
            beam_size=quality_retry_beam_size,
            vad_filter=False,
            condition_on_previous_text=False,
            on_partial=on_partial,
        )
        if _transcript_quality_score(retry_text) >= _transcript_quality_score(text):
            text = retry_text
    return text


def transcribe_streaming(audio_file: str, on_partial=None) -> str:
    try:
        backend = _resolve_stt_backend()

        text = ""
        if backend == "huggingface":
            try:
                text = _transcribe_huggingface(audio_file, on_partial=on_partial)
            except Exception as exc:
                logger.error("Hugging Face STT failed: %s", exc)
                if WhisperModel is None:
                    return ""
                logger.warning("Falling back to faster-whisper STT")

        if not text:
            text = _transcribe_faster_whisper(audio_file, on_partial=on_partial)

        if not text:
            logger.warning("STT produced empty transcript")
        return text
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        return ""
