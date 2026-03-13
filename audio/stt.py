from core.config import (
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

_model = None
_MOJIBAKE_CHARS = set("ØÙÃÂÐ")


def _get_model():
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


def transcribe_streaming(audio_file: str, on_partial=None) -> str:
    try:
        model = _get_model()
        text = _transcribe_once(
            model,
            audio_file,
            language=WHISPER_LANGUAGE,
            beam_size=WHISPER_BEAM_SIZE,
            vad_filter=WHISPER_VAD_FILTER,
            condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            on_partial=on_partial,
        )

        # Retry strategy for realtime reliability if first pass is empty.
        if not text and WHISPER_VAD_FILTER:
            logger.info("Retrying STT without internal VAD after empty transcript")
            text = _transcribe_once(
                model,
                audio_file,
                language=WHISPER_LANGUAGE,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=False,
                condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                on_partial=on_partial,
            )

        if not text and WHISPER_LANGUAGE is not None:
            logger.info("Retrying STT with auto language detection after empty transcript")
            text = _transcribe_once(
                model,
                audio_file,
                language=None,
                beam_size=WHISPER_BEAM_SIZE,
                vad_filter=False,
                condition_on_previous_text=WHISPER_CONDITION_ON_PREVIOUS_TEXT,
                on_partial=on_partial,
            )

        if not text:
            logger.warning("STT produced empty transcript")
        return text
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        return ""
