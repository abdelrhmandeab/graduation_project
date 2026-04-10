import re

from core.config import (
    STT_ARABIC_POST_NORMALIZATION,
    STT_BACKEND,
    STT_EGYPTIAN_DIALECT_ONLY,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE_TYPE,
    WHISPER_CONDITION_ON_PREVIOUS_TEXT,
    WHISPER_DEVICE,
    WHISPER_LANGUAGE_HINT,
    WHISPER_MODEL,
    WHISPER_VAD_FILTER,
)
from core.language_gate import detect_supported_language
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
_runtime_stt_backend = str(STT_BACKEND or "faster_whisper").strip().lower()
_last_transcription_meta = {
    "text": "",
    "language": "en",
    "backend": "faster_whisper",
    "language_confidence": 0.0,
}


def _normalize_stt_backend(value) -> str:
    raw = str(value or "faster_whisper").strip().lower().replace("-", "_")
    aliases = {
        "fw": "faster_whisper",
        "faster": "faster_whisper",
        "whisper": "faster_whisper",
    }
    backend = aliases.get(raw, raw)
    if backend != "faster_whisper":
        return "faster_whisper"
    return backend


def get_runtime_stt_backend() -> str:
    return _normalize_stt_backend(_runtime_stt_backend)


def set_runtime_stt_backend(backend: str) -> str:
    global _runtime_stt_backend
    _runtime_stt_backend = _normalize_stt_backend(backend)
    return _runtime_stt_backend


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
    return None


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


def transcribe_backend_direct_with_meta(audio_file: str, *, backend: str, on_partial=None, language_hint=None):
    requested_backend = _normalize_stt_backend(backend)
    text, detected_language, detected_language_confidence = _transcribe_faster_whisper_with_meta(
        audio_file,
        on_partial=on_partial,
        language_hint=language_hint,
    )
    normalized_language = _coerce_supported_language(
        detected_language,
        text,
        fallback=language_hint or "",
    )
    return {
        "text": str(text or ""),
        "language": normalized_language,
        "language_confidence": max(0.0, min(1.0, float(detected_language_confidence or 0.0))),
        "backend": requested_backend,
    }


def transcribe_streaming_with_meta(audio_file: str, on_partial=None, language_hint=None):
    try:
        direct_result = transcribe_backend_direct_with_meta(
            audio_file,
            backend="faster_whisper",
            on_partial=on_partial,
            language_hint=language_hint,
        )
        text = str((direct_result or {}).get("text") or "")
        detected_language = str((direct_result or {}).get("language") or "")
        try:
            language_confidence = float((direct_result or {}).get("language_confidence") or 0.0)
        except (TypeError, ValueError):
            language_confidence = 0.0

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

        normalized_language = _coerce_supported_language(
            detected_language,
            text,
            fallback=language_hint or "",
        )
        _update_last_transcription_meta(
            text,
            normalized_language,
            "faster_whisper",
            language_confidence=language_confidence,
        )
        return {
            "text": text,
            "language": normalized_language,
            "language_confidence": max(0.0, min(1.0, float(language_confidence or 0.0))),
            "backend": "faster_whisper",
        }
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        fallback_language = _coerce_supported_language(language_hint, "", fallback="en")
        _update_last_transcription_meta(
            "",
            fallback_language,
            "faster_whisper",
            language_confidence=0.0,
        )
        return {
            "text": "",
            "language": fallback_language,
            "language_confidence": 0.0,
            "backend": "faster_whisper",
        }


def transcribe_streaming(audio_file: str, on_partial=None, language_hint=None) -> str:
    result = transcribe_streaming_with_meta(
        audio_file,
        on_partial=on_partial,
        language_hint=language_hint,
    )
    return str((result or {}).get("text") or "")
