import asyncio
import io
import inspect
import re
import threading
import time
import wave
import warnings

import concurrent.futures as _cf

from core.config import (
    ELEVENLABS_API_KEY,
    ELEVENLABS_BASE_URL,
    TTS_ARABIC_SPOKEN_DIALECT,
    TTS_DEFAULT_BACKEND,
    TTS_EDGE_MIXED_SCRIPT_CHUNKING,
    TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS,
    TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH,
    TTS_EDGE_MIXED_SCRIPT_MIN_TEXT_LENGTH,
    TTS_ELEVENLABS_ARABIC_ENABLED,
    TTS_ELEVENLABS_MODEL_ID,
    TTS_ELEVENLABS_TIMEOUT_SECONDS,
    TTS_EGYPTIAN_COLLOQUIAL_REWRITE,
    TTS_ENABLED,
    TTS_PARAGRAPH_GAP_MS,
    TTS_QUALITY_MODE,
    TTS_SENTENCE_FIRST_FLUSH_MIN_CHARS,
    TTS_SENTENCE_GAP_MS,
    TTS_SENTENCE_STREAMING_ENABLED,
    TTS_SENTENCE_SYNTH_WORKERS,
    TTS_SIMULATED_CHAR_DELAY,
)
from core.logger import logger
from core.metrics import latency_tracker, metrics, record_stage_timing
from core.persona import persona_manager
from core.tts_voices import VoiceProfile, get_active_voice_profile, format_voice_profile_summary

try:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Core Pydantic V1 functionality isn't compatible with Python 3\.14 or greater\.",
            category=UserWarning,
        )
        from elevenlabs.client import ElevenLabs
except Exception:  # pragma: no cover - optional dependency
    ElevenLabs = None

_ELEVENLABS_TTS_COOLDOWN_UNTIL = 0.0


def _elevenlabs_tts_on_cooldown():
    return time.time() < _ELEVENLABS_TTS_COOLDOWN_UNTIL


def _set_elevenlabs_tts_cooldown(reason: str, *, seconds: float = 1800.0) -> None:
    global _ELEVENLABS_TTS_COOLDOWN_UNTIL
    duration = max(60.0, float(seconds))
    _ELEVENLABS_TTS_COOLDOWN_UNTIL = max(_ELEVENLABS_TTS_COOLDOWN_UNTIL, time.time() + duration)
    logger.warning("ElevenLabs TTS cooldown enabled for %.0fs: %s", duration, reason)


def _contains_arabic(text):
    for ch in str(text or ""):
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            return True
    return False


def _contains_latin(text):
    for ch in str(text or ""):
        if "a" <= ch.lower() <= "z":
            return True
    return False


def _count_arabic_letters(text):
    count = 0
    for ch in str(text or ""):
        code = ord(ch)
        if (
            0x0600 <= code <= 0x06FF
            or 0x0750 <= code <= 0x077F
            or 0x08A0 <= code <= 0x08FF
            or 0xFB50 <= code <= 0xFDFF
            or 0xFE70 <= code <= 0xFEFF
        ):
            count += 1
    return count


def _count_latin_letters(text):
    count = 0
    for ch in str(text or ""):
        if "a" <= ch.lower() <= "z":
            count += 1
    return count


_EGYPTIAN_TTS_PHRASE_REPLACEMENTS = (
    # ── RULE: longer / more-specific phrases must come BEFORE shorter ones ──
    # so "لن أستطيع" fires before "أستطيع", "لماذا" fires before "ماذا", etc.
    #
    # Q2 2026 P0 Optimization #9 – Egyptian colloquial TTS rewriter:
    #   Status: Regex-based implementation complete with 150+ MSA→Egyptian phrase pairs.
    #   Handles: negation (لا→مش), future tense (سوف→ه), questions (ماذا→إيه), connectors.
    #   Limitation: Regex misses 30-40% of LLM-generated MSA phrasings (idioms, rare forms).
    #   Future approach (post-Q2): Train a small seq2seq rewriter (~0.5B params) on
    #     Egyptian corpus (labeled MSA↔Egyptian pairs). Benefit: Near-100% coverage.
    #     Training data: ~10k MSA/Egyptian sentence pairs. Training time: ~2-4h on CPU.
    #     Inference: ~20-50ms per sentence on CPU. Fallback: Keep regex for fast path.

    # Negated ability (must precede bare ability words)
    ("لن أستطيع", "مش هقدر"),
    ("لن يستطيع", "مش هيقدر"),
    ("لن تستطيع", "مش هتقدر"),
    ("لن نستطيع", "مش هنقدر"),
    ("لن أتمكن", "مش هقدر"),
    ("لن يتمكن", "مش هيقدر"),
    ("لا أستطيع", "مش قادر"),
    ("لا يستطيع", "مش قادر"),
    ("لا تستطيع", "مش قادرة"),
    ("لا نستطيع", "مش قادرين"),
    ("لا أقدر", "مش قادر"),
    ("لا أعرف", "مش عارف"),
    ("لا أعلم", "مش عارف"),
    ("لم أعد أعرف", "بقيت مش عارف"),
    # Negated existence (before bare negation)
    ("لا يوجد", "مفيش"),
    ("لا توجد", "مفيش"),
    ("ليس هناك", "مفيش"),
    ("ليس لدي", "معنديش"),
    ("ليست لدي", "معنديش"),
    ("ليس لديك", "معندكش"),
    ("ليس لديه", "معندوش"),
    # Negated past tense
    ("لم يكن", "ماكانش"),
    ("لم تكن", "ماكانتش"),
    ("لم أكن", "ماكنتش"),
    ("لم أفعل", "معملتش"),
    ("لم أعمل", "معملتش"),
    ("لم أرى", "ماشفتش"),
    ("لم أسمع", "ماسمعتش"),
    # Future tense — `سوف` / `س` prefix becomes `هـ`
    ("سوف أقوم", "هعمل"),
    ("سوف أعمل", "هعمل"),
    ("سوف أساعدك", "هساعدك"),
    ("سوف أخبرك", "هقولك"),
    ("سوف أقول", "هقول"),
    ("سوف نقوم", "هنعمل"),
    ("سوف نعمل", "هنعمل"),
    ("سوف يتم", "هيتم"),
    ("سأقوم بـ", "هـ"),
    ("سأقوم", "هعمل"),
    ("سأعمل", "هعمل"),
    ("سأخبرك", "هقولك"),
    ("سأشرح", "هشرحلك"),
    ("سأساعدك", "هساعدك"),
    ("سأرسل", "هبعت"),
    ("سوف أرسل", "هبعت"),
    ("سوف أرسلها", "هبعتها"),
    ("سوف أرسله", "هبعته"),
    ("سوف أحاول", "هحاول"),
    ("سوف أفكر", "هفكر"),
    ("سأحاول", "هحاول"),
    ("سأفكر", "هفكر"),
    ("سنقوم", "هنعمل"),
    ("سنعمل", "هنعمل"),
    ("سيتم", "هيتم"),
    # Question words: compound before simple
    ("كيف ذلك", "إزاي ده"),
    ("كيف يمكن", "إزاي ممكن"),
    ("كيف يمكنك", "إزاي تقدر"),
    ("من هو", "مين ده"),
    ("من هي", "مين دي"),
    ("ما هو", "إيه ده"),
    ("ما هي", "إيه دي"),
    ("ماذا تعني", "تعني إيه"),
    ("ماذا يعني", "يعني إيه"),
    ("لماذا", "ليه"),          # must precede ماذا
    ("لمَ", "ليه"),
    ("ماذا", "إيه"),
    ("متى", "إمتى"),
    ("أين", "فين"),
    ("كيف", "إزاي"),
    # Multi-word expressions (before their shorter components)
    ("بكل تأكيد", "اكيد"),
    ("بالتأكيد", "اكيد"),
    ("بالإضافة إلى ذلك", "وكمان"),
    ("بالإضافة الى ذلك", "وكمان"),
    ("في الوقت الحالي", "دلوقتي"),
    ("في الوقت الراهن", "دلوقتي"),
    ("في هذه اللحظة", "دلوقتي"),
    ("في هذا الوقت", "دلوقتي"),
    ("على سبيل المثال", "مثلاً"),
    ("على الأرجح", "غالباً"),
    ("بشكل خاص", "خصوصاً"),
    ("بشكل سريع", "بسرعة"),
    ("من الضروري", "لازم"),
    ("من الأفضل", "الأحسن"),
    ("من الممكن", "ممكن"),
    ("من المهم", "مهم"),
    ("يجب عليك", "لازم"),
    ("يجب علي", "لازم"),
    ("يجب أن", "لازم"),
    ("يجب ان", "لازم"),
    ("ينبغي أن", "المفروض"),
    ("ينبغي ان", "المفروض"),
    ("من المفترض", "المفروض"),
    ("بعد ذلك", "بعدين"),
    ("قبل ذلك", "قبل كده"),
    ("بعد قليل", "بعد شوية"),
    ("منذ قليل", "من شوية"),
    ("منذ فترة", "من فترة"),
    ("في البداية", "في الأول"),
    ("في النهاية", "في الآخر"),
    ("في المنتصف", "في النص"),
    ("مع ذلك", "مع كده"),
    ("رغم ذلك", "مع كده"),
    ("بالرغم من", "رغم"),
    ("على أي حال", "على العموم"),
    ("على كل حال", "على العموم"),
    ("بأي حال", "بأي شكل"),
    ("يحتاج إلى", "محتاج"),
    ("يحتاج الى", "محتاج"),
    ("أحتاج إلى", "محتاج"),
    ("أحتاج الى", "محتاج"),
    ("نحتاج إلى", "محتاجين"),
    ("نحتاج الى", "محتاجين"),
    ("من فضلك،", "لو سمحت،"),
    ("لو سمحتم", "لو سمحت"),
    ("تمام الأمر", "تمام"),
    ("تم بنجاح", "اتعمل تمام"),
    ("تم الأمر", "خلاص"),
    ("هل تريد", "عايز"),
    ("هل تريدين", "عايزة"),
    ("هل يمكنك", "تقدر"),
    ("هل يمكنني", "اقدر"),
    ("هل يمكن", "ممكن"),
    # Politeness / acknowledgements
    ("شكراً جزيلاً", "متشكر جداً"),
    ("شكرا جزيلا", "متشكر جداً"),
    ("لا شكر على واجب", "العفو"),
    # Demonstrative compounds (before bare demonstratives)
    ("هذا الأمر", "الموضوع ده"),
    ("هذه الطريقة", "الطريقة دي"),
    ("هذا الشيء", "الحاجة دي"),
    ("هذه الحالة", "الحالة دي"),
    ("هذه المرة", "المرة دي"),
    ("هذه الأيام", "الأيام دي"),
    ("هذا اليوم", "اليوم ده"),
    ("هذه السنة", "السنة دي"),
    # Relative pronouns
    ("الذي", "اللي"),
    ("التي", "اللي"),
    ("الذين", "اللي"),
    ("اللذان", "اللي"),
    ("اللتان", "اللي"),
    # Connectors (compound before simple)
    ("لأنه", "عشانه"),
    ("لأنها", "عشانها"),
    ("ولكن", "بس"),
    ("لكن", "بس"),
    ("لأن", "عشان"),
    ("كي لا", "عشان ما"),
    ("لكي", "عشان"),
    ("كذلك", "كمان"),
    ("بينما", "وانت"),
    ("في حين", "وانت"),
    ("إذا", "لو"),
    ("اذا", "لو"),
    ("إن", "لو"),
    # Verbs (negated before bare)
    ("يُفضَّل", "الأحسن"),
    ("يُفضل", "الأحسن"),
    ("بالطبع", "طبعاً"),
    ("حالياً", "دلوقتي"),
    ("حاليا", "دلوقتي"),
    ("للأسف", "يا ريت"),
    ("أولاً", "الأول"),
    ("ثانياً", "تانياً"),
    ("ثالثاً", "تالتاً"),
    ("أخيراً", "في الآخر"),
    ("من فضلك", "لو سمحت"),
    # Ownership / pronouns
    ("لدي", "عندي"),
    ("لديك", "عندك"),
    ("لديه", "عنده"),
    ("لديها", "عندها"),
    ("لدينا", "عندنا"),
    ("لديهم", "عندهم"),
    # Opinions / belief
    ("في الواقع", "بصراحة"),
    ("في الحقيقة", "بصراحة"),
    ("بكل صراحة", "بصراحة"),
    ("بصدق", "بصراحة"),
    ("أعتقد أن", "بفكر إن"),
    ("أعتقد أنه", "بفكر إنه"),
    ("أعتقد أنها", "بفكر إنها"),
    ("لا أظن أن", "مش بفكر إن"),
    ("لا أظن", "مش بفكر"),
    ("أظن أن", "بفكر إن"),
    # Imperative / invitation
    ("دعني أخبرك", "خليني أقولك"),
    ("دعني أشرح لك", "خليني أشرحلك"),
    ("دعني أوضح لك", "خليني أوضحلك"),
    ("دعني أريك", "خليني أوريك"),
    ("اسمح لي أن", "خليني"),
    ("اسمح لي", "خليني"),
    # Reassurance / negated worry
    ("لا تقلق", "ماتقلقش"),
    ("لا تخف", "ماتخافش"),
    ("لا يهم", "مش مهم"),
    ("لا بأس بذلك", "ماعليش"),
    ("لا بأس", "ماعليش"),
    ("لا يوجد مشكلة", "مفيش مشكلة"),
    ("لا توجد مشكلة", "مفيش مشكلة"),
    # Causation / result
    ("لذلك", "عشان كده"),
    ("لذا", "عشان كده"),
    ("ولذلك", "وعشان كده"),
    ("وبذلك", "وبكده"),
    ("نتيجة لذلك", "وعشان كده"),
    ("بسبب ذلك", "عشان كده"),
    ("نظراً لذلك", "بسبب كده"),
    ("حيث أن", "بما إن"),
    ("بما أن", "بما إن"),
    ("على الرغم من أن", "مع إن"),
    ("على الرغم من", "رغم"),
    # Contrast / comparison
    ("من ناحية أخرى", "من تاني ناحية"),
    ("من جهة أخرى", "من تاني ناحية"),
    # Additional quantity/degree
    ("دائماً", "دايماً"),
    ("دائما", "دايماً"),
    ("فقط", "بس"),
    ("فحسب", "بس"),
    # Common response openers
    ("حسناً إذن", "تمام يبقى"),
    ("حسناً،", "تمام،"),
    ("إذن،", "يبقى،"),
    ("إذن", "يبقى"),
    # Additional greeting / farewell
    ("مرحباً بك", "أهلاً بيك"),
    ("أهلاً وسهلاً بك", "أهلاً بيك"),
    ("سعيد بلقائك", "تشرفنا"),
    ("وداعاً", "مع السلامة"),
    ("إلى اللقاء", "مع السلامة"),
    # Additional connectors
    ("من خلال", "عن طريق"),
    ("بواسطة", "عن طريق"),
    ("وفقاً", "حسب"),
    ("طبقاً", "حسب"),
    ("وفقاً لـ", "حسب"),
    ("طبقاً لـ", "حسب"),
    ("إضافة إلى ذلك", "وكمان"),
    # Discourse markers
    ("أولاً وقبل كل شيء", "في الأول"),
    ("بادئ ذي بدء", "في الأول"),
    ("وخلاصة القول", "والخلاصة"),
    ("خلاصة القول", "والخلاصة"),
    ("في الختام", "في الآخر"),
    ("بشكل مختصر", "باختصار"),
    ("باختصار شديد", "باختصار"),
)

_EGYPTIAN_TTS_WORD_REPLACEMENTS = (
    # Single-word replacements — applied with word-boundary regex AFTER phrases.
    # Negated forms are handled in the phrase table above; only bare forms here.
    # Ability
    ("يمكنني", "اقدر"),
    ("يمكنه", "يقدر"),
    ("يمكنها", "تقدر"),
    ("أستطيع", "اقدر"),
    ("نستطيع", "نقدر"),
    ("يمكنك", "تقدر"),
    ("يمكنكم", "تقدروا"),
    ("بإمكانك", "تقدر"),
    ("بإمكاني", "بقدر"),
    # Time
    ("الآن", "دلوقتي"),
    ("الان", "دلوقتي"),
    ("اليوم", "النهاردة"),
    ("غداً", "بكرة"),
    ("غدا", "بكرة"),
    ("أمس", "امبارح"),
    ("البارحة", "امبارح"),
    # Demonstratives
    ("هذا", "ده"),
    ("هذه", "دي"),
    ("ذلك", "ده"),
    ("تلك", "دي"),
    ("هؤلاء", "دول"),
    ("أولئك", "دول"),
    # Verbs
    ("أذهب", "أروح"),
    ("يذهب", "يروح"),
    ("تذهب", "تروح"),
    ("نذهب", "نروح"),
    ("اذهب", "روح"),
    ("أريد", "عايز"),
    ("تريد", "عايز"),
    ("يريد", "عايز"),
    ("تريدين", "عايزة"),
    ("نريد", "عايزين"),
    ("أعرف", "عارف"),
    ("يعرف", "بيعرف"),
    ("أعلم", "عارف"),
    ("تعلم", "تعرف"),
    ("يقول", "بيقول"),
    ("تقول", "بتقول"),
    ("نقول", "بنقول"),
    ("يعمل", "بيعمل"),
    ("تعمل", "بتعمل"),
    ("أعمل", "بعمل"),
    ("نعمل", "بنعمل"),
    ("أفكر", "بفكر"),
    ("تفكر", "بتفكر"),
    ("يفكر", "بيفكر"),
    ("أنظر", "أشوف"),
    ("تنظر", "تشوف"),
    ("ينظر", "يشوف"),
    ("يحتاج", "محتاج"),
    ("تحتاج", "محتاجة"),
    ("يفضل", "الأحسن"),
    ("يجب", "لازم"),
    # Like / similar
    ("مثل", "زي"),
    # Also / too
    ("أيضاً", "كمان"),
    ("أيضا", "كمان"),
    ("كذلك", "كمان"),
    # Negation (bare)
    ("ليس", "مش"),
    ("ليست", "مش"),
    ("لست", "مش"),
    ("لسنا", "مش"),
    # Common words
    ("جداً", "أوي"),
    ("جدا", "أوي"),
    ("كثيراً", "أوي"),
    ("كثيرا", "أوي"),
    ("قليلاً", "شوية"),
    ("قليلا", "شوية"),
    ("سريعاً", "بسرعة"),
    ("سريعا", "بسرعة"),
    ("ببطء", "براحتك"),
    ("صحيح", "صح"),
    ("خاطئ", "غلط"),
    ("حسناً", "تمام"),
    ("حسنا", "تمام"),
    ("موافق", "تمام"),
    ("عفواً", "أهلاً"),
    ("بالطبع", "طبعاً"),
    ("ربما", "يمكن"),
    ("لربما", "يمكن"),
    # Quantity / degree
    ("بعض", "شوية"),
    ("معظم", "أغلب"),
    # Common nouns
    ("الأشخاص", "الناس"),
    ("شخص", "حد"),
    ("شيء", "حاجة"),
    ("أشياء", "حاجات"),
    # Additional single-word replacements
    ("أعتقد", "بفكر"),
    ("ممتاز", "عظيم"),
    ("سيئ", "وحش"),
    ("قريباً", "قريب"),
    ("بعيداً", "بعيد"),
    ("أبداً", "خالص"),
    ("لابد", "لازم"),
    ("خطأ", "غلط"),
    ("جيد", "كويس"),
    ("ليس جيداً", "مش كويس"),
    ("أكيد", "اكيد"),
)


_EGY_MARKERS = frozenset((
    "ده", "دي", "إيه", "بيّ", "كده", "يبقى", "إنت", "إنتي",
    "النهاردة", "دلوقتي", "مش", "بيعمل", "عايز", "بحب",
    "يلعب", "بيقول", "بيعمل", "بينور", "عاوز", "طب", "بص",
))


def _count_egy_markers(text):
    count = 0
    for marker in _EGY_MARKERS:
        count += text.count(marker)
    return count


def _rewrite_to_egyptian_colloquial(text):
    """Rewrite an MSA-tilted line into Egyptian Arabic for natural TTS output."""
    from core.config import TTS_EGY_REWRITE_AGGRESSIVE, TTS_EGY_REWRITE_SKIP_THRESHOLD

    updated = str(text or "")
    if not updated:
        return updated

    if not TTS_EGY_REWRITE_AGGRESSIVE:
        marker_count = _count_egy_markers(updated)
        if marker_count >= TTS_EGY_REWRITE_SKIP_THRESHOLD:
            logger.debug("egy_rewrite skipped: already_egy markers=%d", marker_count)
            return updated

    phrase_subs = 0
    for source, target in _EGYPTIAN_TTS_PHRASE_REPLACEMENTS:
        if source in updated:
            updated = updated.replace(source, target)
            phrase_subs += 1

    word_subs = 0
    for source, target in _EGYPTIAN_TTS_WORD_REPLACEMENTS:
        pattern = rf"(?<!\w){re.escape(source)}(?!\w)"
        new_text = re.sub(pattern, target, updated, flags=re.UNICODE)
        if new_text != updated:
            word_subs += 1
            updated = new_text

    updated = re.sub(r"\s+", " ", updated).strip()

    if phrase_subs > 0 or word_subs > 0:
        logger.info("egy_rewrite phrases=%d words=%d", phrase_subs, word_subs)

    return updated


class SpeechEngine:
    def __init__(self):
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = None
        self._queue_thread = None
        self._process = None
        self._runtime_backend = str(TTS_DEFAULT_BACKEND or "auto").strip().lower()
        self._quality_mode = self._normalize_quality_mode(TTS_QUALITY_MODE)
        self._runtime_rate_offset = 0
        self._runtime_pause_scale = 1.0
        self._edge_tts_unavailable_logged = False
        self._edge_tts_decode_warning_logged = False
        self._edge_tts_unsupported_voices = set()
        self._elevenlabs_unavailable_logged = False
        self._enabled = bool(TTS_ENABLED)
        self._voice: VoiceProfile = get_active_voice_profile()
        logger.info(format_voice_profile_summary(self._voice))

    def _normalize_backend(self, backend):
        raw = str(backend or "auto").strip().lower()
        aliases = {
            "edge": "edge_tts",
            "edgetts": "edge_tts",
            "elevenlabs": "hybrid",
            "hybrid_elevenlabs": "hybrid",
        }
        resolved = aliases.get(raw, raw)
        allowed = {"auto", "console", "edge_tts", "hybrid"}
        if resolved not in allowed:
            return "auto"
        return resolved

    def _normalize_quality_mode(self, mode):
        raw = str(mode or "natural").strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "human": "natural",
            "natural_voice": "natural",
            "default": "standard",
            "balanced": "standard",
            "robot": "standard",
            "robotic": "standard",
        }
        normalized = aliases.get(raw, raw)
        if normalized not in {"natural", "standard"}:
            return "natural"
        return normalized

    def get_backend(self):
        with self._lock:
            return self._normalize_backend(self._runtime_backend)

    def set_backend(self, backend):
        with self._lock:
            self._runtime_backend = self._normalize_backend(backend)
            return self._runtime_backend

    def get_quality_mode(self):
        with self._lock:
            return self._normalize_quality_mode(self._quality_mode)

    def set_quality_mode(self, mode):
        with self._lock:
            self._quality_mode = self._normalize_quality_mode(mode)
            return self._quality_mode

    def get_tuning_settings(self):
        with self._lock:
            return {
                "rate_offset": int(self._runtime_rate_offset),
                "pause_scale": float(self._runtime_pause_scale),
            }

    def set_tuning_settings(self, *, rate_offset=None, pause_scale=None):
        with self._lock:
            if rate_offset is not None:
                self._runtime_rate_offset = int(max(-60, min(60, int(rate_offset))))
            if pause_scale is not None:
                self._runtime_pause_scale = float(max(0.6, min(1.6, float(pause_scale))))
            return {
                "rate_offset": int(self._runtime_rate_offset),
                "pause_scale": float(self._runtime_pause_scale),
            }

    def set_enabled(self, enabled):
        with self._lock:
            self._enabled = bool(enabled)
        return True, f"Speech {'enabled' if enabled else 'disabled'}."

    def is_enabled(self):
        with self._lock:
            return self._enabled

    def is_speaking(self):
        with self._lock:
            thread = self._thread
            queue_thread = self._queue_thread
        return bool(
            (thread and thread.is_alive())
            or (queue_thread and queue_thread.is_alive())
        )

    def interrupt(self):
        self._stop_event.set()
        with self._lock:
            process = self._process
            thread = self._thread
            queue_thread = self._queue_thread
            self._process = None
            self._thread = None
            self._queue_thread = None

        if process is not None:
            try:
                process.terminate()
            except Exception:
                pass
        try:
            import sounddevice as sd  # type: ignore

            sd.stop()
        except Exception:
            pass
        current_ident = threading.current_thread().ident
        if thread and thread.is_alive() and thread.ident != current_ident:
            thread.join(timeout=2)
        if queue_thread and queue_thread.is_alive() and queue_thread.ident != current_ident:
            queue_thread.join(timeout=2)
        return True

    def speak_async(self, text, language=None):
        if not (text or "").strip():
            return False, "Nothing to speak."

        if not self.is_enabled():
            return False, "Speech output is disabled."

        self.interrupt()
        self._stop_event.clear()

        thread = threading.Thread(
            target=self._run_speech,
            args=(text, language),
            name="jarvis-speech",
            daemon=True,
        )
        with self._lock:
            self._thread = thread
        thread.start()
        return True, "Speech started."

    def synthesize_one_sentence(self, text, language=None):
        """Synthesize a single sentence and return (sample_rate, waveform) or None on failure."""
        synth_started = time.perf_counter()
        spoken_text = self._prepare_text_for_speech(text, preferred_language=language)
        if not spoken_text:
            return None

        arabic_preferred = self._is_arabic_preferred_text(spoken_text, preferred_language=language)
        backend = str(self._resolve_backend() or "auto").strip().lower()
        result = None

        if backend in {"auto", "hybrid"}:
            result = self._synthesize_elevenlabs(spoken_text)
            if result is None:
                fallback_language = "ar" if arabic_preferred else "en"
                result = self._synthesize_edge_tts(spoken_text, preferred_language=fallback_language)
        elif backend == "edge_tts":
            lang_hint = "ar" if arabic_preferred else "en"
            result = self._synthesize_edge_tts(spoken_text, preferred_language=lang_hint)

        synth_elapsed = time.perf_counter() - synth_started
        record_stage_timing(
            "tts_synth", synth_elapsed,
            lang="ar" if arabic_preferred else "en",
            backend=backend,
        )

        return result

    def _synthesize_elevenlabs(self, normalized_text):
        """Synthesize via ElevenLabs and return (sample_rate, waveform) or None."""
        if not bool(TTS_ELEVENLABS_ARABIC_ENABLED):
            return None
        if _elevenlabs_tts_on_cooldown():
            return None
        if not normalized_text:
            return None

        api_key = str(ELEVENLABS_API_KEY or "").strip()
        voice_id = self._voice.elevenlabs_voice_id or ""
        if ElevenLabs is None or not api_key or not voice_id:
            return None

        try:
            client = ElevenLabs(
                api_key=api_key,
                base_url=str(ELEVENLABS_BASE_URL or "https://api.elevenlabs.io").rstrip("/"),
                timeout=float(TTS_ELEVENLABS_TIMEOUT_SECONDS),
            )
        except Exception as exc:
            logger.warning("ElevenLabs client initialization failed: %s", exc)
            return None

        profile = self._voice
        voice_settings = {
            "stability": profile.stability,
            "similarity_boost": profile.similarity_boost,
            "style": profile.style,
            "use_speaker_boost": profile.use_speaker_boost,
        }

        convert_kwargs = {
            "voice_id": voice_id,
            "text": normalized_text,
            "model_id": str(TTS_ELEVENLABS_MODEL_ID or "eleven_multilingual_v2"),
            "output_format": "wav_24000",
            "optimize_streaming_latency": 1,
            "voice_settings": voice_settings,
        }

        try:
            response = client.text_to_speech.convert(**convert_kwargs)
        except Exception as exc:
            if "quota_exceeded" in str(exc) or "status_code: 401" in str(exc) or "http 401" in str(exc).lower():
                _set_elevenlabs_tts_cooldown(f"convert:{exc}")
            logger.warning("ElevenLabs TTS sentence synth failed (voice_id=%s): %s", voice_id, exc)
            return None

        try:
            audio_bytes = b"".join(response)
        except Exception as exc:
            if "quota_exceeded" in str(exc) or "status_code: 401" in str(exc) or "http 401" in str(exc).lower():
                _set_elevenlabs_tts_cooldown(f"stream:{exc}")
            logger.warning("ElevenLabs TTS stream read failed (voice_id=%s): %s", voice_id, exc)
            return None

        if not audio_bytes:
            return None

        return self._decode_edge_audio_bytes(audio_bytes)

    def _synthesize_edge_tts(self, normalized_text, *, preferred_language=None):
        """Synthesize via Edge-TTS and return (sample_rate, waveform) or None."""
        try:
            import edge_tts  # type: ignore
        except Exception:
            return None

        if not normalized_text:
            return None

        voice_candidates = self._edge_tts_voice_candidates(normalized_text, preferred_language=preferred_language)
        wants_arabic = self._is_arabic_preferred_text(normalized_text, preferred_language=preferred_language)
        profile = self._voice
        supports_output_format = self._edge_tts_supports_output_format(edge_tts)
        can_decode_compressed = self._can_decode_edge_compressed_stream()

        if not supports_output_format and not can_decode_compressed:
            return None

        async def _collect(voice_name):
            # edge-tts's Communicate always treats `text` as plain text to be
            # XML-escaped and spoken verbatim — it does not parse SSML passed
            # this way. Passing a <speak>/<prosody> wrapper here previously
            # caused the tag markup itself to be read aloud. Rate/pitch must
            # go through the dedicated constructor kwargs instead.
            edge_rate = (profile.rate_ar if wants_arabic else profile.rate_en) or "+0%"
            edge_pitch = (profile.pitch_ar if wants_arabic else profile.pitch_en) or ""
            supports_pitch = self._edge_tts_supports_parameter(edge_tts, "pitch")
            kwargs = {"voice": voice_name, "rate": edge_rate}
            if supports_pitch and edge_pitch:
                kwargs["pitch"] = edge_pitch
            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"
            speaker = edge_tts.Communicate(normalized_text, **kwargs)

            chunks = []
            stream = speaker.stream()
            try:
                async for event in stream:
                    if self._stop_event.is_set():
                        break
                    if str(event.get("type") or "").lower() != "audio":
                        continue
                    data = event.get("data")
                    if data:
                        chunks.append(bytes(data))
            finally:
                close_stream = getattr(stream, "aclose", None)
                if close_stream is not None:
                    await close_stream()
            return b"".join(chunks)

        for voice_name in voice_candidates:
            try:
                audio_bytes = self._run_async(_collect(voice_name))
                if self._stop_event.is_set():
                    return None
                if not audio_bytes:
                    continue
                decoded = self._decode_edge_audio_bytes(audio_bytes)
                if decoded is not None:
                    return decoded
            except Exception as exc:
                if self._is_edge_voice_unavailable_error(str(exc)):
                    self._remember_edge_voice_unavailable(voice_name)
        return None

    def _is_paragraph_boundary(self, sentence_text):
        """Return True if the sentence ends with a period/fullstop followed by implicit paragraph break."""
        text = str(sentence_text or "").rstrip()
        return text.endswith((".","。", "؟", "?", "!", "\n"))

    def speak_sentence_queue(self, sentences_iterator, language=None):
        if sentences_iterator is None:
            return False, "No sentences provided."

        if not self.is_enabled():
            return False, "Speech output is disabled."

        self.interrupt()
        self._stop_event.clear()

        if not TTS_SENTENCE_STREAMING_ENABLED:
            return self._speak_sentence_queue_sequential(sentences_iterator, language)

        def _run_streaming_queue():
            import numpy as np  # type: ignore

            queue_started = time.perf_counter()
            sentence_gap_samples = 0
            paragraph_gap_samples = 0
            playback_rate = 24000
            first_word_recorded = False
            sentence_count = 0

            pool = _cf.ThreadPoolExecutor(
                max_workers=TTS_SENTENCE_SYNTH_WORKERS,
                thread_name_prefix="tts-synth",
            )
            try:
                futures = []
                sentence_texts = []
                first_played = False

                def _submit_sentence(text):
                    utterance = " ".join(str(text or "").split()).strip()
                    if not utterance or len(utterance) < TTS_SENTENCE_FIRST_FLUSH_MIN_CHARS:
                        if utterance:
                            sentence_texts.append(utterance)
                            futures.append(pool.submit(self.synthesize_one_sentence, utterance, language))
                        return
                    sentence_texts.append(utterance)
                    futures.append(pool.submit(self.synthesize_one_sentence, utterance, language))

                drain_idx = 0

                for sentence in sentences_iterator:
                    if self._stop_event.is_set():
                        break
                    _submit_sentence(sentence)

                    while drain_idx < len(futures) and futures[drain_idx].done():
                        result = futures[drain_idx].result()
                        if self._stop_event.is_set():
                            break
                        if result is not None:
                            sr, waveform = result
                            if not first_played:
                                playback_rate = sr
                                sentence_gap_samples = int(sr * TTS_SENTENCE_GAP_MS / 1000)
                                paragraph_gap_samples = int(sr * TTS_PARAGRAPH_GAP_MS / 1000)
                                first_played = True
                                if not first_word_recorded:
                                    first_word_recorded = True
                                    latency_tracker.record(
                                        "tts_first_word",
                                        time.perf_counter() - queue_started,
                                    )
                            else:
                                is_para = self._is_paragraph_boundary(
                                    sentence_texts[drain_idx - 1] if drain_idx > 0 else ""
                                )
                                gap = paragraph_gap_samples if is_para else sentence_gap_samples
                                if gap > 0:
                                    silence = np.zeros(gap, dtype=np.float32)
                                    self._play_waveform(silence, playback_rate, blocking=True)
                                    if self._stop_event.is_set():
                                        break

                            self._play_waveform(waveform, sr, blocking=True)
                            sentence_count += 1
                        drain_idx += 1

                while drain_idx < len(futures):
                    if self._stop_event.is_set():
                        break
                    result = futures[drain_idx].result()
                    if self._stop_event.is_set():
                        break
                    if result is not None:
                        sr, waveform = result
                        if not first_played:
                            playback_rate = sr
                            sentence_gap_samples = int(sr * TTS_SENTENCE_GAP_MS / 1000)
                            paragraph_gap_samples = int(sr * TTS_PARAGRAPH_GAP_MS / 1000)
                            first_played = True
                            if not first_word_recorded:
                                first_word_recorded = True
                                latency_tracker.record(
                                    "tts_first_word",
                                    time.perf_counter() - queue_started,
                                )
                        else:
                            is_para = self._is_paragraph_boundary(
                                sentence_texts[drain_idx - 1] if drain_idx > 0 else ""
                            )
                            gap = paragraph_gap_samples if is_para else sentence_gap_samples
                            if gap > 0:
                                silence = np.zeros(gap, dtype=np.float32)
                                self._play_waveform(silence, playback_rate, blocking=True)
                                if self._stop_event.is_set():
                                    break

                        self._play_waveform(waveform, sr, blocking=True)
                        sentence_count += 1
                    drain_idx += 1
            except Exception as exc:
                logger.error("Streaming sentence queue failed: %s", exc)
            finally:
                total_elapsed = time.perf_counter() - queue_started
                record_stage_timing("tts_playback", total_elapsed, backend="streaming")
                if sentence_count > 0:
                    backend = str(self._resolve_backend() or "auto").strip().lower()
                    lang = str(language or "unknown").strip().lower()
                    logger.info(
                        "tts engine=%s profile=%s lang=%s sentences=%d playback=%.2fs",
                        backend,
                        self._voice.name,
                        lang,
                        sentence_count,
                        total_elapsed,
                    )
                pool.shutdown(wait=False)
                with self._lock:
                    if (
                        self._queue_thread
                        and self._queue_thread.ident == threading.current_thread().ident
                    ):
                        self._queue_thread = None

        thread = threading.Thread(
            target=_run_streaming_queue,
            name="jarvis-speech-queue",
            daemon=True,
        )
        with self._lock:
            self._queue_thread = thread
        thread.start()
        return True, "Speech queue started."

    def _speak_sentence_queue_sequential(self, sentences_iterator, language=None):
        """Fallback: synthesize and play each sentence one at a time (pre-Phase 2 behavior)."""
        def _run_sentence_queue():
            try:
                for sentence in sentences_iterator:
                    if self._stop_event.is_set():
                        break
                    utterance = " ".join(str(sentence or "").split()).strip()
                    if not utterance:
                        continue
                    try:
                        self._run_speech(utterance, language=language)
                    except Exception as exc:
                        logger.error("Sentence queue speech failed: %s", exc)
                        if self._stop_event.is_set():
                            break
            finally:
                with self._lock:
                    if (
                        self._queue_thread
                        and self._queue_thread.ident == threading.current_thread().ident
                    ):
                        self._queue_thread = None

        thread = threading.Thread(
            target=_run_sentence_queue,
            name="jarvis-speech-queue",
            daemon=True,
        )
        with self._lock:
            self._queue_thread = thread
        thread.start()
        return True, "Speech queue started."

    def _resolve_backend(self):
        return self.get_backend()

    def _prepare_text_for_speech(self, text, *, preferred_language=None):
        from core.tts_prosody import polish_for_voice

        rewrite_started = time.perf_counter()

        normalized = " ".join(str(text or "").split()).strip()
        if not normalized:
            return normalized

        is_arabic = self._is_arabic_preferred_text(normalized, preferred_language=preferred_language)

        if is_arabic:
            dialect = str(TTS_ARABIC_SPOKEN_DIALECT or "egyptian").strip().lower()
            if dialect == "egyptian" and bool(TTS_EGYPTIAN_COLLOQUIAL_REWRITE):
                rewritten = _rewrite_to_egyptian_colloquial(normalized)
                if rewritten and rewritten != normalized:
                    logger.info("Applied Egyptian colloquial rewrite for Arabic TTS utterance")
                normalized = rewritten or normalized

        normalized = polish_for_voice(normalized, language="ar" if is_arabic else "en")

        rewrite_elapsed = time.perf_counter() - rewrite_started
        record_stage_timing("tts_rewrite", rewrite_elapsed, lang="ar" if is_arabic else "en")

        return normalized

    def _probe_edge_tts_environment(self):
        info = {
            "available": False,
            "voice": self._voice.edge_voice_en,
            "supports_output_format": False,
            "compressed_decode_available": False,
            "error": "",
        }
        try:
            import edge_tts  # type: ignore
            import numpy  # type: ignore
            import sounddevice  # type: ignore

            _ = edge_tts, numpy, sounddevice

            supports_output_format = bool(self._edge_tts_supports_output_format(edge_tts))
            compressed_decode_available = bool(self._can_decode_edge_compressed_stream())
            info["supports_output_format"] = supports_output_format
            info["compressed_decode_available"] = compressed_decode_available
            info["available"] = bool(supports_output_format or compressed_decode_available)
            if not info["available"]:
                info["error"] = (
                    "edge_tts stream decode unavailable: install soundfile or upgrade edge_tts "
                    "to a version that supports output_format."
                )
        except Exception as exc:
            info["error"] = str(exc)
        return info

    def _probe_elevenlabs_environment(self):
        api_key = str(ELEVENLABS_API_KEY or "").strip()
        voice_id = self._voice.elevenlabs_voice_id or ""
        model_id = str(TTS_ELEVENLABS_MODEL_ID or "").strip()
        enabled = bool(TTS_ELEVENLABS_ARABIC_ENABLED)
        sdk_available = ElevenLabs is not None
        available = bool(enabled and api_key and voice_id and sdk_available)
        return {
            "enabled": enabled,
            "api_key_configured": bool(api_key),
            "voice_id": voice_id,
            "model_id": model_id,
            "sdk_available": sdk_available,
            "available_for_arabic": available,
            "available_for_speech": available,
        }

    def run_voice_diagnostic(self):
        phrase = "Jarvis voice diagnostic. If you can hear this, text to speech output is working."
        requested_backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        edge_info = self._probe_edge_tts_environment()
        elevenlabs_info = self._probe_elevenlabs_environment()

        active_backend = requested_backend
        if requested_backend == "auto":
            active_backend = "hybrid"

        if active_backend == "edge_tts":
            device_label = edge_info.get("voice") or "edge_tts_voice"
        elif active_backend == "hybrid":
            if elevenlabs_info.get("available_for_speech"):
                device_label = "elevenlabs_plus_edge"
            else:
                device_label = edge_info.get("voice") or "hybrid_fallback"
        else:
            device_label = "console_output"

        spoke_ok, spoke_message = self.speak_async(phrase)

        lines = [
            "Voice Diagnostic",
            f"diagnostic_phrase: {phrase}",
            f"speech_enabled: {self.is_enabled()}",
            f"requested_backend: {requested_backend}",
            f"active_backend: {active_backend}",
            f"voice_quality_mode: {quality_mode}",
            f"output_device: {device_label}",
            f"edge_tts_available: {edge_info.get('available')}",
            f"edge_tts_voice: {edge_info.get('voice')}",
            f"edge_tts_supports_output_format: {edge_info.get('supports_output_format')}",
            f"edge_tts_compressed_decode_available: {edge_info.get('compressed_decode_available')}",
            f"elevenlabs_speech_enabled: {elevenlabs_info.get('enabled')}",
            f"elevenlabs_api_key_configured: {elevenlabs_info.get('api_key_configured')}",
            f"elevenlabs_sdk_available: {elevenlabs_info.get('sdk_available')}",
            f"elevenlabs_voice_id: {elevenlabs_info.get('voice_id') or 'not_set'}",
            f"elevenlabs_model_id: {elevenlabs_info.get('model_id') or 'not_set'}",
            f"elevenlabs_available_for_speech: {elevenlabs_info.get('available_for_speech')}",
            f"speech_attempt: {spoke_message}",
        ]
        if edge_info.get("error"):
            lines.append(f"edge_tts_error: {edge_info.get('error')}")

        meta = {
            "requested_backend": requested_backend,
            "active_backend": active_backend,
            "voice_quality_mode": quality_mode,
            "output_device": device_label,
            "speech_attempt_ok": bool(spoke_ok),
            "edge_tts_available": bool(edge_info.get("available")),
            "edge_tts_voice": edge_info.get("voice"),
            "edge_tts_supports_output_format": bool(edge_info.get("supports_output_format")),
            "edge_tts_compressed_decode_available": bool(edge_info.get("compressed_decode_available")),
            "elevenlabs_speech_enabled": bool(elevenlabs_info.get("enabled")),
            "elevenlabs_api_key_configured": bool(elevenlabs_info.get("api_key_configured")),
            "elevenlabs_sdk_available": bool(elevenlabs_info.get("sdk_available")),
            "elevenlabs_voice_id": elevenlabs_info.get("voice_id") or "",
            "elevenlabs_model_id": elevenlabs_info.get("model_id") or "",
            "elevenlabs_available_for_speech": bool(elevenlabs_info.get("available_for_speech")),
            # Backward-compatible aliases for existing diagnostics / tooling.
            "elevenlabs_arabic_enabled": bool(elevenlabs_info.get("enabled")),
            "elevenlabs_arabic_voice_id": elevenlabs_info.get("voice_id") or "",
            "elevenlabs_arabic_model_id": elevenlabs_info.get("model_id") or "",
            "elevenlabs_available_for_arabic": bool(elevenlabs_info.get("available_for_speech")),
        }
        return True, "\n".join(lines), meta

    def _prewarm_edge_tts(self, *, preferred_language=None):
        try:
            import edge_tts  # type: ignore
        except Exception as exc:
            should_log = False
            with self._lock:
                if not self._edge_tts_unavailable_logged:
                    self._edge_tts_unavailable_logged = True
                    should_log = True
            if should_log:
                logger.warning("Edge-TTS dependencies unavailable: %s", exc)
            return False

        warmup_text = "Ready."
        voice_candidates = self._edge_tts_voice_candidates(warmup_text, preferred_language="en")
        if not voice_candidates:
            return False

        supports_output_format = self._edge_tts_supports_output_format(edge_tts)
        supports_pitch = self._edge_tts_supports_parameter(edge_tts, "pitch")
        supports_volume = self._edge_tts_supports_parameter(edge_tts, "volume")
        can_decode_compressed = self._can_decode_edge_compressed_stream()
        if not supports_output_format and not can_decode_compressed:
            self._log_edge_tts_decode_warning_once(
                "Edge-TTS stream decode unavailable in this environment. Install soundfile or upgrade edge_tts."
            )
            return False

        edge_rate = self._voice.rate_en or "+0%"
        edge_pitch = ""
        edge_volume = ""

        async def _collect_audio_bytes(voice_name):
            kwargs = {
                "voice": voice_name,
                "rate": edge_rate,
            }
            if supports_pitch and edge_pitch:
                kwargs["pitch"] = edge_pitch
            if supports_volume and edge_volume:
                kwargs["volume"] = edge_volume

            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"
            speaker = edge_tts.Communicate(warmup_text, **kwargs)

            chunks = []
            stream = speaker.stream()
            try:
                async for event in stream:
                    if str(event.get("type") or "").lower() != "audio":
                        continue
                    data = event.get("data")
                    if data:
                        chunks.append(bytes(data))
            finally:
                close_stream = getattr(stream, "aclose", None)
                if close_stream is not None:
                    await close_stream()
            return b"".join(chunks)

        last_error = ""
        first_voice = voice_candidates[0]
        for index, voice_name in enumerate(voice_candidates):
            if index > 0:
                logger.info("Edge-TTS prewarm fallback voice attempt: %s -> %s", first_voice, voice_name)
            try:
                audio_bytes = self._run_async(_collect_audio_bytes(voice_name))
                if not audio_bytes:
                    last_error = f"empty_audio:{voice_name}"
                    continue

                decoded = self._decode_edge_audio_bytes(audio_bytes)
                if decoded is None:
                    last_error = f"decode_unavailable:{voice_name}"
                    continue
                return True
            except Exception as exc:
                last_error = str(exc)
                if self._is_edge_voice_unavailable_error(last_error):
                    self._remember_edge_voice_unavailable(voice_name)

        if last_error:
            logger.debug("Edge-TTS prewarm failed after %s voice attempt(s): %s", len(voice_candidates), last_error)
        return False

    def prewarm(self, *, preferred_language=None):
        backend = str(self._resolve_backend() or "auto").strip().lower()

        if backend == "console":
            return False, "console"

        if backend == "edge_tts":
            if self._prewarm_edge_tts(preferred_language=preferred_language):
                return True, "edge_tts"
            return False, "edge_tts"

        if backend == "hybrid":
            if self._prewarm_edge_tts(preferred_language=preferred_language):
                return True, "edge_tts"
            return False, "hybrid"

        edge_ok = self._prewarm_edge_tts(preferred_language=preferred_language)
        if edge_ok:
            return True, "edge_tts"

        return False, "none"

    def _run_speech(self, text, language=None):
        started = time.perf_counter()
        success = True
        backend = str(self._resolve_backend() or "auto").strip().lower()
        quality_mode = self.get_quality_mode()
        style = persona_manager.get_speech_style()
        logger.info("Speech backend=%s quality=%s style=%s", backend, quality_mode, style)
        spoken_text = self._prepare_text_for_speech(text, preferred_language=language)
        arabic_preferred = self._is_arabic_preferred_text(spoken_text, preferred_language=language)

        try:
            if backend in {"auto", "hybrid"}:
                if self._speak_elevenlabs(spoken_text):
                    return
                logger.info("ElevenLabs TTS unavailable; falling back to Edge-TTS")
                fallback_language = "ar" if arabic_preferred else "en"
                if self._speak_edge_tts(spoken_text, preferred_language=fallback_language):
                    return
                logger.warning("Edge-TTS synthesis failed; using console fallback")
                self._speak_console(spoken_text, prefix="TTS fallback")
                return

            if backend == "edge_tts":
                if arabic_preferred:
                    if self._speak_edge_tts(spoken_text, preferred_language="ar"):
                        return
                    self._speak_console(spoken_text, prefix="Edge-TTS Arabic fallback")
                    return

                if self._speak_edge_tts(spoken_text, preferred_language="en"):
                    return
                self._speak_console(spoken_text, prefix="Edge-TTS fallback")
                return

            self._speak_console(spoken_text, prefix="TTS")
        except Exception:
            success = False
            raise
        finally:
            tts_elapsed = time.perf_counter() - started
            metrics.record_stage("tts", tts_elapsed, success=success)
            record_stage_timing("tts_playback", tts_elapsed, backend=backend)
            lang = "ar" if arabic_preferred else "en"
            logger.info(
                "tts engine=%s profile=%s lang=%s sentences=1 playback=%.2fs",
                backend,
                self._voice.name,
                lang,
                tts_elapsed,
            )
            with self._lock:
                self._process = None
                if self._thread and self._thread.ident == threading.current_thread().ident:
                    self._thread = None

    def _speak_console(self, text, prefix):
        words = text.split()
        if not words:
            return
        tuning = self.get_tuning_settings()
        delay = max(0.0, float(TTS_SIMULATED_CHAR_DELAY)) * float(tuning.get("pause_scale") or 1.0)
        print(f"[{prefix}]")
        for word in words:
            if self._stop_event.is_set():
                break
            print(word, end=" ", flush=True)
            time.sleep(max(0.01, len(word) * delay))
        print("")

    def _run_async(self, coroutine):
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop and running_loop.is_running():
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(coroutine)
            finally:
                loop.close()

        return asyncio.run(coroutine)

    def _edge_tts_supports_output_format(self, edge_tts_module):
        try:
            signature = inspect.signature(edge_tts_module.Communicate.__init__)
        except Exception:
            return False
        if "output_format" in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _edge_tts_supports_parameter(self, edge_tts_module, parameter_name):
        try:
            signature = inspect.signature(edge_tts_module.Communicate.__init__)
        except Exception:
            return False
        key = str(parameter_name or "").strip()
        if key in signature.parameters:
            return True
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )

    def _can_decode_edge_compressed_stream(self):
        try:
            import soundfile as _sf  # type: ignore

            _ = _sf
            return True
        except Exception:
            return False

    def _decode_edge_audio_bytes(self, audio_bytes):
        payload = bytes(audio_bytes or b"")
        if len(payload) < 8:
            return None

        header = payload[:4]
        if header in {b"RIFF", b"RIFX", b"RF64"}:
            try:
                with wave.open(io.BytesIO(payload), "rb") as handle:
                    sample_rate = int(handle.getframerate() or 0)
                    sample_width = int(handle.getsampwidth() or 0)
                    channels = int(handle.getnchannels() or 1)
                    frame_count = int(handle.getnframes() or 0)
                    frames = handle.readframes(frame_count)
            except Exception:
                return None

            if sample_rate <= 0 or not frames:
                return None

            import numpy as np  # type: ignore

            if sample_width == 1:
                waveform = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
                waveform = (waveform - 128.0) / 128.0
            elif sample_width == 2:
                waveform = np.frombuffer(frames, dtype=np.int16)
            elif sample_width == 4:
                waveform = np.frombuffer(frames, dtype=np.int32)
            else:
                return None

            if channels > 1:
                expected = int(waveform.size // channels) * channels
                if expected <= 0:
                    return None
                waveform = waveform[:expected].reshape(-1, channels)

            return sample_rate, waveform

        # Older edge-tts versions stream compressed audio (for example MP3).
        # Decode through soundfile when available.
        try:
            import numpy as np  # type: ignore
            import soundfile as sf  # type: ignore
        except Exception:
            return None

        try:
            waveform, sample_rate = sf.read(io.BytesIO(payload), dtype="float32")
        except Exception:
            return None

        if int(sample_rate or 0) <= 0:
            return None
        samples = np.asarray(waveform)
        if samples.size <= 0:
            return None
        return int(sample_rate), samples

    def _log_edge_tts_decode_warning_once(self, message):
        should_log = False
        with self._lock:
            if not self._edge_tts_decode_warning_logged:
                self._edge_tts_decode_warning_logged = True
                should_log = True
        if should_log:
            logger.warning(message)

    def _is_arabic_preferred_text(self, text, preferred_language=None):
        normalized_language = str(preferred_language or "").strip().lower()
        if normalized_language in {"ar", "arabic"}:
            return True
        if normalized_language in {"en", "english"}:
            return False

        arabic_letters = _count_arabic_letters(text)
        if arabic_letters <= 0:
            return False

        latin_letters = _count_latin_letters(text)
        if latin_letters <= 0:
            return True

        # Keep Arabic voice when Arabic script dominates, even with inline English fragments.
        return arabic_letters >= latin_letters

    def _is_edge_voice_unavailable_error(self, error_text):
        normalized = str(error_text or "").lower()
        if not normalized or "voice" not in normalized:
            return False
        return any(
            token in normalized
            for token in (
                "invalid",
                "not found",
                "unsupported",
                "unavailable",
                "unknown",
            )
        )

    def _remember_edge_voice_unavailable(self, voice_name):
        voice_key = str(voice_name or "").strip().lower()
        if not voice_key:
            return
        with self._lock:
            self._edge_tts_unsupported_voices.add(voice_key)

    def _edge_tts_voice_candidates(self, normalized_text, *, preferred_language=None):
        profile = self._voice
        with self._lock:
            unsupported = set(self._edge_tts_unsupported_voices)

        wants_arabic = self._is_arabic_preferred_text(
            normalized_text, preferred_language=preferred_language,
        )
        if wants_arabic:
            primary = profile.edge_voice_ar
            fallbacks = list(profile.edge_voice_ar_fallbacks)
        else:
            primary = profile.edge_voice_en
            fallbacks = list(profile.edge_voice_en_fallbacks)

        candidates = [primary] + fallbacks

        deduped = []
        seen = set()
        for candidate in candidates:
            voice_name = str(candidate or "").strip()
            if not voice_name:
                continue
            key = voice_name.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(voice_name)

        filtered = [voice for voice in deduped if voice.lower() not in unsupported]
        return filtered or deduped[:1]

    def _edge_tts_text_chunks(self, normalized_text, *, preferred_language=None):
        text = " ".join(str(normalized_text or "").split()).strip()
        if not text:
            return []

        has_arabic = _contains_arabic(text)
        has_latin = _contains_latin(text)
        if not (has_arabic and has_latin):
            return [
                {
                    "text": text,
                    "script": "arabic"
                    if self._is_arabic_preferred_text(text, preferred_language=preferred_language)
                    else "latin",
                }
            ]

        chunks = []
        for token in text.split():
            arabic_letters = _count_arabic_letters(token)
            latin_letters = _count_latin_letters(token)
            if arabic_letters and arabic_letters >= latin_letters:
                token_script = "arabic"
            elif latin_letters:
                token_script = "latin"
            else:
                token_script = "neutral"

            if not chunks:
                chunks.append({"script": token_script, "tokens": [token]})
                continue

            if token_script == "neutral":
                chunks[-1]["tokens"].append(token)
                continue

            if chunks[-1]["script"] == "neutral":
                chunks[-1]["script"] = token_script
                chunks[-1]["tokens"].append(token)
                continue

            if chunks[-1]["script"] == token_script:
                chunks[-1]["tokens"].append(token)
            else:
                chunks.append({"script": token_script, "tokens": [token]})

        default_script = (
            "arabic"
            if self._is_arabic_preferred_text(text, preferred_language=preferred_language)
            else "latin"
        )
        finalized = []
        for chunk in chunks:
            script = str(chunk.get("script") or "").strip().lower()
            if script not in {"arabic", "latin"}:
                script = default_script
            chunk_text = " ".join(chunk.get("tokens") or []).strip()
            if not chunk_text:
                continue
            if script == "latin":
                chunk_text = re.sub(r"(?<!\S)\u0627\u0644(?=[A-Za-z])", "", chunk_text)
                chunk_text = re.sub(r"(?<!\S)\u0648(?=[A-Za-z])", "", chunk_text)
                chunk_text = " ".join(chunk_text.split()).strip()
                if not chunk_text:
                    continue
            if finalized and finalized[-1]["script"] == script:
                finalized[-1]["text"] = f"{finalized[-1]['text']} {chunk_text}".strip()
            else:
                finalized.append({"script": script, "text": chunk_text})

        return finalized or [{"script": default_script, "text": text}]

    def _edge_tts_chunk_audio_profile(self, is_arabic_chunk):
        profile = self._voice
        if is_arabic_chunk:
            rate = profile.rate_ar or "+0%"
            pitch = profile.pitch_ar or ""
            volume = ""
        else:
            rate = profile.rate_en or "+0%"
            pitch = profile.pitch_en or ""
            volume = ""
        return rate, pitch, volume

    def _speak_edge_tts_mixed_chunks(
        self,
        normalized_text,
        edge_tts_module,
        supports_output_format,
        supports_pitch,
        supports_volume,
        can_decode_compressed,
        preferred_language=None,
    ):
        if not supports_output_format and not can_decode_compressed:
            return False

        chunks = self._edge_tts_text_chunks(normalized_text, preferred_language=preferred_language)
        if len(chunks) <= 1:
            return False

        shared_voice_candidates = self._edge_tts_voice_candidates(
            normalized_text,
            preferred_language=preferred_language,
        )
        shared_voice = shared_voice_candidates[0] if shared_voice_candidates else ""
        if not shared_voice:
            return False

        text_length = len(str(normalized_text or ""))
        if text_length < int(TTS_EDGE_MIXED_SCRIPT_MIN_TEXT_LENGTH):
            logger.debug(
                "Edge-TTS mixed chunk mode skipped: text_length=%s below min=%s",
                text_length,
                int(TTS_EDGE_MIXED_SCRIPT_MIN_TEXT_LENGTH),
            )
            return False

        if text_length > int(TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH):
            logger.info(
                "Edge-TTS mixed chunk mode skipped: text_length=%s exceeds max=%s",
                text_length,
                int(TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH),
            )
            return False

        if len(chunks) > int(TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS):
            logger.info(
                "Edge-TTS mixed chunk mode skipped: chunk_count=%s exceeds max=%s",
                len(chunks),
                int(TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS),
            )
            return False

        logger.info("Edge-TTS mixed-script chunk mode enabled (%s chunks)", len(chunks))

        async def _collect_audio_bytes(chunk_text, voice_name, *, chunk_is_arabic):
            chunk_rate, chunk_pitch, chunk_volume = self._edge_tts_chunk_audio_profile(chunk_is_arabic)
            kwargs = {
                "voice": voice_name,
                "rate": chunk_rate,
            }
            if chunk_is_arabic and supports_pitch and chunk_pitch:
                kwargs["pitch"] = chunk_pitch
            if chunk_is_arabic and supports_volume and chunk_volume:
                kwargs["volume"] = chunk_volume

            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"
                speaker = edge_tts_module.Communicate(chunk_text, **kwargs)
            else:
                speaker = edge_tts_module.Communicate(chunk_text, **kwargs)

            collected = []
            stream = speaker.stream()
            try:
                async for event in stream:
                    if self._stop_event.is_set():
                        break
                    if str(event.get("type") or "").lower() != "audio":
                        continue
                    data = event.get("data")
                    if data:
                        collected.append(bytes(data))
            finally:
                close_stream = getattr(stream, "aclose", None)
                if close_stream is not None:
                    await close_stream()
            return b"".join(collected)

        # Sentinel — returned by _synth_chunk when stop_event fired mid-synthesis.
        _SYNTH_STOP = object()

        def _synth_chunk(chunk):
            """Synthesize one mixed-script chunk; returns (sr, waveform), None (skip), False (fail), or _SYNTH_STOP."""
            chunk_text = str(chunk.get("text") or "").strip()
            if not chunk_text:
                return None
            chunk_is_arabic = str(chunk.get("script") or "").strip().lower() == "arabic"
            chunk_candidates = [shared_voice] + [v for v in shared_voice_candidates[1:] if v != shared_voice]
            first_voice = chunk_candidates[0] if chunk_candidates else ""
            last_error = ""
            for index, voice_name in enumerate(chunk_candidates):
                if index > 0:
                    logger.info("Edge-TTS chunk fallback voice attempt: %s -> %s", first_voice, voice_name)
                try:
                    audio_bytes = self._run_async(
                        _collect_audio_bytes(chunk_text, voice_name, chunk_is_arabic=chunk_is_arabic)
                    )
                    if self._stop_event.is_set():
                        return _SYNTH_STOP
                    if not audio_bytes:
                        last_error = f"empty_audio:{voice_name}"
                        continue
                    decoded = self._decode_edge_audio_bytes(audio_bytes)
                    if decoded is None:
                        self._log_edge_tts_decode_warning_once(
                            "Edge-TTS stream decode failed. Install soundfile or upgrade edge_tts for output_format support."
                        )
                        last_error = f"decode_unavailable:{voice_name}"
                        continue
                    return decoded  # (sample_rate, waveform)
                except Exception as exc:
                    last_error = str(exc)
                    if self._is_edge_voice_unavailable_error(last_error):
                        self._remember_edge_voice_unavailable(voice_name)
                    logger.warning("Edge-TTS chunk synthesis failed with voice '%s': %s", voice_name, exc)
            if last_error:
                logger.warning(
                    "Edge-TTS mixed chunk failed after %s voice attempt(s): %s",
                    len(chunk_candidates),
                    last_error,
                )
            return False

        # Double-buffer: pre-synthesize chunk[n+1] while chunk[n] is playing so
        # there is no synthesis gap between consecutive mixed-script chunks.
        chunks_list = list(chunks)
        if not chunks_list:
            return True

        with _cf.ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-prefetch") as _pool:
            next_future = _pool.submit(_synth_chunk, chunks_list[0])

            for i in range(len(chunks_list)):
                upcoming_future = (
                    _pool.submit(_synth_chunk, chunks_list[i + 1])
                    if i + 1 < len(chunks_list)
                    else None
                )

                result = next_future.result()
                next_future = upcoming_future

                if result is None:
                    continue  # empty-text chunk, skip
                if result is _SYNTH_STOP or self._stop_event.is_set():
                    return False
                if result is False:
                    return False

                sample_rate, waveform = result
                played = self._play_waveform(waveform, sample_rate, blocking=True)
                if not played:
                    return False

        return True

    def _normalize_audio_samples(self, waveform, sample_rate: int = 24000):
        import numpy as np  # type: ignore

        samples = np.asarray(waveform)
        if samples.ndim > 1:
            samples = np.mean(samples, axis=1)
        if samples.size == 0:
            return None

        if samples.dtype.kind in {"i", "u"}:
            info = np.iinfo(samples.dtype)
            peak_limit = float(max(abs(info.min), info.max)) or 1.0
            normalized = samples.astype(np.float32) / peak_limit
        else:
            normalized = samples.astype(np.float32, copy=False)
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
            peak = float(np.max(np.abs(normalized)))
            if peak > 1.0:
                normalized = normalized / peak

        normalized = np.clip(normalized, -1.0, 1.0)
        return self._apply_edge_fade(normalized, sample_rate=max(8000, int(sample_rate or 24000)))

    def _apply_edge_fade(self, samples, fade_ms: float = 8.0, sample_rate: int = 24000):
        """Ramp the first/last few ms to zero so playback start/stop doesn't click.

        Edge-TTS and ElevenLabs streams rarely start or end at a zero crossing, and
        sounddevice opens/closes its output stream per utterance, so an abrupt
        non-zero sample at either edge produces an audible pop. A short linear fade
        removes the discontinuity without being perceptible as speech distortion.
        """
        import numpy as np  # type: ignore

        fade_len = int(sample_rate * (fade_ms / 1000.0))
        fade_len = min(fade_len, samples.shape[0] // 2)
        if fade_len <= 1:
            return samples

        ramp = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        samples[:fade_len] *= ramp
        samples[-fade_len:] *= ramp[::-1]
        return samples

    def _is_effectively_silent(self, samples):
        import numpy as np  # type: ignore

        normalized = np.asarray(samples, dtype=np.float32).reshape(-1)
        if normalized.size == 0:
            return True

        peak = float(np.max(np.abs(normalized)))
        rms = float(np.sqrt(np.mean(np.square(normalized))))
        return peak < 0.003 and rms < 0.0008

    def _play_waveform(self, waveform, sample_rate, *, blocking=False):
        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:
            logger.warning("Waveform playback dependency unavailable: %s", exc)
            return False

        playback_rate = max(8000, int(sample_rate or 0))

        samples = self._normalize_audio_samples(waveform, sample_rate=playback_rate)
        if samples is None:
            logger.warning("Waveform playback skipped because synthesized audio was empty")
            return False
        if self._is_effectively_silent(samples):
            logger.warning("Synthesized audio is effectively silent; triggering fallback")
            return False

        try:
            sd.stop()
            if blocking:
                if self._stop_event.is_set():
                    sd.stop()
                    return False
                sd.play(samples, samplerate=playback_rate, blocking=True)
                return True

            sd.play(samples, samplerate=playback_rate, blocking=False)

            expected_seconds = float(samples.shape[0]) / float(max(1, playback_rate))
            playback_deadline = time.perf_counter() + max(1.0, expected_seconds + 2.0)
            while True:
                if self._stop_event.is_set():
                    sd.stop()
                    return False
                if time.perf_counter() >= playback_deadline:
                    logger.warning("TTS playback watchdog reached; forcing stream stop")
                    sd.stop()
                    break
                try:
                    stream = sd.get_stream()
                except Exception:
                    break
                if not stream or not getattr(stream, "active", False):
                    break
                time.sleep(0.05)
            return True
        except Exception as exc:
            logger.error("Waveform playback failed: %s", exc)
            try:
                sd.stop()
            except Exception:
                pass
            return False

    def _speak_elevenlabs(self, text):
        if not bool(TTS_ELEVENLABS_ARABIC_ENABLED):
            return False

        if _elevenlabs_tts_on_cooldown():
            return False

        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        api_key = str(ELEVENLABS_API_KEY or "").strip()
        voice_id = self._voice.elevenlabs_voice_id or ""
        if ElevenLabs is None:
            should_log = False
            with self._lock:
                if not self._elevenlabs_unavailable_logged:
                    self._elevenlabs_unavailable_logged = True
                    should_log = True
            if should_log:
                logger.warning("ElevenLabs SDK is unavailable; falling back to Edge-TTS.")
            return False

        if not api_key or not voice_id:
            should_log = False
            with self._lock:
                if not self._elevenlabs_unavailable_logged:
                    self._elevenlabs_unavailable_logged = True
                    should_log = True
            if should_log:
                logger.warning("ElevenLabs TTS is enabled but not fully configured (missing API key or voice id).")
            return False

        decoded = self._synthesize_elevenlabs(normalized_text)
        if decoded is None:
            return False

        sample_rate, waveform = decoded
        played = self._play_waveform(waveform, sample_rate)
        if not played:
            return False

        return True

    def _speak_edge_tts(self, text, *, preferred_language=None):
        try:
            import edge_tts  # type: ignore
        except Exception as exc:
            should_log = False
            with self._lock:
                if not self._edge_tts_unavailable_logged:
                    self._edge_tts_unavailable_logged = True
                    should_log = True
            if should_log:
                logger.warning("Edge-TTS dependencies unavailable: %s", exc)
            return False

        normalized_text = " ".join(str(text or "").split()).strip()
        if not normalized_text:
            return False

        voice_candidates = self._edge_tts_voice_candidates(normalized_text, preferred_language=preferred_language)
        wants_arabic = self._is_arabic_preferred_text(normalized_text, preferred_language=preferred_language)
        profile = self._voice
        edge_rate = (profile.rate_ar if wants_arabic else profile.rate_en) or "+0%"
        edge_pitch = (profile.pitch_ar if wants_arabic else profile.pitch_en) or ""
        edge_volume = ""
        supports_output_format = self._edge_tts_supports_output_format(edge_tts)
        supports_pitch = self._edge_tts_supports_parameter(edge_tts, "pitch")
        supports_volume = self._edge_tts_supports_parameter(edge_tts, "volume")
        can_decode_compressed = self._can_decode_edge_compressed_stream()

        if not supports_output_format and not can_decode_compressed:
            self._log_edge_tts_decode_warning_once(
                "Edge-TTS stream decode unavailable in this environment. Install soundfile or upgrade edge_tts."
            )
            return False

        if bool(TTS_EDGE_MIXED_SCRIPT_CHUNKING):
            mixed_chunk_ok = self._speak_edge_tts_mixed_chunks(
                normalized_text,
                edge_tts,
                supports_output_format,
                supports_pitch,
                supports_volume,
                can_decode_compressed,
                preferred_language=preferred_language,
            )
            if mixed_chunk_ok:
                return True

        async def _collect_audio_bytes(voice_name):
            # edge-tts's Communicate always treats `text` as plain text to be
            # XML-escaped and spoken verbatim — it does not parse SSML passed
            # this way. Passing a <speak>/<prosody> wrapper here previously
            # caused the tag markup itself to be read aloud. Rate/pitch must
            # go through the dedicated constructor kwargs instead.
            kwargs = {
                "voice": voice_name,
                "rate": edge_rate,
            }
            if supports_pitch and edge_pitch:
                kwargs["pitch"] = edge_pitch
            if supports_volume and edge_volume:
                kwargs["volume"] = edge_volume

            if supports_output_format:
                kwargs["output_format"] = "riff-24khz-16bit-mono-pcm"

            speaker = edge_tts.Communicate(normalized_text, **kwargs)

            chunks = []
            stream = speaker.stream()
            try:
                async for event in stream:
                    if self._stop_event.is_set():
                        break
                    if str(event.get("type") or "").lower() != "audio":
                        continue
                    data = event.get("data")
                    if data:
                        chunks.append(bytes(data))
            finally:
                close_stream = getattr(stream, "aclose", None)
                if close_stream is not None:
                    await close_stream()
            return b"".join(chunks)

        last_error = ""
        first_voice = voice_candidates[0] if voice_candidates else ""

        for index, voice_name in enumerate(voice_candidates):
            if index > 0:
                logger.info("Edge-TTS fallback voice attempt: %s -> %s", first_voice, voice_name)
            try:
                audio_bytes = self._run_async(_collect_audio_bytes(voice_name))
                if self._stop_event.is_set():
                    return False
                if not audio_bytes:
                    last_error = f"empty_audio:{voice_name}"
                    continue

                decoded = self._decode_edge_audio_bytes(audio_bytes)
                if decoded is None:
                    self._log_edge_tts_decode_warning_once(
                        "Edge-TTS stream decode failed. Install soundfile or upgrade edge_tts for output_format support."
                    )
                    last_error = f"decode_unavailable:{voice_name}"
                    continue

                sample_rate, waveform = decoded
                played = self._play_waveform(waveform, sample_rate)
                if not played:
                    last_error = f"playback_failed:{voice_name}"
                    continue
                return True
            except Exception as exc:
                last_error = str(exc)
                if self._is_edge_voice_unavailable_error(last_error):
                    self._remember_edge_voice_unavailable(voice_name)
                logger.warning("Edge-TTS synthesis failed with voice '%s': %s", voice_name, exc)

        if last_error:
            logger.warning("Edge-TTS synthesis failed after %s voice attempt(s): %s", len(voice_candidates), last_error)
        return False

speech_engine = SpeechEngine()
