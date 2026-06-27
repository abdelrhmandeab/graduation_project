import os
import getpass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


# Load .env from project root (next to core/) and let it override ambient vars
# so local workspace edits take effect consistently during development.
load_dotenv(PROJECT_ROOT / ".env", override=True)


def _env(key, default=""):
    """Read an environment variable with a fallback default."""
    return os.environ.get(key, default)


def _env_int(key, default):
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key, default):
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key, default):
    value = os.environ.get(key)
    if value is None:
        return bool(default)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _env_list(key, default_values):
    raw = os.environ.get(key)
    if raw is None:
        return tuple(default_values)
    text = str(raw).replace(";", ",")
    parts = [item.strip() for item in text.split(",") if item.strip()]
    if not parts:
        return tuple(default_values)
    return tuple(parts)


# Audio
SAMPLE_RATE = 16000
MAX_RECORD_DURATION = max(3.0, min(20.0, _env_float("JARVIS_MAX_RECORD_DURATION", 8.0)))
AUDIO_CHUNK_SIZE = 1024
VAD_ENERGY_THRESHOLD = _env_float("JARVIS_VAD_ENERGY_THRESHOLD", 0.014)
VAD_BACKEND = _env("JARVIS_VAD_BACKEND", "silero")  # "silero" | "energy"
VAD_SILERO_THRESHOLD = _env_float("JARVIS_VAD_SILERO_THRESHOLD", 0.5)
# Minimum intent confidence (0–1) before a partial transcript is eligible for
# early execution.  The system's assess_intent_confidence typically scores
# well-formed commands in the 0.80–0.88 range, so 0.82 is a practical floor
# that still rejects ambiguous partials while letting clear commands fire early.
EARLY_EXEC_CONFIDENCE_THRESHOLD = _env_float("JARVIS_EARLY_EXEC_CONFIDENCE_THRESHOLD", 0.78)
VAD_COMMAND_SILENCE_SECONDS = _env_float("JARVIS_VAD_COMMAND_SILENCE_SECONDS", 0.50)
VAD_CHAT_SILENCE_SECONDS = _env_float("JARVIS_VAD_CHAT_SILENCE_SECONDS", 1.50)
VAD_SILENCE_SECONDS = _env_float("JARVIS_VAD_SILENCE_SECONDS", VAD_COMMAND_SILENCE_SECONDS)
VAD_MIN_SPEECH_SECONDS = _env_float("JARVIS_VAD_MIN_SPEECH_SECONDS", 0.30)
VAD_PREROLL_SECONDS = 0.2
VAD_START_TIMEOUT_SECONDS = _env_float("JARVIS_VAD_START_TIMEOUT_SECONDS", 3.2)
REALTIME_MAX_PENDING_UTTERANCES = 1
REALTIME_DROP_WHEN_BUSY = True
REALTIME_BACKPRESSURE_POLL_SECONDS = 0.25
# Latency toggle: when mic VAD already detected speech, optionally skip the
# second file-based speech guard even for non-responsive audio UX profiles.
SPEECH_GUARD_SKIP_NON_RESPONSIVE_PROFILES = _env_bool(
    "JARVIS_SPEECH_GUARD_SKIP_NON_RESPONSIVE_PROFILES",
    False,
)

# Wake Word
WAKE_WORD_UNIFIED_ONNX_PATH = _env(
    "JARVIS_WAKE_WORD_UNIFIED_ONNX_PATH",
    "models/jarvis_unified/jarvis_unified.onnx",
).strip()
WAKE_WORD_THRESHOLD = max(-100.0, min(1.0, _env_float("JARVIS_WAKE_WORD_THRESHOLD", 0.45)))
WAKE_WORD_CONFIRM_FRAMES = max(
    1,
    _env_int("JARVIS_WAKE_WORD_CONFIRM_FRAMES", 1),
)
WAKE_WORD_CHUNK_SIZE = max(320, _env_int("JARVIS_WAKE_WORD_CHUNK_SIZE", 1280))
WAKE_WORD_INPUT_DEVICE = _env("JARVIS_WAKE_WORD_INPUT_DEVICE", "").strip() or None
WAKE_WORD_AUDIO_GAIN = max(0.5, min(3.0, _env_float("JARVIS_WAKE_WORD_AUDIO_GAIN", 1.4)))
WAKE_WORD_MIN_RMS = max(0.0, _env_float("JARVIS_WAKE_WORD_MIN_RMS", 0.015))
WAKE_WORD_USER_SPEAKER_ID = _env(
    "JARVIS_WAKE_WORD_SPEAKER_ID",
    getpass.getuser(),
).strip()
WAKE_WORD_USER_SAMPLES_DIR = _env(
    "JARVIS_WAKE_WORD_USER_SAMPLES_DIR",
    _project_path("data", "wake_samples", "user_positive"),
).strip()
WAKE_WORD_SCORE_DEBUG = _env_bool("JARVIS_WAKE_WORD_SCORE_DEBUG", False)
WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS = max(
    1.0,
    _env_float("JARVIS_WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS", 10.0),
)
WAKE_WORD_DETECTION_COOLDOWN_SECONDS = max(
    0.2,
    _env_float("JARVIS_WAKE_WORD_DETECTION_COOLDOWN_SECONDS", 1.5),
)
# Adaptive wake-word retraining: accumulate confirmed detections and
# periodically retrain the ONNX model in the background.
ADAPTIVE_WAKE_ENABLED = _env_bool("JARVIS_ADAPTIVE_WAKE_ENABLED", True)
ADAPTIVE_WAKE_MIN_CONFIRMED = max(5, _env_int("JARVIS_ADAPTIVE_WAKE_MIN_CONFIRMED", 15))
ADAPTIVE_WAKE_RETRAIN_INTERVAL_SECONDS = max(
    300.0,
    _env_float("JARVIS_ADAPTIVE_WAKE_RETRAIN_INTERVAL_SECONDS", 3600.0),
)
ADAPTIVE_WAKE_EPOCHS = max(3, _env_int("JARVIS_ADAPTIVE_WAKE_EPOCHS", 10))
ADAPTIVE_WAKE_MIN_VAL_ACC = max(0.80, min(1.0, _env_float("JARVIS_ADAPTIVE_WAKE_MIN_VAL_ACC", 0.95)))
ADAPTIVE_WAKE_CONFIRMED_DIR = _env(
    "JARVIS_ADAPTIVE_WAKE_CONFIRMED_DIR",
    _project_path("data", "wake_samples", "confirmed_positive"),
).strip()
ADAPTIVE_WAKE_FALSE_POSITIVE_DIR = _env(
    "JARVIS_ADAPTIVE_WAKE_FALSE_POSITIVE_DIR",
    _project_path("data", "wake_samples", "confirmed_negative"),
).strip()

# Legacy language modes all map to the single bilingual model.
WAKE_WORD_MODE = "unified"
WAKE_WORD_DEPRECATED_KEYS = tuple(
    key
    for key in (
        "JARVIS_WAKE_WORD",
        "JARVIS_WAKE_WORD_AR_ONNX_PATH",
        "JARVIS_WAKE_WORD_AR_ENABLED",
        "JARVIS_WAKE_WORD_MODE",
        "JARVIS_WAKE_MODE",
        "JARVIS_WAKE_WORD_EN_THRESHOLD",
        "JARVIS_WAKE_WORD_AR_THRESHOLD",
    )
    if key in os.environ
)

# STT
STT_BACKEND = _env(
    "JARVIS_STT_BACKEND",
    "hybrid_elevenlabs",
)  # hybrid_elevenlabs | faster_whisper
if STT_BACKEND not in {"hybrid_elevenlabs", "faster_whisper"}:
    STT_BACKEND = "hybrid_elevenlabs"

ELEVENLABS_BASE_URL = _env("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").strip() or "https://api.elevenlabs.io"
ELEVENLABS_API_KEY = _env("ELEVENLABS_API_KEY", "").strip()

# Hard language lock: STT is confined to English + Egyptian Arabic only.
STT_LANGUAGE_LOCK = _env_bool("JARVIS_STT_LANGUAGE_LOCK", True)
STT_FORBID_OTHER_LANGUAGES = _env_bool("JARVIS_STT_FORBID_OTHER_LANGUAGES", True)
STT_VALIDATION_DOMINANT_SCRIPT_MIN = max(
    0.50,
    _env_float("JARVIS_STT_VALIDATION_DOMINANT_SCRIPT_MIN", 0.70),
)
STT_AR_INITIAL_PROMPT = _env("JARVIS_STT_AR_INITIAL_PROMPT", "محادثة بالعامية المصرية:")
STT_EN_INITIAL_PROMPT = _env("JARVIS_STT_EN_INITIAL_PROMPT", "")
STT_RETRY_OPPOSITE_LANGUAGE = _env_bool("JARVIS_STT_RETRY_OPPOSITE_LANGUAGE", True)
# Default language hint passed to STT. Prefer "auto"; when locked, "auto" is
# resolved to ar/en before any heavy decode and Whisper never receives language=None.
STT_LANGUAGE_HINT = str(_env("JARVIS_STT_LANGUAGE_HINT", "auto")).strip().lower() or "auto"

# Minimum seconds of captured speech before emitting a partial transcript.
# Lowering this value makes partials appear sooner but may increase noisy/unstable fragments.
STT_PARTIAL_MIN_SECONDS = max(0.15, _env_float("JARVIS_STT_PARTIAL_MIN_SECONDS", 0.35))
# Maximum recent-audio window (seconds) used for partial transcription.
# Smaller windows reduce partial latency and CPU usage; larger windows improve stability.
STT_PARTIAL_WINDOW_SECONDS = max(0.6, _env_float("JARVIS_STT_PARTIAL_WINDOW_SECONDS", 1.6))
# Minimum spacing (seconds) between partial transcription attempts.
STT_PARTIAL_INTERVAL_SECONDS = max(0.2, _env_float("JARVIS_STT_PARTIAL_INTERVAL_SECONDS", 0.35))
# Whisper model used for partial transcriptions (defaults to tiny for speed).
STT_PARTIAL_WHISPER_MODEL = _env("JARVIS_STT_PARTIAL_WHISPER_MODEL", "auto").strip() or "auto"

STT_ELEVENLABS_ENABLED = _env_bool("JARVIS_STT_ELEVENLABS_ENABLED", True)
STT_ELEVENLABS_STT_MODEL = _env("JARVIS_STT_ELEVENLABS_MODEL", "scribe_v2").strip() or "scribe_v2"
STT_ELEVENLABS_CONNECT_TIMEOUT_SECONDS = max(
    0.5,
    _env_float("JARVIS_STT_ELEVENLABS_CONNECT_TIMEOUT_SECONDS", 2.0),
)
STT_ELEVENLABS_READ_TIMEOUT_SECONDS = max(
    3.0,
    _env_float("JARVIS_STT_ELEVENLABS_READ_TIMEOUT_SECONDS", 15.0),
)
STT_ELEVENLABS_HTTP2 = _env_bool("JARVIS_STT_ELEVENLABS_HTTP2", True)
STT_ELEVENLABS_COOLDOWN_SECONDS = max(60, _env_int("JARVIS_STT_ELEVENLABS_COOLDOWN_SECONDS", 1800))
STT_MAX_AUDIO_SECONDS = max(3, _env_int("JARVIS_STT_MAX_AUDIO_SECONDS", 12))
STT_CLOUD_RACE_LANGUAGES = _env_bool("JARVIS_STT_CLOUD_RACE_LANGUAGES", False)
STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS = max(2, _env_int("JARVIS_STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS", 5))

# Local fallback backend settings.
WHISPER_MODEL = _env("JARVIS_WHISPER_MODEL", "auto").strip() or "auto"
WHISPER_COMPUTE_TYPE = _env("JARVIS_WHISPER_COMPUTE_TYPE", "auto").strip().lower() or "auto"
WHISPER_DEVICE = _env("JARVIS_WHISPER_DEVICE", "auto").strip().lower() or "auto"
STT_BEAM_SIZE_SHORT = max(1, _env_int("JARVIS_STT_BEAM_SIZE_SHORT", 1))
STT_BEAM_SIZE_LONG = max(1, _env_int("JARVIS_STT_BEAM_SIZE_LONG", 5))
STT_BEAM_SIZE_SHORT_THRESHOLD_SECONDS = max(
    0.5,
    _env_float("JARVIS_STT_BEAM_SIZE_SHORT_THRESHOLD_SECONDS", 2.0),
)
STT_NO_SPEECH_THRESHOLD = max(0.0, min(1.0, _env_float("JARVIS_STT_NO_SPEECH_THRESHOLD", 0.70)))
STT_MIN_AUDIO_RMS = max(0.0, _env_float("JARVIS_STT_MIN_AUDIO_RMS", 0.005))

# LLM
# LLM_BACKEND: "claude" uses Anthropic Claude API; "ollama" uses local Ollama (default).
LLM_BACKEND = str(_env("JARVIS_LLM_BACKEND", "ollama")).strip().lower()
if LLM_BACKEND not in {"claude", "ollama"}:
    LLM_BACKEND = "ollama"

# Claude API (used when LLM_BACKEND=claude)
ANTHROPIC_API_KEY = _env("ANTHROPIC_API_KEY", "").strip()
CLAUDE_DEFAULT_MODEL = _env("JARVIS_CLAUDE_DEFAULT_MODEL", "claude-haiku-4-5").strip()
CLAUDE_QUALITY_MODEL = _env("JARVIS_CLAUDE_QUALITY_MODEL", "claude-sonnet-4-6").strip()
CLAUDE_MAX_TOKENS_COMMAND = max(64, _env_int("JARVIS_CLAUDE_MAX_TOKENS_COMMAND", 256))
CLAUDE_MAX_TOKENS_QUESTION = max(128, _env_int("JARVIS_CLAUDE_MAX_TOKENS_QUESTION", 600))

LLM_MODEL = _env("JARVIS_LLM_MODEL", "qwen3:4b")
LLM_AUTO_SELECT_MODEL = _env_bool("JARVIS_LLM_AUTO_SELECT", True)
LLM_FALLBACK_MODELS = tuple(
    m.strip()
    for m in _env("JARVIS_LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
)
LLM_TIMEOUT_SECONDS = max(10, _env_int("JARVIS_LLM_TIMEOUT_SECONDS", 30))
# Context values are now autosize ceilings; per-turn ctx is selected from prompt size.
LLM_OLLAMA_NUM_CTX = max(512, _env_int("JARVIS_LLM_OLLAMA_NUM_CTX", 4096))
LLM_OLLAMA_BASE_URL = _env("JARVIS_LLM_OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
LLM_OLLAMA_AUTOSTART = _env_bool("JARVIS_LLM_OLLAMA_AUTOSTART", True)
LLM_OLLAMA_EXECUTABLE = _env("JARVIS_LLM_OLLAMA_EXECUTABLE", "ollama").strip() or "ollama"
LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS = max(
    3.0,
    _env_float("JARVIS_LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS", 25.0),
)
LLM_LIGHTWEIGHT_NUM_CTX = max(256, _env_int("JARVIS_LLM_LIGHTWEIGHT_NUM_CTX", 2048))
LLM_CTX_AUTOSIZE = _env_bool("JARVIS_LLM_CTX_AUTOSIZE", True)
LLM_FEWSHOT_MIN = max(0, _env_int("JARVIS_LLM_FEWSHOT_MIN", 2))
LLM_FEWSHOT_MAX = max(0, _env_int("JARVIS_LLM_FEWSHOT_MAX", 4))
LLM_RESPONSE_CACHE_ENABLED = _env_bool("JARVIS_LLM_RESPONSE_CACHE_ENABLED", True)
LLM_RESPONSE_CACHE_TTL_SECONDS = max(10, _env_int("JARVIS_LLM_RESPONSE_CACHE_TTL_SECONDS", 600))
LLM_RESPONSE_CACHE_TTL_FACTUAL_SECONDS = max(
    10, _env_int("JARVIS_LLM_RESPONSE_CACHE_TTL_FACTUAL_SECONDS", 3600)
)
LLM_RESPONSE_CACHE_TTL_OPINION_SECONDS = max(
    10, _env_int("JARVIS_LLM_RESPONSE_CACHE_TTL_OPINION_SECONDS", 300)
)
LLM_RESPONSE_CACHE_KEY_INCLUDES_PERSONA = _env_bool("JARVIS_LLM_RESPONSE_CACHE_KEY_INCLUDES_PERSONA", True)
LLM_RESPONSE_CACHE_MAX_SIZE = max(16, _env_int("JARVIS_LLM_RESPONSE_CACHE_MAX_SIZE", 256))
LLM_RESPONSE_CACHE_MAX_QUERY_WORDS = max(1, _env_int("JARVIS_LLM_RESPONSE_CACHE_MAX_QUERY_WORDS", 8))
LLM_APPEND_SOURCE_CITATIONS = True
LLM_DEFAULT_LANGUAGE = (_env("JARVIS_LLM_DEFAULT_LANGUAGE", "en") or "en").strip().lower()
if LLM_DEFAULT_LANGUAGE not in {"en", "ar"}:
    LLM_DEFAULT_LANGUAGE = "en"
LLM_LANG_PIN_ENABLED = _env_bool("JARVIS_LLM_LANG_PIN_ENABLED", True)
LLM_TEMPERATURE = max(0.0, min(1.5, _env_float("JARVIS_LLM_TEMPERATURE", 0.4)))
LLM_TOP_P = max(0.1, min(1.0, _env_float("JARVIS_LLM_TOP_P", 0.9)))
LLM_REPEAT_PENALTY = max(1.0, min(2.0, _env_float("JARVIS_LLM_REPEAT_PENALTY", 1.1)))
LLM_MAX_RESPONSE_TOKENS = max(32, _env_int("JARVIS_LLM_MAX_RESPONSE_TOKENS", 160))
LLM_STOP_TOKENS = [
    s for s in (_env("JARVIS_LLM_STOP_TOKENS", "\\n\\nUSER:,\\nUSER:,</s>,<think>").split(","))
    if s
]
# NLU (Phase 1 + Phase 2)
NLU_INTENT_ROUTING_ENABLED = _env_bool("JARVIS_NLU_INTENT_ROUTING_ENABLED", True)
NLU_ENTITY_EXTRACTION_ENABLED = _env_bool("JARVIS_NLU_ENTITY_EXTRACTION_ENABLED", True)
RESPONSE_SHAPER_ENABLED = _env_bool("JARVIS_RESPONSE_SHAPER_ENABLED", True)
NLU_INTENT_CONFIDENCE_THRESHOLD = _env_float("JARVIS_NLU_INTENT_CONFIDENCE_THRESHOLD", 0.75)
NLU_PARSER_FASTPATH_ENABLED = _env_bool("JARVIS_NLU_PARSER_FASTPATH_ENABLED", True)
NLU_PARSER_FASTPATH_CONFIDENCE_FLOOR = _env_float("JARVIS_NLU_PARSER_FASTPATH_CONFIDENCE_FLOOR", 0.55)
NLU_LLM_QUERY_EXTRACTION_ENABLED = _env_bool("JARVIS_NLU_LLM_QUERY_EXTRACTION_ENABLED", False)
SEMANTIC_ROUTER_ENABLED = _env_bool("JARVIS_SEMANTIC_ROUTER_ENABLED", True)
SEMANTIC_ROUTER_CONFIDENCE_THRESHOLD = _env_float("JARVIS_SEMANTIC_ROUTER_CONFIDENCE_THRESHOLD", 0.75)
NLU_INTENT_CACHE_ENABLED = _env_bool("JARVIS_NLU_INTENT_CACHE_ENABLED", True)
NLU_INTENT_CACHE_MAX_SIZE = _env_int("JARVIS_NLU_INTENT_CACHE_MAX_SIZE", 256)
NLU_INTENT_CACHE_TTL_SECONDS = _env_int("JARVIS_NLU_INTENT_CACHE_TTL_SECONDS", 600)

NLU_INTENT_THRESHOLD_BY_FAMILY = {
    "OS_APP_OPEN": _env_float("JARVIS_NLU_THRESHOLD_OS_APP_OPEN", 0.72),
    "OS_APP_CLOSE": _env_float("JARVIS_NLU_THRESHOLD_OS_APP_CLOSE", 0.74),
    "OS_FILE_SEARCH": _env_float("JARVIS_NLU_THRESHOLD_OS_FILE_SEARCH", 0.73),
    "OS_FILE_NAVIGATION": _env_float("JARVIS_NLU_THRESHOLD_OS_FILE_NAVIGATION", 0.74),
    "OS_SYSTEM_COMMAND": _env_float("JARVIS_NLU_THRESHOLD_OS_SYSTEM_COMMAND", 0.70),  # Lowered from 0.85 to allow system actions (WiFi, Bluetooth, media) without excessive clarification
    "JOB_QUEUE_COMMAND": _env_float("JARVIS_NLU_THRESHOLD_JOB_QUEUE_COMMAND", 0.70),
    "VOICE_COMMAND": _env_float("JARVIS_NLU_THRESHOLD_VOICE_COMMAND", 0.82),
}

# Phase 2 confidence/ranking tuning
ENTITY_CLARIFICATION_THRESHOLD_BY_INTENT = {
    "OS_APP_OPEN": _env_float("JARVIS_ENTITY_THRESHOLD_OS_APP_OPEN", 0.58),
    "OS_APP_CLOSE": _env_float("JARVIS_ENTITY_THRESHOLD_OS_APP_CLOSE", 0.60),
    "OS_FILE_SEARCH": _env_float("JARVIS_ENTITY_THRESHOLD_OS_FILE_SEARCH", 0.56),
    "OS_FILE_NAVIGATION": _env_float("JARVIS_ENTITY_THRESHOLD_OS_FILE_NAVIGATION", 0.56),
    "OS_SYSTEM_COMMAND": _env_float("JARVIS_ENTITY_THRESHOLD_OS_SYSTEM_COMMAND", 0.55),
    "JOB_QUEUE_COMMAND": _env_float("JARVIS_ENTITY_THRESHOLD_JOB_QUEUE_COMMAND", 0.52),
}

ENTITY_CLARIFICATION_LANGUAGE_ADJUSTMENT = {
    "en": _env_float("JARVIS_ENTITY_THRESHOLD_ADJUST_EN", 0.00),
    "ar": _env_float("JARVIS_ENTITY_THRESHOLD_ADJUST_AR", -0.02),
}

ENTITY_CLARIFICATION_MIXED_LANGUAGE_BONUS = _env_float(
    "JARVIS_ENTITY_THRESHOLD_MIXED_LANGUAGE_BONUS",
    0.03,
)

CLARIFICATION_PREFERENCE_MAX_AGE_SECONDS = _env_int(
    "JARVIS_CLARIFICATION_PREF_MAX_AGE_SECONDS",
    1209600,
)
CLARIFICATION_FALLBACK_AFTER_MISSES = _env_int("JARVIS_CLARIFICATION_FALLBACK_AFTER_MISSES", 2)

APP_RESOLUTION_USAGE_BOOST_PER_HIT = _env_float("JARVIS_APP_USAGE_BOOST_PER_HIT", 0.02)
APP_RESOLUTION_USAGE_BOOST_CAP = _env_float("JARVIS_APP_USAGE_BOOST_CAP", 0.12)
APP_RESOLUTION_RECENT_BONUS_SECONDS = _env_int("JARVIS_APP_RECENT_BONUS_SECONDS", 1800)
APP_RESOLUTION_RUNNING_BONUS_OPEN = _env_float("JARVIS_APP_RUNNING_BONUS_OPEN", 0.04)
APP_RESOLUTION_RUNNING_BONUS_CLOSE = _env_float("JARVIS_APP_RUNNING_BONUS_CLOSE", 0.16)
APP_RESOLUTION_AVAILABLE_BONUS = _env_float("JARVIS_APP_AVAILABLE_BONUS", 0.03)

# Speech / TTS
TTS_ENABLED = True
TTS_DEFAULT_BACKEND = _env("JARVIS_TTS_BACKEND", "hybrid")  # hybrid | edge_tts | auto | console
TTS_QUALITY_MODE = _env("JARVIS_TTS_QUALITY_MODE", "natural")  # natural | standard
TTS_EDGE_VOICE = _env("JARVIS_TTS_EDGE_VOICE", "en-US-AriaNeural")
TTS_EDGE_RATE = _env("JARVIS_TTS_EDGE_RATE", "+0%")
TTS_EDGE_ARABIC_VOICE = _env("JARVIS_TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural")
TTS_EDGE_ARABIC_VOICE_FALLBACKS = _env_list(
    "JARVIS_TTS_EDGE_ARABIC_VOICE_FALLBACKS",
    ("ar-EG-ShakirNeural", "ar-SA-HamedNeural"),
)
TTS_EDGE_ARABIC_RATE = _env("JARVIS_TTS_EDGE_ARABIC_RATE", "-4%")
TTS_EDGE_ARABIC_PITCH = _env("JARVIS_TTS_EDGE_ARABIC_PITCH", "-8Hz")
TTS_EDGE_ARABIC_VOLUME = _env("JARVIS_TTS_EDGE_ARABIC_VOLUME", "+4%")
TTS_EDGE_MIXED_SCRIPT_CHUNKING = _env_bool("JARVIS_TTS_EDGE_MIXED_SCRIPT_CHUNKING", True)
TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS = max(2, _env_int("JARVIS_TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS", 6))
TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH = max(80, _env_int("JARVIS_TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH", 220))
# Minimum text length (characters) required to enable mixed-script chunking.
# Short responses (shorter than this) will be synthesized in a single shot
# to avoid unnecessary chunk boundaries and playback gaps.
TTS_EDGE_MIXED_SCRIPT_MIN_TEXT_LENGTH = max(24, _env_int("JARVIS_TTS_EDGE_MIXED_SCRIPT_MIN_TEXT_LENGTH", 120))
TTS_ELEVENLABS_ARABIC_ENABLED = _env_bool("JARVIS_TTS_ELEVENLABS_ARABIC_ENABLED", False)
TTS_ELEVENLABS_ARABIC_VOICE_ID = _env("JARVIS_TTS_ELEVENLABS_ARABIC_VOICE_ID", "").strip()
TTS_ELEVENLABS_ARABIC_MODEL_ID = _env("JARVIS_TTS_ELEVENLABS_ARABIC_MODEL_ID", "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
# Arabic TTS backend: "edge" uses edge-tts ar-EG-SalmaNeural (default); "elevenlabs" uses ElevenLabs.
TTS_ARABIC_BACKEND = str(_env("JARVIS_TTS_ARABIC_BACKEND", "edge")).strip().lower()
if TTS_ARABIC_BACKEND not in {"edge", "elevenlabs"}:
    TTS_ARABIC_BACKEND = "edge"
TTS_ELEVENLABS_TIMEOUT_SECONDS = max(3.0, _env_float("JARVIS_TTS_ELEVENLABS_TIMEOUT_SECONDS", 15.0))
TTS_PREWARM_ENABLED = _env_bool("JARVIS_TTS_PREWARM_ENABLED", True)
STARTUP_PARSER_NLP_PREWARM_ENABLED = _env_bool("JARVIS_STARTUP_PARSER_NLP_PREWARM_ENABLED", True)
STARTUP_BACKGROUND_PREWARM_ENABLED = _env_bool("JARVIS_STARTUP_BACKGROUND_PREWARM_ENABLED", True)
PREWARM_SEMANTIC_ROUTER_BLOCKING = _env_bool("JARVIS_PREWARM_SEMANTIC_ROUTER_BLOCKING", False)
PREWARM_LLM_BLOCKING = _env_bool("JARVIS_PREWARM_LLM_BLOCKING", False)
GREETING_ENABLED = _env_bool("JARVIS_GREETING_ENABLED", True)
GREETING_LANGUAGE = _env("JARVIS_GREETING_LANGUAGE", "en").strip().lower()
GREETING_TEXT_EN = _env("JARVIS_GREETING_TEXT_EN", "Jarvis is online and ready.")
GREETING_TEXT_AR = _env("JARVIS_GREETING_TEXT_AR", "أهلاً بيك، جارفيس جاهز معاك.")
TTS_ARABIC_SPOKEN_DIALECT = str(_env("JARVIS_TTS_ARABIC_SPOKEN_DIALECT", "egyptian")).strip().lower()
if TTS_ARABIC_SPOKEN_DIALECT not in {"egyptian", "msa", "auto"}:
    TTS_ARABIC_SPOKEN_DIALECT = "egyptian"
TTS_EGYPTIAN_COLLOQUIAL_REWRITE = _env_bool("JARVIS_TTS_EGYPTIAN_COLLOQUIAL_REWRITE", True)
TTS_SIMULATED_CHAR_DELAY = 0.02
# Wake-word interrupt replaces the old VAD-based barge-in.  The wake word
# itself is now the only mechanism that can interrupt TTS or LLM streaming.
WAKE_INTERRUPT_ACK_SOUND = _env_bool("JARVIS_WAKE_INTERRUPT_ACK_SOUND", True)
WAKE_INTERRUPT_ACK_FREQ_HZ = max(200, min(2000, _env_int("JARVIS_WAKE_INTERRUPT_ACK_FREQ_HZ", 880)))
WAKE_INTERRUPT_ACK_DURATION_MS = max(40, min(400, _env_int("JARVIS_WAKE_INTERRUPT_ACK_DURATION_MS", 100)))
WAKE_INTERRUPT_BLOCKED_TONE_ENABLED = _env_bool("JARVIS_WAKE_INTERRUPT_BLOCKED_TONE_ENABLED", False)
WAKE_INTERRUPT_BLOCKED_TONE_FREQ_HZ = max(100, min(1000, _env_int("JARVIS_WAKE_INTERRUPT_BLOCKED_TONE_FREQ_HZ", 220)))
WAKE_INTERRUPT_BLOCKED_TONE_DURATION_MS = max(30, min(300, _env_int("JARVIS_WAKE_INTERRUPT_BLOCKED_TONE_DURATION_MS", 80)))


# Dialogue state machine — follow-up window
# When enabled, Jarvis enters a FOLLOW_UP window after each response.  Within
# that window the user can speak again without repeating the wake word.
FOLLOWUP_ENABLED = _env_bool("JARVIS_FOLLOWUP_ENABLED", True)
FOLLOWUP_WINDOW_SECONDS = max(3.0, _env_float("JARVIS_FOLLOWUP_WINDOW_SECONDS", 10.0))
# Optional: play a short chime when Jarvis enters the follow-up window so the
# user knows they can speak freely.  Disabled by default.
FOLLOWUP_CHIME_ENABLED = _env_bool("JARVIS_FOLLOWUP_CHIME_ENABLED", False)

# Persona
PERSONA_DEFAULT = "friendly"
PERSONA_PROFILE = (_env("JARVIS_PERSONA_PROFILE", "jarvis_classic") or "jarvis_classic").strip().lower()
if PERSONA_PROFILE not in {"jarvis_classic", "jarvis_warm", "custom"}:
    PERSONA_PROFILE = "jarvis_classic"
PERSONA_NAME = _env("JARVIS_PERSONA_NAME", "Jarvis").strip() or "Jarvis"
PERSONA_ADDRESSEE_EN = _env("JARVIS_PERSONA_ADDRESSEE_EN", "").strip()
PERSONA_ADDRESSEE_AR = _env("JARVIS_PERSONA_ADDRESSEE_AR", "").strip()
PERSONA_STYLE_EN = _env("JARVIS_PERSONA_STYLE_EN", "calm, dry, useful").strip() or "calm, dry, useful"
PERSONA_STYLE_AR = _env("JARVIS_PERSONA_STYLE_AR", "هادي، خفيف، عملي").strip() or "هادي، خفيف، عملي"
PERSONA_VOICE_LENGTH_EN = _env("JARVIS_PERSONA_VOICE_LENGTH_EN", "1-3 short sentences").strip() or "1-3 short sentences"
PERSONA_VOICE_LENGTH_AR = _env("JARVIS_PERSONA_VOICE_LENGTH_AR", "جملة لاتنين قصيرة").strip() or "جملة لاتنين قصيرة"
PERSONA_FORBIDDEN_EN = tuple(
    item.strip()
    for item in _env("JARVIS_PERSONA_FORBIDDEN_EN", "As an AI,as a language model,I'm just,I cannot").split(",")
    if item.strip()
)
PERSONA_FORBIDDEN_AR = tuple(
    item.strip()
    for item in _env("JARVIS_PERSONA_FORBIDDEN_AR", "بصفتي,كذكاء صناعي,يسعدني").split(",")
    if item.strip()
)
PERSONA_LENGTH_TARGET_ENABLED = _env_bool("JARVIS_PERSONA_LENGTH_TARGET_ENABLED", True)
TONE_ADAPTATION_ENABLED = _env_bool("JARVIS_TONE_ADAPTATION_ENABLED", True)
TONE_SENSITIVE_NEUTRAL_ENABLED = _env_bool("JARVIS_TONE_SENSITIVE_NEUTRAL_ENABLED", True)
RESPONSE_MODE_FEATURE_ENABLED = _env_bool("JARVIS_RESPONSE_MODE_FEATURE_ENABLED", True)
CODE_SWITCH_CONTINUITY_ENABLED = _env_bool("JARVIS_CODE_SWITCH_CONTINUITY_ENABLED", True)
CODE_SWITCH_CONTINUITY_WINDOW = max(2, _env_int("JARVIS_CODE_SWITCH_CONTINUITY_WINDOW", 6))
CODE_SWITCH_DOMINANT_RATIO = max(0.50, min(0.90, _env_float("JARVIS_CODE_SWITCH_DOMINANT_RATIO", 0.70)))

PERSONA_RESPONSE_MAX_WORDS = {
    "assistant": _env_int("JARVIS_PERSONA_MAX_WORDS_ASSISTANT", 48),
    "formal": _env_int("JARVIS_PERSONA_MAX_WORDS_FORMAL", 44),
    "casual": _env_int("JARVIS_PERSONA_MAX_WORDS_CASUAL", 56),
    "professional": _env_int("JARVIS_PERSONA_MAX_WORDS_PROFESSIONAL", 36),
    "friendly": _env_int("JARVIS_PERSONA_MAX_WORDS_FRIENDLY", 38),
    "brief": _env_int("JARVIS_PERSONA_MAX_WORDS_BRIEF", 16),
}

# Weather (Open-Meteo — free, no API key)
WEATHER_DEFAULT_LATITUDE = _env_float("JARVIS_WEATHER_LATITUDE", 30.04)
WEATHER_DEFAULT_LONGITUDE = _env_float("JARVIS_WEATHER_LONGITUDE", 31.24)
WEATHER_DEFAULT_CITY = _env("JARVIS_WEATHER_CITY", "Cairo")

# Web search
WEB_SEARCH_ENABLED = _env_bool("JARVIS_WEB_SEARCH_ENABLED", True)
WEB_SEARCH_MAX_RESULTS = max(1, _env_int("JARVIS_WEB_SEARCH_MAX_RESULTS", 3))
LIVE_DATA_FORCE_QUESTIONS = _env_bool("JARVIS_LIVE_DATA_FORCE_QUESTIONS", False)
VOICE_NORMALIZER_ENABLED = _env_bool("JARVIS_VOICE_NORMALIZER_ENABLED", True)
VOICE_NORMALIZER_PERSONA_SIGNATURE_PROB = max(
    0.0,
    min(1.0, _env_float("JARVIS_VOICE_NORMALIZER_PERSONA_SIGNATURE_PROB", 0.2)),
)
VOICE_NORMALIZER_MAX_SEARCH_RESULTS = max(1, _env_int("JARVIS_VOICE_NORMALIZER_MAX_SEARCH_RESULTS", 2))
VOICE_NORMALIZER_KEEP_URLS = _env_bool("JARVIS_VOICE_NORMALIZER_KEEP_URLS", False)
# Trusted domains promoted by the recency/quality scorer. Empty = no preference.
_DEFAULT_WEB_SEARCH_TRUSTED_DOMAINS = (
    "wikipedia.org",
    "bbc.com",
    "bbc.co.uk",
    "reuters.com",
    "apnews.com",
    "aljazeera.com",
    "aljazeera.net",
    "ahram.org.eg",
    "youm7.com",
    "masrawy.com",
    "elwatannews.com",
    "github.com",
    "stackoverflow.com",
    "python.org",
    "docs.python.org",
    "developer.mozilla.org",
    "msdn.microsoft.com",
    "learn.microsoft.com",
    "openmeteo.com",
    "metoffice.gov.uk",
)
WEB_SEARCH_TRUSTED_DOMAINS = _env_list(
    "JARVIS_WEB_SEARCH_TRUSTED_DOMAINS",
    _DEFAULT_WEB_SEARCH_TRUSTED_DOMAINS,
)
# Domains that should be filtered out unconditionally. Empty by default.
WEB_SEARCH_BLOCKED_DOMAINS = _env_list(
    "JARVIS_WEB_SEARCH_BLOCKED_DOMAINS",
    (),
)


# -----------------------------
# Feature flags (Phase 7)
# -----------------------------
# Toggle higher-level features for staged rollout and quick rollback during
# deployment. Flags can be overridden via environment variables prefixed with
# JARVIS_FEATURE_ for testing and gradual enablement.
FEATURE_FLAGS = {
    "NUMERIC_PARSING_ENABLED": _env_bool("JARVIS_FEATURE_NUMERIC_PARSING_ENABLED", True),
    "AUTO_APP_DISCOVERY_ENABLED": _env_bool("JARVIS_FEATURE_AUTO_APP_DISCOVERY_ENABLED", True),
    "MEDIA_DIRECT_DISPATCH_ENABLED": _env_bool("JARVIS_FEATURE_MEDIA_DIRECT_DISPATCH_ENABLED", True),
    "SYSTEM_VOLUME_CONTROL": _env_bool("JARVIS_FEATURE_SYSTEM_VOLUME_CONTROL", True),
}
# Score blending weights — all values clamped between 0 and 1.
WEB_SEARCH_TRUSTED_DOMAIN_BOOST = max(
    0.0, min(1.0, _env_float("JARVIS_WEB_SEARCH_TRUSTED_DOMAIN_BOOST", 0.35))
)
WEB_SEARCH_RECENCY_BOOST = max(
    0.0, min(1.0, _env_float("JARVIS_WEB_SEARCH_RECENCY_BOOST", 0.25))
)

# Offline knowledge base (Phase 4)
KB_ENABLED = True
KB_RETRIEVAL_ENABLED = True
KB_STORAGE_DIR = ".jarvis_kb"
KB_FAISS_INDEX_FILE = os.path.join(KB_STORAGE_DIR, "index.faiss")
KB_META_FILE = os.path.join(KB_STORAGE_DIR, "meta.json")
KB_SOURCE_STATE_FILE = os.path.join(KB_STORAGE_DIR, "sources.json")
KB_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
KB_EMBEDDING_DIM = 256
KB_CHUNK_SIZE = 600
KB_CHUNK_OVERLAP = 120
KB_TOP_K = max(1, _env_int("JARVIS_KB_TOP_K", 3))
KB_MAX_CONTEXT_CHARS = max(600, _env_int("JARVIS_KB_MAX_CONTEXT_CHARS", 1400))
KB_MIN_PROMPT_SCORE = 0.45
KB_MIN_SEMANTIC_ONLY_SCORE = 0.58
KB_RERANK_CANDIDATE_MULTIPLIER = 4
KB_LEXICAL_RERANK_WEIGHT = 0.35
KB_EMBEDDING_RERANK_WEIGHT = 0.65
KB_AUTO_SYNC_ENABLED = _env_bool("JARVIS_KB_AUTO_SYNC_ENABLED", False)
KB_AUTO_SYNC_INTERVAL_SECONDS = max(1.0, _env_float("JARVIS_KB_AUTO_SYNC_INTERVAL_SECONDS", 4.0))
KB_AUTO_SYNC_PATHS = _env_list("JARVIS_KB_AUTO_SYNC_PATHS", ())
KB_BLOCKED_CONTEXT_PATTERNS = (
    "ignore previous instruction",
    "ignore all previous instruction",
    "system prompt",
    "you are assistant",
    "assistant:",
    "developer:",
    "system:",
)

# Session memory
MEMORY_ENABLED = True
MEMORY_FILE = _project_path("jarvis_memory.json")
# Phase 2.8 — primary persistence is SQLite; the JSON file above is now used
# only for legacy migration on first launch and as a debug-export target.
MEMORY_DB_FILE = _env("JARVIS_MEMORY_DB_FILE", _project_path("jarvis_memory.db"))
MEMORY_BACKEND = str(_env("JARVIS_MEMORY_BACKEND", "sqlite")).strip().lower() or "sqlite"
if MEMORY_BACKEND not in {"sqlite", "json"}:
    MEMORY_BACKEND = "sqlite"
MEMORY_MAX_TURNS = 10
MEMORY_MAX_CONTEXT_CHARS = max(300, _env_int("JARVIS_MEMORY_MAX_CONTEXT_CHARS", 900))
# Phase 2.9 — persist last N language detections + explicit user preference
# across app restarts so STT language hinting survives reboots.
MEMORY_PERSIST_LANGUAGE_HISTORY = _env_bool("JARVIS_MEMORY_PERSIST_LANGUAGE_HISTORY", True)
MEMORY_LANGUAGE_HISTORY_PERSIST_LIMIT = max(
    1, _env_int("JARVIS_MEMORY_LANGUAGE_HISTORY_PERSIST_LIMIT", 3)
)
CLARIFICATION_PREFERENCE_HALF_LIFE_SECONDS = _env_int(
    "JARVIS_CLARIFICATION_PREFERENCE_HALF_LIFE_SECONDS",
    1209600,
)
CLARIFICATION_PREFERENCE_MIN_SCORE = _env_float("JARVIS_CLARIFICATION_PREFERENCE_MIN_SCORE", 0.34)
CLARIFICATION_CORRECTION_WINDOW_SECONDS = _env_int("JARVIS_CLARIFICATION_CORRECTION_WINDOW_SECONDS", 45)
FOLLOWUP_REFERENCE_MAX_AGE_SECONDS = _env_int("JARVIS_FOLLOWUP_REFERENCE_MAX_AGE_SECONDS", 1800)
FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS = _env_int(
    "JARVIS_FOLLOWUP_APP_REFERENCE_MAX_AGE_SECONDS",
    FOLLOWUP_REFERENCE_MAX_AGE_SECONDS,
)
FOLLOWUP_FILE_REFERENCE_MAX_AGE_SECONDS = _env_int(
    "JARVIS_FOLLOWUP_FILE_REFERENCE_MAX_AGE_SECONDS",
    FOLLOWUP_REFERENCE_MAX_AGE_SECONDS,
)
FOLLOWUP_PENDING_CONFIRMATION_MAX_AGE_SECONDS = _env_int(
    "JARVIS_FOLLOWUP_PENDING_CONFIRMATION_MAX_AGE_SECONDS",
    180,
)
FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS = _env_int("JARVIS_FOLLOWUP_APP_REFERENCE_HALF_LIFE_SECONDS", 900)
FOLLOWUP_FILE_REFERENCE_HALF_LIFE_SECONDS = _env_int("JARVIS_FOLLOWUP_FILE_REFERENCE_HALF_LIFE_SECONDS", 720)
FOLLOWUP_PENDING_CONFIRMATION_HALF_LIFE_SECONDS = _env_int(
    "JARVIS_FOLLOWUP_PENDING_CONFIRMATION_HALF_LIFE_SECONDS",
    75,
)
FOLLOWUP_REFERENCE_MIN_CONFIDENCE = _env_float("JARVIS_FOLLOWUP_REFERENCE_MIN_CONFIDENCE", 0.20)
FOLLOWUP_REFERENCE_CONFLICT_WINDOW_SECONDS = _env_float("JARVIS_FOLLOWUP_REFERENCE_CONFLICT_WINDOW_SECONDS", 0.0)
FOLLOWUP_DESTRUCTIVE_REFERENCE_MIN_CONFIDENCE = _env_float(
    "JARVIS_FOLLOWUP_DESTRUCTIVE_REFERENCE_MIN_CONFIDENCE",
    0.55,
)
FOLLOWUP_DESTRUCTIVE_REQUIRE_EXPLICIT_REFERENCE = _env_bool(
    "JARVIS_FOLLOWUP_DESTRUCTIVE_REQUIRE_EXPLICIT_REFERENCE",
    True,
)

# Observability / diagnostics
DOCTOR_STARTUP_ENABLED = True
DOCTOR_STARTUP_ASYNC = _env_bool("JARVIS_DOCTOR_STARTUP_ASYNC", True)
DOCTOR_SCHEDULE_INTERVAL_SECONDS = 900
DOCTOR_INCLUDE_MODEL_LOAD_CHECKS = False

# OS
MAX_FILE_RESULTS = 5
DEFAULT_WORKING_DIRECTORY = os.path.expanduser("~")
DEFAULT_SEARCH_PATH = DEFAULT_WORKING_DIRECTORY
POWERSHELL_EXECUTABLE = _env("JARVIS_POWERSHELL_EXECUTABLE", "powershell")
ACTION_LOG_FILE = _project_path("jarvis_actions.log")
ROLLBACK_DIR_NAME = ".jarvis_rollback"
CONFIRMATION_TIMEOUT_SECONDS = 45
CONFIRMATION_TOKEN_BYTES = max(4, min(32, _env_int("JARVIS_CONFIRMATION_TOKEN_BYTES", 8)))
CONFIRMATION_TOKEN_MIN_HEX_LEN = max(
    6,
    min(
        CONFIRMATION_TOKEN_BYTES * 2,
        _env_int("JARVIS_CONFIRMATION_TOKEN_MIN_HEX_LEN", 6),
    ),
)
CONFIRMATION_MAX_ATTEMPTS_PER_TOKEN = _env_int("JARVIS_CONFIRMATION_MAX_ATTEMPTS_PER_TOKEN", 6)
CONFIRMATION_LOCKOUT_SECONDS = _env_int("JARVIS_CONFIRMATION_LOCKOUT_SECONDS", 120)
ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS = False
ALLOW_PERMANENT_DELETE = False
STATE_DB_FILE = _project_path("jarvis_state.db")
SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE = True
SECOND_FACTOR_PIN = _env("JARVIS_SECOND_FACTOR_PIN", "")
SECOND_FACTOR_PASSPHRASE = _env("JARVIS_SECOND_FACTOR_PASSPHRASE", "")
SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN = _env_int("JARVIS_SECOND_FACTOR_MAX_ATTEMPTS_PER_TOKEN", 3)
SECOND_FACTOR_LOCKOUT_SECONDS = _env_int("JARVIS_SECOND_FACTOR_LOCKOUT_SECONDS", 90)
SEARCH_INDEX_DB_FILE = _project_path("jarvis_index.db")
SEARCH_INDEX_REFRESH_SECONDS = 60
SEARCH_INDEX_MAX_RESULTS = 20
JOB_MAX_RETRIES_DEFAULT = 1

POLICY_READ_ONLY_MODE = False
POLICY_DRY_RUN_MODE = _env_bool("JARVIS_POLICY_DRY_RUN_MODE", False)
POLICY_ALLOWED_PATHS = (
    os.path.abspath(DEFAULT_WORKING_DIRECTORY),
    os.path.abspath(os.path.join(DEFAULT_WORKING_DIRECTORY, "Desktop")),
)
POLICY_BLOCKED_PATH_PREFIXES = (
    r"C:\Windows\System32\config",
    r"C:\Windows\System32\drivers\etc",
)
POLICY_ALLOW_READ_OUTSIDE_ALLOWLIST = True
POLICY_COMMAND_PERMISSIONS = {
    "confirmation": True,
    "rollback": True,
    "file_search": True,
    "file_navigation": True,
    "file_write": True,
    "app_open": True,
    "app_close": True,
    "system_command": True,
    "metrics": True,
    "audit_log": True,
    "policy": True,
    "batch": True,
    "job_queue": True,
    "search_index": True,
    "persona": True,
    "speech": True,
    "knowledge_base": True,
    "memory": True,
    "observability": True,
}

POLICY_PROFILES = {
    "strict": {
        "read_only_mode": True,
        "dry_run_mode": True,
        "command_permissions": {
            "confirmation": True,
            "rollback": True,
            "file_search": True,
            "file_navigation": True,
            "file_write": False,
            "app_open": False,
            "app_close": False,
            "system_command": False,
            "metrics": True,
            "audit_log": True,
            "policy": True,
            "batch": False,
            "job_queue": False,
            "search_index": True,
            "persona": True,
            "speech": False,
            "knowledge_base": True,
            "memory": True,
            "observability": True,
        },
    },
    "normal": {
        "read_only_mode": POLICY_READ_ONLY_MODE,
        "dry_run_mode": POLICY_DRY_RUN_MODE,
        "command_permissions": POLICY_COMMAND_PERMISSIONS,
    },
    "demo": {
        "read_only_mode": True,
        "dry_run_mode": True,
        "command_permissions": {
            "confirmation": True,
            "rollback": True,
            "file_search": True,
            "file_navigation": True,
            "file_write": False,
            "app_open": True,
            "app_close": True,
            "system_command": False,
            "metrics": True,
            "audit_log": True,
            "policy": True,
            "batch": True,
            "job_queue": True,
            "search_index": True,
            "persona": True,
            "speech": False,
            "knowledge_base": True,
            "memory": True,
            "observability": True,
        },
    },
}

# Streaming sentence-splitter
SENTENCE_BUFFER_EN_SOFT_WORDS = max(1, _env_int("JARVIS_SENTENCE_BUFFER_EN_SOFT_WORDS", 7))
SENTENCE_BUFFER_EN_HARD_WORDS = max(
    SENTENCE_BUFFER_EN_SOFT_WORDS,
    _env_int("JARVIS_SENTENCE_BUFFER_EN_HARD_WORDS", 15),
)
SENTENCE_BUFFER_AR_SOFT_WORDS = max(1, _env_int("JARVIS_SENTENCE_BUFFER_AR_SOFT_WORDS", 6))
SENTENCE_BUFFER_AR_HARD_WORDS = max(
    SENTENCE_BUFFER_AR_SOFT_WORDS,
    _env_int("JARVIS_SENTENCE_BUFFER_AR_HARD_WORDS", 18),
)
SENTENCE_BUFFER_HOLD_CONNECTORS = _env_bool("JARVIS_SENTENCE_BUFFER_HOLD_CONNECTORS", True)

# Demo mode: shows intent/confidence overlay in console (set via --demo-mode flag or env var)
DEMO_MODE = _env_bool("JARVIS_DEMO_MODE", False)

# Logging
LOG_FILE = _project_path("jarvis.log")
LOG_CONSOLE_LEVEL = _env("JARVIS_LOG_CONSOLE_LEVEL", "INFO")
LOG_FILE_LEVEL = _env("JARVIS_LOG_FILE_LEVEL", "DEBUG")
LOG_ROTATE_MAX_BYTES = _env_int("JARVIS_LOG_ROTATE_MAX_BYTES", 2_000_000)
LOG_ROTATE_BACKUPS = _env_int("JARVIS_LOG_ROTATE_BACKUPS", 3)
TIMING_LOG_ENABLED = _env_bool("JARVIS_TIMING_LOG_ENABLED", True)


