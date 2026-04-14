import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _project_path(*parts):
    return str(PROJECT_ROOT.joinpath(*parts))


# Load .env from project root (next to core/)
load_dotenv(PROJECT_ROOT / ".env")


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
INPUT_AUDIO_FILE = _project_path("input.wav")
VAD_ENERGY_THRESHOLD = _env_float("JARVIS_VAD_ENERGY_THRESHOLD", 0.014)
VAD_SILENCE_SECONDS = _env_float("JARVIS_VAD_SILENCE_SECONDS", 0.55)
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
WAKE_WORD = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.35
WAKE_WORD_CHUNK_SIZE = 1280
WAKE_WORD_INPUT_DEVICE = None  # None, device index (int), or name substring (str)
WAKE_WORD_AUDIO_GAIN = 1.4
WAKE_WORD_SCORE_DEBUG = False
WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS = 1.0
WAKE_WORD_DETECTION_COOLDOWN_SECONDS = 1.0
WAKE_WORD_MODE = str(_env("JARVIS_WAKE_MODE", "both")).strip().lower()
if WAKE_WORD_MODE not in {"english", "arabic", "both"}:
    WAKE_WORD_MODE = "both"

WAKE_WORD_AR_ENABLED = _env_bool("JARVIS_WAKE_WORD_AR_ENABLED", True)
_DEFAULT_WAKE_WORD_AR_TRIGGERS = (
    "hey jarvis",
    "hello jarvis",
    "jarvis",
    "\u0627\u0647\u0644\u0627 \u062c\u0627\u0631\u0641\u064a\u0633",
    "\u0645\u0631\u062d\u0628\u0627 \u062c\u0627\u0631\u0641\u064a\u0633",
    "\u064a\u0627 \u062c\u0627\u0631\u0641\u064a\u0633",
    "\u062c\u0627\u0631\u0641\u064a\u0633",
    # Keep common ASR spelling variants for reliability.
    "\u064a\u0627 \u062c\u0627\u0631\u0641\u0633",
    "\u0645\u0631\u062d\u0628\u0627 \u062c\u0627\u0631\u0641\u0633",
    "\u062c\u0627\u0631\u0641\u0633",
)
WAKE_WORD_AR_TRIGGERS = _env_list("JARVIS_WAKE_WORD_AR_TRIGGERS", _DEFAULT_WAKE_WORD_AR_TRIGGERS)
WAKE_WORD_AR_STT_MODEL = _env("JARVIS_WAKE_WORD_AR_STT_MODEL", "tiny")
WAKE_WORD_AR_CHUNK_SECONDS = max(0.8, _env_float("JARVIS_WAKE_WORD_AR_CHUNK_SECONDS", 1.5))
WAKE_WORD_AR_CHECK_INTERVAL_SECONDS = max(0.5, _env_float("JARVIS_WAKE_WORD_AR_CHECK_INTERVAL_SECONDS", 2.0))
WAKE_WORD_AR_CONSECUTIVE_HITS_REQUIRED = max(1, _env_int("JARVIS_WAKE_WORD_AR_CONSECUTIVE_HITS_REQUIRED", 2))
WAKE_WORD_AR_CONFIRM_WINDOW_SECONDS = max(1.0, _env_float("JARVIS_WAKE_WORD_AR_CONFIRM_WINDOW_SECONDS", 3.0))

# STT
STT_BACKEND = _env(
    "JARVIS_STT_BACKEND",
    "hybrid_elevenlabs",
)  # hybrid_elevenlabs | faster_whisper
if STT_BACKEND not in {"hybrid_elevenlabs", "faster_whisper"}:
    STT_BACKEND = "hybrid_elevenlabs"

ELEVENLABS_BASE_URL = _env("ELEVENLABS_BASE_URL", "https://api.elevenlabs.io").strip() or "https://api.elevenlabs.io"
ELEVENLABS_API_KEY = _env("ELEVENLABS_API_KEY", "").strip()

STT_LANGUAGE_DETECT_MODEL = _env("JARVIS_STT_LANGUAGE_DETECT_MODEL", "tiny").strip() or "tiny"
STT_MIXED_TREAT_AS_ARABIC = _env_bool("JARVIS_STT_MIXED_TREAT_AS_ARABIC", True)

STT_ELEVENLABS_ENABLED = _env_bool("JARVIS_STT_ELEVENLABS_ENABLED", True)
STT_ELEVENLABS_STT_MODEL = _env("JARVIS_STT_ELEVENLABS_MODEL", "scribe_v1").strip() or "scribe_v1"
STT_ELEVENLABS_ARABIC_LANGUAGE = _env("JARVIS_STT_ELEVENLABS_ARABIC_LANG", "ara").strip() or "ara"
STT_ELEVENLABS_TIMEOUT_SECONDS = max(3.0, _env_float("JARVIS_STT_ELEVENLABS_TIMEOUT_SECONDS", 15.0))
STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS = max(2, _env_int("JARVIS_STT_ELEVENLABS_WEAK_TEXT_MIN_CHARS", 5))

# Local fallback backend settings.
WHISPER_MODEL = _env("JARVIS_WHISPER_MODEL", "small")

# LLM
LLM_MODEL = _env("JARVIS_LLM_MODEL", "qwen2.5:3b")
LLM_FALLBACK_MODELS = tuple(
    m.strip()
    for m in _env("JARVIS_LLM_FALLBACK_MODELS", "").split(",")
    if m.strip()
)
LLM_TIMEOUT_SECONDS = max(10, _env_int("JARVIS_LLM_TIMEOUT_SECONDS", 30))
LLM_OLLAMA_NUM_CTX = max(512, _env_int("JARVIS_LLM_OLLAMA_NUM_CTX", 2048))
LLM_OLLAMA_BASE_URL = _env("JARVIS_LLM_OLLAMA_BASE_URL", "http://localhost:11434").strip() or "http://localhost:11434"
LLM_OLLAMA_AUTOSTART = _env_bool("JARVIS_LLM_OLLAMA_AUTOSTART", True)
LLM_OLLAMA_EXECUTABLE = _env("JARVIS_LLM_OLLAMA_EXECUTABLE", "ollama").strip() or "ollama"
LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS = max(
    3.0,
    _env_float("JARVIS_LLM_OLLAMA_AUTOSTART_TIMEOUT_SECONDS", 25.0),
)
LLM_LIGHTWEIGHT_NUM_CTX = max(256, _env_int("JARVIS_LLM_LIGHTWEIGHT_NUM_CTX", 1024))
LLM_RESPONSE_CACHE_ENABLED = _env_bool("JARVIS_LLM_RESPONSE_CACHE_ENABLED", True)
LLM_RESPONSE_CACHE_TTL_SECONDS = max(10, _env_int("JARVIS_LLM_RESPONSE_CACHE_TTL_SECONDS", 600))
LLM_RESPONSE_CACHE_MAX_SIZE = max(16, _env_int("JARVIS_LLM_RESPONSE_CACHE_MAX_SIZE", 256))
LLM_RESPONSE_CACHE_MAX_QUERY_WORDS = max(1, _env_int("JARVIS_LLM_RESPONSE_CACHE_MAX_QUERY_WORDS", 8))
LLM_CPU_UPGRADE_TEST_ENABLED = _env_bool("JARVIS_LLM_CPU_UPGRADE_TEST_ENABLED", False)
LLM_CPU_UPGRADE_MODEL = _env("JARVIS_LLM_CPU_UPGRADE_MODEL", "qwen2.5:3b")
LLM_CPU_UPGRADE_MAX_LATENCY_SECONDS = max(0.5, _env_float("JARVIS_LLM_CPU_UPGRADE_MAX_LATENCY_SECONDS", 3.0))
LLM_REALTIME_REWRITE_ENABLED = _env_bool("JARVIS_LLM_REALTIME_REWRITE_ENABLED", False)
LLM_APPEND_SOURCE_CITATIONS = True

# NLU (Phase 1)
NLU_INTENT_ROUTING_ENABLED = _env_bool("JARVIS_NLU_INTENT_ROUTING_ENABLED", True)
NLU_INTENT_CONFIDENCE_THRESHOLD = _env_float("JARVIS_NLU_INTENT_CONFIDENCE_THRESHOLD", 0.75)
NLU_PARSER_FASTPATH_ENABLED = _env_bool("JARVIS_NLU_PARSER_FASTPATH_ENABLED", True)
NLU_PARSER_FASTPATH_CONFIDENCE_FLOOR = _env_float("JARVIS_NLU_PARSER_FASTPATH_CONFIDENCE_FLOOR", 0.55)
NLU_LLM_QUERY_EXTRACTION_ENABLED = _env_bool("JARVIS_NLU_LLM_QUERY_EXTRACTION_ENABLED", False)
NLU_INTENT_CACHE_ENABLED = _env_bool("JARVIS_NLU_INTENT_CACHE_ENABLED", True)
NLU_INTENT_CACHE_MAX_SIZE = _env_int("JARVIS_NLU_INTENT_CACHE_MAX_SIZE", 256)
NLU_INTENT_CACHE_TTL_SECONDS = _env_int("JARVIS_NLU_INTENT_CACHE_TTL_SECONDS", 600)

NLU_INTENT_THRESHOLD_BY_FAMILY = {
    "OS_APP_OPEN": _env_float("JARVIS_NLU_THRESHOLD_OS_APP_OPEN", 0.72),
    "OS_APP_CLOSE": _env_float("JARVIS_NLU_THRESHOLD_OS_APP_CLOSE", 0.74),
    "OS_FILE_SEARCH": _env_float("JARVIS_NLU_THRESHOLD_OS_FILE_SEARCH", 0.73),
    "OS_FILE_NAVIGATION": _env_float("JARVIS_NLU_THRESHOLD_OS_FILE_NAVIGATION", 0.74),
    "OS_SYSTEM_COMMAND": _env_float("JARVIS_NLU_THRESHOLD_OS_SYSTEM_COMMAND", 0.77),
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
TTS_ELEVENLABS_ARABIC_ENABLED = _env_bool("JARVIS_TTS_ELEVENLABS_ARABIC_ENABLED", True)
TTS_ELEVENLABS_ARABIC_VOICE_ID = _env("JARVIS_TTS_ELEVENLABS_ARABIC_VOICE_ID", "").strip()
TTS_ELEVENLABS_ARABIC_MODEL_ID = _env("JARVIS_TTS_ELEVENLABS_ARABIC_MODEL_ID", "eleven_multilingual_v2").strip() or "eleven_multilingual_v2"
TTS_ELEVENLABS_TIMEOUT_SECONDS = max(3.0, _env_float("JARVIS_TTS_ELEVENLABS_TIMEOUT_SECONDS", 15.0))
TTS_PREWARM_ENABLED = _env_bool("JARVIS_TTS_PREWARM_ENABLED", True)
STARTUP_PARSER_NLP_PREWARM_ENABLED = _env_bool("JARVIS_STARTUP_PARSER_NLP_PREWARM_ENABLED", True)
TTS_ARABIC_SPOKEN_DIALECT = str(_env("JARVIS_TTS_ARABIC_SPOKEN_DIALECT", "egyptian")).strip().lower()
if TTS_ARABIC_SPOKEN_DIALECT not in {"egyptian", "msa", "auto"}:
    TTS_ARABIC_SPOKEN_DIALECT = "egyptian"
TTS_EGYPTIAN_COLLOQUIAL_REWRITE = _env_bool("JARVIS_TTS_EGYPTIAN_COLLOQUIAL_REWRITE", True)
TTS_DEFAULT_RATE = 175
TTS_SIMULATED_CHAR_DELAY = 0.02
BARGE_IN_INTERRUPT_ON_WAKE = True
WAKE_WORD_IGNORE_WHILE_SPEAKING = True

# Persona
PERSONA_DEFAULT = "assistant"
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
MEMORY_MAX_TURNS = 10
MEMORY_MAX_CONTEXT_CHARS = max(300, _env_int("JARVIS_MEMORY_MAX_CONTEXT_CHARS", 900))
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
        "command_permissions": POLICY_COMMAND_PERMISSIONS,
    },
    "demo": {
        "read_only_mode": True,
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

# Logging
LOG_FILE = _project_path("jarvis.log")


