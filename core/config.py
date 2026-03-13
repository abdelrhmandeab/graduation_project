import os

# Audio
SAMPLE_RATE = 16000
MAX_RECORD_DURATION = 10
AUDIO_CHUNK_SIZE = 1024
INPUT_AUDIO_FILE = "input.wav"
VAD_ENERGY_THRESHOLD = 0.012
VAD_SILENCE_SECONDS = 0.8
VAD_MIN_SPEECH_SECONDS = 0.35
VAD_PREROLL_SECONDS = 0.2
VAD_START_TIMEOUT_SECONDS = 4.0
REALTIME_MAX_PENDING_UTTERANCES = 1
REALTIME_DROP_WHEN_BUSY = True
REALTIME_BACKPRESSURE_POLL_SECONDS = 0.25

# Wake Word
WAKE_WORD = "hey_jarvis"
WAKE_WORD_THRESHOLD = 0.35
WAKE_WORD_CHUNK_SIZE = 1280
WAKE_WORD_INPUT_DEVICE = None  # None, device index (int), or name substring (str)
WAKE_WORD_AUDIO_GAIN = 1.4
WAKE_WORD_SCORE_DEBUG = False
WAKE_WORD_SCORE_DEBUG_INTERVAL_SECONDS = 1.0
WAKE_WORD_DETECTION_COOLDOWN_SECONDS = 1.0

# STT
WHISPER_MODEL = "small"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"
WHISPER_LANGUAGE = None  # None enables auto language detection
WHISPER_BEAM_SIZE = 2
WHISPER_VAD_FILTER = False  # Mic VAD already trims silence; double VAD can cause empty transcripts
WHISPER_CONDITION_ON_PREVIOUS_TEXT = False

# LLM
LLM_MODEL = "qwen2.5:1.5b"
LLM_FALLBACK_MODELS = ("qwen2.5:0.5b",)
LLM_TIMEOUT_SECONDS = 120
LLM_APPEND_SOURCE_CITATIONS = True

# Speech / TTS
TTS_ENABLED = True
TTS_DEFAULT_BACKEND = "console"  # console | pyttsx3 | xtts | voicecraft
TTS_DEFAULT_RATE = 175
TTS_SIMULATED_CHAR_DELAY = 0.02
TTS_EXTERNAL_TIMEOUT_SECONDS = 45
BARGE_IN_INTERRUPT_ON_WAKE = True
WAKE_WORD_IGNORE_WHILE_SPEAKING = True

# Voice cloning
VOICE_CLONE_ENABLED = False
VOICE_CLONE_PROVIDER = "xtts"  # xtts | voicecraft
VOICE_CLONE_REFERENCE_AUDIO = ""
XTTS_CLI_PATH = ""
VOICECRAFT_CLI_PATH = ""

# Persona
PERSONA_DEFAULT = "assistant"

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
KB_TOP_K = 4
KB_MAX_CONTEXT_CHARS = 2200
KB_MIN_PROMPT_SCORE = 0.45
KB_MIN_SEMANTIC_ONLY_SCORE = 0.58
KB_RERANK_CANDIDATE_MULTIPLIER = 4
KB_LEXICAL_RERANK_WEIGHT = 0.35
KB_EMBEDDING_RERANK_WEIGHT = 0.65
KB_BLOCKED_CONTEXT_PATTERNS = (
    "ignore previous instruction",
    "ignore all previous instruction",
    "system prompt",
    "you are chatgpt",
    "assistant:",
    "developer:",
    "system:",
)

# Session memory
MEMORY_ENABLED = True
MEMORY_FILE = "jarvis_memory.json"
MEMORY_MAX_TURNS = 10
MEMORY_MAX_CONTEXT_CHARS = 1400

# Observability / benchmarking
OBSERVABILITY_RESOURCE_SAMPLING_SECONDS = 10
BENCHMARK_OUTPUT_FILE = "jarvis_benchmark.json"
RESILIENCE_OUTPUT_FILE = "jarvis_resilience.json"
BENCHMARK_SLA_P95_MS = 1500.0
BENCHMARK_SLA_SUCCESS_RATE_MIN = 0.95
RESILIENCE_SLA_P95_MS = 2000.0
RESILIENCE_SLA_SUCCESS_RATE_MIN = 0.8

# OS
MAX_FILE_RESULTS = 5
DEFAULT_WORKING_DIRECTORY = os.path.expanduser("~")
DEFAULT_SEARCH_PATH = DEFAULT_WORKING_DIRECTORY
POWERSHELL_EXECUTABLE = "powershell"
ACTION_LOG_FILE = "jarvis_actions.log"
ROLLBACK_DIR_NAME = ".jarvis_rollback"
CONFIRMATION_TIMEOUT_SECONDS = 45
ALLOW_DESTRUCTIVE_SYSTEM_COMMANDS = False
STATE_DB_FILE = "jarvis_state.db"
SECOND_FACTOR_REQUIRED_FOR_DESTRUCTIVE = True
SECOND_FACTOR_PIN = "2468"
SECOND_FACTOR_PASSPHRASE = "jarvis-confirm"
SEARCH_INDEX_DB_FILE = "jarvis_index.db"
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
    "benchmark": True,
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
            "benchmark": False,
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
            "benchmark": True,
        },
    },
}

# Logging
LOG_FILE = "jarvis.log"
