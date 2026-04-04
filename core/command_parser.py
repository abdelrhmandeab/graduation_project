import os
import re
from dataclasses import dataclass, field

from os_control.system_ops import normalize_system_action


@dataclass
class ParsedCommand:
    intent: str
    raw: str
    normalized: str
    action: str = ""
    args: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------

_COLLAPSE_WS_RE = re.compile(r"\s+")
_MATCH_SANITIZE_RE = re.compile(r"[^a-z0-9_\s:\\/.\-\u0600-\u06FF]")
_DRIVE_COLON_RE = re.compile(r"\b([a-z])\s*:", flags=re.IGNORECASE)
_DRIVE_WORD_RE = re.compile(r"\b([a-z])\s+(?:drive|partition)\b", flags=re.IGNORECASE)
_SEA_C_DRIVE_RE = re.compile(r"\b(?:sea|see|cee)\s+(?:drive|partition)\b", flags=re.IGNORECASE)
_OPEN_FILLER_PREFIXES = (
    r"^(?:for me|for us|for me now|for me please)\s+",
    r"^(?:\u0645\u0646 \u0641\u0636\u0644\u0643|\u0644\u0648 \u0633\u0645\u062d\u062a|\u0631\u062c\u0627\u0621|\u0627\u0644\u0631\u062c\u0627\u0621)\s+",
    r"^(?:the)\s+",
    r"^(?:\u0627\u0644)\s+",
)
_FILESYSTEM_OPEN_HINTS = (
    "drive",
    "partition",
    "folder",
    "directory",
    "desktop",
    "downloads",
    "documents",
    "pictures",
    "music",
    "videos",
    "file explorer",
    "\u0642\u0631\u0635",
    "\u0628\u0627\u0631\u062a\u0634\u0646",
    "\u0642\u0633\u0645",
    "\u062f\u0631\u0627\u064a\u0641",
    "\u0645\u062c\u0644\u062f",
    "\u0645\u0644\u0641",
    "\u0633\u0637\u062d \u0627\u0644\u0645\u0643\u062a\u0628",
    "\u0627\u0644\u062a\u062d\u0645\u064a\u0644\u0627\u062a",
    "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a",
    "\u0627\u0644\u0635\u0648\u0631",
    "\u0627\u0644\u0645\u0648\u0633\u064a\u0642\u0649",
    "\u0627\u0644\u0641\u064a\u062f\u064a\u0648\u0647\u0627\u062a",
)
_SPECIAL_FOLDER_ALIASES = {
    "desktop": "Desktop",
    "\u0633\u0637\u062d \u0627\u0644\u0645\u0643\u062a\u0628": "Desktop",
    "downloads": "Downloads",
    "download": "Downloads",
    "\u0627\u0644\u062a\u062d\u0645\u064a\u0644\u0627\u062a": "Downloads",
    "\u0627\u0644\u062a\u0646\u0632\u064a\u0644\u0627\u062a": "Downloads",
    "documents": "Documents",
    "document": "Documents",
    "\u0627\u0644\u0645\u0633\u062a\u0646\u062f\u0627\u062a": "Documents",
    "pictures": "Pictures",
    "picture": "Pictures",
    "\u0627\u0644\u0635\u0648\u0631": "Pictures",
    "music": "Music",
    "\u0627\u0644\u0645\u0648\u0633\u064a\u0642\u0649": "Music",
    "videos": "Videos",
    "video": "Videos",
    "\u0627\u0644\u0641\u064a\u062f\u064a\u0648\u0647\u0627\u062a": "Videos",
}
_SEARCH_PATH_ALIASES = {
    **_SPECIAL_FOLDER_ALIASES,
    "\u0627\u0644\u0645\u0643\u062a\u0628": "Desktop",
}


def _normalize_for_match(text: str) -> str:
    lowered = " ".join((text or "").lower().split()).strip()
    cleaned = _MATCH_SANITIZE_RE.sub(" ", lowered)
    return _COLLAPSE_WS_RE.sub(" ", cleaned).strip()


def _strip_spoken_prefixes(normalized_text: str) -> str:
    candidate = (normalized_text or "").strip()
    patterns = (
        r"^(?:hey|ok|okay)\s+jarvis\s+",
        r"^(?:hey|ok|okay)\s+",
        r"^jarvis\s+",
        r"^please\s+",
        r"^(?:please\s+)?(?:can|could|would|will)\s+you\s+",
        r"^(?:please\s+)?(?:i need you to|i want you to|i want to)\s+",
        r"^(?:\u064a\u0627\s+)?\u062c\u0627\u0631\u0641\u064a\u0633\s+",
        r"^(?:\u0645\u0646 \u0641\u0636\u0644\u0643|\u0644\u0648 \u0633\u0645\u062d\u062a|\u0631\u062c\u0627\u0621|\u0627\u0644\u0631\u062c\u0627\u0621)\s+",
        r"^(?:\u0647\u0644 \u064a\u0645\u0643\u0646\u0643|\u0647\u0644 \u062a\u0633\u062a\u0637\u064a\u0639|\u0645\u0645\u0643\u0646)\s+",
        r"^(?:\u0627\u0631\u064a\u062f\u0643 \u0627\u0646|\u0623\u0631\u064a\u062f\u0643 \u0623\u0646|\u0627\u0631\u064a\u062f|\u0623\u0631\u064a\u062f|\u0639\u0627\u064a\u0632\u0643|\u0639\u0627\u064a\u0632)\s+(?:\u0627\u0646|\u0623\u0646)?\s*",
    )
    for pattern in patterns:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _extract_drive_letter(text: str):
    if _SEA_C_DRIVE_RE.search(text or ""):
        return "C"
    for pattern in (_DRIVE_COLON_RE, _DRIVE_WORD_RE):
        match = pattern.search(text or "")
        if match:
            return match.group(1).upper()
    return None


def _is_drive_open_request(text: str) -> bool:
    lowered = (text or "").lower()
    explicit_verbs = (
        "open",
        "show",
        "browse",
        "access",
        "enter",
        "\u0627\u0641\u062a\u062d",
        "\u0627\u0639\u0631\u0636",
        "\u0627\u0638\u0647\u0631",
        "\u062a\u0635\u0641\u062d",
        "\u0627\u062f\u062e\u0644",
    )
    if any(verb in lowered for verb in explicit_verbs):
        return True
    if "go to" in lowered and ("drive" in lowered or "partition" in lowered):
        return True
    if (
        ("\u0627\u0630\u0647\u0628 \u0627\u0644\u0649" in lowered or "\u0627\u0646\u062a\u0642\u0644 \u0627\u0644\u0649" in lowered)
        and (
            "\u0642\u0631\u0635" in lowered
            or "\u0628\u0627\u0631\u062a\u0634\u0646" in lowered
            or "\u0642\u0633\u0645" in lowered
            or "\u062f\u0631\u0627\u064a\u0641" in lowered
        )
    ):
        return True
    return False


def _strip_open_fillers(text: str) -> str:
    candidate = (text or "").strip()
    for pattern in _OPEN_FILLER_PREFIXES:
        candidate = re.sub(pattern, "", candidate, flags=re.IGNORECASE).strip()
    return candidate


def _special_folder_path(text: str):
    lowered = (text or "").lower()
    user_home = os.path.expanduser("~")
    for key, folder_name in _SPECIAL_FOLDER_ALIASES.items():
        if key in lowered:
            return os.path.join(user_home, folder_name)
    return None


def _looks_like_filesystem_target(text: str) -> bool:
    lowered = (text or "").lower()
    if any(hint in lowered for hint in _FILESYSTEM_OPEN_HINTS):
        return True
    if "\\" in lowered or "/" in lowered:
        return True
    if re.search(r"\b[a-z]:\\", lowered):
        return True
    return False


def _collapse_repeated_phrase(text: str) -> str:
    candidate = " ".join((text or "").split()).strip()
    if not candidate:
        return ""

    tokens = candidate.split(" ")
    if len(tokens) >= 2 and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            return " ".join(tokens[:half])

    lower = candidate.lower()
    for sep in (" in ", " on ", " inside ", " \u0641\u064a ", " \u062f\u0627\u062e\u0644 "):
        parts = [segment.strip() for segment in lower.split(sep) if segment.strip()]
        if len(parts) >= 2 and len(set(parts)) == 1:
            return parts[0]
    return candidate


def _normalize_search_path_hint(path_hint: str):
    candidate = _collapse_repeated_phrase(path_hint)
    if not candidate:
        return None

    lowered = candidate.lower().strip()
    alias = _SEARCH_PATH_ALIASES.get(lowered)
    if alias:
        return os.path.join(os.path.expanduser("~"), alias)
    return candidate


# ---------------------------------------------------------------------------
# Table-driven keyword matching
# ---------------------------------------------------------------------------
# Each entry: (set_of_keywords, intent, action)
# Matched against `normalized`.

_KEYWORD_TABLE = [
    # Observability
    ({"observability", "observability report", "show observability", "dashboard"}, "OBSERVABILITY_REPORT", ""),
    # Benchmark
    ({"benchmark run", "run benchmark", "benchmark quick"}, "BENCHMARK_COMMAND", "run"),
    ({"resilience demo", "run resilience demo", "failure demo"}, "BENCHMARK_COMMAND", "resilience_demo"),
    # Persona
    ({"persona status", "persona show"}, "PERSONA_COMMAND", "status"),
    ({"persona list", "list personas"}, "PERSONA_COMMAND", "list"),
    ({"persona voice status"}, "PERSONA_COMMAND", "voice_status"),
    ({"assistant mode", "assistant mode on"}, "PERSONA_COMMAND", "set", {"profile": "assistant"}),
    # Voice
    ({"voice status", "speech status", "حالة الصوت", "حالة النطق"}, "VOICE_COMMAND", "status"),
    ({"voice diagnostic", "voice diagnostics", "speech diagnostic", "tts diagnostic"}, "VOICE_COMMAND", "diagnostic"),
    ({"audio ux status", "audio profile status", "voice audio status", "حالة تجربة الصوت", "حالة ملف تجربة الصوت"}, "VOICE_COMMAND", "audio_ux_status"),
    ({"audio ux profiles", "audio ux profile list", "list audio ux profiles", "قائمة ملفات تجربة الصوت", "ملفات تجربة الصوت"}, "VOICE_COMMAND", "audio_ux_profiles"),
    ({"audio ux profile balanced", "audio profile balanced", "set audio profile balanced", "ملف تجربة الصوت متوازن", "وضع تجربة الصوت متوازن", "وضع الصوت متوازن"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "balanced"}),
    ({"audio ux profile responsive", "audio profile responsive", "set audio profile responsive", "ملف تجربة الصوت سريع", "وضع تجربة الصوت سريع", "وضع الصوت سريع"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "responsive"}),
    ({"audio ux profile robust", "audio profile robust", "set audio profile robust", "ملف تجربة الصوت قوي", "وضع تجربة الصوت قوي", "وضع الصوت قوي", "وضع الصوت ثابت"}, "VOICE_COMMAND", "audio_ux_profile_set", {"profile": "robust"}),
    ({"voice quality status", "speech quality status", "tts quality status", "حالة جودة الصوت", "حالة جودة النطق"}, "VOICE_COMMAND", "voice_quality_status"),
    ({"voice quality natural", "speech quality natural", "tts quality natural", "natural voice mode", "جودة الصوت طبيعي", "وضع الصوت طبيعي", "وضع النطق طبيعي"}, "VOICE_COMMAND", "voice_quality_set", {"mode": "natural"}),
    ({"voice quality standard", "speech quality standard", "tts quality standard", "robot voice mode", "robotic voice mode", "جودة الصوت قياسي", "وضع الصوت قياسي", "وضع الصوت روبوتي"}, "VOICE_COMMAND", "voice_quality_set", {"mode": "standard"}),
    ({"stt profile status", "speech profile status", "voice stt profile status", "حالة ملف الاستماع"}, "VOICE_COMMAND", "stt_profile_status"),
    ({"stt profile quiet", "speech profile quiet", "ملف الاستماع هادئ", "وضع الاستماع هادئ"}, "VOICE_COMMAND", "stt_profile_set", {"profile": "quiet"}),
    ({"stt profile noisy", "speech profile noisy", "ملف الاستماع ضوضاء", "وضع الاستماع ضوضاء"}, "VOICE_COMMAND", "stt_profile_set", {"profile": "noisy"}),
    ({"voice clone on", "enable voice clone"}, "VOICE_COMMAND", "clone_on"),
    ({"voice clone off", "disable voice clone"}, "VOICE_COMMAND", "clone_off"),
    ({"hf profile status", "huggingface profile status", "voice hf profile status"}, "VOICE_COMMAND", "hf_profile_status"),
    ({"hf profile arabic", "huggingface profile arabic", "speech hf profile arabic"}, "VOICE_COMMAND", "hf_profile_set", {"profile": "arabic"}),
    ({"hf profile english", "huggingface profile english", "speech hf profile english"}, "VOICE_COMMAND", "hf_profile_set", {"profile": "english"}),
    ({"stop speaking", "interrupt speech", "be quiet", "stop talking"}, "VOICE_COMMAND", "interrupt"),
    ({"speech on", "enable speech"}, "VOICE_COMMAND", "speech_on"),
    ({"speech off", "disable speech"}, "VOICE_COMMAND", "speech_off"),
    # Knowledge base
    ({"kb status", "knowledge status", "knowledge base status"}, "KNOWLEDGE_BASE_COMMAND", "status"),
    ({"kb quality", "knowledge quality", "kb quality report"}, "KNOWLEDGE_BASE_COMMAND", "quality"),
    ({"kb clear", "knowledge clear"}, "KNOWLEDGE_BASE_COMMAND", "clear"),
    ({"kb retrieval on", "knowledge retrieval on"}, "KNOWLEDGE_BASE_COMMAND", "retrieval_on"),
    ({"kb retrieval off", "knowledge retrieval off"}, "KNOWLEDGE_BASE_COMMAND", "retrieval_off"),
    # Memory
    ({"memory status", "session memory status"}, "MEMORY_COMMAND", "status"),
    ({"memory clear", "session memory clear"}, "MEMORY_COMMAND", "clear"),
    ({"memory on", "enable memory"}, "MEMORY_COMMAND", "on"),
    ({"memory off", "disable memory"}, "MEMORY_COMMAND", "off"),
    ({"memory show", "show memory"}, "MEMORY_COMMAND", "show"),
    # Demo
    ({"demo mode on", "demo on"}, "DEMO_MODE", "on"),
    ({"demo mode off", "demo off"}, "DEMO_MODE", "off"),
    ({"demo mode status", "demo status"}, "DEMO_MODE", "status"),
    # Metrics
    ({"show metrics", "metrics", "metrics report"}, "METRICS_REPORT", ""),
    # Audit
    ({"verify audit", "verify audit log", "audit verify"}, "AUDIT_VERIFY", ""),
    ({"audit reseal", "reseal audit", "repair audit chain"}, "AUDIT_RESEAL", ""),
    # Policy
    ({"policy status"}, "POLICY_COMMAND", "status"),
    # Batch
    ({"batch plan", "batch start", "batch begin"}, "BATCH_COMMAND", "plan"),
    ({"batch preview", "batch show"}, "BATCH_COMMAND", "preview"),
    ({"batch status"}, "BATCH_COMMAND", "status"),
    ({"batch commit", "batch run"}, "BATCH_COMMAND", "commit"),
    ({"batch abort", "batch cancel", "batch clear"}, "BATCH_COMMAND", "abort"),
    # Search index
    ({"index status", "search index status"}, "SEARCH_INDEX_COMMAND", "status"),
    ({"index start", "start index"}, "SEARCH_INDEX_COMMAND", "start"),
    # Job queue
    ({"job worker start"}, "JOB_QUEUE_COMMAND", "worker_start"),
    ({"job worker stop"}, "JOB_QUEUE_COMMAND", "worker_stop"),
    ({"job worker status"}, "JOB_QUEUE_COMMAND", "worker_status"),
    # Rollback
    (
        {
            "undo",
            "rollback",
            "undo last action",
            "\u062a\u0631\u0627\u062c\u0639",
            "\u0627\u0644\u063a\u0627\u0621 \u0627\u062e\u0631 \u0639\u0645\u0644\u064a\u0629",
            "\u0627\u0644\u063a\u0627\u0621 \u0627\u062e\u0631 \u0627\u0645\u0631",
        },
        "OS_ROLLBACK",
        "",
    ),
    # File nav
    (
        {
            "current directory",
            "pwd",
            "\u0627\u0644\u0645\u062c\u0644\u062f \u0627\u0644\u062d\u0627\u0644\u064a",
            "\u0627\u064a\u0646 \u0627\u0646\u0627",
        },
        "OS_FILE_NAVIGATION",
        "pwd",
    ),
    (
        {
            "list drives",
            "drive list",
            "\u0627\u0639\u0631\u0636 \u0627\u0644\u0627\u0642\u0631\u0627\u0635",
            "\u0627\u0638\u0647\u0631 \u0627\u0644\u0627\u0642\u0631\u0627\u0635",
            "\u0642\u0627\u0626\u0645\u0629 \u0627\u0644\u0627\u0642\u0631\u0627\u0635",
        },
        "OS_FILE_NAVIGATION",
        "list_drives",
    ),
]


def _try_keyword_table(normalized, raw):
    for entry in _KEYWORD_TABLE:
        keywords, intent, action = entry[0], entry[1], entry[2]
        if normalized in keywords:
            args = entry[3] if len(entry) > 3 else {}
            return ParsedCommand(intent, raw, normalized, action=action, args=dict(args))
    return None


# ---------------------------------------------------------------------------
# Table-driven regex matching
# ---------------------------------------------------------------------------
# Each entry: (compiled_regex, use_raw, intent, action, args_builder)
# If use_raw is True, the regex is matched against `raw` (case-insensitive).
# Otherwise it's matched against `normalized`.
# args_builder is a callable: (match) -> dict

_REGEX_TABLE = [
    # Persona
    (
        re.compile(r"^persona voice clone\s+([a-z0-9_-]+)\s+(on|off)$"),
        False,
        "PERSONA_COMMAND",
        "set_profile_clone_enabled",
        lambda m: {"profile": m.group(1), "enabled": m.group(2) == "on"},
    ),
    (
        re.compile(r"^persona voice provider\s+([a-z0-9_-]+)\s+(xtts|voicecraft)$"),
        False,
        "PERSONA_COMMAND",
        "set_profile_clone_provider",
        lambda m: {"profile": m.group(1), "provider": m.group(2)},
    ),
    (
        re.compile(r"^persona voice reference\s+([a-z0-9_-]+)\s+(.+)$", re.IGNORECASE),
        True,
        "PERSONA_COMMAND",
        "set_profile_clone_reference",
        lambda m: {"profile": m.group(1).strip().lower(), "path": m.group(2).strip()},
    ),
    (
        re.compile(r"^persona set\s+([a-z0-9_-]+)$"),
        False,
        "PERSONA_COMMAND",
        "set",
        lambda m: {"profile": m.group(1)},
    ),
    # Voice
    (
        re.compile(r"^voice clone provider\s+(xtts|voicecraft)$"),
        False,
        "VOICE_COMMAND",
        "set_provider",
        lambda m: {"provider": m.group(1)},
    ),
    (
        re.compile(r"^(?:set\s+)?(?:voice\s+)?(?:stt|speech)\s+profile(?:\s+to)?\s+(quiet|noisy)(?:\s+room)?$"),
        False,
        "VOICE_COMMAND",
        "stt_profile_set",
        lambda m: {"profile": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech)\s+)?(?:hf|huggingface)\s+profile(?:\s+to)?\s+(arabic|english|ar|en|عربي|العربية|انجليزي|الانجليزية|الإنجليزية)(?:\s+mode)?$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "hf_profile_set",
        lambda m: {"profile": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:voice|speech|tts)\s+quality(?:\s+to)?\s+(natural|standard|balanced|default|human|robot|robotic)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "voice_quality_set",
        lambda m: {"mode": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:اضبط|حدد|غير|غيّر|اجعل)\s+(?:جودة|وضع)\s+(?:الصوت|النطق)(?:\s+(?:الى|إلى))?\s+(طبيعي|قياسي|افتراضي|روبوت|روبوتي)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "voice_quality_set",
        lambda m: {"mode": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio|voice)\s+(?:ux\s+)?profile(?:\s+to)?\s+(balanced|responsive|robust|fast|low\s*latency|low_latency|stable|reliable|noisy)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": m.group(1).replace(" ", "_")},
    ),
    (
        re.compile(
            r"^(?:اضبط|حدد|غير|غيّر|اجعل)\s+(?:ملف|وضع)\s+(?:تجربة\s+)?(?:الصوت|النطق)(?:\s+(?:الى|إلى))?\s+(متوازن|سريع(?:\s*الاستجابة)?|منخفض\s*الكمون|قوي|ثابت|موثوق)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_profile_set",
        lambda m: {"profile": m.group(1).replace(" ", "_")},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio\s+ux\s+)?(?:mic|microphone|vad)\s+(?:energy\s+)?threshold(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_mic_threshold_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio\s+ux\s+)?(?:wake(?:\s*[-_]?word)?\s+threshold)(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_wake_threshold_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:audio\s+ux\s+)?(?:wake(?:\s*[-_]?word)?\s+gain)(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_wake_gain_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|tts|audio\s+ux)\s+)?pause\s+scale(?:\s+to)?\s+([0-9]+(?:\.[0-9]+)?)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_pause_scale_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(
            r"^(?:set\s+)?(?:(?:voice|speech|tts|audio\s+ux)\s+)?rate\s+offset(?:\s+to)?\s+([+-]?\d+)$",
            re.IGNORECASE,
        ),
        True,
        "VOICE_COMMAND",
        "audio_ux_rate_offset_set",
        lambda m: {"value": m.group(1)},
    ),
    (
        re.compile(r"^voice clone reference\s+(.+)$", re.IGNORECASE),
        True,
        "VOICE_COMMAND",
        "set_reference",
        lambda m: {"path": m.group(1).strip()},
    ),
    # Knowledge base
    (
        re.compile(r"^(?:kb sync|knowledge sync)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "sync_dir",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb add|knowledge add)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "add_file",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb index|knowledge index)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "index_dir",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:kb search|knowledge search)\s+(.+)$", re.IGNORECASE),
        True,
        "KNOWLEDGE_BASE_COMMAND",
        "search",
        lambda m: {"query": m.group(1).strip()},
    ),
    # Audit
    (
        re.compile(r"^show audit log(?:\s+(\d+))?$"),
        False,
        "AUDIT_LOG_REPORT",
        "",
        lambda m: {"limit": int(m.group(1)) if m.group(1) else 10},
    ),
    # Policy
    (
        re.compile(r"^policy profile\s+([a-z0-9_-]+)$"),
        False,
        "POLICY_COMMAND",
        "set_profile",
        lambda m: {"profile": m.group(1)},
    ),
    (
        re.compile(r"^policy (?:read only|readonly)\s+(on|off)$"),
        False,
        "POLICY_COMMAND",
        "set_read_only",
        lambda m: {"enabled": m.group(1) == "on"},
    ),
    (
        re.compile(r"^policy permission\s+([a-z_]+)\s+(on|off)$"),
        False,
        "POLICY_COMMAND",
        "set_permission",
        lambda m: {"permission": m.group(1), "enabled": m.group(2) == "on"},
    ),
    # Batch
    (
        re.compile(r"^batch add\s+(.+)$", re.IGNORECASE),
        True,
        "BATCH_COMMAND",
        "add",
        lambda m: {"command_text": m.group(1).strip()},
    ),
    # Search index
    (
        re.compile(r"^index refresh(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "refresh",
        lambda m: {"root": (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(r"^(?:indexed find|index find|search indexed)\s+(.+?)(?:\s+in\s+(.+))?$", re.IGNORECASE),
        True,
        "SEARCH_INDEX_COMMAND",
        "search",
        lambda m: {"query": m.group(1).strip(), "root": (m.group(2) or "").strip() or None},
    ),
    # Job queue
    (
        re.compile(r"^(?:queue job|job add)\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?\s+(.+)$", re.IGNORECASE),
        True,
        "JOB_QUEUE_COMMAND",
        "enqueue",
        lambda m: {"delay_seconds": int(m.group(1)), "command_text": m.group(2).strip()},
    ),
    (
        re.compile(r"^(?:queue job|job add)\s+(.+)$", re.IGNORECASE),
        True,
        "JOB_QUEUE_COMMAND",
        "enqueue",
        lambda m: {"delay_seconds": 0, "command_text": m.group(1).strip()},
    ),
    (
        re.compile(r"^job status\s+(\d+)$"),
        False,
        "JOB_QUEUE_COMMAND",
        "status",
        lambda m: {"job_id": int(m.group(1))},
    ),
    (
        re.compile(r"^job cancel\s+(\d+)$"),
        False,
        "JOB_QUEUE_COMMAND",
        "cancel",
        lambda m: {"job_id": int(m.group(1))},
    ),
    (
        re.compile(r"^job retry\s+(\d+)(?:\s+in\s+(\d+)\s*(?:s|sec|secs|seconds)?)?$"),
        False,
        "JOB_QUEUE_COMMAND",
        "retry",
        lambda m: {"job_id": int(m.group(1)), "delay_seconds": int(m.group(2) or 0)},
    ),
    (
        re.compile(r"^job list(?:\s+([a-z]+|\d+))?(?:\s+(\d+))?$"),
        False,
        "JOB_QUEUE_COMMAND",
        "list",
        lambda m: _parse_job_list_args(m),
    ),
    # Confirmation
    (
        re.compile(
            r"^(?:confirm|\u062a\u0627\u0643\u064a\u062f|\u062a\u0623\u0643\u064a\u062f)\s+([0-9a-f]{6})(?:\s+(?:with\s+)?(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_CONFIRMATION",
        "",
        lambda m: {"token": m.group(1).lower(), "second_factor": (m.group(2) or "").strip() or None},
    ),
    # File search
    (
        re.compile(
            r"^(?:find file|search file|\u0627\u0628\u062d\u062b \u0639\u0646 \u0645\u0644\u0641|\u0627\u0628\u062d\u062b \u0645\u0644\u0641|\u062f\u0648\u0631 \u0639\u0644\u0649 \u0645\u0644\u0641)\s+(.+?)(?:\s+(?:in|\u0641\u064a)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_SEARCH",
        "",
        lambda m: {"filename": m.group(1).strip(), "search_path": (m.group(2) or "").strip() or None},
    ),
    # File nav - regex-based
    (
        re.compile(
            r"^(?:list files|list directory|show files|show directory|\u0627\u0639\u0631\u0636 \u0627\u0644\u0645\u0644\u0641\u0627\u062a|\u0627\u0638\u0647\u0631 \u0627\u0644\u0645\u0644\u0641\u0627\u062a|\u0627\u0639\u0631\u0636 \u0627\u0644\u0645\u062c\u0644\u062f|\u0627\u0638\u0647\u0631 \u0627\u0644\u0645\u062c\u0644\u062f)(?:\s+(?:in|\u0641\u064a)\s+(.+))?$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "list_directory",
        lambda m: {"path": (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(r"^(?:dir|ls)(?:\s+(.+))?$", re.IGNORECASE),
        True,
        "OS_FILE_NAVIGATION",
        "list_directory",
        lambda m: {"path": (m.group(1) or "").strip() or None},
    ),
    (
        re.compile(
            r"^(?:file info|metadata|\u0645\u0639\u0644\u0648\u0645\u0627\u062a \u0645\u0644\u0641|\u0628\u064a\u0627\u0646\u0627\u062a \u0645\u0644\u0641)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "file_info",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:create folder|make folder|mkdir|\u0627\u0646\u0634\u0626 \u0645\u062c\u0644\u062f|\u0627\u0639\u0645\u0644 \u0645\u062c\u0644\u062f|\u0627\u0635\u0646\u0639 \u0645\u062c\u0644\u062f)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "create_directory",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:delete permanently|permanent delete|force delete|\u0627\u062d\u0630\u0641 \u0646\u0647\u0627\u0626\u064a\u0627|\u062d\u0630\u0641 \u0646\u0647\u0627\u0626\u064a)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item_permanent",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(r"^(?:delete|remove|\u0627\u062d\u0630\u0641|\u0627\u0645\u0633\u062d|\u0627\u0632\u0644)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_FILE_NAVIGATION",
        "delete_item",
        lambda m: {"path": m.group(1).strip()},
    ),
    (
        re.compile(
            r"^(?:move|\u0627\u0646\u0642\u0644|\u062d\u0631\u0643)\s+(.+?)\s+(?:to|\u0627\u0644\u0649|\u0625\u0644\u0649)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "move_item",
        lambda m: {"source": m.group(1).strip(), "destination": m.group(2).strip()},
    ),
    (
        re.compile(
            r"^(?:rename|\u0627\u0639\u062f \u062a\u0633\u0645\u064a\u0629|\u063a\u064a\u0631 \u0627\u0633\u0645|\u063a\u064a\u0651\u0631 \u0627\u0633\u0645)\s+(.+?)\s+(?:to|\u0627\u0644\u0649|\u0625\u0644\u0649)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_FILE_NAVIGATION",
        "rename_item",
        lambda m: {"source": m.group(1).strip(), "new_name": m.group(2).strip()},
    ),
    # Open app explicit
    (
        re.compile(r"^(?:open app|\u0627\u0641\u062a\u062d \u062a\u0637\u0628\u064a\u0642|\u0634\u063a\u0644 \u062a\u0637\u0628\u064a\u0642)\s+(.+)$", re.IGNORECASE),
        True,
        "OS_APP_OPEN",
        "",
        lambda m: {"app_name": m.group(1).strip()},
    ),
    # Close app explicit
    (
        re.compile(
            r"^(?:close app|\u0627\u063a\u0644\u0642 \u062a\u0637\u0628\u064a\u0642|\u0627\u0642\u0641\u0644 \u062a\u0637\u0628\u064a\u0642|\u0633\u0643\u0631 \u062a\u0637\u0628\u064a\u0642|\u0627\u0646\u0647\u064a \u062a\u0637\u0628\u064a\u0642)\s+(.+)$",
            re.IGNORECASE,
        ),
        True,
        "OS_APP_CLOSE",
        "",
        lambda m: {"app_name": m.group(1).strip()},
    ),
]


def _parse_job_list_args(m):
    first = m.group(1)
    second = m.group(2)
    status = None
    limit = 10
    if first:
        if first.isdigit():
            limit = int(first)
        else:
            status = first
    if second:
        limit = int(second)
    return {"status": status, "limit": limit}


def _try_regex_table(normalized, raw):
    for pattern, use_raw, intent, action, args_builder in _REGEX_TABLE:
        text = raw if use_raw else normalized
        m = pattern.match(text)
        if m:
            return ParsedCommand(intent, raw, normalized, action=action, args=args_builder(m))
    return None


# ---------------------------------------------------------------------------
# Heuristic matchers (order-sensitive, cannot be table-driven)
# ---------------------------------------------------------------------------


def _try_drive_open(normalized_match, raw, normalized):
    drive_letter = _extract_drive_letter(normalized_match)
    if drive_letter and _is_drive_open_request(normalized_match):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": f"{drive_letter}:\\"},
        )
    return None


def _try_open_command(raw, normalized):
    open_match = re.match(
        r"^(?:open|launch|start|\u0627\u0641\u062a\u062d|\u0634\u063a\u0644)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if not open_match:
        return None

    target_raw = open_match.group(1).strip()
    target_for_match = _strip_open_fillers(_normalize_for_match(target_raw))

    drive_from_target = _extract_drive_letter(target_for_match)
    if drive_from_target and _is_drive_open_request(f"open {target_for_match}"):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": f"{drive_from_target}:\\"},
        )

    special_folder = _special_folder_path(target_for_match)
    if special_folder:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": special_folder},
        )

    if _looks_like_filesystem_target(target_for_match):
        target_path = target_raw
        if target_path.lower().startswith("the "):
            target_path = target_path[4:].strip()
        if target_path.startswith("\u0627\u0644"):
            target_path = target_path[2:].strip()
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="list_directory",
            args={"path": target_path},
        )

    return ParsedCommand("OS_APP_OPEN", raw, normalized, args={"app_name": target_raw})


def _try_close_command(raw, normalized):
    close_match = re.match(
        (
            r"^(?:close|terminate|kill|quit|exit|\u0627\u063a\u0644\u0642|\u0627\u0642\u0641\u0644|\u0633\u0643\u0631|\u0627\u0646\u0647\u064a)\s+"
            r"(?:app\s+|application\s+|program\s+|\u062a\u0637\u0628\u064a\u0642\s+)?(.+)$"
        ),
        raw,
        flags=re.IGNORECASE,
    )
    if not close_match:
        return None

    target_raw = close_match.group(1).strip()
    if not target_raw:
        return None

    blocked_system_targets = {
        "computer",
        "pc",
        "system",
        "الجهاز",
        "الكمبيوتر",
        "النظام",
    }
    normalized_target = _normalize_for_match(target_raw)
    if normalized_target in blocked_system_targets:
        return None

    return ParsedCommand("OS_APP_CLOSE", raw, normalized, args={"app_name": target_raw})


def _try_natural_file_search(raw, normalized):
    patterns = (
        re.compile(
            r"^(?:find|search|look\s+for|locate)\s+(?:for\s+)?(?:file\s+)?(.+?)(?:\s+(?:in|on|inside)\s+(.+))?$",
            re.IGNORECASE,
        ),
        re.compile(
            (
                r"^(?:(?:i\s+)?(?:want|need)\s+(?:to\s+)?)"
                r"(?:find|search|look\s+for|locate)\s+(?:for\s+)?(?:file\s+)?"
                r"(.+?)(?:\s+(?:in|on|inside)\s+(.+))?$"
            ),
            re.IGNORECASE,
        ),
        re.compile(
            (
                r"^(?:(?:\u0627\u0631\u064a\u062f|\u0623\u0631\u064a\u062f|\u0639\u0627\u064a\u0632|\u0627\u0628\u063a\u0649|\u0623\u0628\u063a\u0649)\s+(?:\u0627\u0646|\u0623\u0646)?\s+)?"
                r"(?:\u0627\u062c\u062f|\u0623\u062c\u062f|\u0627\u062f\u0648\u0631|\u0623\u062f\u0648\u0631|\u0627\u0628\u062d\u062b|\u0623\u0628\u062d\u062b)(?:\s+\u0639\u0646)?\s+(?:\u0645\u0644\u0641\s+)?"
                r"(.+?)(?:\s+(?:\u0641\u064a|\u062f\u0627\u062e\u0644)\s+(.+))?$"
            ),
            re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.match(raw)
        if not match:
            continue

        filename = _collapse_repeated_phrase(match.group(1) or "")
        filename = filename.strip().strip('"').strip("'")
        if not filename:
            return None

        search_path = _normalize_search_path_hint(match.group(2) or "")
        return ParsedCommand(
            "OS_FILE_SEARCH",
            raw,
            normalized,
            args={"filename": filename, "search_path": search_path},
        )
    return None


def _try_system_action(normalized_match, normalized, raw):
    system_action = normalize_system_action(normalized_match) or normalize_system_action(normalized)
    if system_action:
        return ParsedCommand(
            "OS_SYSTEM_COMMAND",
            raw,
            normalized,
            args={"action_key": system_action},
        )
    return None


def _try_cd_commands(normalized, raw):
    if normalized.startswith("go to "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[6:].strip()},
        )
    if normalized.startswith("change directory "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[len("change directory ") :].strip()},
        )
    if normalized.startswith("cd "):
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": raw[3:].strip()},
        )

    arabic_match = re.match(
        r"^(?:\u0627\u0630\u0647\u0628|\u0631\u0648\u062d|\u0627\u0646\u062a\u0642\u0644)\s+(?:\u0627\u0644\u0649|\u0625\u0644\u0649)\s+(.+)$",
        raw,
        flags=re.IGNORECASE,
    )
    if arabic_match:
        return ParsedCommand(
            "OS_FILE_NAVIGATION",
            raw,
            normalized,
            action="cd",
            args={"path": arabic_match.group(1).strip()},
        )
    return None


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------


def parse_command(text: str) -> ParsedCommand:
    raw = text or ""
    normalized = " ".join(raw.lower().split()).strip()
    normalized_match = _normalize_for_match(raw)
    spoken_candidate = _strip_spoken_prefixes(normalized_match)

    # Try stripping spoken prefixes and re-parsing.
    if spoken_candidate and spoken_candidate != normalized_match:
        nested = parse_command(spoken_candidate)
        if nested.intent != "LLM_QUERY":
            return ParsedCommand(
                nested.intent,
                raw,
                normalized,
                action=nested.action,
                args=dict(nested.args),
            )

    # 1. Keyword table (exact match on normalized).
    result = _try_keyword_table(normalized, raw)
    if result:
        return result

    # 2. Regex table.
    result = _try_regex_table(normalized, raw)
    if result:
        return result

    # 3. Natural file search phrasing.
    result = _try_natural_file_search(raw, normalized)
    if result:
        return result

    # 4. Drive open heuristic.
    result = _try_drive_open(normalized_match, raw, normalized)
    if result:
        return result

    # 5. "open ..." disambiguation.
    result = _try_open_command(raw, normalized)
    if result:
        return result

    # 6. System action aliases.
    result = _try_system_action(normalized_match, normalized, raw)
    if result:
        return result

    # 7. Natural close-app phrasing.
    result = _try_close_command(raw, normalized)
    if result:
        return result

    # 8. CD / navigation commands.
    result = _try_cd_commands(normalized, raw)
    if result:
        return result

    # 9. LLM fallback.
    return ParsedCommand("LLM_QUERY", raw, normalized)


