"""Intent schema — single source of truth for every intent Jarvis can route.

Each IntentSpec declares:
  - domain        : broad category (os | file | media | info | chat | meta | safety)
  - required_slots: slots the router MUST have before executing
  - optional_slots: slots that improve execution but are not mandatory
  - risk          : none | low | medium | high
                      none   → no confirmation ever
                      low    → execute directly, log only
                      medium → request_close_app / app-close confirmation gate
                      high   → PIN-confirmed (shutdown, permanent delete, logoff)
  - fast_execute  : eligible for the <100 ms fast lane (parser fast-path)
  - examples_en   : canonical English example utterances (used by eval harness)
  - examples_ar   : canonical Egyptian-Arabic examples

Risk levels are aligned with the existing ``destructive`` flags in
``system_ops.SYSTEM_COMMANDS`` and ``file_ops.request_delete_item``.

SCATTERED LITERALS THIS WILL LATER REPLACE (Phase 7 migration targets):
  - core/intent_confidence.py  : _QUESTION_PENALTY_INTENTS, _SENSITIVE_SYSTEM_ACTION_KEYS,
                                  ENTITY_CLARIFICATION_THRESHOLD_BY_INTENT
  - os_control/system_ops.py   : SYSTEM_COMMANDS[*]["destructive"]
  - os_control/file_ops.py     : permanent=True path
  - core/command_router.py     : _PERMISSION_MAP, _PARSER_FASTPATH_INTENTS
Do NOT delete those literals yet — Phase 7 migrates and removes them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from core.config import NLU_SCHEMA_ENABLED  # noqa: F401 — imported for guard below
from core.logger import get_logger

logger = get_logger("intent_schema")


@dataclass(frozen=True)
class IntentSpec:
    name: str
    domain: str
    required_slots: Tuple[str, ...] = ()
    optional_slots: Tuple[str, ...] = ()
    risk: str = "low"
    fast_execute: bool = False
    examples_en: Tuple[str, ...] = ()
    examples_ar: Tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Schema registry
# ---------------------------------------------------------------------------

SCHEMA: dict[str, IntentSpec] = {}


def _reg(*args, **kwargs) -> IntentSpec:
    spec = IntentSpec(*args, **kwargs)
    SCHEMA[spec.name] = spec
    return spec


# ── OS: application control ────────────────────────────────────────────────

_reg(
    name="OS_APP_OPEN",
    domain="os",
    required_slots=("app_name",),
    optional_slots=(),
    risk="low",
    fast_execute=True,
    examples_en=("open Chrome", "launch Spotify", "start Notepad"),
    examples_ar=("افتح كروم", "شغل سبوتيفاي", "افتح النوت باد"),
)

_reg(
    name="OS_APP_CLOSE",
    domain="os",
    required_slots=("app_name",),
    optional_slots=(),
    risk="medium",
    fast_execute=True,
    examples_en=("close Chrome", "shut down Firefox", "اقفل Spotify"),
    examples_ar=("اقفل كروم", "سكر فايرفوكس", "اقفل سبوتيفاي"),
)

# ── OS: system commands ────────────────────────────────────────────────────

_reg(
    name="OS_SYSTEM_COMMAND",
    domain="os",
    required_slots=("action_key",),
    optional_slots=("volume_level", "brightness_level", "window_query",
                    "tab_query", "url", "search_query", "seek_seconds",
                    "brightness_percent"),
    risk="low",       # individual destructive keys (shutdown/restart/logoff) get risk="high"
    fast_execute=True,
    examples_en=("volume up", "mute", "take a screenshot", "close tab",
                 "lock the computer"),
    examples_ar=("ارفع الصوت", "اكتم", "خد سكرين شوت", "اقفل التاب",
                 "قفل الكمبيوتر"),
)

_reg(
    name="OS_SYSTEM_COMMAND_DESTRUCTIVE",
    domain="os",
    required_slots=("action_key",),
    optional_slots=(),
    risk="high",
    fast_execute=False,
    examples_en=("shut down the computer", "restart", "log off"),
    examples_ar=("اطفي الكمبيوتر", "اعادة تشغيل", "تسجيل خروج"),
)

# ── File system ────────────────────────────────────────────────────────────

_reg(
    name="OS_FILE_NAVIGATION",
    domain="file",
    required_slots=(),
    optional_slots=("path", "action"),
    risk="low",
    fast_execute=True,
    examples_en=("open documents", "go to downloads", "list files on desktop"),
    examples_ar=("افتح المستندات", "روح التنزيلات", "اعرض ملفات سطح المكتب"),
)

_reg(
    name="OS_FILE_NAVIGATION_BATCH",
    domain="file",
    required_slots=("commands",),
    optional_slots=(),
    risk="medium",
    fast_execute=False,
    examples_en=("move all PDFs to documents", "rename all images"),
    examples_ar=("انقل كل الـ PDF للمستندات", "اعد تسمية كل الصور"),
)

_reg(
    name="OS_FILE_SEARCH",
    domain="file",
    required_slots=("filename",),
    optional_slots=("search_path", "extension", "kind"),
    risk="none",
    fast_execute=True,
    examples_en=("find my CV PDF", "search for report.docx", "دور على ملف"),
    examples_ar=("دور على السيرة الذاتية", "دور على report.docx في المستندات"),
)

_reg(
    name="OS_FILE_SEARCH_ADVANCED",
    domain="file",
    required_slots=("query",),
    optional_slots=("path", "extension", "modified_after", "size_min"),
    risk="none",
    fast_execute=False,
    examples_en=("find files modified today", "search for large videos"),
    examples_ar=("دور على ملفات اتعدلت النهارده", "دور على فيديوهات كبيرة"),
)

_reg(
    name="OS_FILE_WRITE",
    domain="file",
    required_slots=("path",),
    optional_slots=("content",),
    risk="medium",
    fast_execute=False,
    examples_en=("create a file called notes.txt", "write to report.txt"),
    examples_ar=("عمل ملف اسمه notes.txt", "اكتب في report.txt"),
)

# ── OS: screen / capture ───────────────────────────────────────────────────

_reg(
    name="OS_SCREEN_DESCRIBE",
    domain="os",
    required_slots=(),
    optional_slots=("mode",),
    risk="none",
    fast_execute=True,
    examples_en=("what's on my screen", "describe the screen", "what app is open"),
    examples_ar=("إيه اللي على الشاشة", "وصف الشاشة", "إيه التطبيق المفتوح"),
)

# ── Productivity ───────────────────────────────────────────────────────────

_reg(
    name="OS_TIMER",
    domain="os",
    required_slots=("seconds",),
    optional_slots=("label",),
    risk="none",
    fast_execute=True,
    examples_en=("set a timer for 5 minutes", "timer 30 seconds", "cancel timer"),
    examples_ar=("اضبط تايمر 5 دقايق", "تايمر 30 ثانية", "الغي التايمر"),
)

_reg(
    name="OS_REMINDER",
    domain="os",
    required_slots=("time_str",),
    optional_slots=("label",),
    risk="none",
    fast_execute=True,
    examples_en=("remind me at 3pm", "set a reminder for tomorrow"),
    examples_ar=("فكرني الساعة 3", "اعملي reminder بكره"),
)

_reg(
    name="OS_NOTE",
    domain="os",
    required_slots=(),
    optional_slots=("body", "name"),
    risk="none",
    fast_execute=True,
    examples_en=("take a note", "note: buy groceries", "create a new note"),
    examples_ar=("اعملي نوت", "نوت: اشتري بقالة", "اكتب نوتة"),
)

_reg(
    name="OS_EMAIL",
    domain="os",
    required_slots=(),
    optional_slots=("to", "subject", "body"),
    risk="low",
    fast_execute=False,
    examples_en=("draft an email to John about the meeting",
                 "open Outlook", "compose email"),
    examples_ar=("اعملي ايميل لـ John", "افتح أوتلوك", "ابعت ايميل"),
)

_reg(
    name="OS_CALENDAR",
    domain="os",
    required_slots=(),
    optional_slots=("event", "date", "time_str"),
    risk="none",
    fast_execute=False,
    examples_en=("add meeting to calendar", "open calendar"),
    examples_ar=("ضيف اجتماع في التقويم", "افتح التقويم"),
)

_reg(
    name="OS_CLIPBOARD",
    domain="os",
    required_slots=(),
    optional_slots=("text",),
    risk="none",
    fast_execute=True,
    examples_en=("read clipboard", "copy this to clipboard", "clear clipboard"),
    examples_ar=("اقرا الكليببورد", "انسخ ده للكليببورد", "امسح الكليببورد"),
)

_reg(
    name="OS_SYSINFO",
    domain="os",
    required_slots=(),
    optional_slots=("metric",),
    risk="none",
    fast_execute=True,
    examples_en=("how much RAM is in use", "what's my battery", "show CPU usage"),
    examples_ar=("كام RAM بيتاخد", "إيه البطارية", "وريني استخدام المعالج"),
)

_reg(
    name="OS_SETTINGS",
    domain="os",
    required_slots=(),
    optional_slots=("setting", "value"),
    risk="low",
    fast_execute=False,
    examples_en=("open settings", "go to display settings"),
    examples_ar=("افتح الإعدادات", "روح إعدادات الشاشة"),
)

_reg(
    name="OS_ROLLBACK",
    domain="os",
    required_slots=(),
    optional_slots=("action_id",),
    risk="medium",
    fast_execute=False,
    examples_en=("undo that", "rollback the last action"),
    examples_ar=("تراجع", "ارجع اللي عملته"),
)

# ── Safety / confirmation ──────────────────────────────────────────────────

_reg(
    name="OS_CONFIRMATION",
    domain="safety",
    required_slots=("token",),
    optional_slots=("second_factor",),
    risk="none",
    fast_execute=True,
    examples_en=("confirm ABC123", "yes 1234"),
    examples_ar=("اتأكد ABC123", "أيوه 1234"),
)

_reg(
    name="OS_PIN_CONFIRM",
    domain="safety",
    required_slots=("pin",),
    optional_slots=(),
    risk="none",
    fast_execute=True,
    examples_en=("1234", "my pin is 5678"),
    examples_ar=("1234", "الرقم 5678"),
)

# ── Meta / assistant control ───────────────────────────────────────────────

_reg(
    name="IDENTITY",
    domain="meta",
    required_slots=(),
    optional_slots=(),
    risk="none",
    fast_execute=True,
    examples_en=("who are you", "what can you do", "introduce yourself"),
    examples_ar=("مين انت", "إيه اللي تعرف تعمله", "عرف نفسك"),
)

_reg(
    name="VOICE_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("backend", "profile", "language"),
    risk="none",
    fast_execute=True,
    examples_en=("switch to hybrid voice", "set voice to Arabic"),
    examples_ar=("غير الصوت للهجين", "خلي الصوت عربي"),
)

_reg(
    name="MEMORY_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("key", "value", "language"),
    risk="none",
    fast_execute=True,
    examples_en=("remember I prefer English", "set language to Arabic"),
    examples_ar=("افتكر إني بفضل عربي", "اعمل اللغة عربي"),
)

_reg(
    name="PERSONA_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("profile",),
    risk="none",
    fast_execute=True,
    examples_en=("switch persona to professional", "persona set casual"),
    examples_ar=("غير الشخصية لـ professional", "خلي الشخصية casual"),
)

_reg(
    name="POLICY_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("policy_key", "value"),
    risk="low",
    fast_execute=True,
    examples_en=("enable dry-run mode", "disable app open"),
    examples_ar=("شغل dry-run", "وقف فتح التطبيقات"),
)

_reg(
    name="DEMO_MODE",
    domain="meta",
    required_slots=(),
    optional_slots=("state",),
    risk="none",
    fast_execute=True,
    examples_en=("demo mode on", "turn off demo mode"),
    examples_ar=("شغل demo mode", "اطفي demo mode"),
)

_reg(
    name="RESPONSE_MODE_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("mode",),
    risk="none",
    fast_execute=True,
    examples_en=("brief mode", "verbose mode", "short answers"),
    examples_ar=("خلي الردود قصيرة", "وسع الردود"),
)

# ── Batch / job queue ──────────────────────────────────────────────────────

_reg(
    name="BATCH_COMMAND",
    domain="meta",
    required_slots=("commands",),
    optional_slots=(),
    risk="medium",
    fast_execute=False,
    examples_en=("do X then Y then Z", "first open Chrome then search for news"),
    examples_ar=("افتح كروم وبعدين دور على الأخبار"),
)

_reg(
    name="COMMAND_CHAIN",
    domain="meta",
    required_slots=("commands",),
    optional_slots=(),
    risk="medium",
    fast_execute=False,
    examples_en=("open Chrome and go to gmail.com"),
    examples_ar=("افتح كروم وروح gmail.com"),
)

_reg(
    name="JOB_QUEUE_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("action", "job_id", "commands"),
    risk="none",
    fast_execute=True,
    examples_en=("add to queue: open Chrome", "show queue", "clear queue"),
    examples_ar=("ضيف للقايمة: افتح كروم", "وريني القايمة", "امسح القايمة"),
)

# ── Observability / audit ──────────────────────────────────────────────────

_reg(
    name="SEARCH_INDEX_COMMAND",
    domain="meta",
    required_slots=(),
    optional_slots=("action", "path"),
    risk="none",
    fast_execute=True,
    examples_en=("rescan apps", "rebuild search index"),
    examples_ar=("إعادة مسح التطبيقات", "ابني سيرش إندكس"),
)

_reg(
    name="AUDIT_VERIFY",
    domain="meta",
    required_slots=(),
    optional_slots=(),
    risk="none",
    fast_execute=True,
    examples_en=("audit verify", "check audit log"),
    examples_ar=("تحقق من سجل التدقيق"),
)

_reg(
    name="AUDIT_RESEAL",
    domain="meta",
    required_slots=(),
    optional_slots=(),
    risk="medium",
    fast_execute=False,
    examples_en=("reseal audit log"),
    examples_ar=("اختم سجل التدقيق"),
)

_reg(
    name="AUDIT_LOG_REPORT",
    domain="meta",
    required_slots=(),
    optional_slots=("limit",),
    risk="none",
    fast_execute=True,
    examples_en=("show audit log", "last 10 audit entries"),
    examples_ar=("وريني سجل التدقيق"),
)

_reg(
    name="OBSERVABILITY_REPORT",
    domain="meta",
    required_slots=(),
    optional_slots=(),
    risk="none",
    fast_execute=True,
    examples_en=("show observability report", "system health"),
    examples_ar=("وريني تقرير النظام"),
)

_reg(
    name="METRICS_REPORT",
    domain="meta",
    required_slots=(),
    optional_slots=(),
    risk="none",
    fast_execute=True,
    examples_en=("show metrics", "performance report"),
    examples_ar=("وريني الأداء"),
)

# ── Fallback ───────────────────────────────────────────────────────────────

_reg(
    name="LLM_QUERY",
    domain="chat",
    required_slots=(),
    optional_slots=("query",),
    risk="none",
    fast_execute=False,
    examples_en=("what is the capital of France", "tell me a joke",
                 "explain machine learning"),
    examples_ar=("إيه عاصمة فرنسا", "قولي نكتة", "اشرح machine learning"),
)


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_spec(intent: str) -> IntentSpec | None:
    """Return the IntentSpec for *intent*, or None if not registered."""
    return SCHEMA.get(str(intent or "").strip().upper())


def required_slots(intent: str) -> tuple[str, ...]:
    spec = get_spec(intent)
    return spec.required_slots if spec else ()


def optional_slots(intent: str) -> tuple[str, ...]:
    spec = get_spec(intent)
    return spec.optional_slots if spec else ()


def risk(intent: str) -> str:
    """Return risk level string ('none'|'low'|'medium'|'high') for *intent*."""
    spec = get_spec(intent)
    return spec.risk if spec else "low"


def is_fast(intent: str) -> bool:
    """True when the intent is eligible for the <100 ms fast-execute lane."""
    spec = get_spec(intent)
    return bool(spec and spec.fast_execute)


def domain(intent: str) -> str:
    spec = get_spec(intent)
    return spec.domain if spec else "unknown"


# ---------------------------------------------------------------------------
# Startup validation (called from orchestrator background prewarm)
# ---------------------------------------------------------------------------

def validate_schema_coverage() -> list[str]:
    """Check that every intent the semantic router and parser can emit has a schema entry.

    Returns a list of intent names that are missing from SCHEMA.
    Logs a WARNING for each missing entry — does NOT raise.
    """
    missing: list[str] = []

    try:
        from nlp.semantic_router import ROUTES
        for route in ROUTES:
            intent_name = str(route.get("name") or "").strip().upper()
            if intent_name and intent_name not in SCHEMA:
                logger.warning("intent_schema: no schema entry for semantic route %r", intent_name)
                missing.append(intent_name)
    except Exception as exc:
        logger.debug("intent_schema: could not import semantic_router for validation: %s", exc)

    try:
        from core.command_router import _PARSER_FASTPATH_INTENTS
        for intent_name in _PARSER_FASTPATH_INTENTS:
            name = str(intent_name or "").strip().upper()
            if name and name not in SCHEMA:
                logger.warning("intent_schema: no schema entry for fast-path intent %r", name)
                if name not in missing:
                    missing.append(name)
    except Exception as exc:
        logger.debug("intent_schema: could not import command_router for validation: %s", exc)

    if not missing:
        logger.debug("intent_schema: all %d intents have schema entries.", len(SCHEMA))

    return missing
