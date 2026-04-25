"""Semantic intent router — multilingual embedding similarity for paraphrase-tolerant NLU.

Uses sentence-transformers with paraphrase-multilingual-MiniLM-L12-v2 (~90MB).
Classifies bilingual (EN + Egyptian Arabic) utterances into command intents
in <5ms per call after initial model load.

Graceful fallback: if dependencies are missing, classify_semantic() returns None
and the cascade falls through to keyword NLP or LLM.
"""

from __future__ import annotations

import time
from typing import Optional, Tuple

from core.logger import logger

# ---------------------------------------------------------------------------
# Lazy-loaded globals — populated by _ensure_loaded()
# ---------------------------------------------------------------------------
_router = None
_loaded = False
_load_failed = False


# ---------------------------------------------------------------------------
# Route definitions — bilingual utterances for each intent family
# ---------------------------------------------------------------------------

_ROUTE_DEFINITIONS: list[dict] = [
    {
        "name": "OS_APP_OPEN",
        "utterances": [
            "open chrome",
            "launch notepad",
            "start excel",
            "open the browser",
            "run word",
            "open spotify",
            "launch firefox",
            "open file explorer",
            "open calculator",
            "افتح كروم",
            "شغل النوت باد",
            "افتحلي البرنامج",
            "ممكن تفتح الوورد",
            "شغللي اكسل",
            "افتح التطبيق",
            "افتحلي سبوتيفاي",
            "شغل الحاسبة",
            "افتح الملفات",
            "ممكن تفتحلي البرنامج بتاع النت",
        ],
    },
    {
        "name": "OS_APP_CLOSE",
        "utterances": [
            "close chrome",
            "quit notepad",
            "exit word",
            "kill the application",
            "close spotify",
            "stop firefox",
            "اقفل كروم",
            "سكر البرنامج",
            "قفل سبوتيفاي",
            "اقفل التطبيق",
            "سكر النوت باد",
        ],
    },
    {
        "name": "OS_FILE_SEARCH",
        "utterances": [
            "find my file report.pdf",
            "search for document",
            "where is my file",
            "look for presentation",
            "find homework assignment",
            "دور على ملف",
            "فين الملف بتاعي",
            "دورلي على الفايل",
            "ابحث عن ملف",
            "فين الدوكيومنت",
        ],
    },
    {
        "name": "OS_SYSTEM_COMMAND",
        "utterances": [
            "turn up the volume",
            "make it louder",
            "I can't hear",
            "raise the volume",
            "lower brightness",
            "lower the sound",
            "make it quieter",
            "mute the sound",
            "take a screenshot",
            "lock the computer",
            "shut down the pc",
            "restart the computer",
            "turn on wifi",
            "disable bluetooth",
            "turn off notifications",
            "maximize window",
            "minimize window",
            "snap window left",
            "next track",
            "pause music",
            "open new tab",
            "close tab",
            "search google for",
            "ارفع الصوت",
            "خفض السطوع",
            "اكتم الصوت",
            "خد سكرين شوت",
            "قفل الكمبيوتر",
            "اطفي الجهاز",
            "اعمل ريستارت",
            "شغل الواي فاي",
            "اطفي البلوتوث",
            "كبر الشباك",
            "صغر الشباك",
            "الاغنية اللي بعد كده",
            "وقف المزيكا",
            "افتح تاب جديد",
        ],
    },
    {
        "name": "OS_TIMER",
        "utterances": [
            "set a timer for 5 minutes",
            "timer 10 seconds",
            "set an alarm",
            "remind me in 30 minutes",
            "cancel the timer",
            "stop the alarm",
            "what timers are running",
            "list active timers",
            "حط تايمر 5 دقايق",
            "تايمر 10 ثواني",
            "صحيني بعد ساعة",
            "الغي التايمر",
            "وقف المنبه",
            "ايه التايمرات اللي شغالة",
            "صحيني بعد نص ساعة",
            "حط تايمر ربع ساعة",
            "تايمر دقيقتين",
            "اعملي تايمر",
            "شغل تايمر",
            "حط منبه بعد 5 دقايق",
        ],
    },
    {
        "name": "OS_CLIPBOARD",
        "utterances": [
            "what's in my clipboard",
            "read clipboard",
            "paste clipboard contents",
            "copy this text",
            "clear clipboard",
            "اللي في الكليب بورد",
            "اقرا الكليب بورد",
            "انسخ النص ده",
            "امسح الكليب بورد",
            "ايه اللي متنسخ",
        ],
    },
    {
        "name": "OS_SYSINFO",
        "utterances": [
            "battery status",
            "how much battery do I have",
            "check battery level",
            "system info",
            "CPU usage",
            "RAM usage",
            "disk space",
            "how much storage is left",
            "البطارية كام",
            "الرام قد ايه",
            "معلومات النظام",
            "استهلاك المعالج",
            "الهارد فاضي قد ايه",
            "الشحن كام في المية",
        ],
    },
    {
        "name": "OS_EMAIL",
        "utterances": [
            "draft an email",
            "compose email",
            "new email",
            "send email to john",
            "write an email about the meeting",
            "open email draft",
            "ابعت ايميل",
            "افتح ايميل جديد",
            "ابعت ايميل عن الميتنج",
            "اكتب ايميل",
        ],
    },
    {
        "name": "OS_CALENDAR",
        "utterances": [
            "create calendar event",
            "add meeting to calendar",
            "schedule event",
            "new calendar event tomorrow at 3pm",
            "add appointment",
            "set up a meeting",
            "اعمل حدث في الكالندر",
            "ضيف ميتنج",
            "اعمل موعد",
            "حط ايفنت بكره الساعة 3",
        ],
    },
    {
        "name": "OS_SETTINGS",
        "utterances": [
            "open settings",
            "open windows settings",
            "show me the settings",
            "go to settings",
            "open display settings",
            "open wifi settings",
            "open bluetooth settings",
            "open sound settings",
            "open privacy settings",
            "open battery settings",
            "open windows update",
            "open notifications settings",
            "open background settings",
            "افتح الاعدادات",
            "افتح الإعدادات",
            "افتحلي الاعدادات",
            "افتح اعدادات الشاشة",
            "افتح اعدادات الواي فاي",
            "افتح اعدادات الصوت",
            "افتح اعدادات البلوتوث",
            "افتح اعدادات البطارية",
            "افتح تحديث ويندوز",
            "روح على الاعدادات",
            "ودّيني للاعدادات",
        ],
    },
    {
        "name": "OS_FILE_NAVIGATION",
        "utterances": [
            "list files in this folder",
            "go to documents folder",
            "change directory to downloads",
            "show folder contents",
            "create a new folder",
            "delete this file",
            "rename the file",
            "move file to desktop",
            "وريني الملفات",
            "روح على فولدر الداونلود",
            "اعمل فولدر جديد",
            "امسح الملف ده",
            "غير اسم الفايل",
        ],
    },
    {
        "name": "VOICE_COMMAND",
        "utterances": [
            "turn speech on",
            "enable voice",
            "disable speech",
            "mute voice output",
            "be quiet",
            "stop talking",
            "voice status",
            "شغل الصوت",
            "فعل النطق",
            "اطفي الصوت",
            "اكتم الكلام",
            "اسكت",
            "حالة الصوت",
        ],
    },
    {
        "name": "JOB_QUEUE_COMMAND",
        "utterances": [
            "in 5 minutes open chrome",
            "remind me in 10 minutes to stretch",
            "after 30 seconds play music",
            "schedule a task",
            "show queued jobs",
            "cancel scheduled task",
            "بعد 5 دقايق افتح كروم",
            "فكرني بعد 10 دقايق",
            "بعد نص ساعة شغل موسيقى",
            "وريني المهام المجدولة",
        ],
    },
    {
        "name": "LLM_QUERY",
        "utterances": [
            "what is quantum computing",
            "tell me about egypt",
            "explain machine learning",
            "who is elon musk",
            "what's the weather like",
            "give me the latest news",
            "how does electricity work",
            "what are the pyramids",
            "tell me a joke",
            "what time is it in tokyo",
            "ايه هو الذكاء الاصطناعي",
            "احكيلي عن التاريخ",
            "اشرحلي الفيزياء",
            "مين هو ايلون ماسك",
            "الجو عامل ازاي",
            "ايه اخر الاخبار",
            "احكيلي نكته",
            "ازاي الكهربا بتشتغل",
            "الاهرامات اتبنت امتى",
        ],
    },
]

# Confidence threshold — below this, fall through to next tier
SEMANTIC_CONFIDENCE_THRESHOLD = 0.75


def _ensure_loaded() -> bool:
    """Lazy-load the sentence-transformers model and build the route index.

    Returns True if ready, False if unavailable.
    """
    global _router, _loaded, _load_failed

    if _loaded:
        return _router is not None
    if _load_failed:
        return False

    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
    except ImportError:
        logger.info("sentence-transformers not installed — semantic router disabled.")
        _load_failed = True
        return False
    except Exception as exc:
        logger.warning(
            "sentence-transformers import failed (%s) — semantic router disabled.",
            exc,
        )
        _load_failed = True
        return False

    started = time.perf_counter()
    try:
        model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )

        # Pre-compute embeddings for all route utterances
        routes = []
        for route_def in _ROUTE_DEFINITIONS:
            name = route_def["name"]
            utterances = route_def["utterances"]
            embeddings = model.encode(utterances, normalize_embeddings=True, show_progress_bar=False)
            routes.append({
                "name": name,
                "embeddings": embeddings,  # shape: (N, dim)
            })

        _router = {
            "model": model,
            "routes": routes,
            "np": np,
        }
        _loaded = True

        elapsed = time.perf_counter() - started
        logger.info(
            "Semantic router loaded in %.2fs (%d routes, %d total utterances).",
            elapsed,
            len(routes),
            sum(len(r["embeddings"]) for r in routes),
        )
        return True

    except Exception as exc:
        logger.warning("Semantic router load failed: %s", exc)
        _load_failed = True
        return False


def classify_semantic(text: str) -> Optional[Tuple[str, float]]:
    """Classify text using embedding similarity against route utterances.

    Returns (intent_name, confidence) or None if unavailable/below threshold.
    Confidence is the cosine similarity to the best-matching route.
    """
    if not text or not _ensure_loaded() or _router is None:
        return None

    try:
        np = _router["np"]
        model = _router["model"]

        # Encode the query
        query_embedding = model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )[0]  # shape: (dim,)

        best_intent = "LLM_QUERY"
        best_score = 0.0

        for route in _router["routes"]:
            # Cosine similarity (embeddings are already normalized → dot product)
            similarities = route["embeddings"] @ query_embedding  # shape: (N,)
            max_sim = float(np.max(similarities))
            if max_sim > best_score:
                best_score = max_sim
                best_intent = route["name"]

        if best_score < SEMANTIC_CONFIDENCE_THRESHOLD:
            return None

        return best_intent, best_score

    except Exception as exc:
        logger.debug("Semantic router classification failed: %s", exc)
        return None


def is_available() -> bool:
    """Check if the semantic router is loaded and ready."""
    return _loaded and _router is not None


def prewarm() -> bool:
    """Force model load. Returns True if successful."""
    return _ensure_loaded()
