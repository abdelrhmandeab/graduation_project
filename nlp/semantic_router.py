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
from nlp.entity_types import EntityType

# ---------------------------------------------------------------------------
# Lazy-loaded globals — populated by _ensure_loaded()
# ---------------------------------------------------------------------------
_router = None
_loaded = False
_load_failed = False
ROUTER_READY = False


_ROUTE_ENTITY_TYPES: dict[str, tuple[EntityType, ...]] = {
    "OS_APP_OPEN": (EntityType.APP,),
    "OS_APP_CLOSE": (EntityType.APP,),
    "OS_FILE_SEARCH": (EntityType.PATH,),
    "OS_FILE_NAVIGATION": (EntityType.PATH,),
    "OS_SYSTEM_COMMAND": (EntityType.SYSTEM_FEATURE,),
    "OS_TIMER": (EntityType.DURATION, EntityType.DATE),
    "OS_CLIPBOARD": (EntityType.PATH,),
    "OS_SYSINFO": (EntityType.SYSTEM_FEATURE,),
    "OS_EMAIL": (EntityType.EMAIL, EntityType.PERSON),
    "OS_CALENDAR": (EntityType.DATE, EntityType.DURATION, EntityType.PERSON),
    "OS_SETTINGS": (EntityType.SYSTEM_FEATURE,),
    "VOICE_COMMAND": (EntityType.SYSTEM_FEATURE,),
    "JOB_QUEUE_COMMAND": (EntityType.DATE, EntityType.DURATION),
}


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
            "شغللي الانترنت",
            "افتح الانترنت",
            "شغل اليوتيوب",
            "افتح الجيميل",
            "افتحلي الجيميل",
            "شغل الفيسبوك",
            "افتح تيك توك",
            "افتح انستاجرام",
            "شغل ديسكورد",
            "افتح تيمز",
            "شغل زووم",
            "افتح الموسيقى",
            "شغل الصور",
            "افتحلي صور",
            "شغللي الفيديوهات",
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
            "اطفي البرنامج",
            "قفل المتصفح",
            "سكر الكروم",
            "اقفل الوورد",
            "اقفل الاكسل",
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
            "close the YouTube tab",
            "close the Chrome tab",
            "close the Facebook tab in the browser",
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
            "كبر الشاشة",
            "صغر الشاشة",
            "كبر النافذة",
            "صغر النافذة",
            "الاغنية اللي بعد كده",
            "وقف المزيكا",
            "افتح تاب جديد",
            "افتح tab جديدة",
            "تاب جديدة في البراوزر",
            "اقفل tab الـ يوتيوب",
            "اقفل tab الـ فيسبوك",
            "سكر تاب الـ browser",
            "اقفل تاب اليوتيوب في البراوزر",
            # additional volume / brightness
            "زود الصوت",
            "قلل الصوت",
            "الصوت واطي",
            "مش سامع خالص",
            "ارفع الفوليم",
            "خفض الفوليم",
            "صوت أعلى",
            "صوت أخفض",
            "اضبط الصوت",
            "ارفع السطوع",
            "زود الاضاءة",
            "قلل الإضاءة",
            "السطوع واطي",
            # screenshot variants
            "صور الشاشة",
            "خد لقطة شاشة",
            "سكرين شوت",
            # lock / sleep / power
            "اقفل الجهاز",
            "قفل الشاشة",
            "نوم الجهاز",
            "اوضع الكمبيوتر في السليب",
            "اوقف الكمبيوتر",
            "شتداون",
            # music controls
            "الاغنية اللي فاتت",
            "شغل المزيكا",
            "شغل الموسيقى",
            "وقف المزيكا",
            "الاغنية الجاية",
            # wifi / bluetooth
            "شغل الواي فاي",
            "وقف الواي فاي",
            "شغل البلوتوث",
            "وقف البلوتوث",
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
            # additional timer variants
            "نبهني بعد 10 دقايق",
            "نبهني بعد ساعة",
            "نبهني بعد نص ساعة",
            "ابدأ العد التنازلي",
            "تايمر ساعة",
            "اعملي منبه بعد دقيقة",
            "حط تايمر نص ساعة",
            "اعملي تايمر على 20 دقيقة",
            "عايز تايمر",
            "محتاج تايمر",
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
            "open outlook",
            "open outlook and draft an email",
            "launch outlook",
            "email someone about the project",
            "write an email to my boss",
            "ابعت ايميل",
            "افتح ايميل جديد",
            "ابعت ايميل عن الميتنج",
            "اكتب ايميل",
            "افتح أوتلوك",
            "افتح اوتلوك",
            "اعملي ايميل",
            "اكتب إيميل لحد",
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
            "reveal in explorer",
            "show in file explorer",
            "open file explorer",
            "open downloads in explorer",
            "open folder in file manager",
            "وريني الملفات",
            "روح على فولدر الداونلود",
            "اعمل فولدر جديد",
            "امسح الملف ده",
            "غير اسم الفايل",
            "افتح المستكشف",
            "وريني مكان الملف",
            "فين الملف ده",
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
        "name": "OS_NOTE",
        "utterances": [
            "create a new note",
            "take a note",
            "make a note",
            "note this down",
            "write a note",
            "new note",
            "save a note",
            "write this down",
            "اعملي نوتة",
            "اعمل نوتة جديدة",
            "نوتة جديدة",
            "اكتب نوتة",
            "سجل نوتة",
            "دون ملاحظة",
            "اكتبلي نوتة",
            "حفظ ملاحظة",
        ],
    },
    {
        "name": "IDENTITY",
        "utterances": [
            "who are you",
            "what are you",
            "introduce yourself",
            "what can you do",
            "tell me about yourself",
            "what is your name",
            "are you an AI",
            "are you jarvis",
            "what kind of assistant are you",
            "how do you work",
            "انت مين",
            "إنت مين",
            "مين انت",
            "عرفني بنفسك",
            "عرفني عليك",
            "بتعمل ايه",
            "تعرف تعمل ايه",
            "اسمك ايه",
            "انت جارفيس",
            "انت مساعد ايه",
            "انت بتشتغل ازاي",
        ],
    },
    {
        "name": "OS_SCREEN_DESCRIBE",
        "utterances": [
            "what's on my screen",
            "what do you see on my screen",
            "describe my screen",
            "show me what's open",
            "what's currently on the screen",
            "tell me what's on screen",
            "what apps are open",
            "what windows are open",
            "what am I looking at",
            "describe what's visible",
            "what's the active window",
            "what program am I in",
            "ايه اللي شايفه",
            "ايه اللي على الشاشة",
            "وصف الشاشة",
            "ايه اللي فاتح",
            "ايه اللي مفتوح دلوقتي",
            "ايه التطبيق اللي شغال",
            "انا فين دلوقتي",
            "ايه اللي بيحصل على الشاشة",
            "قولي ايه اللي شايفه",
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
            # additional Arabic question examples (Egyptian dialect)
            "ايه الفرق بين الذكاء الاصطناعي والبرمجة",
            "عرفني على البرمجة",
            "ازاي بتعمل مواقع انترنت",
            "ممكن تساعدني في الرياضيات",
            "سؤال عندي عن الفيزياء",
            "ليه السما زرقا",
            "ازاي اتعلم بايثون",
            "فين اقدر اشوف مباريات الكرة",
            "ايه هي النسبية",
            "احكيلي عن نابليون",
            "مين اخترع الكمبيوتر",
            "كيف تعمل الشبكات",
            "عايز اعرف اكتر عن مصر",
            "احكيلي عن ثقافة اليابان",
            "ايه الفرق بين المصر والسعودية",
            "ليه الكون كبير قوي",
            "ازاي بتشتغل الطائرات",
            "مين بنى الاهرامات",
            "ايه هو الانترنت",
            "اشرحلي الاقتصاد",
            "ايه معنى الديمقراطية",
            "حكاية الثورة الفرنسية ايه",
            "ازاي بتشتغل الذاكرة في المخ",
            "اقولي عن عمل الكمبيوتر",
        ],
    },
]

# Confidence threshold — below this, fall through to next tier
SEMANTIC_CONFIDENCE_THRESHOLD = 0.75


def _ensure_loaded() -> bool:
    """Lazy-load the sentence-transformers model and build the route index.

    Returns True if ready, False if unavailable.
    """
    global _router, _loaded, _load_failed, ROUTER_READY

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
        ROUTER_READY = True

        elapsed = time.perf_counter() - started
        logger.debug(
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


def classify_semantic_topk(text: str, k: int = 3) -> list[Tuple[str, float]]:
    """Classify text against every route, returning the top-k (intent, score) pairs.

    Sorted descending by score. Each route contributes its single best-matching
    utterance similarity. Returns [] if the router isn't ready or on failure —
    callers should treat an empty list the same as "no match".
    """
    if not text or not is_router_ready() or _router is None:
        return []

    try:
        np = _router["np"]
        model = _router["model"]

        query_embedding = model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )[0]  # shape: (dim,)

        scores = []
        for route in _router["routes"]:
            # Cosine similarity (embeddings are already normalized → dot product)
            similarities = route["embeddings"] @ query_embedding  # shape: (N,)
            max_sim = float(np.max(similarities))
            scores.append((route["name"], max_sim))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[: max(1, int(k))]

    except Exception as exc:
        logger.debug("Semantic router top-k classification failed: %s", exc)
        return []


def classify_semantic(text: str) -> Optional[Tuple[str, float]]:
    """Classify text using embedding similarity against route utterances.

    Returns (intent_name, confidence) or None if unavailable/below threshold.
    Confidence is the cosine similarity to the best-matching route.

    Thin back-compat wrapper over classify_semantic_topk — prefer the top-k
    function for new callers that need margin scoring.
    """
    topk = classify_semantic_topk(text, k=1)
    if not topk:
        return None

    best_intent, best_score = topk[0]
    if best_score < SEMANTIC_CONFIDENCE_THRESHOLD:
        return None

    return best_intent, best_score


def get_route_entity_types(intent_name: str) -> tuple[EntityType, ...]:
    """Return the entity types associated with a semantic route intent."""
    return _ROUTE_ENTITY_TYPES.get(str(intent_name or "").strip().upper(), ())


def is_available() -> bool:
    """Check if the semantic router is loaded and ready."""
    return is_router_ready()


def is_router_ready() -> bool:
    """Return True only after the model and route index are fully loaded."""
    return bool(ROUTER_READY and _loaded and _router is not None)


def prewarm() -> bool:
    """Force model load. Returns True if successful."""
    return _ensure_loaded()
