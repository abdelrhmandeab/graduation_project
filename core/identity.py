"""Jarvis self-introduction pool — varied, funny, Egyptian-Arabic responses.

Pool mode (default): picks a random entry from a curated list, avoiding the
last-used index so consecutive asks always get a different answer.
LLM mode (opt-in): sends a one-shot persona prompt at higher temperature.
"""

from __future__ import annotations

import random
from typing import Optional

from core.config import IDENTITY_MODE, IDENTITY_AVOID_REPEAT, IDENTITY_LLM_TEMPERATURE
from core.logger import logger

# ---------------------------------------------------------------------------
# Intro pools
# ---------------------------------------------------------------------------

INTRO_POOL_EN: list[str] = [
    "I'm Jarvis — your personal voice assistant. I open apps, set timers, control your volume, answer questions, and more. Just talk to me.",
    "Name's Jarvis. I run on your machine, listen for your voice, and get things done fast. Think of me as your hands-free co-pilot.",
    "Jarvis here. I speak English and Egyptian Arabic, work offline, and I'm pretty quick. What do you need?",
    "I'm Jarvis — a local AI assistant. No cloud required. I understand voice commands and answer questions. Fire away.",
    "Jarvis at your service. I can open files, search the web, set alarms, control media, and have a decent conversation. What's up?",
    "The name's Jarvis. I was built to save you clicks. Tell me what you need and I'll handle it — or at least try my best.",
    "I'm Jarvis, your on-device assistant. I understand both English and Egyptian Arabic, which is a flex not many assistants have.",
    "Jarvis speaking. I run locally on this machine, so I'm fast and private. Apps, timers, volume, web queries — just ask.",
    "I'm Jarvis. I was trained to understand Egyptian Arabic colloquial speech — no MSA needed. Talk to me naturally.",
    "Hi, Jarvis here. I live on your computer, not in the cloud. That means I'm fast, private, and always available. What can I do for you?",
    "I'm Jarvis. Technically I'm a collection of models and scripts, but practically I'm the one who opens your apps when you're too lazy to click.",
    "Jarvis, reporting for duty. I handle voice commands, answer questions, and occasionally make dad jokes. Mostly the first two though.",
]

INTRO_POOL_AR: list[str] = [
    "أنا جارفيس، مساعدك الصوتي الشخصي. بشغل برامج، بضبط تايمرات، بتحكم في الصوت، وبرد على أسئلتك. قولي اللي محتاجه.",
    "اسمي جارفيس. بشتغل على جهازك مباشرةً، بسمع صوتك، وبنفذ الأوامر بسرعة. فكرني زي المساعد اللي بيشتغل من غير ما تحرك أوتار.",
    "أنا جارفيس — بتكلم عربي مصري وانجليزي، وبشتغل من غير نت. سريع وموثوق. إيه اللي تحتاجه؟",
    "جارفيس بخدمتك. بفتح ملفات، بدور على حاجات، بضبط منبهات، وبتحكم في الميديا. مش بس كده — بتكلم كمان.",
    "أنا جارفيس. اتصنع عشان يريحك من الضغط على الأزرار. قولي عايز إيه وأنا باخد دوري.",
    "جارفيس موجود. بشتغل على الجهاز ده مباشرةً، يعني سريع وخصوصيتك محفوظة. برامج، تايمر، صوت، أسئلة — كلها عندي.",
    "أنا جارفيس، المساعد اللي اتبنى يفهم العامية المصرية من غير ما تلخبط في الفصحى. كلمني بالطبيعي.",
    "جارفيس أنا. مش في السحاب، موجود على جهازك. ده معناه إني أسرع وأكتر خصوصية من أغلب المساعدين التانيين.",
    "أنا جارفيس. بفهم اللي بتقوله، بنفذ الأوامر، وبحاول أكون مفيد في أكتر من موضوع. الساعة دي أنا كويس جداً.",
    "اسمي جارفيس، وأنا مساعدك الصوتي. تقنياً أنا مجموعة موديلات وكود، بس عملياً أنا اللي بيفتحلك البرامج لما تكون كسلان.",
    "جارفيس هنا. بتكلم مصري، بشتغل أوفلاين، وبنفذ أوامر صوتية بسرعة. قولي بتحتاج إيه وخليني أبقى مفيد.",
    "أنا جارفيس — مساعد ذكي محلي على جهازك. بأتكلم معاك، بفتحلك حاجات، وبضبطلك إعدادات. وبعمل نكت أحياناً بس ده مش الأهم.",
]

# Module-level state for avoid-repeat tracking
_last_en_index: int = -1
_last_ar_index: int = -1


def _pick_avoiding_last(pool: list[str], last_idx: int) -> tuple[str, int]:
    """Pick a random entry from pool, avoiding last_idx when possible."""
    if len(pool) <= 1:
        return pool[0], 0
    candidates = [i for i in range(len(pool)) if i != last_idx]
    idx = random.choice(candidates)
    return pool[idx], idx


def get_identity_reply(language: str, persona: Optional[dict] = None) -> str:
    """Return a self-introduction string in the requested language.

    In pool mode (default): picks randomly from INTRO_POOL_EN / INTRO_POOL_AR,
    avoiding the previously returned entry.
    In llm mode: calls the LLM with a high-temperature one-shot prompt.
    """
    global _last_en_index, _last_ar_index

    is_ar = str(language or "").strip().lower().startswith("ar")
    mode = str(IDENTITY_MODE or "pool").strip().lower()

    if mode == "llm":
        return _get_llm_identity_reply(language, persona)

    # Pool mode
    if is_ar:
        pool = INTRO_POOL_AR
        if IDENTITY_AVOID_REPEAT:
            text, _last_ar_index = _pick_avoiding_last(pool, _last_ar_index)
        else:
            text = random.choice(pool)
    else:
        pool = INTRO_POOL_EN
        if IDENTITY_AVOID_REPEAT:
            text, _last_en_index = _pick_avoiding_last(pool, _last_en_index)
        else:
            text = random.choice(pool)

    return text


def _get_llm_identity_reply(language: str, persona: Optional[dict] = None) -> str:
    """One-shot LLM call for a varied self-intro at higher temperature."""
    is_ar = str(language or "").strip().lower().startswith("ar")
    try:
        from llm.ollama_client import ask_llm

        if is_ar:
            prompt = (
                "أنت جارفيس، مساعد صوتي ذكي بالعامية المصرية. "
                "قدم نفسك في جملة أو اتنين بأسلوب طريف وإنساني، "
                "من غير تكرار ونص مكتوب. رد بالعامية المصرية فقط."
            )
        else:
            prompt = (
                "You are Jarvis, a local AI voice assistant. "
                "Introduce yourself in one or two sentences in a witty, human way. "
                "Be brief, natural, and a little funny. English only."
            )

        response = ask_llm(
            prompt,
            temperature=float(IDENTITY_LLM_TEMPERATURE),
            num_ctx=128,
        )
        return str(response or "").strip() or get_identity_reply(language)
    except Exception as exc:
        logger.debug("LLM identity reply failed, falling back to pool: %s", exc)
        return get_identity_reply(language)
