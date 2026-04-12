import re


_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_LATIN_CHAR_RE = re.compile(r"[A-Za-z]")


def normalize_language(language):
    key = (language or "").strip().lower()
    return "ar" if key == "ar" else "en"


def detect_language_hint(text, fallback="en"):
    value = str(text or "")
    has_ar = bool(_ARABIC_CHAR_RE.search(value))
    has_en = bool(_LATIN_CHAR_RE.search(value))
    if has_ar and not has_en:
        return "ar"
    if has_en and not has_ar:
        return "en"
    return normalize_language(fallback)


_TEMPLATES = {
    "please_clarify_intent": {
        "en": "Please clarify your intent.",
        "ar": "وضحلي طلبك اكتر.",
    },
    "clarification_cancelled": {
        "en": "Clarification cancelled.",
        "ar": "تمام، لغيت التوضيح.",
    },
    "confirmation_failed": {
        "en": "Confirmation failed: {message}",
        "ar": "التأكيد فشل: {message}",
    },
    "confirmation_failed_with_usage": {
        "en": "Confirmation failed: {message} Use `confirm {token} <PIN_or_passphrase>`.",
        "ar": "التأكيد فشل: {message} استخدم `confirm {token} <PIN_or_passphrase>`.",
    },
    "unsupported_confirmation_payload": {
        "en": "Unsupported confirmation payload.",
        "ar": "محتوى التأكيد مش مدعوم.",
    },
    "confirmation_cancelled": {
        "en": "Pending confirmation cancelled.",
        "ar": "تمام، لغيت التأكيد المعلق.",
    },
    "missing_pending_confirmation": {
        "en": "There is no pending confirmation token right now.",
        "ar": "مفيش كود تأكيد معلق دلوقتي.",
    },
    "missing_last_file_delete": {
        "en": "I need a recent file reference first. Say `delete <path>` once, then you can say `delete it`.",
        "ar": "محتاج مرجع قريب لملف الاول. قول `delete <path>` مرة واحدة وبعدها تقدر تقول `امسحه`.",
    },
    "missing_last_file_rename": {
        "en": "I do not know which file to rename yet. Specify a file first, then use `rename it to <name>`.",
        "ar": "لسه مش عارف انهي ملف عايز تعيد تسميته. حدد ملف الاول وبعدها استخدم `rename it to <name>`.",
    },
    "missing_last_file_move": {
        "en": "I do not know which file to move yet. Specify a file first, then use `move it to <destination>`.",
        "ar": "لسه مش عارف انهي ملف عايز تنقله. حدد ملف الاول وبعدها استخدم `move it to <destination>`.",
    },
    "missing_last_app_open": {
        "en": "I need a recent app reference first. Open or mention an app, then you can say `open it`.",
        "ar": "محتاج مرجع قريب لبرنامج الاول. افتح او اذكر برنامج وبعدها تقدر تقول `افتحه`.",
    },
    "missing_last_app_close": {
        "en": "I need a recent app reference first. Open or mention an app, then you can say `close it`.",
        "ar": "محتاج مرجع قريب لبرنامج الاول. افتح او اذكر برنامج وبعدها تقدر تقول `اقفله`.",
    },
    "stale_followup_reference": {
        "en": "That reference is too old, so I do not want to guess. Please mention the app or file again.",
        "ar": "المرجع ده قديم قوي، فمش هخمن. قول اسم البرنامج او الملف تاني.",
    },
    "destructive_followup_requires_explicit_target": {
        "en": "For safety, I do not delete with vague references like `delete it`. Please say the exact file or path.",
        "ar": "عشان الامان، مش همسح بحاجة مبهمة زي `delete it`. قول الملف او المسار بشكل واضح.",
    },
    "destructive_followup_low_confidence": {
        "en": "I need a more certain file target before deleting. Please specify the file/path explicitly.",
        "ar": "محتاج هدف ملف اوضح قبل المسح. حدد الملف او المسار بشكل واضح.",
    },
    "missing_followup_reference": {
        "en": "I need a recent app or file reference first. Mention one, then try again.",
        "ar": "محتاج مرجع قريب لبرنامج او ملف الاول. اذكره وبعدين جرب تاني.",
    },
    "followup_reference_conflict": {
        "en": "I found both a recent app and file with similar recency. Say `open the app` or `open the file`.",
        "ar": "لقيت برنامج وملف قريبين في نفس الوقت. قول `open the app` او `open the file`.",
    },
    "missing_second_recent_app": {
        "en": "I only have one recent app. Mention another app first, then say `open both` or `close them`.",
        "ar": "عندي برنامج حديث واحد بس. اذكر برنامج تاني الاول وبعدها قول `open both` او `close them`.",
    },
    "missing_filename_search": {
        "en": "Please provide a filename to search for.",
        "ar": "قول اسم الملف اللي عايز تدور عليه.",
    },
    "missing_app_name_open": {
        "en": "Please provide an app name to open.",
        "ar": "قول اسم البرنامج اللي عايز تفتحه.",
    },
    "missing_app_name_close": {
        "en": "Please provide an app name to close.",
        "ar": "قول اسم البرنامج اللي عايز تقفله.",
    },
    "file_not_found": {
        "en": "File not found.",
        "ar": "الملف غير موجود.",
    },
    "low_confidence_action_like_query": {
        "en": "I am not fully sure which action you want. Please rephrase as one clear command (for example: `open app notepad`, `find file notes.txt`, or `delete file.txt`).",
        "ar": "لسه مش متأكد انت عايز انهي إجراء. صيغه كأمر واحد واضح (مثال: `open app notepad` او `find file notes.txt` او `delete file.txt`).",
    },
    "low_confidence_unclear_query": {
        "en": "I did not fully catch that short phrase. Please say one clear request (for example: `search web for market news` or `open app calculator`).",
        "ar": "العبارة القصيرة دي موضحتش كفاية. قول طلب واحد واضح (مثال: `دور في الويب على اخبار السوق` او `افتح البرنامج الحاسبة`).",
    },
    "low_entity_confidence": {
        "en": "I need a clearer target before I execute this. Please include the exact app name, file name, or path.",
        "ar": "محتاج هدف اوضح قبل التنفيذ. قول اسم البرنامج او اسم الملف او المسار بدقة.",
    },
    "multiple_actions_detected": {
        "en": "I detected more than one action in one command. Please split it into one step at a time.",
        "ar": "لقيت اكتر من إجراء في نفس الأمر. قسمه خطوة خطوة.",
    },
    "open_target_ambiguous_intro": {
        "en": "I want to confirm before acting. Did you mean:",
        "ar": "عايز أتأكد قبل التنفيذ. تقصد:",
    },
    "open_target_option_app": {
        "en": "Open application '{target}'",
        "ar": "فتح البرنامج '{target}'",
    },
    "open_target_option_path": {
        "en": "Open folder/path '{target}'",
        "ar": "فتح المجلد/المسار '{target}'",
    },
    "reply_with_1_2_app_folder_or_cancel": {
        "en": "Reply with `1`, `2`, `app`, `folder`, or `cancel`.",
        "ar": "اكتب `1` او `2` او `app` او `folder` او `cancel`.",
    },
    "app_ambiguous_open_intro": {
        "en": "I found multiple app matches. Which one should I open?",
        "ar": "لقيت اكتر من برنامج مطابق. انهي واحد افتحه؟",
    },
    "app_ambiguous_close_intro": {
        "en": "I found multiple app matches. Which one should I close?",
        "ar": "لقيت اكتر من برنامج مطابق. انهي واحد اقفله؟",
    },
    "file_ambiguous_intro": {
        "en": "I found multiple files for '{filename}'. Which one do you mean?",
        "ar": "وجدت عدة ملفات مطابقة لـ '{filename}'. اي واحد تقصد؟",
    },
    "reply_with_number_or_cancel": {
        "en": "Reply with the number (for example `1`) or `cancel`.",
        "ar": "اكتب الرقم (مثال `1`) او `cancel`.",
    },
    "reply_with_number_cancel_or_more": {
        "en": "Reply with a number, `show more`, or `cancel`.",
        "ar": "اكتب رقما او `show more` او `cancel`.",
    },
    "clarification_page_indicator": {
        "en": "Showing options page {page} of {total_pages}.",
        "ar": "اعرض خيارات الصفحة {page} من {total_pages}.",
    },
    "clarification_confidence_line": {
        "en": "I am {confidence_percent}% confident this is the right disambiguation.",
        "ar": "انا واثق بنسبة {confidence_percent}% إن ده التوضيح الصح.",
    },
    "no_more_options": {
        "en": "There are no more options to show.",
        "ar": "مفيش اختيارات تانية تتعرض.",
    },
    "clarification_none_selected": {
        "en": "Understood. None of these options match. Please restate your target.",
        "ar": "تمام. ولا اختيار من دول مناسب. صيّغ هدفك تاني.",
    },
    "clarification_retry_with_examples": {
        "en": "I still need a direct selection. Reply like `1`, `the chrome one`, `the first app`, or `cancel`.",
        "ar": "لسه محتاج اختيار مباشر. اكتب زي `1` او `برنامج كروم` او `الاختيار الاول` او `cancel`.",
    },
    "clarification_retry_unclear_query": {
        "en": "I still need one clear request. For example: `explain how the ozone layer works` or `search web for ozone layer`.",
        "ar": "لسه محتاج طلب واحد واضح. مثال: `اشرح ازاي طبقة الاوزون بتشتغل` او `دور في الويب على طبقة الاوزون`.",
    },
    "response_mode_explain_on": {
        "en": "Explain mode is on. I will include short reasoning with command outputs.",
        "ar": "تم تفعيل وضع الشرح. سأضيف سببا مختصرا مع نواتج الاوامر.",
    },
    "response_mode_concise_on": {
        "en": "Concise mode is on. I will keep outputs short.",
        "ar": "تم تفعيل الوضع المختصر. سأجعل المخرجات اقصر.",
    },
    "response_mode_default_on": {
        "en": "Default mode restored.",
        "ar": "رجعت للوضع الافتراضي.",
    },
    "response_explain_bridge": {
        "en": "Why:",
        "ar": "السبب:",
    },
    "response_mode_explain_suffix": {
        "en": "{bridge} interpreted as intent={intent}, action={action}.",
        "ar": "{bridge} تم تفسير الطلب كـ intent={intent} و action={action}.",
    },
    "confirmation_prompt_base": {
        "en": "Confirmation required (risk: {risk_tier}) for: {description}. Say `confirm {token}`",
        "ar": "التأكيد مطلوب (مستوى المخاطر: {risk_tier}) للعملية: {description}. قول `confirm {token}`",
    },
    "confirmation_prompt_second_factor": {
        "en": " and provide PIN/passphrase as second factor.",
        "ar": " مع ادخال الرقم السري/عبارة المرور كعامل ثان.",
    },
    "confirmation_prompt_timeout": {
        "en": " within {timeout_seconds} seconds.",
        "ar": " خلال {timeout_seconds} ثانية.",
    },
}


_ANTI_REPETITION_PREFIXES = {
    "en": {
        "assistant": ["Understood. ", "Got it. "],
        "formal": ["Certainly. ", "Understood. "],
        "professional": ["Certainly. ", "Noted. "],
        "casual": ["Sure thing. ", "Got you. "],
        "friendly": ["Happy to help. ", "Absolutely. "],
        "brief": ["Noted. ", "Okay. "],
    },
    "ar": {
        "assistant": ["حاضر. ", "تمام. "],
        "formal": ["بكل تاكيد. ", "مفهوم. "],
        "professional": ["بكل تاكيد. ", "تم. "],
        "casual": ["تمام. ", "حاضر. "],
        "friendly": ["بكل سرور. ", "تم. "],
        "brief": ["تم. ", "حاضر. "],
    },
}


def render_template(key, language="en", **kwargs):
    lang = normalize_language(language)
    table = _TEMPLATES.get(key) or {}
    template = table.get(lang) or table.get("en") or ""
    try:
        return template.format(**kwargs)
    except Exception:
        return template


def anti_repetition_prefixes(language, persona):
    lang = normalize_language(language)
    by_persona = _ANTI_REPETITION_PREFIXES.get(lang) or _ANTI_REPETITION_PREFIXES["en"]
    key = (persona or "").strip().lower()
    return list(by_persona.get(key) or by_persona.get("assistant") or [""])


def format_confirmation_prompt(
    description,
    token,
    *,
    risk_tier="medium",
    timeout_seconds=45,
    require_second_factor=False,
    language="en",
):
    message = render_template(
        "confirmation_prompt_base",
        language,
        risk_tier=str(risk_tier or "medium"),
        description=str(description or ""),
        token=str(token or ""),
    )
    if require_second_factor:
        message += render_template("confirmation_prompt_second_factor", language)
    message += render_template(
        "confirmation_prompt_timeout",
        language,
        timeout_seconds=max(1, int(timeout_seconds or 1)),
    )
    return message
