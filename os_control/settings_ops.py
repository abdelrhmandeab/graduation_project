"""Open Windows Settings pages via the ms-settings: URI scheme.

Works on Windows 10 and 11 without admin privileges, no GPU, no internet.
Falls back gracefully if the Settings app cannot be launched (e.g. on Server SKUs).
"""

import os
import subprocess

from core.logger import logger


# Mapping of friendly aliases (EN + Egyptian Arabic) to ms-settings: URIs.
# Source: https://learn.microsoft.com/windows/uwp/launch-resume/launch-settings-app
_SETTINGS_PAGES = {
    # Display / appearance
    "display": "ms-settings:display",
    "screen": "ms-settings:display",
    "العرض": "ms-settings:display",
    "الشاشة": "ms-settings:display",
    "السطوع": "ms-settings:display",

    "personalization": "ms-settings:personalization",
    "themes": "ms-settings:themes",
    "wallpaper": "ms-settings:personalization-background",
    "background": "ms-settings:personalization-background",
    "خلفية": "ms-settings:personalization-background",
    "الخلفية": "ms-settings:personalization-background",
    "الثيم": "ms-settings:themes",

    "lockscreen": "ms-settings:lockscreen",
    "lock screen": "ms-settings:lockscreen",
    "شاشة القفل": "ms-settings:lockscreen",

    # Network / connectivity
    "network": "ms-settings:network",
    "internet": "ms-settings:network",
    "الشبكة": "ms-settings:network",
    "النت": "ms-settings:network",
    "الانترنت": "ms-settings:network",

    "wifi": "ms-settings:network-wifi",
    "wi-fi": "ms-settings:network-wifi",
    "wireless": "ms-settings:network-wifi",
    "واي فاي": "ms-settings:network-wifi",
    "وايفاي": "ms-settings:network-wifi",

    "bluetooth": "ms-settings:bluetooth",
    "بلوتوث": "ms-settings:bluetooth",

    "vpn": "ms-settings:network-vpn",
    "في بي ان": "ms-settings:network-vpn",

    "airplane": "ms-settings:network-airplanemode",
    "airplane mode": "ms-settings:network-airplanemode",
    "وضع الطيران": "ms-settings:network-airplanemode",

    # Sound / mouse / devices
    "sound": "ms-settings:sound",
    "audio": "ms-settings:sound",
    "الصوت": "ms-settings:sound",

    "mouse": "ms-settings:mousetouchpad",
    "touchpad": "ms-settings:devices-touchpad",
    "الماوس": "ms-settings:mousetouchpad",
    "الفأرة": "ms-settings:mousetouchpad",

    "keyboard": "ms-settings:keyboard",
    "الكيبورد": "ms-settings:keyboard",
    "لوحة المفاتيح": "ms-settings:keyboard",

    "printers": "ms-settings:printers",
    "printer": "ms-settings:printers",
    "الطابعة": "ms-settings:printers",
    "الطابعات": "ms-settings:printers",

    "devices": "ms-settings:bluetooth",
    "الاجهزة": "ms-settings:bluetooth",
    "الأجهزة": "ms-settings:bluetooth",

    # System
    "battery": "ms-settings:batterysaver",
    "البطارية": "ms-settings:batterysaver",
    "الشحن": "ms-settings:batterysaver",

    "power": "ms-settings:powersleep",
    "sleep": "ms-settings:powersleep",
    "الطاقة": "ms-settings:powersleep",
    "السكون": "ms-settings:powersleep",

    "storage": "ms-settings:storagesense",
    "disk": "ms-settings:storagesense",
    "التخزين": "ms-settings:storagesense",
    "الهارد": "ms-settings:storagesense",

    "about": "ms-settings:about",
    "system info": "ms-settings:about",
    "معلومات النظام": "ms-settings:about",

    # Privacy / security
    "privacy": "ms-settings:privacy",
    "الخصوصية": "ms-settings:privacy",

    "windows update": "ms-settings:windowsupdate",
    "update": "ms-settings:windowsupdate",
    "updates": "ms-settings:windowsupdate",
    "تحديث ويندوز": "ms-settings:windowsupdate",
    "التحديثات": "ms-settings:windowsupdate",

    "windows security": "ms-settings:windowsdefender",
    "defender": "ms-settings:windowsdefender",
    "antivirus": "ms-settings:windowsdefender",
    "ويندوز سيكيوريتي": "ms-settings:windowsdefender",

    "camera privacy": "ms-settings:privacy-webcam",
    "microphone privacy": "ms-settings:privacy-microphone",

    # Apps
    "apps": "ms-settings:appsfeatures",
    "applications": "ms-settings:appsfeatures",
    "default apps": "ms-settings:defaultapps",
    "البرامج": "ms-settings:appsfeatures",
    "التطبيقات": "ms-settings:appsfeatures",

    "startup": "ms-settings:startupapps",
    "startup apps": "ms-settings:startupapps",
    "بدء التشغيل": "ms-settings:startupapps",

    # Notifications / focus
    "notifications": "ms-settings:notifications",
    "الاشعارات": "ms-settings:notifications",
    "الإشعارات": "ms-settings:notifications",

    "focus": "ms-settings:quiethours",
    "focus assist": "ms-settings:quiethours",
    "do not disturb": "ms-settings:quiethours",
    "عدم الازعاج": "ms-settings:quiethours",
    "عدم الإزعاج": "ms-settings:quiethours",

    # Accounts
    "accounts": "ms-settings:yourinfo",
    "account": "ms-settings:yourinfo",
    "user account": "ms-settings:yourinfo",
    "الحساب": "ms-settings:yourinfo",
    "حسابي": "ms-settings:yourinfo",

    "sign in options": "ms-settings:signinoptions",
    "password": "ms-settings:signinoptions",
    "كلمة السر": "ms-settings:signinoptions",
    "كلمة المرور": "ms-settings:signinoptions",

    # Top-level fallback
    "settings": "ms-settings:",
    "الاعدادات": "ms-settings:",
    "الإعدادات": "ms-settings:",
    "الضبط": "ms-settings:",
}


def _normalize(value):
    return " ".join(str(value or "").strip().lower().split())


def resolve_settings_uri(query):
    """Map a free-form query to an ms-settings: URI, or empty string if no match.

    Tries exact match first, then substring containment so "open display settings"
    or "روح على اعدادات الشبكة" both resolve cleanly.
    """
    text = _normalize(query)
    if not text:
        return ""

    direct = _SETTINGS_PAGES.get(text)
    if direct:
        return direct

    # Strip common verbs to improve match rate.
    for prefix in (
        "open ", "launch ", "show ", "go to ", "take me to ", "settings ",
        "افتح ", "افتحلي ", "روح على ", "ودّيني ", "وريني ", "اعدادات ", "إعدادات ",
    ):
        if text.startswith(prefix):
            stripped = text[len(prefix):].strip()
            if not stripped:
                continue
            uri = _SETTINGS_PAGES.get(stripped)
            if uri:
                return uri

    # Substring match — prefer the longest specific alias.
    # Generic top-level aliases ("settings", "الاعدادات", ...) are skipped here
    # so phrases like "open display settings" resolve to the display page,
    # not the top-level Settings root.
    generic_aliases = {
        "settings",
        "windows settings",
        "الاعدادات",
        "الإعدادات",
        "الضبط",
    }
    best_alias = ""
    for alias in _SETTINGS_PAGES:
        if alias in generic_aliases:
            continue
        if alias in text and len(alias) > len(best_alias):
            best_alias = alias
    if best_alias:
        return _SETTINGS_PAGES[best_alias]

    # Fall back to the top-level Settings root only if a generic alias appears
    # and no specific page matched.
    for alias in generic_aliases:
        if alias in text:
            return _SETTINGS_PAGES[alias]

    return ""


def list_known_pages():
    """Return a deduplicated list of (alias, uri) pairs — useful for help/docs."""
    seen_uris = {}
    for alias, uri in _SETTINGS_PAGES.items():
        seen_uris.setdefault(uri, alias)
    return [(alias, uri) for uri, alias in seen_uris.items()]


def open_settings_page(query):
    """Open the Windows Settings page that best matches `query`.

    Returns a status message string.
    """
    uri = resolve_settings_uri(query)
    if not uri:
        return f"Could not find a Settings page matching '{query}'."

    try:
        # os.startfile is the standard, no-admin Windows way to launch URIs.
        # Falls back to `cmd /c start ...` on environments where startfile is
        # unavailable (e.g. when running under WSL).
        if hasattr(os, "startfile"):
            os.startfile(uri)
        else:
            subprocess.Popen(
                ["cmd", "/c", "start", "", uri],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("Opened settings page: %s", uri)
        return f"Opened {uri}."
    except Exception as exc:
        logger.warning("Failed to open settings page '%s': %s", uri, exc)
        return f"Could not open settings: {exc}."
