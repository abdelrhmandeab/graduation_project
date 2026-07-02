"""Unified TTS voice profile registry.

One knob (JARVIS_TTS_VOICE_PROFILE) selects an ElevenLabs voice ID *and* a
matched bilingual edge-tts pair (EN + AR) as a single unit.  Three built-in
profiles ship; a "custom" slot lets every field be overridden from .env.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from core.logger import get_logger

_log = get_logger("tts")


@dataclass(frozen=True)
class VoiceProfile:
    name: str
    elevenlabs_voice_id: str
    edge_voice_en: str
    edge_voice_ar: str
    edge_voice_en_fallbacks: tuple[str, ...] = ()
    edge_voice_ar_fallbacks: tuple[str, ...] = ()
    rate_en: str = "+0%"
    rate_ar: str = "+0%"
    pitch_en: str = ""
    pitch_ar: str = ""
    stability: float = 0.45
    similarity_boost: float = 0.75
    style: float = 0.0
    use_speaker_boost: bool = False


DEFAULT_PROFILES: dict[str, VoiceProfile] = {
    "jarvis_male_classic": VoiceProfile(
        name="jarvis_male_classic",
        elevenlabs_voice_id="",
        edge_voice_en="en-US-GuyNeural",
        edge_voice_ar="ar-EG-ShakirNeural",
        edge_voice_en_fallbacks=("en-US-AriaNeural", "en-GB-RyanNeural"),
        edge_voice_ar_fallbacks=("ar-EG-SalmaNeural", "ar-SA-HamedNeural"),
        rate_en="+5%",
        rate_ar="+5%",
        pitch_en="+0Hz",
        pitch_ar="+0Hz",
        stability=0.40,
        similarity_boost=0.80,
        style=0.20,
        use_speaker_boost=True,
    ),
    "jarvis_female_warm": VoiceProfile(
        name="jarvis_female_warm",
        elevenlabs_voice_id="",
        edge_voice_en="en-US-AriaNeural",
        edge_voice_ar="ar-EG-SalmaNeural",
        edge_voice_en_fallbacks=("en-US-JennyNeural", "en-GB-SoniaNeural"),
        edge_voice_ar_fallbacks=("ar-EG-ShakirNeural", "ar-SA-HamedNeural"),
        rate_en="+3%",
        rate_ar="+3%",
        pitch_en="+0Hz",
        pitch_ar="+1Hz",
        stability=0.30,
        similarity_boost=0.80,
        style=0.40,
        use_speaker_boost=True,
    ),
    "jarvis_male_calm": VoiceProfile(
        name="jarvis_male_calm",
        elevenlabs_voice_id="",
        edge_voice_en="en-US-ChristopherNeural",
        edge_voice_ar="ar-EG-ShakirNeural",
        edge_voice_en_fallbacks=("en-US-GuyNeural", "en-GB-RyanNeural"),
        edge_voice_ar_fallbacks=("ar-EG-SalmaNeural", "ar-SA-HamedNeural"),
        rate_en="+0%",
        rate_ar="+0%",
        pitch_en="+0Hz",
        pitch_ar="-2Hz",
        stability=0.50,
        similarity_boost=0.80,
        style=0.10,
        use_speaker_boost=True,
    ),
}


def _split_csv(raw: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in raw.split(",") if s.strip())


def _read_deprecated_alias(new_key: str, old_env_key: str) -> str | None:
    """Return the old env value if set, logging a one-time deprecation warning."""
    value = os.environ.get(old_env_key)
    if value is not None:
        _log.warning(
            "Deprecated env '%s' is set — migrate to '%s' or use JARVIS_TTS_VOICE_PROFILE. "
            "This key will be removed in a future release.",
            old_env_key,
            new_key,
        )
    return value


def get_active_voice_profile() -> VoiceProfile:
    """Resolve the active voice profile from env config.

    Priority:
      1. Per-field overrides from JARVIS_TTS_* env vars (always applied on top).
      2. Built-in profile selected by JARVIS_TTS_VOICE_PROFILE.
      3. Deprecated legacy env vars (alias bridge, with warning).
    """
    profile_name = (
        os.environ.get("JARVIS_TTS_VOICE_PROFILE", "jarvis_male_classic")
        .strip()
        .lower()
    )

    if profile_name == "custom":
        base = DEFAULT_PROFILES["jarvis_male_classic"]
    else:
        base = DEFAULT_PROFILES.get(profile_name)
        if base is None:
            _log.warning(
                "Unknown voice profile '%s'; falling back to jarvis_male_classic",
                profile_name,
            )
            base = DEFAULT_PROFILES["jarvis_male_classic"]

    el_voice_id = os.environ.get("JARVIS_TTS_ELEVENLABS_VOICE_ID", "").strip()
    if not el_voice_id:
        legacy = _read_deprecated_alias(
            "JARVIS_TTS_ELEVENLABS_VOICE_ID",
            "JARVIS_TTS_ELEVENLABS_ARABIC_VOICE_ID",
        )
        if legacy is not None:
            el_voice_id = legacy.strip()
    if not el_voice_id:
        el_voice_id = base.elevenlabs_voice_id

    edge_en = os.environ.get("JARVIS_TTS_EDGE_VOICE_EN", "").strip()
    if not edge_en:
        legacy = _read_deprecated_alias(
            "JARVIS_TTS_EDGE_VOICE_EN", "JARVIS_TTS_EDGE_VOICE"
        )
        if legacy is not None:
            edge_en = legacy.strip()
    if not edge_en:
        edge_en = base.edge_voice_en

    edge_ar = os.environ.get("JARVIS_TTS_EDGE_VOICE_AR", "").strip()
    if not edge_ar:
        legacy = _read_deprecated_alias(
            "JARVIS_TTS_EDGE_VOICE_AR", "JARVIS_TTS_EDGE_ARABIC_VOICE"
        )
        if legacy is not None:
            edge_ar = legacy.strip()
    if not edge_ar:
        edge_ar = base.edge_voice_ar

    en_fb_raw = os.environ.get("JARVIS_TTS_EDGE_VOICE_EN_FALLBACKS", "").strip()
    edge_en_fallbacks = _split_csv(en_fb_raw) if en_fb_raw else base.edge_voice_en_fallbacks

    ar_fb_raw = os.environ.get("JARVIS_TTS_EDGE_VOICE_AR_FALLBACKS", "").strip()
    if not ar_fb_raw:
        legacy = _read_deprecated_alias(
            "JARVIS_TTS_EDGE_VOICE_AR_FALLBACKS",
            "JARVIS_TTS_EDGE_ARABIC_VOICE_FALLBACKS",
        )
        if legacy is not None:
            ar_fb_raw = legacy.strip()
    edge_ar_fallbacks = _split_csv(ar_fb_raw) if ar_fb_raw else base.edge_voice_ar_fallbacks

    rate_en = os.environ.get("JARVIS_TTS_EDGE_RATE_EN", "").strip()
    if not rate_en:
        legacy_rate = os.environ.get("JARVIS_TTS_EDGE_RATE", "").strip()
        rate_en = legacy_rate if legacy_rate else base.rate_en

    rate_ar = os.environ.get("JARVIS_TTS_EDGE_RATE_AR", "").strip()
    if not rate_ar:
        legacy_rate = os.environ.get("JARVIS_TTS_EDGE_ARABIC_RATE", "").strip()
        rate_ar = legacy_rate if legacy_rate else base.rate_ar

    pitch_en = os.environ.get("JARVIS_TTS_EDGE_PITCH_EN", "").strip() or base.pitch_en

    pitch_ar = os.environ.get("JARVIS_TTS_EDGE_PITCH_AR", "").strip()
    if not pitch_ar:
        legacy_pitch = os.environ.get("JARVIS_TTS_EDGE_ARABIC_PITCH", "").strip()
        pitch_ar = legacy_pitch if legacy_pitch else base.pitch_ar

    def _env_float_or(key: str, default: float) -> float:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except ValueError:
            return default

    def _env_bool_or(key: str, default: bool) -> bool:
        raw = os.environ.get(key, "").strip().lower()
        if not raw:
            return default
        return raw in {"1", "true", "yes", "on"}

    stability = _env_float_or("JARVIS_TTS_ELEVENLABS_STABILITY", base.stability)
    similarity_boost = _env_float_or("JARVIS_TTS_ELEVENLABS_SIMILARITY_BOOST", base.similarity_boost)
    style = _env_float_or("JARVIS_TTS_ELEVENLABS_STYLE", base.style)
    use_speaker_boost = _env_bool_or("JARVIS_TTS_ELEVENLABS_USE_SPEAKER_BOOST", base.use_speaker_boost)

    return VoiceProfile(
        name=profile_name,
        elevenlabs_voice_id=el_voice_id,
        edge_voice_en=edge_en,
        edge_voice_ar=edge_ar,
        edge_voice_en_fallbacks=edge_en_fallbacks,
        edge_voice_ar_fallbacks=edge_ar_fallbacks,
        rate_en=rate_en,
        rate_ar=rate_ar,
        pitch_en=pitch_en,
        pitch_ar=pitch_ar,
        stability=stability,
        similarity_boost=similarity_boost,
        style=style,
        use_speaker_boost=use_speaker_boost,
    )


def format_voice_profile_summary(profile: VoiceProfile) -> str:
    parts = [
        f"voice_profile={profile.name}",
        f"el_voice={profile.elevenlabs_voice_id or '(none)'}",
        f"edge_en={profile.edge_voice_en}",
        f"edge_ar={profile.edge_voice_ar}",
        f"stability={profile.stability:.2f}",
        f"style={profile.style:.2f}",
        f"speaker_boost={profile.use_speaker_boost}",
    ]
    return "TTS " + " ".join(parts)
