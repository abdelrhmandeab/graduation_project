"""Short audible cues (tones/bleeps) for wake-word feedback.

Tones are synthesised as int16 PCM at the project sample rate and cached
so repeated plays don't re-compute.  Playback is fire-and-forget via
``sounddevice.play`` on the default output device.
"""

from __future__ import annotations

import numpy as np

from core.config import SAMPLE_RATE

_cache: dict[tuple[int, int], np.ndarray] = {}


def _generate_tone(freq_hz: int, duration_ms: int) -> np.ndarray:
    key = (freq_hz, duration_ms)
    if key in _cache:
        return _cache[key]

    sr = int(SAMPLE_RATE)
    n_samples = max(1, int(sr * duration_ms / 1000))
    t = np.arange(n_samples, dtype=np.float32) / sr
    sine = np.sin(2.0 * np.pi * freq_hz * t)

    # Short fade in/out to avoid clicks
    fade_samples = min(n_samples // 4, int(sr * 0.005))
    if fade_samples > 0:
        fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
        fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
        sine[:fade_samples] *= fade_in
        sine[-fade_samples:] *= fade_out

    tone = (sine * 16000).astype(np.int16)
    _cache[key] = tone
    return tone


def play_cue(freq_hz: int, duration_ms: int) -> None:
    try:
        import sounddevice as sd
    except Exception:
        return
    try:
        samples = _generate_tone(freq_hz, duration_ms)
        sd.play(samples, samplerate=int(SAMPLE_RATE), blocking=False)
    except Exception:
        pass
