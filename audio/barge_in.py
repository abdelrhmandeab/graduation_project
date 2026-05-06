"""Phase 2.11 — VAD-based barge-in monitor.

While the assistant is speaking, this module opens a lightweight microphone
stream and watches for a sustained energy spike. When the user starts talking
over the assistant we treat it as an implicit interrupt: the active TTS
playback is stopped immediately so the user does not have to wait for the
sentence to finish.

Design notes
------------
* The monitor only runs while ``speech_engine.is_speaking()``. The
  orchestrator starts it right before kicking off TTS and the monitor itself
  exits as soon as the speech-engine stop event fires (set by ``interrupt()``).
* Activation is gated by ``BARGE_IN_VAD_GRACE_SECONDS``: cheap laptop mics
  often pick up the speaker output, so we ignore the first half second of
  audio to avoid self-triggering before the user could realistically respond.
* Echo cancellation via ``set_tts_rms()``: the TTS engine reports its current
  playback RMS so we can reject blocks where the mic just mirrors the speaker.
  A block is only counted when mic_rms > tts_rms * ECHO_CANCEL_RATIO (1.5).
* Silero VAD double-confirmation: after the energy streak meets the minimum
  duration, the accumulated samples are re-checked by SileroVAD before the
  callback fires, preventing false triggers from impulsive noise.
* Post-interrupt cooldown of 500 ms prevents the monitor from re-triggering
  during the tail of the interrupt itself.
* The barge-in callback is invoked from the monitor thread — keep it short.
  In practice the orchestrator just calls ``speech_engine.interrupt()``.
"""

from __future__ import annotations

import threading
import time

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover — optional in headless test envs
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

from core.config import (
    BARGE_IN_VAD_ENABLED,
    BARGE_IN_VAD_ENERGY_THRESHOLD,
    BARGE_IN_VAD_GRACE_SECONDS,
    BARGE_IN_VAD_MIN_SPEECH_SECONDS,
    SAMPLE_RATE,
    WAKE_WORD_INPUT_DEVICE,
)
from core.logger import logger

_MONITOR_BLOCK_SAMPLES = 480  # 30 ms @ 16 kHz — small enough for a tight reaction
_POST_INTERRUPT_COOLDOWN_SECONDS = 0.5
_ECHO_CANCEL_RATIO = 1.5  # mic_rms must exceed tts_rms * ratio to count as real speech


class BargeInMonitor:
    """Background thread that watches the mic for user speech during TTS."""

    def __init__(self, on_barge_in, *, is_active=None):
        self._on_barge_in = on_barge_in
        self._is_active = is_active or (lambda: True)
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        # TTS playback RMS reported by the speech engine for echo cancellation.
        self._tts_rms: float = 0.0
        self._tts_rms_lock = threading.Lock()

    def set_tts_rms(self, rms: float) -> None:
        """Called by the TTS engine with its current playback chunk RMS."""
        with self._tts_rms_lock:
            self._tts_rms = max(0.0, float(rms))

    def _get_tts_rms(self) -> float:
        with self._tts_rms_lock:
            return self._tts_rms

    def is_running(self) -> bool:
        with self._lock:
            thread = self._thread
        return bool(thread and thread.is_alive())

    def start(self) -> bool:
        if not BARGE_IN_VAD_ENABLED:
            return False
        if sd is None:
            logger.debug("Barge-in monitor disabled: sounddevice unavailable (%s)", _SOUNDDEVICE_IMPORT_ERROR)
            return False
        with self._lock:
            if self._thread and self._thread.is_alive():
                return True
            self._stop_event.clear()
            thread = threading.Thread(
                target=self._run,
                name="jarvis-barge-in-vad",
                daemon=True,
            )
            self._thread = thread
            thread.start()
        return True

    def stop(self, *, join_timeout: float = 0.5) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
            self._thread = None
        if thread and thread.is_alive() and thread.ident != threading.current_thread().ident:
            thread.join(timeout=float(join_timeout))

    def _run(self) -> None:
        threshold = max(0.001, float(BARGE_IN_VAD_ENERGY_THRESHOLD))
        min_speech_seconds = max(0.05, float(BARGE_IN_VAD_MIN_SPEECH_SECONDS))
        grace_seconds = max(0.0, float(BARGE_IN_VAD_GRACE_SECONDS))
        sample_rate = int(SAMPLE_RATE or 16000)

        seconds_per_block = float(_MONITOR_BLOCK_SAMPLES) / float(sample_rate)
        required_blocks = max(1, int(round(min_speech_seconds / max(seconds_per_block, 1e-6))))

        # Lazy-load Silero VAD for double-confirmation; fall back gracefully.
        silero_vad = None
        try:
            from audio.vad import SileroVAD
            silero_vad = SileroVAD(sample_rate=sample_rate)
        except Exception as exc:
            logger.debug("Barge-in: Silero VAD unavailable, using energy-only detection: %s", exc)

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16,
                device=_resolve_input_device(),
                blocksize=_MONITOR_BLOCK_SAMPLES,
            ) as stream:
                started_ts = time.perf_counter()
                voiced_blocks = 0
                voiced_samples: list[np.ndarray] = []

                while not self._stop_event.is_set():
                    if not self._is_active():
                        return

                    try:
                        block, _ = stream.read(_MONITOR_BLOCK_SAMPLES)
                    except Exception as exc:
                        logger.debug("Barge-in stream read failed: %s", exc)
                        return

                    if (time.perf_counter() - started_ts) < grace_seconds:
                        continue

                    samples = np.asarray(block, dtype=np.int16).reshape(-1)
                    if samples.size == 0:
                        continue

                    samples_float = samples.astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(np.square(samples_float))))

                    # Echo cancellation: reject blocks where the mic is just
                    # mirroring TTS speaker output.
                    tts_rms = self._get_tts_rms()
                    if tts_rms > 0.0 and rms < tts_rms * _ECHO_CANCEL_RATIO:
                        voiced_blocks = 0
                        voiced_samples.clear()
                        continue

                    if rms >= threshold:
                        voiced_blocks += 1
                        voiced_samples.append(samples_float)
                        if voiced_blocks >= required_blocks:
                            # Silero VAD double-confirmation before firing.
                            confirmed = True
                            if silero_vad is not None:
                                try:
                                    accumulated = np.concatenate(voiced_samples)
                                    confirmed = silero_vad.is_speech(accumulated)
                                except Exception as exc:
                                    logger.debug("Silero VAD check failed: %s", exc)

                            if confirmed:
                                logger.info(
                                    "Barge-in detected (rms=%.4f, threshold=%.4f, %d blocks, vad=%s).",
                                    rms,
                                    threshold,
                                    voiced_blocks,
                                    "silero" if silero_vad is not None else "energy",
                                )
                                try:
                                    self._on_barge_in()
                                except Exception as exc:
                                    logger.warning("Barge-in callback raised: %s", exc)
                                # Cooldown: hold the thread alive so the monitor
                                # cannot re-fire during the interrupt tail.
                                time.sleep(_POST_INTERRUPT_COOLDOWN_SECONDS)
                                return
                            else:
                                # VAD rejected — reset streak and try again.
                                logger.debug("Barge-in energy streak rejected by Silero VAD (rms=%.4f).", rms)
                                voiced_blocks = 0
                                voiced_samples.clear()
                    else:
                        # Single quiet block resets the streak — we only react to
                        # sustained voice, not isolated clicks.
                        voiced_blocks = 0
                        voiced_samples.clear()
        except Exception as exc:
            logger.debug("Barge-in monitor exited: %s", exc)


def _resolve_input_device():
    """Reuse the wake-word device hint so both modules listen on the same mic."""
    cfg = WAKE_WORD_INPUT_DEVICE
    if cfg is None or str(cfg).strip() == "":
        return None
    if isinstance(cfg, int):
        return cfg

    if sd is None:
        return None

    name_query = str(cfg).strip().lower()
    try:
        devices = sd.query_devices()
    except Exception:
        return None

    for idx, device in enumerate(devices):
        if int(device.get("max_input_channels", 0)) <= 0:
            continue
        if name_query in str(device.get("name", "")).lower():
            return idx
    return None
