"""Echo cancellation refinements — Task 5.3.

Three sub-systems, each exposed as a module-level singleton:

* baseline_noise   (BaselineNoiseSampler)  — records 2 s of ambient silence on
  startup; the measured RMS is added to the echo-rejection threshold so that
  background noise does not count as user speech.

* noise_reducer    (NoiseReducer)          — optional noisereduce pre-processing
  for STT audio.  Stationary noise suppression helps in noisy environments.
  Gated by JARVIS_NOISE_REDUCE_ENABLED (off by default, adds ~50 ms).

* echo_ratio_adapter (EchoRatioAdapter)   — starts at BARGE_IN_ENERGY_RATIO and
  slowly converges toward the ratio actually observed on confirmed barge-in
  events.  Handles both laptop speakers (close mic) and external speakers.
"""

from __future__ import annotations

import threading
from typing import Optional

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:
    sd = None
    _SOUNDDEVICE_IMPORT_ERROR = exc
else:
    _SOUNDDEVICE_IMPORT_ERROR = None

from core.config import (
    BARGE_IN_ENERGY_RATIO,
    BASELINE_NOISE_CALIBRATION_SECONDS,
    NOISE_REDUCE_ENABLED,
    SAMPLE_RATE,
)
from core.logger import logger

_BLOCK_SIZE = 480  # 30 ms @ 16 kHz


# ---------------------------------------------------------------------------
# BaselineNoiseSampler
# ---------------------------------------------------------------------------

class BaselineNoiseSampler:
    """Records a brief window of ambient silence to estimate the noise floor.

    The baseline RMS is added to the barge-in echo-rejection threshold so that
    background noise is not mistaken for user speech:

        mic_rms > tts_rms * ratio + baseline_rms

    This makes the monitor work correctly in both quiet rooms (baseline ≈ 0)
    and noisy environments (baseline absorbs the constant noise floor).
    """

    def __init__(self) -> None:
        self._baseline_rms: float = 0.0
        self._calibrated: bool = False
        self._lock = threading.Lock()

    def calibrate(
        self,
        seconds: Optional[float] = None,
        device=None,
    ) -> float:
        """Open the mic, record *seconds* of ambient audio, compute median RMS.

        Returns the baseline RMS (0.0 on failure).  Safe to call from a
        background thread; re-entrant calls are serialised by *_lock*.
        """
        if sd is None:
            logger.debug(
                "Baseline noise calibration skipped: sounddevice unavailable (%s)",
                _SOUNDDEVICE_IMPORT_ERROR,
            )
            return 0.0

        duration = max(0.5, float(seconds if seconds is not None else BASELINE_NOISE_CALIBRATION_SECONDS))
        sample_rate = int(SAMPLE_RATE or 16000)
        total_blocks = max(1, int(round(duration * sample_rate / _BLOCK_SIZE)))
        rms_values: list[float] = []

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype=np.int16,
                device=device,
                blocksize=_BLOCK_SIZE,
            ) as stream:
                for _ in range(total_blocks):
                    block, _ = stream.read(_BLOCK_SIZE)
                    samples = (
                        np.asarray(block, dtype=np.int16)
                        .reshape(-1)
                        .astype(np.float32)
                        / 32768.0
                    )
                    rms_values.append(float(np.sqrt(np.mean(np.square(samples)))))
        except Exception as exc:
            logger.warning("Baseline noise calibration failed: %s", exc)
            return 0.0

        if not rms_values:
            return 0.0

        # Median is more robust than mean against transient noise spikes.
        baseline = float(np.median(rms_values))
        with self._lock:
            self._baseline_rms = baseline
            self._calibrated = True

        logger.info(
            "Baseline noise calibration complete: rms=%.5f (%.0f ms, %d blocks)",
            baseline,
            duration * 1000,
            len(rms_values),
        )
        return baseline

    def get_baseline_rms(self) -> float:
        with self._lock:
            return self._baseline_rms

    def is_calibrated(self) -> bool:
        with self._lock:
            return self._calibrated

    def adjusted_echo_threshold(self, tts_rms: float, ratio: float) -> float:
        """Return the echo-rejection threshold for one barge-in monitor block.

        Formula: ``tts_rms * ratio + baseline_rms``
        When TTS is silent (tts_rms=0), returns baseline_rms so random noise
        does not trigger barge-in at the end of a sentence.
        """
        baseline = self.get_baseline_rms()
        return max(0.0, float(tts_rms)) * max(1.0, float(ratio)) + baseline


# ---------------------------------------------------------------------------
# NoiseReducer
# ---------------------------------------------------------------------------

class NoiseReducer:
    """Optional stationary noise suppression using the *noisereduce* library.

    Only active when JARVIS_NOISE_REDUCE_ENABLED=true.  If the library is not
    installed the audio passes through unchanged.
    """

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._lock = threading.Lock()

    def _is_available(self) -> bool:
        if self._available is not None:
            return self._available
        with self._lock:
            if self._available is not None:
                return self._available
            try:
                import noisereduce  # noqa: F401
                self._available = True
                logger.info("noisereduce available — noise suppression ready.")
            except ImportError:
                self._available = False
                logger.debug(
                    "noisereduce not installed; noise suppression disabled. "
                    "Install with: pip install noisereduce>=3.0.0"
                )
        return bool(self._available)

    def reduce(self, audio_int16: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Apply stationary noise reduction.

        Returns the original array unchanged if the feature is disabled, the
        library is unavailable, or an error occurs during processing.
        """
        if not NOISE_REDUCE_ENABLED:
            return audio_int16
        if not self._is_available():
            return audio_int16
        try:
            import noisereduce as nr
            audio_float = audio_int16.astype(np.float32) / 32768.0
            reduced = nr.reduce_noise(y=audio_float, sr=int(sample_rate), stationary=True)
            return np.clip(reduced * 32768.0, -32768, 32767).astype(np.int16)
        except Exception as exc:
            logger.debug("noisereduce processing failed, using original audio: %s", exc)
            return audio_int16


# ---------------------------------------------------------------------------
# EchoRatioAdapter
# ---------------------------------------------------------------------------

class EchoRatioAdapter:
    """Adapts the echo-rejection ratio from confirmed barge-in events.

    Starts at ``BARGE_IN_ENERGY_RATIO`` (default 1.8).  Each confirmed
    interrupt provides a measured ``mic_rms / tts_rms`` sample; the ratio
    converges slowly (EWMA α=0.15) toward the real acoustic environment:

    * Laptop speakers (close mic): typical ratio ~1.3–1.8
    * External speakers (far mic):  typical ratio ~2.0–2.5
    * Echoey room:                  ratio may reach 3.0+

    This removes the need to manually tweak JARVIS_BARGE_IN_ENERGY_RATIO for
    each hardware configuration — the system self-tunes over several minutes.
    """

    _MIN_RATIO: float = 1.0
    _MAX_RATIO: float = 3.5
    _EWMA_ALPHA: float = 0.15  # slow convergence — reacts over ~7 events
    _MIN_TTS_RMS: float = 0.02  # ignore events where TTS was near-silent

    def __init__(self) -> None:
        self._ratio: float = float(BARGE_IN_ENERGY_RATIO)
        self._initial_ratio: float = float(BARGE_IN_ENERGY_RATIO)
        self._event_count: int = 0
        self._lock = threading.Lock()

    def get_current_ratio(self) -> float:
        with self._lock:
            return self._ratio

    def record_interrupt_event(self, mic_rms: float, tts_rms: float) -> None:
        """Update the ratio estimate from one confirmed barge-in event.

        Only updates when TTS was audible enough to compute a meaningful ratio.
        Outliers (ratio outside [MIN, MAX]) are silently discarded.
        """
        tts = max(0.0, float(tts_rms))
        mic = max(0.0, float(mic_rms))
        if tts < self._MIN_TTS_RMS:
            return
        measured = mic / tts
        if measured < self._MIN_RATIO or measured > self._MAX_RATIO:
            return

        with self._lock:
            prev = self._ratio
            self._ratio = (
                (1.0 - self._EWMA_ALPHA) * self._ratio
                + self._EWMA_ALPHA * measured
            )
            self._ratio = max(self._MIN_RATIO, min(self._MAX_RATIO, self._ratio))
            self._event_count += 1
            new_ratio = self._ratio
            count = self._event_count

        if abs(new_ratio - prev) > 0.1:
            logger.info(
                "EchoRatioAdapter: ratio %.2f → %.2f after %d events (measured=%.2f)",
                prev,
                new_ratio,
                count,
                measured,
            )

    def get_calibration_status(self) -> dict:
        with self._lock:
            return {
                "initial_ratio": round(self._initial_ratio, 3),
                "current_ratio": round(self._ratio, 3),
                "event_count": self._event_count,
                "adapted": abs(self._ratio - self._initial_ratio) > 0.05,
            }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------

baseline_noise = BaselineNoiseSampler()
noise_reducer = NoiseReducer()
echo_ratio_adapter = EchoRatioAdapter()
