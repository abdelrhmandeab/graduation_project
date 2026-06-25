"""Wake word enrollment — record user samples and compute a per-user threshold offset.

Run once to calibrate Jarvis to your voice:

    python audio/wake_enrollment.py

Records 5 samples of you saying "Hey Jarvis" (or the configured Arabic trigger),
passes them through the wake-word model to measure your personal peak scores,
then saves a calibration file at:

    data/wake_calibration/<username>.json

The command prints the unified threshold to add to ``.env`` after scoring.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import wave

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_SAMPLES_TO_RECORD = 5
_RECORD_SECONDS = 2.5
_SAMPLE_RATE = 16000
_CHANNELS = 1
_CHUNK_SIZE = 1280
_CALIBRATION_DIR = pathlib.Path(__file__).parent.parent / "data" / "wake_calibration"
_SAMPLES_DIR = pathlib.Path(__file__).parent.parent / "data" / "wake_samples" / "user_positive"


def _get_username() -> str:
    import getpass
    return getpass.getuser()


def _record_sample(index: int, total: int, device=None) -> np.ndarray | None:
    """Record one 2.5s sample. Returns int16 numpy array or None on error."""
    try:
        import sounddevice as sd
    except ImportError:
        print("ERROR: sounddevice not installed. Run: pip install sounddevice")
        return None

    print(f"\nSample {index}/{total}: Say 'Hey Jarvis' clearly...")
    time.sleep(0.4)  # brief pause before recording

    frames = int(_RECORD_SECONDS * _SAMPLE_RATE)
    try:
        audio = sd.rec(
            frames,
            samplerate=_SAMPLE_RATE,
            channels=_CHANNELS,
            dtype="int16",
            device=device,
        )
        # Countdown indicator
        for remaining in range(int(_RECORD_SECONDS), 0, -1):
            print(f"  Recording... {remaining}s ", end="\r", flush=True)
            time.sleep(1.0)
        sd.wait()
        print("  Done.           ")
        return audio.reshape(-1)
    except Exception as exc:
        print(f"  Recording failed: {exc}")
        return None


def _save_sample(audio: np.ndarray, username: str, index: int) -> pathlib.Path:
    """Save sample as WAV to the user's positive samples directory."""
    target_dir = _SAMPLES_DIR / username
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"enroll_{timestamp}_{index}.wav"
    path = target_dir / filename
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(_CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(audio.astype(np.int16).tobytes())
    return path


def _score_samples(samples: list[np.ndarray]) -> list[float]:
    """Run each sample through the openWakeWord model and return peak scores."""
    try:
        from audio.wake_word import _get_unified_model
        from core.config import WAKE_WORD_UNIFIED_ONNX_PATH
    except Exception as exc:
        print(f"WARNING: Could not load wake-word model: {exc}")
        return []

    try:
        model = _get_unified_model(WAKE_WORD_UNIFIED_ONNX_PATH)
    except Exception as exc:
        print(f"WARNING: Wake-word model unavailable: {exc}")
        return []

    scores = []
    for i, audio in enumerate(samples):
        peak = 0.0
        chunks = [
            audio[j : j + _CHUNK_SIZE]
            for j in range(0, len(audio) - _CHUNK_SIZE + 1, _CHUNK_SIZE)
        ]
        for chunk in chunks:
            if len(chunk) < _CHUNK_SIZE:
                break
            try:
                preds = model.predict(chunk)
                prediction_key = next(iter(getattr(model, "prediction_buffer", {}) or {}), None)
                score = preds.get(prediction_key) if prediction_key else None
                if score is None and preds:
                    score = next(iter(preds.values()))
                if score is not None:
                    peak = max(peak, float(score))
            except Exception:
                pass
        scores.append(peak)
        print(f"  Sample {i + 1} peak score: {peak:.4f}")

    return scores


def _save_calibration(username: str, scores: list[float], base_threshold: float) -> pathlib.Path:
    """Compute and save calibration offset."""
    _CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    calibration_path = _CALIBRATION_DIR / f"{username}.json"

    if scores:
        avg_peak = float(np.mean(scores))
        min_peak = float(np.min(scores))
        # Offset = how much to shift the threshold so even the quietest sample
        # would trigger. We target the threshold at 85% of the minimum peak.
        target_threshold = min_peak * 0.85
        offset = target_threshold - base_threshold
    else:
        avg_peak = 0.0
        min_peak = 0.0
        offset = 0.0

    calibration = {
        "username": username,
        "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "samples": len(scores),
        "avg_peak_score": round(avg_peak, 4),
        "min_peak_score": round(min_peak, 4),
        "base_threshold": round(base_threshold, 4),
        "threshold_offset": round(offset, 4),
        "recommended_threshold": round(max(0.10, base_threshold + offset), 4),
    }

    calibration_path.write_text(json.dumps(calibration, indent=2, ensure_ascii=False), encoding="utf-8")
    return calibration_path


def _apply_calibration_to_env(username: str, calibration: dict) -> None:
    """Print the .env line the user should add, and optionally apply it."""
    recommended = calibration.get("recommended_threshold", 0.55)
    print(f"\nRecommended threshold for your voice: {recommended:.3f}")
    print("Add this to your .env to apply permanently:")
    print(f"  JARVIS_WAKE_WORD_THRESHOLD={recommended:.3f}")


def run_enrollment() -> None:
    """Main enrollment flow — interactive CLI."""
    from core.config import WAKE_WORD_THRESHOLD, WAKE_WORD_USER_SPEAKER_ID

    username = str(WAKE_WORD_USER_SPEAKER_ID or _get_username()).strip() or _get_username()
    base_threshold = float(WAKE_WORD_THRESHOLD)

    print("=" * 60)
    print("  Jarvis Wake-Word Enrollment")
    print("=" * 60)
    print(f"  User: {username}")
    print(f"  Samples: {_SAMPLES_TO_RECORD}")
    print(f"  Current threshold: {base_threshold:.3f}")
    print()
    print("You will be asked to say 'Hey Jarvis' five times.")
    print("Speak clearly, at a natural distance from your microphone.")
    input("Press Enter when ready...")

    samples_audio: list[np.ndarray] = []
    for i in range(1, _SAMPLES_TO_RECORD + 1):
        audio = _record_sample(i, _SAMPLES_TO_RECORD)
        if audio is not None and len(audio) > 0:
            samples_audio.append(audio)
            _save_sample(audio, username, i)
        else:
            print(f"  Skipping sample {i} (recording failed).")

    if not samples_audio:
        print("\nERROR: No samples recorded. Enrollment failed.")
        sys.exit(1)

    print(f"\nScoring {len(samples_audio)} sample(s) against wake-word model...")
    scores = _score_samples(samples_audio)

    cal_path = _save_calibration(username, scores, base_threshold)
    calibration = json.loads(cal_path.read_text(encoding="utf-8"))

    print(f"\nCalibration saved: {cal_path}")
    print(f"  Average peak score : {calibration['avg_peak_score']:.4f}")
    print(f"  Minimum peak score : {calibration['min_peak_score']:.4f}")

    _apply_calibration_to_env(username, calibration)
    print("\nEnrollment complete. Restart Jarvis for changes to take effect.")


if __name__ == "__main__":
    run_enrollment()
