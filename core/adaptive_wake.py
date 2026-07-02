"""Background adaptive retrainer for the unified wake-word model.

Accumulates confirmed-positive and confirmed-negative samples from runtime
detections, then periodically retrains the ONNX model and hot-swaps it into
the live listener without restarting Jarvis.

Lifecycle:
    1. Orchestrator calls ``record_confirmed()`` / ``record_false_positive()``
       after each wake-word cycle based on whether speech followed.
    2. A daemon thread checks every ``ADAPTIVE_WAKE_RETRAIN_INTERVAL_SECONDS``
       whether enough new samples have accumulated.
    3. When triggered, the retrainer runs feature extraction + training
       (reusing the existing training-data clips on disk via ``--reuse-clips``
       semantics), exports a candidate ONNX, validates it against a held-out
       set, and — if good enough — atomically replaces the live model.
"""

from __future__ import annotations

import pathlib
import shutil
import threading
import time
import wave

import numpy as np

from core.config import (
    ADAPTIVE_WAKE_CONFIRMED_DIR,
    ADAPTIVE_WAKE_ENABLED,
    ADAPTIVE_WAKE_EPOCHS,
    ADAPTIVE_WAKE_FALSE_POSITIVE_DIR,
    ADAPTIVE_WAKE_MIN_CONFIRMED,
    ADAPTIVE_WAKE_MIN_VAL_ACC,
    ADAPTIVE_WAKE_RETRAIN_INTERVAL_SECONDS,
    SAMPLE_RATE,
    WAKE_WORD_USER_SAMPLES_DIR,
    WAKE_WORD_USER_SPEAKER_ID,
)
from core.logger import get_logger, kv

log = get_logger("adaptive_wake")

_confirmed_since_last_train = 0
_false_positives_since_last_train = 0
_lock = threading.Lock()
_thread: threading.Thread | None = None
_stop_event = threading.Event()
_retrain_count = 0


# ---------------------------------------------------------------------------
# Public API called by the orchestrator after each wake cycle
# ---------------------------------------------------------------------------

def record_confirmed(audio_chunks) -> None:
    """A wake detection was followed by real speech — confirmed true positive."""
    global _confirmed_since_last_train
    if not ADAPTIVE_WAKE_ENABLED:
        return
    _save_sample(audio_chunks, pathlib.Path(ADAPTIVE_WAKE_CONFIRMED_DIR), "confirmed")
    with _lock:
        _confirmed_since_last_train += 1


def record_false_positive(audio_chunks) -> None:
    """A wake detection was NOT followed by speech — likely false positive."""
    global _false_positives_since_last_train
    if not ADAPTIVE_WAKE_ENABLED:
        return
    _save_sample(audio_chunks, pathlib.Path(ADAPTIVE_WAKE_FALSE_POSITIVE_DIR), "fp")
    with _lock:
        _false_positives_since_last_train += 1


def get_stats() -> dict:
    with _lock:
        return {
            "confirmed_since_last_train": _confirmed_since_last_train,
            "false_positives_since_last_train": _false_positives_since_last_train,
            "retrain_count": _retrain_count,
            "enabled": ADAPTIVE_WAKE_ENABLED,
        }


# ---------------------------------------------------------------------------
# Sample persistence
# ---------------------------------------------------------------------------

def _save_sample(audio_chunks, directory: pathlib.Path, prefix: str) -> None:
    try:
        speaker = str(WAKE_WORD_USER_SPEAKER_ID or "speaker").strip() or "speaker"
        speaker = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in speaker)
        target_dir = directory / speaker
        target_dir.mkdir(parents=True, exist_ok=True)

        if not audio_chunks:
            return
        audio = np.concatenate(list(audio_chunks), axis=0).astype(np.int16, copy=False).reshape(-1)
        if audio.size < 4000:
            return

        ts = time.strftime("%Y%m%d-%H%M%S")
        ms = int(time.time() * 1000) % 1000
        filename = f"{prefix}_{ts}_{ms:03d}.wav"
        path = target_dir / filename

        with wave.open(str(path), "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(int(SAMPLE_RATE))
            f.writeframes(audio.tobytes())
    except Exception as exc:
        log.warning("Failed to save adaptive sample: %s", exc)


# ---------------------------------------------------------------------------
# Background retrain daemon
# ---------------------------------------------------------------------------

def start_daemon() -> None:
    global _thread
    if not ADAPTIVE_WAKE_ENABLED:
        log.info("Adaptive wake-word retraining is disabled.")
        return
    if _thread is not None and _thread.is_alive():
        return
    _stop_event.clear()
    _thread = threading.Thread(target=_daemon_loop, daemon=True, name="adaptive-wake")
    _thread.start()
    log.info(
        "Adaptive wake-word daemon started (interval=%ds, min_confirmed=%d).",
        int(ADAPTIVE_WAKE_RETRAIN_INTERVAL_SECONDS),
        ADAPTIVE_WAKE_MIN_CONFIRMED,
    )


def stop_daemon() -> None:
    _stop_event.set()
    if _thread is not None:
        _thread.join(timeout=5.0)


def _daemon_loop() -> None:
    while not _stop_event.is_set():
        _stop_event.wait(timeout=float(ADAPTIVE_WAKE_RETRAIN_INTERVAL_SECONDS))
        if _stop_event.is_set():
            break
        with _lock:
            pending = _confirmed_since_last_train
        if pending < ADAPTIVE_WAKE_MIN_CONFIRMED:
            log.debug(
                "Adaptive retrain skipped: %d confirmed < %d minimum.",
                pending, ADAPTIVE_WAKE_MIN_CONFIRMED,
            )
            continue
        try:
            _run_retrain()
        except Exception as exc:
            log.error("Adaptive retrain failed: %s", exc, exc_info=True)


def _run_retrain() -> None:
    global _confirmed_since_last_train, _false_positives_since_last_train, _retrain_count

    log.info("Adaptive retrain starting...")
    t0 = time.perf_counter()

    project_root = pathlib.Path(__file__).resolve().parents[1]
    work_dir = project_root / "data" / "jarvis_unified_training"
    feature_dir = work_dir / "features"
    candidate_dir = project_root / "models" / "jarvis_unified"
    candidate_path = candidate_dir / "jarvis_unified_candidate.onnx"
    live_path = candidate_dir / "jarvis_unified.onnx"
    backup_path = candidate_dir / "jarvis_unified_backup.onnx"

    clip_dirs = {
        "positive_train": work_dir / "positive_train" / "pos",
        "positive_val": work_dir / "positive_val" / "pos",
        "negative_train": work_dir / "negative_train" / "neg",
        "negative_val": work_dir / "negative_val" / "neg",
    }

    for label, d in clip_dirs.items():
        if not any(d.glob("*.wav")):
            log.warning("Retrain aborted: no base training clips in %s.", label)
            return

    # ---- Step 1: Copy confirmed positives into training set ----
    confirmed_dir = pathlib.Path(ADAPTIVE_WAKE_CONFIRMED_DIR)
    confirmed_wavs = sorted(confirmed_dir.glob("**/*.wav")) if confirmed_dir.is_dir() else []
    copied_positive = 0
    for wav in confirmed_wavs:
        dest = clip_dirs["positive_train"] / f"adaptive_{wav.name}"
        if not dest.exists():
            shutil.copy2(wav, dest)
            copied_positive += 1
    # Copy a portion into validation
    val_wavs = confirmed_wavs[: max(1, len(confirmed_wavs) // 5)]
    for wav in val_wavs:
        dest = clip_dirs["positive_val"] / f"adaptive_{wav.name}"
        if not dest.exists():
            shutil.copy2(wav, dest)

    # ---- Step 2: Copy false positives into negative set ----
    fp_dir = pathlib.Path(ADAPTIVE_WAKE_FALSE_POSITIVE_DIR)
    fp_wavs = sorted(fp_dir.glob("**/*.wav")) if fp_dir.is_dir() else []
    copied_negative = 0
    for wav in fp_wavs:
        dest = clip_dirs["negative_train"] / f"adaptive_fp_{wav.name}"
        if not dest.exists():
            shutil.copy2(wav, dest)
            copied_negative += 1
    fp_val_wavs = fp_wavs[: max(1, len(fp_wavs) // 5)]
    for wav in fp_val_wavs:
        dest = clip_dirs["negative_val"] / f"adaptive_fp_{wav.name}"
        if not dest.exists():
            shutil.copy2(wav, dest)

    # ---- Step 3: Also re-ingest user_positive samples ----
    user_pos_dir = pathlib.Path(WAKE_WORD_USER_SAMPLES_DIR)
    if user_pos_dir.is_dir():
        import sys
        scripts_dir = project_root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from train_arabic_wake_model import _ingest_user_positive_clips
        for old_clip in clip_dirs["positive_train"].glob("**/user_default_*.wav"):
            old_clip.unlink()
        _ingest_user_positive_clips(
            source_dir=user_pos_dir,
            train_dir=clip_dirs["positive_train"],
            val_dir=clip_dirs["positive_val"],
            sample_rate=16000,
            val_ratio=0.2,
            source_label="default",
        )

    log.info(
        "Adaptive data: +%d confirmed positives, +%d false-positive negatives.",
        copied_positive, copied_negative,
    )

    # ---- Step 4: Feature extraction + training ----
    import sys
    scripts_dir = project_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from train_arabic_wake_model import (
        _build_feature_file,
        _make_dataloaders,
        _train_classifier,
        _export_onnx,
        _write_background_negative_clips,
        _collect_wavs,
    )
    import torch
    from openwakeword.utils import AudioFeatures

    # Refresh background negatives
    phrase_neg_train = len([p for p in _collect_wavs(clip_dirs["negative_train"]) if not p.name.startswith("background_")])
    _write_background_negative_clips(
        clip_dirs["negative_train"],
        count=max(96, len(_collect_wavs(clip_dirs["positive_train"])) - phrase_neg_train),
        sample_rate=16000,
        seed=int(time.time()) % (2**31),
    )
    phrase_neg_val = len([p for p in _collect_wavs(clip_dirs["negative_val"]) if not p.name.startswith("background_")])
    _write_background_negative_clips(
        clip_dirs["negative_val"],
        count=max(32, len(_collect_wavs(clip_dirs["positive_val"])) - phrase_neg_val),
        sample_rate=16000,
        seed=int(time.time() + 1) % (2**31),
    )

    sample_length_samples = int(round(16000 * 2.0))
    target_frames = 41
    batch_size = 32

    feature_dir.mkdir(parents=True, exist_ok=True)
    _build_feature_file(
        wav_dir=clip_dirs["positive_train"],
        output_file=feature_dir / "positive_features_train.npy",
        sample_length_samples=sample_length_samples,
        batch_size=batch_size,
        augment_positive_gain=True,
    )
    _build_feature_file(
        wav_dir=clip_dirs["negative_train"],
        output_file=feature_dir / "negative_features_train.npy",
        sample_length_samples=sample_length_samples,
        batch_size=batch_size,
        augment_negative_windows=True,
    )
    _build_feature_file(
        wav_dir=clip_dirs["positive_val"],
        output_file=feature_dir / "positive_features_test.npy",
        sample_length_samples=sample_length_samples,
        batch_size=batch_size,
        augment_positive_gain=True,
    )
    _build_feature_file(
        wav_dir=clip_dirs["negative_val"],
        output_file=feature_dir / "negative_features_test.npy",
        sample_length_samples=sample_length_samples,
        batch_size=batch_size,
        augment_negative_windows=True,
    )

    train_loader, val_loader, _ = _make_dataloaders(
        feature_root=feature_dir,
        batch_size=batch_size,
        target_frames=target_frames,
    )

    sample_audio = AudioFeatures(inference_framework="onnx", device="cpu")
    feature_dim = int(sample_audio.get_embedding_shape(2.0)[1])

    classifier = _train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(ADAPTIVE_WAKE_EPOCHS),
        learning_rate=1e-3,
        device=torch.device("cpu"),
        target_frames=target_frames,
        feature_dim=feature_dim,
    )

    # ---- Step 5: Validate before swapping ----
    val_acc = _evaluate_accuracy(classifier, val_loader)
    if val_acc < ADAPTIVE_WAKE_MIN_VAL_ACC:
        log.warning(
            "Adaptive retrain rejected: val_acc=%.3f < min=%.3f. Keeping current model.",
            val_acc, ADAPTIVE_WAKE_MIN_VAL_ACC,
        )
        return

    # ---- Step 6: Export candidate and atomic swap ----
    candidate_dir.mkdir(parents=True, exist_ok=True)
    _export_onnx(classifier, candidate_path, target_frames=target_frames, feature_dim=feature_dim)

    if live_path.exists():
        shutil.copy2(live_path, backup_path)

    shutil.move(str(candidate_path), str(live_path))
    log.info("Candidate model promoted to live: %s", live_path)

    # ---- Step 7: Hot-reload the model in the listener ----
    from audio.wake_word import invalidate_model_cache
    invalidate_model_cache()

    elapsed = time.perf_counter() - t0
    with _lock:
        _confirmed_since_last_train = 0
        _false_positives_since_last_train = 0
        _retrain_count += 1

    kv(
        "adaptive_wake",
        retrain_count=_retrain_count,
        val_acc=f"{val_acc:.3f}",
        elapsed_s=f"{elapsed:.1f}",
        confirmed_positives=copied_positive,
        false_positives=copied_negative,
    )
    log.info("Adaptive retrain #%d completed in %.1fs (val_acc=%.3f).", _retrain_count, elapsed, val_acc)


def _evaluate_accuracy(classifier, val_loader) -> float:
    import torch
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            logits = classifier(inputs)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            correct += int((preds.squeeze(-1) == labels).sum().item())
            total += inputs.shape[0]
    return correct / max(1, total)
