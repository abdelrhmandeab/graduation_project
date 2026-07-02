"""Train and export the unified English/Arabic wake-word ONNX model.

Despite its legacy filename, this script replaces the Arabic-only training
flow. It uses bilingual generated wake phrases plus optional real recordings,
creates adversarial negative clips, computes openWakeWord features, trains a
compact PyTorch classifier, and exports the result as ONNX.

It is intentionally self-contained so it can run on Windows without the Linux-
only openWakeWord notebook flow.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import scipy.io.wavfile as wavfile
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from openwakeword.utils import AudioFeatures

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.config import WAKE_WORD_USER_SAMPLES_DIR

from generate_arabic_wake_data import (
    AR_POSITIVE_PHRASES,
    AR_VOICES,
    EN_POSITIVE_PHRASES,
    EN_VOICES,
    generate_samples,
)


EN_NEGATIVE_PHRASES = [
    "harvest",
    "service",
    "jarvis song",
    "javelin",
    "garbage",
    "jarvis the movie",
    "charles",
    "drive us",
    "drive us home",
    "jar of his",
    "jaw of his",
    "java is",
    "jar of jam",
    "open chrome",
    "what time is it",
    "set a timer",
    "play some music",
    "turn on bluetooth",
    "what's the weather",
]
AR_NEGATIVE_PHRASES = [
    "جزر",
    "يا حارس",
    "يا فارس",
    "جارح",
    "جراج",
    "جارف",
    "جرس",
    "يا حارث",
    "افتح اليوتيوب",
    "شغل اغنية",
    "كام الساعة",
    "ايه اخبار الطقس",
    "اقفل النور",
    "افتح كروم",
    "ابعت ايميل",
]
DEFAULT_NEGATIVE_PHRASES = EN_NEGATIVE_PHRASES + AR_NEGATIVE_PHRASES


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_wavs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("**/*.wav") if p.is_file())


def _pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    if audio.shape[0] > target_samples:
        return audio[-target_samples:]
    if audio.shape[0] < target_samples:
        padded = np.zeros(target_samples, dtype=np.int16)
        # Runtime inference classifies the most recent feature window. Align
        # short clips to the right so the spoken phrase occupies that same
        # position instead of training mostly on trailing silence.
        if audio.size:
            padded[-audio.shape[0] :] = audio
        return padded
    return audio


def _pad_or_trim_frames(features: np.ndarray, target_frames: int) -> np.ndarray:
    if features.shape[0] > target_frames:
        return features[-target_frames:, :]
    if features.shape[0] < target_frames:
        pad = np.zeros((target_frames - features.shape[0], features.shape[1]), dtype=features.dtype)
        return np.concatenate([pad, features], axis=0)
    return features


def _resample_linear(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return audio.astype(np.int16, copy=False)
    if audio.size == 0:
        return audio.astype(np.int16, copy=False)

    audio_float = np.asarray(audio, dtype=np.float32).reshape(-1)
    duration = audio_float.shape[0] / float(src_sr)
    target_len = max(1, int(round(duration * float(target_sr))))
    x_old = np.linspace(0.0, 1.0, num=audio_float.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio_float)
    return np.clip(np.round(resampled), -32768, 32767).astype(np.int16, copy=False)


def _normalize_peak(audio: np.ndarray, peak: float = 0.97) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.int16, copy=False)

    audio_float = np.asarray(audio, dtype=np.float32).reshape(-1)
    max_val = float(np.max(np.abs(audio_float)))
    if max_val <= 0:
        return np.zeros_like(audio_float, dtype=np.int16)
    scale = min(1.0, float(peak) * 32767.0 / max_val)
    return np.clip(np.round(audio_float * scale), -32768, 32767).astype(np.int16, copy=False)


def _collect_user_positive_wavs(directory: Path, *, min_samples: int = 8000) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    results: list[Path] = []
    for path in sorted(directory.glob("**/*.wav")):
        if not path.is_file():
            continue
        # Skip auto-captured samples that live under the auto_captured/ subdirectory
        # — those are unverified runtime detections, not curated positives.
        if "auto_captured" in path.parts:
            continue
        try:
            sr, audio = wavfile.read(path)
            if audio.ndim > 1:
                audio = audio[:, 0]
            if audio.size < min_samples:
                continue
        except Exception:
            continue
        results.append(path)
    return results


def _write_background_negative_clips(
    directory: Path,
    *,
    count: int,
    sample_rate: int,
    seed: int,
) -> int:
    """Add silence and low-level background noise matching runtime startup buffers."""
    directory.mkdir(parents=True, exist_ok=True)
    for stale_path in directory.glob("background_*.wav"):
        stale_path.unlink()
    rng = np.random.default_rng(seed)
    sample_count = int(sample_rate * 4)
    for index in range(max(1, int(count))):
        variant = index % 4
        if variant == 0:
            audio = np.zeros(sample_count, dtype=np.int16)
        elif variant == 1:
            audio = rng.integers(-1000, 1001, sample_count, dtype=np.int16)
        elif variant == 2:
            stddev = float(rng.uniform(25.0, 500.0))
            audio = np.clip(rng.normal(0.0, stddev, sample_count), -32768, 32767).astype(np.int16)
        else:
            frequency = float(rng.uniform(45.0, 180.0))
            amplitude = float(rng.uniform(30.0, 400.0))
            timeline = np.arange(sample_count, dtype=np.float32) / float(sample_rate)
            hum = np.sin(2.0 * np.pi * frequency * timeline) * amplitude
            noise = rng.normal(0.0, amplitude * 0.25, sample_count)
            audio = np.clip(hum + noise, -32768, 32767).astype(np.int16)
        wavfile.write(directory / f"background_{index:04d}.wav", sample_rate, audio)
    return max(1, int(count))


def _sanitize_path_component(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return "speaker"
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _ingest_user_positive_clips(
    *,
    source_dir: Path,
    train_dir: Path,
    val_dir: Path,
    sample_rate: int,
    val_ratio: float,
    source_label: str,
) -> tuple[int, int]:
    wav_paths = _collect_user_positive_wavs(source_dir)
    if not wav_paths:
        return 0, 0

    val_ratio = min(0.9, max(0.0, float(val_ratio)))
    val_count = max(1, int(round(len(wav_paths) * val_ratio))) if len(wav_paths) > 1 else 0
    train_count = 0
    val_written = 0

    for index, wav_path in enumerate(wav_paths):
        sr, audio = wavfile.read(wav_path)
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = np.asarray(audio, dtype=np.int16).reshape(-1)
        audio = _resample_linear(audio, int(sr), sample_rate)
        audio = _normalize_peak(audio)

        is_validation = index < val_count
        target_dir = val_dir if is_validation else train_dir
        relative_parent = Path()
        try:
            relative_parent = wav_path.parent.relative_to(source_dir)
        except ValueError:
            relative_parent = Path()
        if str(relative_parent) not in {"", "."}:
            relative_parent = Path(*(_sanitize_path_component(part) for part in relative_parent.parts))
        target_dir = target_dir / relative_parent if str(relative_parent) not in {"", "."} else target_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in wav_path.stem)
        target_path = target_dir / f"user_{source_label}_{index:04d}_{safe_stem}.wav"
        wavfile.write(target_path, sample_rate, audio)

        if is_validation:
            val_written += 1
        else:
            train_count += 1

    return train_count, val_written


def _batch_generator(file_paths: Sequence[Path], batch_size: int, target_samples: int) -> Iterator[np.ndarray]:
    batch: list[np.ndarray] = []
    for path in file_paths:
        sr, audio = wavfile.read(path)
        if sr != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sr} Hz for {path}")
        if audio.ndim > 1:
            audio = audio[:, 0]
        batch.append(_pad_or_trim(audio, target_samples))
        if len(batch) >= batch_size:
            yield np.stack(batch, axis=0)
            batch = []
    if batch:
        yield np.stack(batch, axis=0)


def _write_positive_and_negative_clips(
    *,
    output_root: Path,
    train_count: int,
    val_count: int,
    sample_rate: int,
    user_positive_dir: Path | None = None,
    user_positive_dir_en: Path | None = None,
    user_positive_val_ratio: float = 0.2,
) -> dict[str, Path]:
    """Generate the training and validation WAV clips used by the trainer."""
    positive_train_dir = output_root / "positive_train"
    positive_val_dir = output_root / "positive_val"
    negative_train_dir = output_root / "negative_train"
    negative_val_dir = output_root / "negative_val"

    for path in (positive_train_dir, positive_val_dir, negative_train_dir, negative_val_dir):
        if path.exists():
            shutil.rmtree(path)
        _ensure_dir(path)

    async def _generate_all():
        async def _generate_grouped(
            *,
            phrase_groups: dict[str, list[str]],
            output_dir: Path,
            keyword: str,
            target_count: int,
            snr_db_choices: list[float],
        ) -> None:
            total_phrases = sum(len(phrases) for phrases in phrase_groups.values())
            voice_groups = {"en": EN_VOICES, "ar": AR_VOICES}
            for language, phrases in phrase_groups.items():
                voices = voice_groups[language]
                group_target = round(target_count * len(phrases) / max(1, total_phrases))
                samples_per_phrase = max(1, round(group_target / max(1, len(phrases) * len(voices))))
                await generate_samples(
                    phrases=phrases,
                    voices=voices,
                    output_dir=output_dir,
                    keyword=keyword,
                    samples_per_phrase=samples_per_phrase,
                    sample_rate=sample_rate,
                    snr_db_choices=snr_db_choices,
                    apply_reverb=True,
                )

        positive_groups = {"en": EN_POSITIVE_PHRASES, "ar": AR_POSITIVE_PHRASES}
        negative_groups = {"en": EN_NEGATIVE_PHRASES, "ar": AR_NEGATIVE_PHRASES}
        await _generate_grouped(
            phrase_groups=positive_groups,
            output_dir=positive_train_dir,
            keyword="pos",
            target_count=train_count,
            snr_db_choices=[16.0, 20.0, 24.0],
        )
        await _generate_grouped(
            phrase_groups=positive_groups,
            output_dir=positive_val_dir,
            keyword="pos",
            target_count=val_count,
            snr_db_choices=[18.0, 22.0],
        )
        await _generate_grouped(
            phrase_groups=negative_groups,
            output_dir=negative_train_dir,
            keyword="neg",
            target_count=train_count,
            snr_db_choices=[14.0, 18.0, 22.0],
        )
        await _generate_grouped(
            phrase_groups=negative_groups,
            output_dir=negative_val_dir,
            keyword="neg",
            target_count=val_count,
            snr_db_choices=[16.0, 20.0],
        )

    asyncio.run(_generate_all())

    user_train_count = 0
    user_val_count = 0
    if user_positive_dir is not None:
        user_train_count, user_val_count = _ingest_user_positive_clips(
            source_dir=Path(user_positive_dir),
            train_dir=positive_train_dir / "pos",
            val_dir=positive_val_dir / "pos",
            sample_rate=sample_rate,
            val_ratio=user_positive_val_ratio,
            source_label="default",
        )
        if user_train_count or user_val_count:
            print(
                f"Imported {user_train_count} user wake clips into train and {user_val_count} into val "
                f"from {Path(user_positive_dir).resolve()}"
            )

    if user_positive_dir_en is not None:
        en_train_count, en_val_count = _ingest_user_positive_clips(
            source_dir=Path(user_positive_dir_en),
            train_dir=positive_train_dir / "pos",
            val_dir=positive_val_dir / "pos",
            sample_rate=sample_rate,
            val_ratio=user_positive_val_ratio,
            source_label="en",
        )
        user_train_count += en_train_count
        user_val_count += en_val_count
        if en_train_count or en_val_count:
            print(
                f"Imported {en_train_count} English user wake clips into train and {en_val_count} into val "
                f"from {Path(user_positive_dir_en).resolve()}"
            )

    positive_train_count = len(_collect_wavs(positive_train_dir / "pos"))
    positive_val_count = len(_collect_wavs(positive_val_dir / "pos"))
    phrase_negative_train_count = len(_collect_wavs(negative_train_dir / "neg"))
    phrase_negative_val_count = len(_collect_wavs(negative_val_dir / "neg"))
    background_train_count = _write_background_negative_clips(
        negative_train_dir / "neg",
        count=max(96, positive_train_count - phrase_negative_train_count),
        sample_rate=sample_rate,
        seed=20260624,
    )
    background_val_count = _write_background_negative_clips(
        negative_val_dir / "neg",
        count=max(32, positive_val_count - phrase_negative_val_count),
        sample_rate=sample_rate,
        seed=20260625,
    )
    print(
        f"Added {background_train_count} train and {background_val_count} validation "
        "silence/background negatives"
    )

    return {
        "positive_train": positive_train_dir / "pos",
        "positive_val": positive_val_dir / "pos",
        "negative_train": negative_train_dir / "neg",
        "negative_val": negative_val_dir / "neg",
    }


def _augment_gain(audio: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(audio.astype(np.float32) * gain, -32768, 32767).astype(np.int16)


def _augment_noise(audio: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    audio_f = audio.astype(np.float32)
    rms = float(np.sqrt(np.mean(np.square(audio_f))))
    if rms < 1.0:
        return audio
    noise_rms = rms / (10.0 ** (snr_db / 20.0))
    noise = rng.normal(0.0, noise_rms, audio.shape).astype(np.float32)
    return np.clip(audio_f + noise, -32768, 32767).astype(np.int16)


_POSITIVE_GAIN_VARIANTS = (1.5, 2.5, 4.0)
_POSITIVE_NOISE_SNR_DB = (10.0, 20.0)


def _build_feature_file(
    *,
    wav_dir: Path,
    output_file: Path,
    sample_length_samples: int,
    batch_size: int,
    augment_negative_windows: bool = False,
    augment_positive_gain: bool = False,
) -> None:
    wav_paths = _collect_wavs(wav_dir)
    if not wav_paths:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    feature_extractor = AudioFeatures(inference_framework="onnx", device="cpu")
    features: list[np.ndarray] = []
    target_samples = max(sample_length_samples, 64000)
    target_frames = 41
    rng = np.random.default_rng(42)
    for wav_path in wav_paths:
        sample_rate, audio = wavfile.read(wav_path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz for {wav_path}")
        if audio.ndim > 1:
            audio = audio[:, 0]
        raw_audio = np.asarray(audio, dtype=np.int16).reshape(-1)
        audio_variants = [_pad_or_trim(raw_audio, target_samples)]

        if augment_positive_gain and raw_audio.size:
            base = _pad_or_trim(raw_audio, target_samples)
            for gain in _POSITIVE_GAIN_VARIANTS:
                audio_variants.append(_augment_gain(base, gain))
            for snr in _POSITIVE_NOISE_SNR_DB:
                audio_variants.append(_augment_noise(base, snr, rng))

        if augment_negative_windows and not wav_path.name.startswith("background_") and raw_audio.size:
            for fraction in (0.4, 0.7):
                prefix_length = max(1, int(round(raw_audio.size * fraction)))
                audio_variants.append(_pad_or_trim(raw_audio[:prefix_length], target_samples))
            trailing_silence = int(round(sample_rate * 0.5))
            shifted = np.zeros(target_samples, dtype=np.int16)
            clipped = raw_audio[-max(1, target_samples - trailing_silence) :]
            end = target_samples - trailing_silence
            shifted[max(0, end - clipped.size) : end] = clipped[-end:]
            audio_variants.append(shifted)
            base = _pad_or_trim(raw_audio, target_samples)
            for gain in _POSITIVE_GAIN_VARIANTS:
                audio_variants.append(_augment_gain(base, gain))

        for audio_variant in audio_variants:
            extracted = feature_extractor._get_embeddings(audio_variant)
            if extracted.ndim != 2:
                raise ValueError(f"Unexpected feature shape {extracted.shape} for {wav_path}")
            features.append(_pad_or_trim_frames(extracted.astype(np.float32, copy=False), target_frames))

    np.save(output_file, np.stack(features, axis=0).astype(np.float32, copy=False))


def _load_feature_windows(feature_file: Path, target_frames: int) -> np.ndarray:
    features = np.load(feature_file)
    if features.ndim != 3:
        raise ValueError(f"Expected feature tensor with 3 dimensions in {feature_file}, got {features.shape}")

    if features.shape[1] > target_frames:
        features = features[:, -target_frames:, :]
    elif features.shape[1] < target_frames:
        pad_width = target_frames - features.shape[1]
        pad_frame = np.zeros((features.shape[0], pad_width, features.shape[2]), dtype=features.dtype)
        features = np.concatenate([pad_frame, features], axis=1)

    return features.astype(np.float32, copy=False)


def _build_loader(features: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool = True) -> DataLoader:
    x = torch.from_numpy(features.astype(np.float32, copy=False))
    y = torch.from_numpy(labels.astype(np.float32, copy=False))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def _make_dataloaders(*, feature_root: Path, batch_size: int, target_frames: int) -> tuple[DataLoader, DataLoader, DataLoader]:
    pos_train = _load_feature_windows(feature_root / "positive_features_train.npy", target_frames)
    neg_train = _load_feature_windows(feature_root / "negative_features_train.npy", target_frames)
    pos_val = _load_feature_windows(feature_root / "positive_features_test.npy", target_frames)
    neg_val = _load_feature_windows(feature_root / "negative_features_test.npy", target_frames)

    if pos_train.shape[0] < neg_train.shape[0]:
        repeats = int(np.ceil(neg_train.shape[0] / max(1, pos_train.shape[0])))
        balanced_pos_train = np.tile(pos_train, (repeats, 1, 1))[: neg_train.shape[0]]
    else:
        balanced_pos_train = pos_train
        if neg_train.shape[0] < pos_train.shape[0]:
            repeats = int(np.ceil(pos_train.shape[0] / max(1, neg_train.shape[0])))
            neg_train = np.tile(neg_train, (repeats, 1, 1))[: pos_train.shape[0]]

    train_features = np.vstack((balanced_pos_train, neg_train))
    train_labels = np.hstack((np.ones(balanced_pos_train.shape[0]), np.zeros(neg_train.shape[0])))

    val_features = np.vstack((pos_val, neg_val))
    val_labels = np.hstack((np.ones(pos_val.shape[0]), np.zeros(neg_val.shape[0])))

    fp_loader = _build_loader(neg_val, np.zeros(neg_val.shape[0], dtype=np.float32), batch_size=batch_size, shuffle=False)
    train_loader = _build_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_features, val_labels, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, fp_loader


class WakeClassifier(nn.Module):
    def __init__(self, target_frames: int, feature_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.target_frames = target_frames
        self.feature_dim = feature_dim
        input_dim = target_frames * feature_dim
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def _train_classifier(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
    target_frames: int,
    feature_dim: int,
) -> WakeClassifier:
    model = WakeClassifier(target_frames=target_frames, feature_dim=feature_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(max(1, epochs)):
        model.train()
        train_loss = 0.0
        train_examples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = inputs.shape[0]
            train_loss += float(loss.item()) * batch_size
            train_examples += batch_size

        model.eval()
        val_loss = 0.0
        val_examples = 0
        val_correct = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).unsqueeze(1)
                logits = model(inputs)
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                predictions = (probs >= 0.5).float()
                val_correct += int((predictions == labels).sum().item())
                batch_size = inputs.shape[0]
                val_loss += float(loss.item()) * batch_size
                val_examples += batch_size

        print(
            f"epoch={epoch + 1}/{epochs} train_loss={train_loss / max(1, train_examples):.4f} "
            f"val_loss={val_loss / max(1, val_examples):.4f} val_acc={val_correct / max(1, val_examples):.3f}"
        )

    return model.cpu().eval()


def _export_onnx(model: WakeClassifier, output_path: Path, target_frames: int, feature_dim: int) -> None:
    class ProbabilityOutput(nn.Module):
        def __init__(self, classifier: WakeClassifier):
            super().__init__()
            self.classifier = classifier

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.classifier(x))

    dummy_input = torch.zeros((1, target_frames, feature_dim), dtype=torch.float32)
    torch.onnx.export(
        ProbabilityOutput(model).eval(),
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
        external_data=False,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the unified English/Arabic wake-word model.")
    parser.add_argument("--work-dir", default="data/jarvis_unified_training", help="Training workspace directory")
    parser.add_argument("--output-dir", default="models/jarvis_unified", help="Where the ONNX model will be saved")
    parser.add_argument("--model-name", default="jarvis_unified", help="Exported model name without extension")
    parser.add_argument("--train-count", type=int, default=180, help="Approximate number of training clips per class")
    parser.add_argument("--val-count", type=int, default=60, help="Approximate number of validation clips per class")
    parser.add_argument("--sample-length-seconds", type=float, default=2.0, help="Generated clip length")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for feature generation and training")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Training learning rate")
    parser.add_argument("--target-frames", type=int, default=41, help="Feature window length used by the runtime")
    parser.add_argument(
        "--user-positive-dir",
        default=str(WAKE_WORD_USER_SAMPLES_DIR),
        help="Folder of user-recorded wake-word WAVs to mix into the positive class",
    )
    parser.add_argument(
        "--user-positive-val-ratio",
        type=float,
        default=0.2,
        help="Share of user-recorded positive WAVs to reserve for validation",
    )
    parser.add_argument(
        "--user-positive-dir-en",
        default="",
        help="Optional folder of English user-recorded wake-word WAVs",
    )
    parser.add_argument(
        "--reuse-clips",
        action="store_true",
        help="Reuse existing generated WAV directories and rebuild features/model only",
    )
    parser.add_argument(
        "--capture-negatives",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Record N seconds of user speaking non-wake phrases for negative training (default: 0 = skip)",
    )
    parser.add_argument(
        "--capture-ambient",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Record N seconds of ambient noise from mic and add to negative training set before training",
    )
    return parser.parse_args()


def _capture_ambient_negatives(
    *,
    duration_seconds: int,
    neg_train_dir: Path,
    neg_val_dir: Path,
    sample_rate: int = 16000,
    gains: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> int:
    """Record live ambient noise and split into negative training clips."""
    import sounddevice as _sd

    print(f"Recording {duration_seconds}s of ambient noise. Do NOT speak...")
    audio = _sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    _sd.wait()
    audio = audio.reshape(-1)

    for old in neg_train_dir.glob("real_ambient*.wav"):
        old.unlink()
    for old in neg_val_dir.glob("real_ambient*.wav"):
        old.unlink()

    clip_len = int(2.0 * sample_rate)
    hop = int(1.0 * sample_rate)
    count = 0
    for start in range(0, len(audio) - clip_len, hop):
        clip = audio[start : start + clip_len]
        for gi, gain in enumerate(gains):
            boosted = np.clip(clip.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
            if count < 15:
                wavfile.write(neg_val_dir / f"real_ambient_{count:04d}_g{gi}.wav", sample_rate, boosted)
            else:
                wavfile.write(neg_train_dir / f"real_ambient_{count:04d}_g{gi}.wav", sample_rate, boosted)
            count += 1
    print(f"Captured {count} real ambient negative clips")
    return count


_USER_NEGATIVE_PHRASES = [
    "open chrome",
    "what time is it",
    "play some music",
    "turn on bluetooth",
    "what's the weather",
    "set a timer",
    "hello",
    "hey",
    "hi there",
    "good morning",
    "thank you",
    "stop",
    "cancel",
    "never mind",
    "افتح كروم",
    "كام الساعة",
    "شغل اغنية",
    "ايه اخبار الطقس",
    "اقفل النور",
    "شكرا",
    "صباح الخير",
    "تمام",
    "الغي",
]


def _capture_user_negatives(
    *,
    neg_train_dir: Path,
    neg_val_dir: Path,
    duration_seconds: int = 60,
    sample_rate: int = 16000,
    gains: tuple[float, ...] = (1.0, 2.0, 4.0),
) -> int:
    """Record user speaking non-wake phrases in a single long session.

    The user talks freely for ``duration_seconds``, saying commands,
    greetings, random sentences — anything except the wake word.  The
    recording is split into overlapping 2.5-second clips at multiple
    gain levels and added to the negative training set.
    """
    import sounddevice as _sd

    for old in neg_train_dir.glob("user_neg_*.wav"):
        old.unlink()
    for old in neg_val_dir.glob("user_neg_*.wav"):
        old.unlink()

    phrases_hint = ", ".join(f'"{p}"' for p in _USER_NEGATIVE_PHRASES[:8])
    print(f"\n=== Recording {duration_seconds}s of NON-wake-word speech ===")
    print(f"Say commands, greetings, random sentences — anything EXCEPT the wake word.")
    print(f"Examples: {phrases_hint} ...")
    print("Recording starts NOW.\n")

    audio = _sd.rec(int(duration_seconds * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    _sd.wait()
    audio = audio.reshape(-1)

    clip_len = int(2.5 * sample_rate)
    hop = int(1.0 * sample_rate)
    clips: list[np.ndarray] = []
    for start in range(0, len(audio) - clip_len, hop):
        clip = audio[start:start + clip_len]
        rms = float(np.sqrt(np.mean(np.square(clip.astype(np.float32) / 32768.0))))
        if rms >= 0.004:
            clips.append(clip)

    count = 0
    val_count = max(1, len(clips) // 5)
    for idx, clip in enumerate(clips):
        for gi, gain in enumerate(gains):
            boosted = np.clip(clip.astype(np.float32) * gain, -32768, 32767).astype(np.int16)
            if idx < val_count:
                wavfile.write(neg_val_dir / f"user_neg_{count:04d}_g{gi}.wav", sample_rate, boosted)
            else:
                wavfile.write(neg_train_dir / f"user_neg_{count:04d}_g{gi}.wav", sample_rate, boosted)
            count += 1

    print(f"Created {count} user-voice negative clips from {len(clips)} speech segments\n")
    return count


def main() -> None:
    args = _parse_args()
    work_dir = Path(args.work_dir).resolve()
    feature_dir = work_dir / "features"
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(work_dir)
    _ensure_dir(feature_dir)
    _ensure_dir(output_dir)

    neg_train = work_dir / "negative_train" / "neg"
    neg_val = work_dir / "negative_val" / "neg"

    if args.capture_negatives > 0:
        _ensure_dir(neg_train)
        _ensure_dir(neg_val)
        _capture_user_negatives(
            neg_train_dir=neg_train,
            neg_val_dir=neg_val,
            duration_seconds=args.capture_negatives,
        )

    if args.capture_ambient > 0:
        _ensure_dir(neg_train)
        _ensure_dir(neg_val)
        _capture_ambient_negatives(
            duration_seconds=args.capture_ambient,
            neg_train_dir=neg_train,
            neg_val_dir=neg_val,
        )

    if args.reuse_clips:
        clip_dirs = {
            "positive_train": work_dir / "positive_train" / "pos",
            "positive_val": work_dir / "positive_val" / "pos",
            "negative_train": work_dir / "negative_train" / "neg",
            "negative_val": work_dir / "negative_val" / "neg",
        }
        for label, directory in clip_dirs.items():
            if not _collect_wavs(directory):
                raise RuntimeError(f"Cannot reuse clips; {label} has no WAV files in {directory}")
        for positive_dir in (clip_dirs["positive_train"], clip_dirs["positive_val"]):
            for copied_user_clip in positive_dir.glob("**/user_default_*.wav"):
                copied_user_clip.unlink()
        if str(args.user_positive_dir).strip():
            reused_train_count, reused_val_count = _ingest_user_positive_clips(
                source_dir=Path(args.user_positive_dir).resolve(),
                train_dir=clip_dirs["positive_train"],
                val_dir=clip_dirs["positive_val"],
                sample_rate=16000,
                val_ratio=args.user_positive_val_ratio,
                source_label="default",
            )
            print(
                f"Re-imported {reused_train_count} curated user clips into train and "
                f"{reused_val_count} into val; unverified automatic captures are excluded"
            )
        phrase_negative_train_count = len(
            [path for path in _collect_wavs(clip_dirs["negative_train"]) if not path.name.startswith("background_")]
        )
        phrase_negative_val_count = len(
            [path for path in _collect_wavs(clip_dirs["negative_val"]) if not path.name.startswith("background_")]
        )
        _write_background_negative_clips(
            clip_dirs["negative_train"],
            count=max(96, len(_collect_wavs(clip_dirs["positive_train"])) - phrase_negative_train_count),
            sample_rate=16000,
            seed=20260624,
        )
        _write_background_negative_clips(
            clip_dirs["negative_val"],
            count=max(32, len(_collect_wavs(clip_dirs["positive_val"])) - phrase_negative_val_count),
            sample_rate=16000,
            seed=20260625,
        )
        print("Reusing existing clips and refreshing silence/background negatives")
    else:
        clip_dirs = _write_positive_and_negative_clips(
            output_root=work_dir,
            train_count=args.train_count,
            val_count=args.val_count,
            sample_rate=16000,
            user_positive_dir=Path(args.user_positive_dir).resolve() if str(args.user_positive_dir).strip() else None,
            user_positive_dir_en=(
                Path(args.user_positive_dir_en).resolve() if str(args.user_positive_dir_en).strip() else None
            ),
            user_positive_val_ratio=args.user_positive_val_ratio,
        )

    sample_length_samples = int(round(16000 * args.sample_length_seconds))

    # Build feature arrays from generated WAVs.
    _build_feature_file(
        wav_dir=clip_dirs["positive_train"],
        output_file=feature_dir / "positive_features_train.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
        augment_positive_gain=True,
    )
    _build_feature_file(
        wav_dir=clip_dirs["negative_train"],
        output_file=feature_dir / "negative_features_train.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
        augment_negative_windows=True,
    )
    _build_feature_file(
        wav_dir=clip_dirs["positive_val"],
        output_file=feature_dir / "positive_features_test.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
        augment_positive_gain=True,
    )
    _build_feature_file(
        wav_dir=clip_dirs["negative_val"],
        output_file=feature_dir / "negative_features_test.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
        augment_negative_windows=True,
    )

    train_loader, val_loader, _ = _make_dataloaders(
        feature_root=feature_dir,
        batch_size=args.batch_size,
        target_frames=args.target_frames,
    )

    sample_audio = AudioFeatures(inference_framework="onnx", device="cpu")
    feature_dim = int(sample_audio.get_embedding_shape(args.sample_length_seconds)[1])

    model = _train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=torch.device("cpu"),
        target_frames=args.target_frames,
        feature_dim=feature_dim,
    )

    onnx_path = output_dir / f"{args.model_name}.onnx"
    _export_onnx(model, onnx_path, target_frames=args.target_frames, feature_dim=feature_dim)
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
