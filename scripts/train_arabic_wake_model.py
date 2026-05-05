"""Train and export a custom Arabic wake-word ONNX model.

This script uses the generated Arabic wake phrases, creates a small set of
adversarial negative clips, computes openWakeWord features, trains a compact
PyTorch classifier, and exports the result as ONNX.

It is intentionally self-contained so it can run on Windows without the Linux-
only openWakeWord notebook flow.
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import sys
from pathlib import Path
from typing import Iterable, Iterator, Sequence

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

from generate_arabic_wake_data import DEFAULT_PHRASES, DEFAULT_VOICES, generate_samples


DEFAULT_NEGATIVE_PHRASES = [
    "افتح اليوتيوب",
    "شغل اغنية",
    "what time is it",
    "open browser",
    "where is my phone",
    "كام الساعة",
    "ايه اخبار الطقس",
    "turn on bluetooth",
    "اقفل النور",
    "hello jarvis",
]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _collect_wavs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.glob("**/*.wav") if p.is_file())


def _pad_or_trim(audio: np.ndarray, target_samples: int) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.int16).reshape(-1)
    if audio.shape[0] > target_samples:
        return audio[:target_samples]
    if audio.shape[0] < target_samples:
        padded = np.zeros(target_samples, dtype=np.int16)
        padded[: audio.shape[0]] = audio
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


def _collect_user_positive_wavs(directory: Path) -> list[Path]:
    if not directory.exists() or not directory.is_dir():
        return []
    return sorted(path for path in directory.glob("**/*.wav") if path.is_file())


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

        target_dir = val_dir if index < val_count else train_dir
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
        target_path = target_dir / f"user_{index:04d}_{safe_stem}.wav"
        wavfile.write(target_path, sample_rate, audio)

        if target_dir == val_dir:
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
    sample_length_seconds: float,
    voices: Sequence[str],
    user_positive_dir: Path | None = None,
    user_positive_val_ratio: float = 0.2,
) -> dict[str, Path]:
    """Generate the training and validation WAV clips used by the trainer."""
    sample_length = int(round(sample_rate * sample_length_seconds))

    positive_train_dir = output_root / "positive_train"
    positive_val_dir = output_root / "positive_val"
    negative_train_dir = output_root / "negative_train"
    negative_val_dir = output_root / "negative_val"

    for path in (positive_train_dir, positive_val_dir, negative_train_dir, negative_val_dir):
        if path.exists():
            shutil.rmtree(path)
        _ensure_dir(path)

    async def _generate_all():
        # Positive clips
        await generate_samples(
            phrases=list(DEFAULT_PHRASES),
            voices=list(voices),
            output_dir=positive_train_dir,
            keyword="pos",
            samples_per_phrase=max(1, train_count // max(1, len(DEFAULT_PHRASES) * len(voices))),
            sample_rate=sample_rate,
            snr_db_choices=[16.0, 20.0, 24.0],
            apply_reverb=True,
        )
        await generate_samples(
            phrases=list(DEFAULT_PHRASES),
            voices=list(voices),
            output_dir=positive_val_dir,
            keyword="pos",
            samples_per_phrase=max(1, val_count // max(1, len(DEFAULT_PHRASES) * len(voices))),
            sample_rate=sample_rate,
            snr_db_choices=[18.0, 22.0],
            apply_reverb=True,
        )

        # Negative clips
        await generate_samples(
            phrases=list(DEFAULT_NEGATIVE_PHRASES),
            voices=list(voices),
            output_dir=negative_train_dir,
            keyword="neg",
            samples_per_phrase=max(1, train_count // max(1, len(DEFAULT_NEGATIVE_PHRASES) * len(voices))),
            sample_rate=sample_rate,
            snr_db_choices=[14.0, 18.0, 22.0],
            apply_reverb=True,
        )
        await generate_samples(
            phrases=list(DEFAULT_NEGATIVE_PHRASES),
            voices=list(voices),
            output_dir=negative_val_dir,
            keyword="neg",
            samples_per_phrase=max(1, val_count // max(1, len(DEFAULT_NEGATIVE_PHRASES) * len(voices))),
            sample_rate=sample_rate,
            snr_db_choices=[16.0, 20.0],
            apply_reverb=True,
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
        )
        if user_train_count or user_val_count:
            print(
                f"Imported {user_train_count} user wake clips into train and {user_val_count} into val "
                f"from {Path(user_positive_dir).resolve()}"
            )

    return {
        "positive_train": positive_train_dir / "pos",
        "positive_val": positive_val_dir / "pos",
        "negative_train": negative_train_dir / "neg",
        "negative_val": negative_val_dir / "neg",
    }


def _build_feature_file(
    *,
    wav_dir: Path,
    output_file: Path,
    sample_length_samples: int,
    batch_size: int,
) -> None:
    wav_paths = _collect_wavs(wav_dir)
    if not wav_paths:
        raise RuntimeError(f"No WAV files found in {wav_dir}")

    feature_extractor = AudioFeatures(inference_framework="onnx", device="cpu")
    features: list[np.ndarray] = []
    target_samples = max(sample_length_samples, 64000)
    target_frames = 41
    for wav_path in wav_paths:
        sample_rate, audio = wavfile.read(wav_path)
        if sample_rate != 16000:
            raise ValueError(f"Expected 16 kHz audio, got {sample_rate} Hz for {wav_path}")
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio = _pad_or_trim(np.asarray(audio, dtype=np.int16), target_samples)
        extracted = feature_extractor._get_embeddings(audio)
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

    train_features = np.vstack((pos_train, neg_train))
    train_labels = np.hstack((np.ones(pos_train.shape[0]), np.zeros(neg_train.shape[0])))

    val_features = np.vstack((pos_val, neg_val))
    val_labels = np.hstack((np.ones(pos_val.shape[0]), np.zeros(neg_val.shape[0])))

    fp_loader = _build_loader(neg_val, np.zeros(neg_val.shape[0], dtype=np.float32), batch_size=batch_size, shuffle=False)
    train_loader = _build_loader(train_features, train_labels, batch_size=batch_size, shuffle=True)
    val_loader = _build_loader(val_features, val_labels, batch_size=batch_size, shuffle=True)
    return train_loader, val_loader, fp_loader


class WakeClassifier(nn.Module):
    def __init__(self, target_frames: int, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.target_frames = target_frames
        self.feature_dim = feature_dim
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(target_frames * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
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
    dummy_input = torch.zeros((1, target_frames, feature_dim), dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamic_axes=None,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export a custom Arabic openWakeWord ONNX model.")
    parser.add_argument("--work-dir", default="data/arabic_wake_training", help="Training workspace directory")
    parser.add_argument("--output-dir", default="models/arabic_wake", help="Where the ONNX model will be saved")
    parser.add_argument("--model-name", default="jarvis_ar_custom", help="Exported model name without extension")
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    work_dir = Path(args.work_dir).resolve()
    feature_dir = work_dir / "features"
    output_dir = Path(args.output_dir).resolve()
    _ensure_dir(work_dir)
    _ensure_dir(feature_dir)
    _ensure_dir(output_dir)

    clip_dirs = _write_positive_and_negative_clips(
        output_root=work_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        sample_rate=16000,
        sample_length_seconds=args.sample_length_seconds,
        voices=DEFAULT_VOICES,
        user_positive_dir=Path(args.user_positive_dir).resolve() if str(args.user_positive_dir).strip() else None,
        user_positive_val_ratio=args.user_positive_val_ratio,
    )

    sample_length_samples = int(round(16000 * args.sample_length_seconds))

    # Build feature arrays from generated WAVs.
    _build_feature_file(
        wav_dir=clip_dirs["positive_train"],
        output_file=feature_dir / "positive_features_train.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
    )
    _build_feature_file(
        wav_dir=clip_dirs["negative_train"],
        output_file=feature_dir / "negative_features_train.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
    )
    _build_feature_file(
        wav_dir=clip_dirs["positive_val"],
        output_file=feature_dir / "positive_features_test.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
    )
    _build_feature_file(
        wav_dir=clip_dirs["negative_val"],
        output_file=feature_dir / "negative_features_test.npy",
        sample_length_samples=sample_length_samples,
        batch_size=args.batch_size,
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
    onnx_path = output_dir / f"{args.model_name}.onnx"
    print(f"Exported ONNX model to {onnx_path}")


if __name__ == "__main__":
    main()
