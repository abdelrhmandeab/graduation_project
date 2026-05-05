"""Generate Arabic wake-word training WAVs for openWakeWord.

Outputs 16 kHz mono WAVs organized as:
  <output_dir>/<keyword>/...

The generated audio is synthesized with edge-tts Arabic voices and augmented
with white noise and simple room reverb to improve robustness.
"""

import argparse
import asyncio
import io
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import edge_tts
except Exception as exc:
    raise RuntimeError("edge-tts is required to run this script.") from exc


DEFAULT_VOICES = [
    "ar-EG-SalmaNeural",
    "ar-EG-ShakirNeural",
    "ar-SA-HamedNeural",
]
DEFAULT_PHRASES = [
    "جارفيس",
    "يا جارفيس",
]


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _resample_linear(audio: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return audio
    if audio.size == 0:
        return audio
    duration = audio.shape[0] / float(src_sr)
    target_len = max(1, int(round(duration * float(target_sr))))
    x_old = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    x_new = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32, copy=False)


def _normalize_peak(audio: np.ndarray, peak: float = 0.97) -> np.ndarray:
    if audio.size == 0:
        return audio
    max_val = float(np.max(np.abs(audio)))
    if max_val <= 0:
        return audio
    scale = min(1.0, float(peak) / max_val)
    return (audio * scale).astype(np.float32, copy=False)


def _add_white_noise(audio: np.ndarray, snr_db: float) -> np.ndarray:
    if audio.size == 0:
        return audio
    signal_power = float(np.mean(audio ** 2))
    if signal_power <= 0:
        return audio
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0.0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)
    return (audio + noise).astype(np.float32, copy=False)


def _make_impulse_response(sample_rate: int, length_ms: int = 220) -> np.ndarray:
    length = max(16, int(sample_rate * (length_ms / 1000.0)))
    decay = np.exp(-np.linspace(0.0, 3.2, num=length))
    impulse = np.zeros(length, dtype=np.float32)
    impulse[0] = 1.0
    reflection_count = 6
    for _ in range(reflection_count):
        idx = random.randint(1, length - 1)
        impulse[idx] += random.uniform(0.15, 0.45) * decay[idx]
    impulse *= decay
    impulse /= max(1e-6, float(np.max(np.abs(impulse))))
    return impulse.astype(np.float32, copy=False)


def _add_room_reverb(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size == 0:
        return audio
    impulse = _make_impulse_response(sample_rate)
    wet = np.convolve(audio, impulse, mode="full")[: audio.shape[0]]
    mixed = 0.78 * audio + 0.22 * wet
    return mixed.astype(np.float32, copy=False)


async def _synthesize_phrase(text: str, voice: str) -> np.ndarray:
    communicate = edge_tts.Communicate(text, voice)
    buffer = io.BytesIO()
    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            buffer.write(chunk.get("data") or b"")
    buffer.seek(0)
    audio, sr = sf.read(buffer, dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1).astype(np.float32, copy=False)
    return audio, int(sr)


async def generate_samples(
    *,
    phrases,
    voices,
    output_dir: Path,
    keyword: str,
    samples_per_phrase: int,
    sample_rate: int,
    snr_db_choices,
    apply_reverb: bool,
):
    keyword_dir = output_dir / keyword
    _ensure_dir(keyword_dir)

    total = 0
    for phrase in phrases:
        for voice in voices:
            for idx in range(samples_per_phrase):
                audio, sr = await _synthesize_phrase(phrase, voice)
                audio = _resample_linear(audio, sr, sample_rate)
                audio = _normalize_peak(audio)

                snr_db = random.choice(snr_db_choices)
                audio = _add_white_noise(audio, snr_db)
                if apply_reverb:
                    audio = _add_room_reverb(audio, sample_rate)
                audio = _normalize_peak(audio)

                safe_voice = voice.replace("/", "-")
                safe_phrase = "_".join(phrase.split())
                filename = f"{safe_phrase}_{safe_voice}_{idx:03d}.wav"
                target = keyword_dir / filename
                sf.write(target, audio, sample_rate, subtype="PCM_16")
                total += 1

    return total


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate Arabic wake-word training WAVs.")
    parser.add_argument("--output", default="data/openwakeword", help="Output root directory")
    parser.add_argument("--keyword", default="jarvis_ar", help="Keyword folder name")
    parser.add_argument("--samples-per-phrase", type=int, default=20, help="Samples per phrase+voice")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Target sample rate")
    parser.add_argument("--snr-db", default="18,22,26", help="Comma-separated SNR values")
    parser.add_argument("--no-reverb", action="store_true", help="Disable reverb augmentation")
    return parser.parse_args()


def main():
    args = _parse_args()
    output_dir = Path(args.output).resolve()
    _ensure_dir(output_dir)

    snr_db_choices = [float(item) for item in str(args.snr_db).split(",") if item.strip()]
    if not snr_db_choices:
        snr_db_choices = [22.0]

    total = asyncio.run(
        generate_samples(
            phrases=DEFAULT_PHRASES,
            voices=DEFAULT_VOICES,
            output_dir=output_dir,
            keyword=str(args.keyword),
            samples_per_phrase=max(1, int(args.samples_per_phrase)),
            sample_rate=max(8000, int(args.sample_rate)),
            snr_db_choices=snr_db_choices,
            apply_reverb=not bool(args.no_reverb),
        )
    )
    print(f"Generated {total} WAV files in {output_dir / args.keyword}")


if __name__ == "__main__":
    main()
