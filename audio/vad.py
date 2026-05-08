import pathlib
import threading
import urllib.request
import wave

import numpy as np
from core.config import SAMPLE_RATE, VAD_BACKEND, VAD_ENERGY_THRESHOLD, VAD_SILERO_THRESHOLD
from core.logger import logger

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover - optional in lean environments
    ort = None
    _ONNXRUNTIME_IMPORT_ERROR = exc
else:
    _ONNXRUNTIME_IMPORT_ERROR = None


_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
_DEFAULT_SILERO_MODEL_PATH = _PROJECT_ROOT / "data" / "vad" / "silero_vad.onnx"
_DEFAULT_SILERO_MODEL_URL = (
    "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
)

_ENERGY_FALLBACK_THRESHOLD = max(0.001, float(VAD_ENERGY_THRESHOLD))


def set_energy_fallback_threshold(value):
    global _ENERGY_FALLBACK_THRESHOLD
    _ENERGY_FALLBACK_THRESHOLD = max(0.001, float(value))
    return _ENERGY_FALLBACK_THRESHOLD


def get_energy_fallback_threshold():
    return float(_ENERGY_FALLBACK_THRESHOLD)


def _coerce_audio_array(audio_chunk):
    if audio_chunk is None:
        return np.asarray([], dtype=np.float32)

    if isinstance(audio_chunk, (str, pathlib.Path)):
        try:
            with wave.open(str(audio_chunk), "rb") as handle:
                frames = handle.readframes(handle.getnframes())
                sample_width = int(handle.getsampwidth())
                channels = int(handle.getnchannels())
        except Exception:
            return np.asarray([], dtype=np.float32)

        if not frames:
            return np.asarray([], dtype=np.float32)

        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            return np.asarray([], dtype=np.float32)

        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    audio = np.asarray(audio_chunk)
    if audio.size == 0:
        return np.asarray([], dtype=np.float32)
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    if audio.dtype == np.int32:
        return audio.astype(np.float32) / 2147483648.0
    if audio.dtype == np.uint8:
        return (audio.astype(np.float32) - 128.0) / 128.0
    return audio.astype(np.float32, copy=False).reshape(-1)


def _download_file(url, target_path):
    target = pathlib.Path(target_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    target.write_bytes(data)


def _compute_rms(audio_float):
    if audio_float.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio_float.astype(np.float32)))))


def _energy_fallback_is_speech(audio_path):
    audio = _coerce_audio_array(audio_path)
    if audio.size == 0:
        return False
    energy = _compute_rms(audio)
    return energy >= float(_ENERGY_FALLBACK_THRESHOLD)


class SileroVAD:
    def __init__(
        self,
        model_path=None,
        *,
        sample_rate=16000,
        threshold=0.5,
        energy_threshold=None,
        model_url=_DEFAULT_SILERO_MODEL_URL,
    ):
        self.sample_rate = int(sample_rate or SAMPLE_RATE or 16000)
        if self.sample_rate not in {8000, 16000}:
            self.sample_rate = 16000
        self.threshold = max(0.05, min(0.95, float(threshold)))
        self.energy_threshold = max(0.001, float(
            _ENERGY_FALLBACK_THRESHOLD if energy_threshold is None else energy_threshold
        ))
        self.window_size_samples = 512 if self.sample_rate == 16000 else 256
        self.context_size = 64 if self.sample_rate == 16000 else 32
        self.model_path = pathlib.Path(model_path or _DEFAULT_SILERO_MODEL_PATH)
        self.model_url = str(model_url or _DEFAULT_SILERO_MODEL_URL).strip()
        self._session = None
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        self._fallback_only = True
        self._load_model()

    def _load_model(self):
        if ort is None:
            logger.warning(
                "Silero VAD ONNX unavailable; falling back to energy gating (%s).",
                _ONNXRUNTIME_IMPORT_ERROR,
            )
            return

        try:
            if not self.model_path.exists():
                _download_file(self.model_url, self.model_path)

            session_options = ort.SessionOptions()
            session_options.inter_op_num_threads = 1
            session_options.intra_op_num_threads = 1

            providers = ["CPUExecutionProvider"] if "CPUExecutionProvider" in ort.get_available_providers() else None
            if providers:
                self._session = ort.InferenceSession(str(self.model_path), sess_options=session_options, providers=providers)
            else:
                self._session = ort.InferenceSession(str(self.model_path), sess_options=session_options)
            self._fallback_only = False
            self.reset()
            logger.info("Loaded Silero VAD ONNX model from %s", self.model_path)
        except Exception as exc:
            self._session = None
            self._fallback_only = True
            logger.warning("Silero VAD ONNX load failed; using energy fallback only: %s", exc)

    def reset(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self.context_size), dtype=np.float32)
        return self

    def is_ready(self) -> bool:
        return self._session is not None

    def is_speech_chunk(self, audio_chunk_int16: np.ndarray) -> bool:
        """Process a single streaming chunk (512 int16 samples at 16 kHz → 32 ms)."""
        chunk = np.asarray(audio_chunk_int16, dtype=np.int16).reshape(-1)
        if chunk.size == 0:
            return False
        audio = chunk.astype(np.float32) / 32768.0
        if _compute_rms(audio) < float(self.energy_threshold):
            return False
        if self._fallback_only or self._session is None:
            return True
        result = self._run_silero_window(audio)
        return bool(result) if result is not None else True

    def _run_silero_window(self, window_float):
        if self._session is None:
            return None

        window = np.asarray(window_float, dtype=np.float32).reshape(-1)
        if window.size < self.window_size_samples:
            pad_width = self.window_size_samples - int(window.size)
            window = np.pad(window, (0, pad_width), mode="constant")
        elif window.size > self.window_size_samples:
            window = window[: self.window_size_samples]

        input_frame = np.concatenate([self._context.reshape(-1), window], axis=0).astype(np.float32, copy=False)
        input_frame = input_frame.reshape(1, -1)
        sr_tensor = np.asarray([self.sample_rate], dtype=np.int64)

        try:
            outputs = self._session.run(
                None,
                {
                    "input": input_frame,
                    "state": self._state,
                    "sr": sr_tensor,
                },
            )
        except Exception as exc:
            logger.warning("Silero VAD inference failed; using energy fallback only: %s", exc)
            self._session = None
            self._fallback_only = True
            return None

        speech_probability = float(np.asarray(outputs[0]).reshape(-1)[0])
        self._state = np.asarray(outputs[1], dtype=np.float32)
        self._context = window[-self.context_size :].reshape(1, -1).astype(np.float32, copy=False)
        return speech_probability >= float(self.threshold)

    def is_speech(self, audio_chunk) -> bool:
        audio = _coerce_audio_array(audio_chunk)
        if audio.size == 0:
            return False

        if _compute_rms(audio) < float(self.energy_threshold):
            return False

        if self._fallback_only or self._session is None:
            return True

        padded_length = int(np.ceil(audio.size / float(self.window_size_samples))) * int(self.window_size_samples)
        if padded_length != audio.size:
            audio = np.pad(audio, (0, padded_length - int(audio.size)), mode="constant")

        detected = False
        for start_index in range(0, audio.size, self.window_size_samples):
            window = audio[start_index : start_index + self.window_size_samples]
            result = self._run_silero_window(window)
            if result is None:
                return True
            if result:
                detected = True
        return detected


def should_run_vad(audio_chunk: np.ndarray, energy_threshold: float = 0.015) -> bool:
    """Return True if the chunk has enough energy to warrant Silero inference."""
    audio = np.asarray(audio_chunk)
    if audio.size == 0:
        return False
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    else:
        audio = audio.astype(np.float32, copy=False)
    return _compute_rms(audio) >= max(0.001, float(energy_threshold))


_batch_vad: "SileroVAD | None" = None
_batch_vad_lock = threading.Lock()


def _get_batch_vad() -> "SileroVAD":
    global _batch_vad
    if _batch_vad is None:
        with _batch_vad_lock:
            if _batch_vad is None:
                backend = str(VAD_BACKEND or "silero").strip().lower()
                if backend == "energy":
                    _batch_vad = SileroVAD.__new__(SileroVAD)
                    _batch_vad.sample_rate = int(SAMPLE_RATE or 16000)
                    _batch_vad.threshold = float(VAD_SILERO_THRESHOLD)
                    _batch_vad.energy_threshold = _ENERGY_FALLBACK_THRESHOLD
                    _batch_vad.window_size_samples = 512
                    _batch_vad.context_size = 64
                    _batch_vad.model_path = _DEFAULT_SILERO_MODEL_PATH
                    _batch_vad.model_url = _DEFAULT_SILERO_MODEL_URL
                    _batch_vad._session = None
                    _batch_vad._state = np.zeros((2, 1, 128), dtype=np.float32)
                    _batch_vad._context = np.zeros((1, 64), dtype=np.float32)
                    _batch_vad._fallback_only = True
                else:
                    _batch_vad = SileroVAD(threshold=float(VAD_SILERO_THRESHOLD))
    return _batch_vad


def prewarm_batch_vad() -> bool:
    """Load the batch VAD singleton now so the first speech-guard check is instant."""
    try:
        vad = _get_batch_vad()
        return vad.is_ready()
    except Exception:
        return False


def is_speech(audio_path) -> bool:
    """Batch speech detection for a WAV file. Uses SileroVAD if available."""
    vad = _get_batch_vad()
    vad.reset()
    return vad.is_speech(audio_path)
