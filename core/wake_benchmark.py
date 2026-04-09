import time
from unittest.mock import patch

import numpy as np

from audio import wake_word
from core.config import SAMPLE_RATE, WAKE_BENCHMARK_SCENARIOS_PER_LANGUAGE, WAKE_WORD_CHUNK_SIZE


_STREAM_EXHAUSTED = "WAKE_BENCH_STREAM_EXHAUSTED"


class _FakeClock:
    def __init__(self):
        self._now = 0.0

    def now(self):
        return float(self._now)

    def advance(self, seconds):
        self._now += max(0.0, float(seconds))


class _FakeInputStream:
    def __init__(self, clock, max_reads, chunk_size):
        self._clock = clock
        self._max_reads = max(1, int(max_reads))
        self._chunk_size = max(1, int(chunk_size))
        self._reads = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, frames):
        frame_count = max(1, int(frames or self._chunk_size))
        if self._reads >= self._max_reads:
            raise RuntimeError(_STREAM_EXHAUSTED)

        self._reads += 1
        self._clock.advance(float(frame_count) / float(SAMPLE_RATE))
        chunk = np.zeros((frame_count, 1), dtype=np.int16)
        return chunk, None


class _FakeSoundDevice:
    def __init__(self, stream):
        self._stream = stream

    def InputStream(self, **_kwargs):
        return self._stream


class _FakeEnglishModel:
    def __init__(self, scores):
        self._scores = list(scores or [0.0])
        self._index = 0

    def predict(self, _audio_chunk):
        if self._index < len(self._scores):
            score = float(self._scores[self._index])
        else:
            score = float(self._scores[-1])
        self._index += 1
        return {wake_word.WAKE_WORD: score}


class _FakeArabicTranscriber:
    def __init__(self, transcripts):
        self._transcripts = list(transcripts or [""])
        self._index = 0

    def next_text(self, _audio_window, _model_name):
        if self._index < len(self._transcripts):
            value = str(self._transcripts[self._index] or "")
        else:
            value = str(self._transcripts[-1] or "")
        self._index += 1
        return value


def _percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    index = int(round((float(p) / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _default_scenarios():
    return _build_default_scenario_pack(int(WAKE_BENCHMARK_SCENARIOS_PER_LANGUAGE))


def _build_default_scenario_pack(target_per_language):
    target = max(10, min(40, int(target_per_language or WAKE_BENCHMARK_SCENARIOS_PER_LANGUAGE)))
    positive_count = max(4, target // 2)
    negative_count = max(1, target - positive_count)

    scenarios = []
    for index in range(positive_count):
        scenarios.append(_english_positive_scenario(index + 1))
    for index in range(negative_count):
        scenarios.append(_english_negative_scenario(index + 1))

    for index in range(positive_count):
        scenarios.append(_arabic_positive_scenario(index + 1))
    for index in range(negative_count):
        scenarios.append(_arabic_negative_scenario(index + 1))

    return scenarios


def _english_positive_scenario(index):
    detection_read = 4 + ((index - 1) % 7)  # 4..10
    detection_score = 0.58 + (0.02 * ((index - 1) % 4))
    warmup = [0.08 + (0.02 * ((step + index) % 3)) for step in range(max(0, detection_read - 1))]

    return {
        "name": f"english_detection_latency_{index:02d}",
        "mode": "english",
        "expected_detection": True,
        "english_scores": warmup + [detection_score],
        "max_reads": max(24, detection_read + 10),
    }


def _english_negative_scenario(index):
    length = 40 + (index % 8)
    near_threshold = 0.28 + (0.01 * (index % 3))
    scores = []
    for step in range(length):
        base = 0.07 + (0.03 * ((step + index) % 5))
        if step % 11 == 0:
            base = near_threshold
        scores.append(min(base, 0.34))

    return {
        "name": f"english_false_positive_guard_{index:02d}",
        "mode": "english",
        "expected_detection": False,
        "english_scores": scores,
        "max_reads": len(scores),
    }


def _arabic_positive_scenario(index):
    # Runtime enforces ar_check_interval_seconds >= 0.5, so keep hits early to stay within p95 SLA.
    first_hit_check = 1 + ((index - 1) % 2)  # 1..2
    transcripts = [""] * max(0, first_hit_check - 1) + ["يا جارفيس", "يا جارفيس"]

    return {
        "name": f"arabic_detection_latency_{index:02d}",
        "mode": "arabic",
        "expected_detection": True,
        "arabic_triggers": ["يا جارفيس"],
        "arabic_transcripts": transcripts,
        "ar_check_interval_seconds": 0.50,
        "ar_consecutive_hits_required": 2,
        "max_reads": 64,
    }


def _arabic_negative_scenario(index):
    # Alternate between no-hit and single-hit-only transcripts to stress false-positive guards.
    if index % 2 == 0:
        transcripts = [""] * 12
    else:
        first_hit_check = 2 + ((index - 1) % 5)
        transcripts = [""] * max(0, first_hit_check - 1) + ["يا جارفيس"] + [""] * 5

    return {
        "name": f"arabic_false_positive_guard_{index:02d}",
        "mode": "arabic",
        "expected_detection": False,
        "arabic_triggers": ["يا جارفيس"],
        "arabic_transcripts": transcripts,
        "ar_check_interval_seconds": 0.30,
        "ar_consecutive_hits_required": 2,
        "max_reads": 64,
    }


def _run_single_scenario(spec):
    saved_wake_runtime = wake_word.get_runtime_wake_word_settings()
    saved_phrase_runtime = wake_word.get_runtime_wake_word_phrase_settings()

    wake_word._last_detection_ts = 0.0
    wake_word._ar_last_hit_ts = 0.0
    wake_word._ar_consecutive_hits = 0

    try:
        wake_word.set_runtime_wake_word_settings(
            threshold=float(spec.get("threshold") or 0.35),
            audio_gain=1.0,
            detection_cooldown_seconds=float(spec.get("detection_cooldown_seconds") or 0.2),
        )
        wake_word.set_runtime_wake_word_phrase_settings(
            mode=str(spec.get("mode") or "both"),
            arabic_enabled=True,
            arabic_triggers=list(spec.get("arabic_triggers") or ["يا جارفيس"]),
            ar_stt_model="tiny",
            ar_chunk_seconds=float(spec.get("ar_chunk_seconds") or 1.2),
            ar_check_interval_seconds=float(spec.get("ar_check_interval_seconds") or 0.30),
            ar_consecutive_hits_required=int(spec.get("ar_consecutive_hits_required") or 2),
            ar_confirm_window_seconds=float(spec.get("ar_confirm_window_seconds") or 3.0),
        )

        clock = _FakeClock()
        stream = _FakeInputStream(
            clock=clock,
            max_reads=int(spec.get("max_reads") or 50),
            chunk_size=int(spec.get("chunk_size") or WAKE_WORD_CHUNK_SIZE),
        )
        fake_sd = _FakeSoundDevice(stream)
        fake_model = _FakeEnglishModel(spec.get("english_scores") or [0.0])
        fake_transcriber = _FakeArabicTranscriber(spec.get("arabic_transcripts") or [""])

        started = clock.now()
        detected = False
        detected_source = ""
        error = ""

        with patch.object(wake_word, "sd", fake_sd), patch.object(
            wake_word,
            "_get_model",
            return_value=fake_model,
        ), patch.object(
            wake_word,
            "_resolve_input_device",
            return_value=None,
        ), patch.object(
            wake_word,
            "_get_ar_stt_model",
            return_value=object(),
        ), patch.object(
            wake_word,
            "_transcribe_arabic_window",
            side_effect=fake_transcriber.next_text,
        ), patch.object(
            wake_word.time,
            "perf_counter",
            side_effect=clock.now,
        ):
            try:
                detected_source = str(wake_word.listen_for_wake_word() or "")
                detected = True
            except RuntimeError as exc:
                if str(exc) != _STREAM_EXHAUSTED:
                    error = str(exc)
                    detected = False
            except Exception as exc:
                error = str(exc)
                detected = False

        latency_ms = None
        if detected:
            latency_ms = (clock.now() - started) * 1000.0

        expected_detection = bool(spec.get("expected_detection"))
        scenario_ok = bool(detected == expected_detection and not error)

        return {
            "name": str(spec.get("name") or "wake_scenario"),
            "mode": str(spec.get("mode") or "both"),
            "expected_detection": expected_detection,
            "detected": bool(detected),
            "detected_source": detected_source,
            "latency_ms": float(latency_ms) if latency_ms is not None else None,
            "ok": scenario_ok,
            "error": error,
        }
    finally:
        wake_word.set_runtime_wake_word_settings(**saved_wake_runtime)
        wake_word.set_runtime_wake_word_phrase_settings(
            mode=saved_phrase_runtime.get("mode"),
            arabic_enabled=saved_phrase_runtime.get("arabic_enabled"),
            arabic_triggers=saved_phrase_runtime.get("arabic_triggers"),
            ar_stt_model=saved_phrase_runtime.get("ar_stt_model"),
            ar_chunk_seconds=saved_phrase_runtime.get("ar_chunk_seconds"),
            ar_check_interval_seconds=saved_phrase_runtime.get("ar_check_interval_seconds"),
            ar_consecutive_hits_required=saved_phrase_runtime.get("ar_consecutive_hits_required"),
            ar_confirm_window_seconds=saved_phrase_runtime.get("ar_confirm_window_seconds"),
        )
        wake_word._last_detection_ts = 0.0
        wake_word._ar_last_hit_ts = 0.0
        wake_word._ar_consecutive_hits = 0


def run_wake_reliability_scenarios(scenarios=None, *, target_per_language=None):
    scenario_pack_name = "wake_reliability_pack_v2"
    scenario_pack_version = "2026-04-05-r2"
    pack_target_per_language = None

    if scenarios is None:
        pack_target_per_language = max(
            10,
            min(40, int(target_per_language or WAKE_BENCHMARK_SCENARIOS_PER_LANGUAGE)),
        )
        effective_scenarios = _build_default_scenario_pack(pack_target_per_language)
    else:
        scenario_pack_name = "wake_reliability_custom"
        scenario_pack_version = "custom"
        effective_scenarios = list(scenarios)

    rows = []
    for spec in list(effective_scenarios):
        rows.append(_run_single_scenario(spec))

    positives = [row for row in rows if bool(row.get("expected_detection"))]
    negatives = [row for row in rows if not bool(row.get("expected_detection"))]

    detected_positive = sum(1 for row in positives if bool(row.get("detected")))
    false_positive = sum(1 for row in negatives if bool(row.get("detected")))
    scenario_success = sum(1 for row in rows if bool(row.get("ok")))

    latencies = [float(row.get("latency_ms")) for row in positives if row.get("latency_ms") is not None]

    english_count = sum(1 for row in rows if str(row.get("mode") or "").lower() == "english")
    arabic_count = sum(1 for row in rows if str(row.get("mode") or "").lower() == "arabic")

    return {
        "timestamp": time.time(),
        "scenario_pack": {
            "name": scenario_pack_name,
            "version": scenario_pack_version,
            "target_per_language": pack_target_per_language,
        },
        "scenario_count": len(rows),
        "english_scenario_count": english_count,
        "arabic_scenario_count": arabic_count,
        "success_count": scenario_success,
        "success_rate": (float(scenario_success) / float(len(rows))) if rows else 0.0,
        "positive_scenario_count": len(positives),
        "negative_scenario_count": len(negatives),
        "detected_positive_count": detected_positive,
        "false_positive_count": false_positive,
        "detection_rate": (float(detected_positive) / float(len(positives))) if positives else 0.0,
        "false_positive_rate": (float(false_positive) / float(len(negatives))) if negatives else 0.0,
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "results": rows,
    }
