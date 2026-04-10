import asyncio
import io
import json
import math
import time
from pathlib import Path

from core.config import TTS_BENCHMARK_CORPUS_FILE


def _default_tts_corpus_pack():
    return {
        "name": "tts_quality_pack_v1",
        "version": "2026-04-06",
        "scenarios": [
            {
                "name": "en_status_overview",
                "language": "en",
                "text": "System status is healthy and all benchmark checks passed.",
                "mock_latency_ms": 520,
                "mock_duration_s": 2.2,
                "mock_rms_db": -18.4,
                "mock_clipping_ratio": 0.001,
            },
            {
                "name": "en_calendar_reminder",
                "language": "en",
                "text": "Reminder set for your calendar meeting at ten thirty AM.",
                "mock_latency_ms": 610,
                "mock_duration_s": 2.6,
                "mock_rms_db": -19.0,
                "mock_clipping_ratio": 0.002,
            },
            {
                "name": "en_navigation_help",
                "language": "en",
                "text": "I found three documents in your desktop reports folder.",
                "mock_latency_ms": 680,
                "mock_duration_s": 2.9,
                "mock_rms_db": -18.7,
                "mock_clipping_ratio": 0.001,
            },
            {
                "name": "en_action_confirmation",
                "language": "en",
                "text": "Action completed successfully and the file was moved.",
                "mock_latency_ms": 590,
                "mock_duration_s": 2.4,
                "mock_rms_db": -18.1,
                "mock_clipping_ratio": 0.001,
            },
            {
                "name": "en_short_ack",
                "language": "en",
                "text": "Done. Anything else?",
                "mock_latency_ms": 440,
                "mock_duration_s": 1.1,
                "mock_rms_db": -17.9,
                "mock_clipping_ratio": 0.001,
            },
            {
                "name": "en_long_reply",
                "language": "en",
                "text": "I can also run a full diagnostics report and share any missing dependencies before you continue.",
                "mock_latency_ms": 760,
                "mock_duration_s": 3.5,
                "mock_rms_db": -19.3,
                "mock_clipping_ratio": 0.002,
            },
            {
                "name": "ar_status_overview",
                "language": "ar",
                "text": "حالة النظام ممتازة وجميع اختبارات الاعتمادية نجحت.",
                "mock_latency_ms": 700,
                "mock_duration_s": 2.8,
                "mock_rms_db": -20.1,
                "mock_clipping_ratio": 0.003,
            },
            {
                "name": "ar_calendar_reminder",
                "language": "ar",
                "text": "تم ضبط تذكير لاجتماع التقويم الساعة العاشرة والنصف.",
                "mock_latency_ms": 820,
                "mock_duration_s": 3.2,
                "mock_rms_db": -20.4,
                "mock_clipping_ratio": 0.003,
            },
            {
                "name": "ar_navigation_help",
                "language": "ar",
                "text": "وجدت ثلاثة ملفات في مجلد التقارير على سطح المكتب.",
                "mock_latency_ms": 790,
                "mock_duration_s": 3.1,
                "mock_rms_db": -19.8,
                "mock_clipping_ratio": 0.003,
            },
            {
                "name": "ar_action_confirmation",
                "language": "ar",
                "text": "تم تنفيذ الأمر بنجاح ونقل الملف إلى المكان المطلوب.",
                "mock_latency_ms": 760,
                "mock_duration_s": 3.0,
                "mock_rms_db": -19.7,
                "mock_clipping_ratio": 0.002,
            },
            {
                "name": "ar_short_ack",
                "language": "ar",
                "text": "تم. هل تحتاج شيئاً آخر؟",
                "mock_latency_ms": 540,
                "mock_duration_s": 1.4,
                "mock_rms_db": -19.2,
                "mock_clipping_ratio": 0.002,
            },
            {
                "name": "ar_long_reply",
                "language": "ar",
                "text": "يمكنني أيضاً تشغيل فحص شامل وإظهار أي تبعيات ناقصة قبل المتابعة.",
                "mock_latency_ms": 880,
                "mock_duration_s": 3.6,
                "mock_rms_db": -20.6,
                "mock_clipping_ratio": 0.003,
            },
        ],
    }


def _normalize_mode(mode):
    raw = str(mode or "auto").strip().lower()
    aliases = {
        "deterministic": "mock",
        "synthetic": "mock",
        "ci": "mock",
        "live": "real",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"auto", "mock", "real"}:
        return "auto"
    return normalized


def _normalize_language(value):
    raw = str(value or "").strip().lower()
    if raw in {"ar", "arabic"}:
        return "ar"
    if raw in {"en", "english"}:
        return "en"
    return "auto"


def _normalize_backend(value):
    raw = str(value or "auto").strip().lower()
    aliases = {
        "edge": "edge_tts",
        "edgetts": "edge_tts",
        "kokoro_tts": "kokoro",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"auto", "edge_tts", "kokoro"}:
        return "auto"
    return normalized


def _coerce_float(value, default, minimum=None, maximum=None):
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _read_json_file(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_corpus(corpus_path=None):
    raw_path = str(corpus_path or TTS_BENCHMARK_CORPUS_FILE).strip()
    candidate = Path(raw_path)
    source_path = ""

    if candidate.exists() and candidate.is_file():
        source_path = str(candidate.resolve())
        payload = _read_json_file(candidate)
        if isinstance(payload, dict):
            scenarios = list(payload.get("scenarios") or [])
            if scenarios:
                return {
                    "name": str(payload.get("name") or "tts_quality_pack").strip(),
                    "version": str(payload.get("version") or "unknown").strip(),
                    "scenarios": scenarios,
                }, source_path

    default_pack = _default_tts_corpus_pack()
    return default_pack, source_path or "embedded"


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round((float(p) / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _objective_quality_score(*, rtf, clipping_ratio, rms_db):
    # Score in [0, 1] from latency (RTF), clipping risk, and loudness stability.
    rtf_penalty = min(0.35, max(0.0, float(rtf) - 0.65) * 0.45)
    clipping_penalty = min(0.40, max(0.0, float(clipping_ratio)) * 2.5)
    loudness_penalty = min(0.25, abs(float(rms_db) + 20.0) / 40.0)
    return max(0.0, min(1.0, 1.0 - (rtf_penalty + clipping_penalty + loudness_penalty)))


def _normalize_samples(waveform):
    import numpy as np  # type: ignore

    samples = np.asarray(waveform)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    if samples.size == 0:
        return None

    if samples.dtype.kind in {"i", "u"}:
        info = np.iinfo(samples.dtype)
        peak_limit = float(max(abs(info.min), info.max)) or 1.0
        normalized = samples.astype(np.float32) / peak_limit
    else:
        normalized = samples.astype(np.float32, copy=False)
        peak = float(np.max(np.abs(normalized)))
        if peak > 1.0:
            normalized = normalized / peak

    return np.clip(normalized, -1.0, 1.0)


def _extract_waveform_stats(waveform, sample_rate):
    import numpy as np  # type: ignore

    samples = _normalize_samples(waveform)
    if samples is None:
        raise ValueError("empty_waveform")

    rate = max(8000, int(sample_rate or 0))
    duration_s = float(samples.shape[0]) / float(rate)
    rms = float(np.sqrt(np.mean(np.square(samples)))) if samples.size else 0.0
    rms_db = 20.0 * math.log10(max(rms, 1e-7))
    peak_abs = float(np.max(np.abs(samples))) if samples.size else 0.0
    clipping_ratio = float(np.mean(np.abs(samples) >= 0.99)) if samples.size else 1.0

    return {
        "duration_s": duration_s,
        "rms_db": rms_db,
        "peak_abs": peak_abs,
        "clipping_ratio": clipping_ratio,
    }


def _run_async(coroutine):
    try:
        return asyncio.run(coroutine)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coroutine)
        finally:
            loop.close()


def _backend_available(backend):
    normalized = _normalize_backend(backend)
    try:
        if normalized == "edge_tts":
            import edge_tts  # type: ignore
            from scipy.io import wavfile  # type: ignore

            _ = edge_tts, wavfile
            return True
        if normalized == "kokoro":
            from kokoro import KPipeline  # type: ignore

            _ = KPipeline
            return True
    except Exception:
        return False
    return False


def _choose_auto_backend(preferred=None):
    preferred_backend = _normalize_backend(preferred)
    candidates = []
    if preferred_backend != "auto":
        candidates.append(preferred_backend)
    candidates.extend(["edge_tts", "kokoro"])

    seen = set()
    for backend in candidates:
        if backend in seen:
            continue
        seen.add(backend)
        if _backend_available(backend):
            return backend
    return None


def _synthesize_edge_tts(text, language):
    import edge_tts  # type: ignore
    from scipy.io import wavfile  # type: ignore

    voice_candidates = ["en-US-AriaNeural"]
    edge_rate = "+0%"
    if language == "ar":
        voice_candidates = ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural", "ar-SA-HamedNeural"]
        edge_rate = "-4%"

    async def _collect_audio_bytes(voice_name):
        try:
            speaker = edge_tts.Communicate(
                text,
                voice=voice_name,
                rate=edge_rate,
                output_format="riff-24khz-16bit-mono-pcm",
            )
        except TypeError:
            speaker = edge_tts.Communicate(text, voice=voice_name, rate=edge_rate)

        chunks = []
        async for event in speaker.stream():
            if str(event.get("type") or "").lower() == "audio":
                data = event.get("data")
                if data:
                    chunks.append(bytes(data))
        return b"".join(chunks)

    started = time.perf_counter()
    last_error = "edge_tts_failed"
    for voice_name in voice_candidates:
        try:
            audio_bytes = _run_async(_collect_audio_bytes(voice_name))
            if not audio_bytes:
                last_error = f"edge_tts_empty_audio:{voice_name}"
                continue

            sample_rate, waveform = wavfile.read(io.BytesIO(audio_bytes))
            latency_ms = (time.perf_counter() - started) * 1000.0
            return waveform, int(sample_rate), latency_ms
        except Exception as exc:
            last_error = f"edge_tts_error:{voice_name}:{exc}"

    raise RuntimeError(last_error)


def _synthesize_kokoro(text, language):
    import numpy as np  # type: ignore
    from kokoro import KPipeline  # type: ignore

    if language == "ar":
        candidates = ["z", "a"]
    else:
        candidates = ["a"]

    pipeline = None
    for lang_code in candidates:
        try:
            pipeline = KPipeline(lang_code=lang_code)
            break
        except Exception:
            pipeline = None
    if pipeline is None:
        raise RuntimeError("kokoro_pipeline_unavailable")

    started = time.perf_counter()
    chunks = []
    generator = pipeline(text, voice="af_heart", speed=1.0)
    for row in generator:
        audio = row[2] if isinstance(row, tuple) and len(row) >= 3 else row
        if audio is None:
            continue
        chunk = np.asarray(audio, dtype=np.float32).reshape(-1)
        if chunk.size:
            chunks.append(chunk)

    latency_ms = (time.perf_counter() - started) * 1000.0
    if not chunks:
        raise RuntimeError("kokoro_empty_audio")

    waveform = np.concatenate(chunks)
    return waveform, 24000, latency_ms


def _synthesize_case(text, language, backend):
    if backend == "edge_tts":
        return _synthesize_edge_tts(text, language)
    if backend == "kokoro":
        return _synthesize_kokoro(text, language)
    raise RuntimeError("unsupported_backend")


def _run_case(case, *, mode, requested_backend):
    name = str(case.get("name") or "tts_case").strip()
    language = _normalize_language(case.get("language"))
    text = " ".join(str(case.get("text") or "").split()).strip()

    quality_min = _coerce_float(case.get("quality_score_min"), default=0.70, minimum=0.0, maximum=1.0)
    rtf_max = _coerce_float(case.get("rtf_max"), default=1.40, minimum=0.10, maximum=4.0)

    backend_from_case = _normalize_backend(case.get("backend"))
    backend_requested = _normalize_backend(requested_backend or "auto")

    if backend_requested != "auto":
        backend_hint = backend_requested
    elif backend_from_case != "auto":
        backend_hint = backend_from_case
    else:
        backend_hint = "auto"

    if backend_hint == "auto":
        backend_initial = _choose_auto_backend(preferred=backend_from_case)
        if backend_initial is None:
            backend_initial = "edge_tts"
    else:
        backend_initial = backend_hint

    if backend_initial not in {"edge_tts", "kokoro"}:
        backend_initial = "edge_tts"

    fallback_attempted = False
    fallback_used = False
    fallback_success = False
    fallback_reason = ""

    candidate_backends = [backend_initial]
    candidate_backends.extend(
        backend_name
        for backend_name in ["edge_tts", "kokoro"]
        if backend_name != backend_initial
    )

    if mode == "real":
        candidate_backends = candidate_backends[:1]

    error = ""
    source = "mock"
    backend_used = backend_initial
    latency_ms = None
    duration_s = None
    rms_db = None
    peak_abs = None
    clipping_ratio = None
    quality_score = None

    should_try_real = mode in {"auto", "real"}
    first_failure_reason = ""
    if should_try_real:
        for index, backend_candidate in enumerate(candidate_backends):
            if not _backend_available(backend_candidate):
                if index == 0:
                    fallback_attempted = True
                    first_failure_reason = "initial_backend_unavailable"
                    fallback_reason = first_failure_reason
                continue

            try:
                waveform, sample_rate, synth_latency_ms = _synthesize_case(text, language, backend_candidate)
                waveform_stats = _extract_waveform_stats(waveform, sample_rate)
                latency_ms = float(synth_latency_ms)
                duration_s = float(waveform_stats["duration_s"])
                rms_db = float(waveform_stats["rms_db"])
                peak_abs = float(waveform_stats["peak_abs"])
                clipping_ratio = float(waveform_stats["clipping_ratio"])
                source = "real"
                backend_used = backend_candidate

                if index > 0:
                    fallback_attempted = True
                    fallback_used = True
                    fallback_reason = first_failure_reason or "initial_backend_failed"
                break
            except Exception as exc:
                if index == 0:
                    fallback_attempted = True
                    first_failure_reason = f"initial_backend_failed:{exc}"
                    fallback_reason = first_failure_reason
                    if mode == "real":
                        error = first_failure_reason
                        break
                elif mode == "real":
                    error = f"synthesis_error:{exc}"
                    break

    if source != "real":
        if mode == "real" and error:
            return {
                "name": name,
                "language": language,
                "backend_requested": backend_requested or "auto",
                "backend_initial": backend_initial,
                "backend_used": backend_used,
                "source": "real",
                "text_char_count": len(text),
                "text_word_count": len(text.split()),
                "ok": False,
                "latency_ms": None,
                "duration_s": None,
                "rtf": None,
                "rms_db": None,
                "peak_abs": None,
                "clipping_ratio": None,
                "quality_score": None,
                "quality_score_min": float(quality_min),
                "rtf_max": float(rtf_max),
                "fallback_attempted": fallback_attempted,
                "fallback_used": fallback_used,
                "fallback_success": False,
                "fallback_reason": fallback_reason,
                "error": error or "real_synthesis_failed",
            }

        latency_ms = _coerce_float(case.get("mock_latency_ms"), default=700.0, minimum=1.0)
        duration_s = _coerce_float(case.get("mock_duration_s"), default=2.5, minimum=0.10)
        rms_db = _coerce_float(case.get("mock_rms_db"), default=-19.0, minimum=-90.0, maximum=0.0)
        peak_abs = _coerce_float(case.get("mock_peak_abs"), default=0.86, minimum=0.0, maximum=1.0)
        clipping_ratio = _coerce_float(case.get("mock_clipping_ratio"), default=0.003, minimum=0.0, maximum=1.0)

    rtf = float(latency_ms) / float(max(0.001, float(duration_s) * 1000.0))

    provided_quality = case.get("mock_quality_score") if source == "mock" else None
    if provided_quality is not None:
        quality_score = _coerce_float(provided_quality, default=0.75, minimum=0.0, maximum=1.0)
    else:
        quality_score = _objective_quality_score(
            rtf=rtf,
            clipping_ratio=clipping_ratio,
            rms_db=rms_db,
        )

    ok = bool(
        text
        and float(quality_score) >= float(quality_min)
        and float(rtf) <= float(rtf_max)
    )

    if fallback_attempted:
        if fallback_used:
            fallback_success = bool(ok and source == "real")
        else:
            fallback_success = bool(ok and source == "mock")

    return {
        "name": name,
        "language": language,
        "backend_requested": backend_requested or "auto",
        "backend_initial": backend_initial,
        "backend_used": backend_used,
        "source": source,
        "text_char_count": len(text),
        "text_word_count": len(text.split()),
        "ok": ok,
        "latency_ms": float(latency_ms),
        "duration_s": float(duration_s),
        "rtf": float(rtf),
        "rms_db": float(rms_db),
        "peak_abs": float(peak_abs),
        "clipping_ratio": float(clipping_ratio),
        "quality_score": float(quality_score),
        "quality_score_min": float(quality_min),
        "rtf_max": float(rtf_max),
        "fallback_attempted": fallback_attempted,
        "fallback_used": fallback_used,
        "fallback_success": fallback_success,
        "fallback_reason": fallback_reason,
        "error": error,
    }


def _summarize_languages(rows):
    grouped = {}
    for row in rows:
        language = str(row.get("language") or "auto").strip().lower() or "auto"
        grouped.setdefault(language, []).append(row)

    summary = {}
    for language, language_rows in sorted(grouped.items()):
        quality_scores = [float(row.get("quality_score")) for row in language_rows if row.get("quality_score") is not None]
        latencies = [float(row.get("latency_ms")) for row in language_rows if row.get("latency_ms") is not None]
        rtfs = [float(row.get("rtf")) for row in language_rows if row.get("rtf") is not None]
        success_count = sum(1 for row in language_rows if bool(row.get("ok")))

        summary[language] = {
            "count": len(language_rows),
            "success_count": success_count,
            "success_rate": (float(success_count) / float(len(language_rows))) if language_rows else 0.0,
            "avg_quality_score": (sum(quality_scores) / float(len(quality_scores))) if quality_scores else 0.0,
            "avg_rtf": (sum(rtfs) / float(len(rtfs))) if rtfs else 0.0,
            "p95_latency_ms": _percentile(latencies, 95) if latencies else 0.0,
        }
    return summary


def _summarize_backends(rows):
    grouped = {}
    for row in rows:
        backend = str(row.get("backend_used") or "unknown").strip().lower() or "unknown"
        grouped.setdefault(backend, []).append(row)

    summary = {}
    for backend, backend_rows in sorted(grouped.items()):
        quality_scores = [float(row.get("quality_score")) for row in backend_rows if row.get("quality_score") is not None]
        latencies = [float(row.get("latency_ms")) for row in backend_rows if row.get("latency_ms") is not None]
        rtfs = [float(row.get("rtf")) for row in backend_rows if row.get("rtf") is not None]
        real_count = sum(1 for row in backend_rows if str(row.get("source") or "") == "real")

        summary[backend] = {
            "count": len(backend_rows),
            "real_count": real_count,
            "mock_count": len(backend_rows) - real_count,
            "avg_quality_score": (sum(quality_scores) / float(len(quality_scores))) if quality_scores else 0.0,
            "avg_rtf": (sum(rtfs) / float(len(rtfs))) if rtfs else 0.0,
            "p95_latency_ms": _percentile(latencies, 95) if latencies else 0.0,
        }
    return summary


def run_tts_quality_scenarios(*, corpus_path=None, mode="auto", backend="auto"):
    normalized_mode = _normalize_mode(mode)
    normalized_backend = _normalize_backend(backend)

    pack, source_path = _load_corpus(corpus_path=corpus_path)
    cases = list(pack.get("scenarios") or [])

    rows = [
        _run_case(case, mode=normalized_mode, requested_backend=normalized_backend)
        for case in cases
    ]

    scenario_count = len(rows)
    success_count = sum(1 for row in rows if bool(row.get("ok")))
    quality_scores = [float(row.get("quality_score")) for row in rows if row.get("quality_score") is not None]
    latency_values = [float(row.get("latency_ms")) for row in rows if row.get("latency_ms") is not None]
    rtf_values = [float(row.get("rtf")) for row in rows if row.get("rtf") is not None]
    clipping_values = [float(row.get("clipping_ratio")) for row in rows if row.get("clipping_ratio") is not None]

    real_scenario_count = sum(1 for row in rows if str(row.get("source") or "") == "real")
    mock_scenario_count = sum(1 for row in rows if str(row.get("source") or "") == "mock")
    fallback_attempted_count = sum(1 for row in rows if bool(row.get("fallback_attempted")))
    fallback_used_count = sum(1 for row in rows if bool(row.get("fallback_used")))
    fallback_success_count = sum(1 for row in rows if bool(row.get("fallback_success")))
    fallback_reliability = (
        float(fallback_success_count) / float(fallback_attempted_count)
        if fallback_attempted_count
        else 1.0
    )

    return {
        "timestamp": time.time(),
        "scenario_count": scenario_count,
        "evaluated_count": scenario_count,
        "success_count": success_count,
        "success_rate": (float(success_count) / float(scenario_count)) if scenario_count else 0.0,
        "avg_quality_score": (sum(quality_scores) / float(len(quality_scores))) if quality_scores else 0.0,
        "p50_quality_score": _percentile(quality_scores, 50) if quality_scores else 0.0,
        "p95_quality_score": _percentile(quality_scores, 95) if quality_scores else 0.0,
        "avg_rtf": (sum(rtf_values) / float(len(rtf_values))) if rtf_values else 0.0,
        "p95_rtf": _percentile(rtf_values, 95) if rtf_values else 0.0,
        "avg_clipping_ratio": (sum(clipping_values) / float(len(clipping_values))) if clipping_values else 0.0,
        "p50_latency_ms": _percentile(latency_values, 50) if latency_values else 0.0,
        "p95_latency_ms": _percentile(latency_values, 95) if latency_values else 0.0,
        "real_scenario_count": real_scenario_count,
        "mock_scenario_count": mock_scenario_count,
        "fallback_attempted_count": fallback_attempted_count,
        "fallback_used_count": fallback_used_count,
        "fallback_success_count": fallback_success_count,
        "fallback_reliability": float(fallback_reliability),
        "corpus": {
            "name": str(pack.get("name") or "tts_quality_pack").strip() or "tts_quality_pack",
            "version": str(pack.get("version") or "unknown").strip() or "unknown",
            "mode": normalized_mode,
            "backend": normalized_backend,
            "source": source_path,
        },
        "languages": _summarize_languages(rows),
        "backends": _summarize_backends(rows),
        "results": rows,
    }
