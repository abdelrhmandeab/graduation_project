import json
import os
import re
import time
from pathlib import Path

from audio.stt import transcribe_streaming
from core.config import STT_BENCHMARK_CORPUS_FILE


_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DEFAULT_CASE_WER_MAX = 0.35
_DEFAULT_CASE_CER_MAX = 0.40
_DEFAULT_MOCK_LATENCY_MS = 650.0


def _default_stt_corpus_pack():
    return {
        "name": "stt_reliability_pack_v1",
        "version": "2026-04-05",
        "scenarios": [
            {
                "name": "en_open_calendar",
                "language": "en",
                "expected_text": "open calendar",
                "mock_transcript": "open calendar",
                "mock_latency_ms": 520,
            },
            {
                "name": "en_play_spotify",
                "language": "en",
                "expected_text": "play music on spotify",
                "mock_transcript": "play music on spotify",
                "mock_latency_ms": 590,
            },
            {
                "name": "en_volume_set",
                "language": "en",
                "expected_text": "set volume to fifty percent",
                "mock_transcript": "set volume to fifty",
                "mock_latency_ms": 650,
            },
            {
                "name": "en_open_chrome",
                "language": "en",
                "expected_text": "open chrome",
                "mock_transcript": "open chrome",
                "mock_latency_ms": 610,
            },
            {
                "name": "en_find_report",
                "language": "en",
                "expected_text": "find quarterly report in documents",
                "mock_transcript": "find quarterly report in documents",
                "mock_latency_ms": 700,
            },
            {
                "name": "en_wifi_off",
                "language": "en",
                "expected_text": "turn off wifi",
                "mock_transcript": "turn off wifi",
                "mock_latency_ms": 640,
            },
            {
                "name": "ar_open_notepad",
                "language": "ar",
                "expected_text": "افتح المفكرة",
                "mock_transcript": "افتح المفكرة",
                "mock_latency_ms": 680,
            },
            {
                "name": "ar_play_spotify",
                "language": "ar",
                "expected_text": "شغل سبوتيفاي",
                "mock_transcript": "شغل سبوتيفاي",
                "mock_latency_ms": 760,
            },
            {
                "name": "ar_volume_set",
                "language": "ar",
                "expected_text": "اضبط الصوت على خمسين بالمئة",
                "mock_transcript": "اضبط الصوت على خمسين بالمية",
                "mock_latency_ms": 740,
            },
            {
                "name": "ar_wifi_off",
                "language": "ar",
                "expected_text": "اطفئ الواي فاي",
                "mock_transcript": "اطفئ الواي فاي",
                "mock_latency_ms": 720,
            },
            {
                "name": "ar_open_settings",
                "language": "ar",
                "expected_text": "افتح الاعدادات",
                "mock_transcript": "افتح الاعدادات",
                "mock_latency_ms": 670,
            },
            {
                "name": "ar_take_screenshot",
                "language": "ar",
                "expected_text": "خذ لقطة شاشة",
                "mock_transcript": "خذ لقطة للشاشة",
                "mock_latency_ms": 690,
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
    if raw in {"arabic", "ar"}:
        return "ar"
    if raw in {"english", "en"}:
        return "en"
    return "auto"


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


def _load_corpus(corpus_path=None):
    raw_path = str(corpus_path or STT_BENCHMARK_CORPUS_FILE).strip()
    candidate = Path(raw_path)
    source_path = ""
    if candidate.exists() and candidate.is_file():
        source_path = str(candidate.resolve())
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                scenarios = list(payload.get("scenarios") or [])
                if scenarios:
                    return {
                        "name": str(payload.get("name") or "stt_reliability_pack").strip(),
                        "version": str(payload.get("version") or "unknown").strip(),
                        "scenarios": scenarios,
                    }, candidate.parent, source_path
        except Exception:
            pass

    default_pack = _default_stt_corpus_pack()
    return default_pack, Path("."), source_path or "embedded"


def _tokenize(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    return _WORD_PATTERN.findall(raw)


def _levenshtein_distance(reference_tokens, hypothesis_tokens):
    ref_len = len(reference_tokens)
    hyp_len = len(hypothesis_tokens)
    if ref_len == 0:
        return hyp_len
    if hyp_len == 0:
        return ref_len

    previous = list(range(hyp_len + 1))
    for ref_index in range(1, ref_len + 1):
        current = [ref_index] + [0] * hyp_len
        ref_token = reference_tokens[ref_index - 1]
        for hyp_index in range(1, hyp_len + 1):
            hyp_token = hypothesis_tokens[hyp_index - 1]
            substitution_cost = 0 if ref_token == hyp_token else 1
            current[hyp_index] = min(
                previous[hyp_index] + 1,
                current[hyp_index - 1] + 1,
                previous[hyp_index - 1] + substitution_cost,
            )
        previous = current
    return previous[hyp_len]


def _wer_details(reference_text, hypothesis_text):
    reference_tokens = _tokenize(reference_text)
    hypothesis_tokens = _tokenize(hypothesis_text)

    reference_len = len(reference_tokens)
    hypothesis_len = len(hypothesis_tokens)
    if reference_len == 0:
        if hypothesis_len == 0:
            return {
                "wer": 0.0,
                "edit_distance": 0,
                "reference_words": 0,
                "hypothesis_words": 0,
            }
        return {
            "wer": 1.0,
            "edit_distance": hypothesis_len,
            "reference_words": 0,
            "hypothesis_words": hypothesis_len,
        }

    edit_distance = _levenshtein_distance(reference_tokens, hypothesis_tokens)
    return {
        "wer": float(edit_distance) / float(reference_len),
        "edit_distance": int(edit_distance),
        "reference_words": int(reference_len),
        "hypothesis_words": int(hypothesis_len),
    }


def _char_sequence(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    normalized = "".join(_WORD_PATTERN.findall(raw))
    return list(normalized)


def _cer_details(reference_text, hypothesis_text):
    reference_chars = _char_sequence(reference_text)
    hypothesis_chars = _char_sequence(hypothesis_text)

    reference_len = len(reference_chars)
    hypothesis_len = len(hypothesis_chars)
    if reference_len == 0:
        if hypothesis_len == 0:
            return {
                "cer": 0.0,
                "char_edit_distance": 0,
                "reference_chars": 0,
                "hypothesis_chars": 0,
            }
        return {
            "cer": 1.0,
            "char_edit_distance": hypothesis_len,
            "reference_chars": 0,
            "hypothesis_chars": hypothesis_len,
        }

    edit_distance = _levenshtein_distance(reference_chars, hypothesis_chars)
    return {
        "cer": float(edit_distance) / float(reference_len),
        "char_edit_distance": int(edit_distance),
        "reference_chars": int(reference_len),
        "hypothesis_chars": int(hypothesis_len),
    }


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = int(round((float(p) / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _resolve_audio_path(base_dir, audio_file):
    raw_path = str(audio_file or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _transcribe_audio_case(transcriber, audio_path, language_hint):
    started = time.perf_counter()
    transcript = transcriber(str(audio_path), language_hint=language_hint)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return str(transcript or "").strip(), float(latency_ms)


def _run_case(case, *, base_dir, mode, transcriber):
    name = str(case.get("name") or "stt_case").strip()
    language = _normalize_language(case.get("language"))
    expected_text = str(case.get("expected_text") or "").strip()
    wer_max = _coerce_float(case.get("wer_max"), default=_DEFAULT_CASE_WER_MAX, minimum=0.0, maximum=1.0)
    cer_max = _coerce_float(case.get("cer_max"), default=_DEFAULT_CASE_CER_MAX, minimum=0.0, maximum=1.0)

    audio_file = str(case.get("audio_file") or "").strip()
    audio_path = _resolve_audio_path(base_dir, audio_file)
    audio_available = bool(audio_path is not None and audio_path.exists() and audio_path.is_file())

    mock_transcript = str(case.get("mock_transcript") or "").strip()
    mock_latency_ms = _coerce_float(case.get("mock_latency_ms"), default=_DEFAULT_MOCK_LATENCY_MS, minimum=1.0)

    source = "none"
    transcript = ""
    latency_ms = None
    error = ""

    if mode == "real":
        if audio_available:
            source = "audio"
            try:
                transcript, latency_ms = _transcribe_audio_case(transcriber, audio_path, language_hint=language)
            except Exception as exc:
                error = f"transcription_error:{exc}"
        else:
            error = "audio_file_missing"
    elif mode == "mock":
        if mock_transcript:
            source = "mock"
            transcript = mock_transcript
            latency_ms = float(mock_latency_ms)
        else:
            error = "mock_transcript_missing"
    else:
        if audio_available:
            source = "audio"
            try:
                transcript, latency_ms = _transcribe_audio_case(transcriber, audio_path, language_hint=language)
            except Exception as exc:
                error = f"transcription_error:{exc}"
        elif mock_transcript:
            source = "mock"
            transcript = mock_transcript
            latency_ms = float(mock_latency_ms)
        else:
            error = "no_audio_or_mock_transcript"

    evaluated = bool(expected_text) and not bool(error)
    details = _wer_details(expected_text, transcript) if evaluated else {
        "wer": None,
        "edit_distance": None,
        "reference_words": len(_tokenize(expected_text)),
        "hypothesis_words": len(_tokenize(transcript)),
    }
    cer_details = _cer_details(expected_text, transcript) if evaluated else {
        "cer": None,
        "char_edit_distance": None,
        "reference_chars": len(_char_sequence(expected_text)),
        "hypothesis_chars": len(_char_sequence(transcript)),
    }
    wer = details.get("wer")
    cer = cer_details.get("cer")
    ok = bool(
        evaluated
        and wer is not None
        and cer is not None
        and float(wer) <= wer_max
        and float(cer) <= cer_max
    )

    return {
        "name": name,
        "language": language,
        "source": source,
        "audio_file": str(audio_path) if audio_path is not None else audio_file,
        "expected_text": expected_text,
        "transcript": transcript,
        "wer": float(wer) if wer is not None else None,
        "wer_max": float(wer_max),
        "cer": float(cer) if cer is not None else None,
        "cer_max": float(cer_max),
        "ok": ok,
        "latency_ms": float(latency_ms) if latency_ms is not None else None,
        "expected_word_count": details.get("reference_words"),
        "transcript_word_count": details.get("hypothesis_words"),
        "edit_distance": details.get("edit_distance"),
        "expected_char_count": cer_details.get("reference_chars"),
        "transcript_char_count": cer_details.get("hypothesis_chars"),
        "char_edit_distance": cer_details.get("char_edit_distance"),
        "error": error,
    }


def _summarize_languages(rows):
    grouped = {}
    for row in rows:
        language = str(row.get("language") or "auto").strip().lower() or "auto"
        grouped.setdefault(language, []).append(row)

    summary = {}
    for language, language_rows in sorted(grouped.items()):
        evaluated_rows = [row for row in language_rows if row.get("wer") is not None]
        wer_values = [float(row.get("wer")) for row in evaluated_rows]
        cer_values = [float(row.get("cer")) for row in evaluated_rows if row.get("cer") is not None]
        latencies = [float(row.get("latency_ms")) for row in evaluated_rows if row.get("latency_ms") is not None]
        success_count = sum(1 for row in evaluated_rows if bool(row.get("ok")))

        summary[language] = {
            "count": len(language_rows),
            "evaluated_count": len(evaluated_rows),
            "success_count": success_count,
            "success_rate": (float(success_count) / float(len(evaluated_rows))) if evaluated_rows else 0.0,
            "avg_wer": (sum(wer_values) / float(len(wer_values))) if wer_values else 1.0,
            "p95_wer": _percentile(wer_values, 95) if wer_values else 1.0,
            "avg_cer": (sum(cer_values) / float(len(cer_values))) if cer_values else 1.0,
            "p95_cer": _percentile(cer_values, 95) if cer_values else 1.0,
            "p95_latency_ms": _percentile(latencies, 95) if latencies else 0.0,
        }
    return summary


def run_stt_reliability_scenarios(*, corpus_path=None, mode="auto", transcriber=None):
    normalized_mode = _normalize_mode(mode)
    pack, base_dir, source_path = _load_corpus(corpus_path=corpus_path)

    run_cases = list(pack.get("scenarios") or [])
    runner = transcriber or transcribe_streaming

    rows = [
        _run_case(case, base_dir=base_dir, mode=normalized_mode, transcriber=runner)
        for case in run_cases
    ]

    scenario_count = len(rows)
    evaluated_rows = [row for row in rows if row.get("wer") is not None]
    wer_values = [float(row.get("wer")) for row in evaluated_rows]
    cer_values = [float(row.get("cer")) for row in evaluated_rows if row.get("cer") is not None]
    latency_values = [float(row.get("latency_ms")) for row in evaluated_rows if row.get("latency_ms") is not None]

    success_count = sum(1 for row in evaluated_rows if bool(row.get("ok")))
    evaluated_count = len(evaluated_rows)

    real_audio_scenario_count = sum(1 for row in rows if str(row.get("source") or "") == "audio")
    mock_scenario_count = sum(1 for row in rows if str(row.get("source") or "") == "mock")

    avg_wer = (sum(wer_values) / float(len(wer_values))) if wer_values else 1.0
    avg_cer = (sum(cer_values) / float(len(cer_values))) if cer_values else 1.0

    return {
        "timestamp": time.time(),
        "scenario_count": scenario_count,
        "evaluated_count": evaluated_count,
        "success_count": success_count,
        "success_rate": (float(success_count) / float(evaluated_count)) if evaluated_count else 0.0,
        "avg_wer": float(avg_wer),
        "avg_cer": float(avg_cer),
        "p50_wer": _percentile(wer_values, 50) if wer_values else 1.0,
        "p95_wer": _percentile(wer_values, 95) if wer_values else 1.0,
        "p50_cer": _percentile(cer_values, 50) if cer_values else 1.0,
        "p95_cer": _percentile(cer_values, 95) if cer_values else 1.0,
        "p50_latency_ms": _percentile(latency_values, 50) if latency_values else 0.0,
        "p95_latency_ms": _percentile(latency_values, 95) if latency_values else 0.0,
        "real_audio_scenario_count": real_audio_scenario_count,
        "mock_scenario_count": mock_scenario_count,
        "corpus": {
            "name": str(pack.get("name") or "stt_reliability_pack").strip() or "stt_reliability_pack",
            "version": str(pack.get("version") or "unknown").strip() or "unknown",
            "mode": normalized_mode,
            "source": source_path,
        },
        "languages": _summarize_languages(rows),
        "results": rows,
    }
