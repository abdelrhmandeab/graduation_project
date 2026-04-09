import json
import re
import time
from pathlib import Path

from audio import stt as stt_runtime
from audio.stt import normalize_arabic_post_transcript
from core.config import (
    STT_EGYPTIAN_BENCHMARK_CORPUS_FILE,
)


_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def _default_pack():
    return {
        "name": "stt_egyptian_dialect_pack_v1",
        "version": "embedded",
        "baseline_setup": "fw_tiny_cpu",
        "latency_budget_ms_low_mid_cpu": 800,
        "setups": [],
        "scenarios": [],
    }


def _coerce_float(value, default=0.0, minimum=None, maximum=None):
    try:
        result = float(value)
    except Exception:
        result = float(default)
    if minimum is not None:
        result = max(float(minimum), result)
    if maximum is not None:
        result = min(float(maximum), result)
    return result


def _load_pack(corpus_path=None):
    raw_path = str(corpus_path or STT_EGYPTIAN_BENCHMARK_CORPUS_FILE).strip()
    path = Path(raw_path)
    if path.exists() and path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload, path.parent, str(path.resolve())
        except Exception:
            pass
    return _default_pack(), Path("."), "embedded"


def _tokenize(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    return _WORD_PATTERN.findall(raw)


def _char_sequence(text):
    raw = str(text or "").strip().lower()
    if not raw:
        return []
    normalized = "".join(_WORD_PATTERN.findall(raw))
    return list(normalized)


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


def _wer(reference_text, hypothesis_text):
    reference_tokens = _tokenize(reference_text)
    hypothesis_tokens = _tokenize(hypothesis_text)
    if not reference_tokens:
        return 0.0 if not hypothesis_tokens else 1.0
    distance = _levenshtein_distance(reference_tokens, hypothesis_tokens)
    return float(distance) / float(len(reference_tokens))


def _cer(reference_text, hypothesis_text):
    reference_chars = _char_sequence(reference_text)
    hypothesis_chars = _char_sequence(hypothesis_text)
    if not reference_chars:
        return 0.0 if not hypothesis_chars else 1.0
    distance = _levenshtein_distance(reference_chars, hypothesis_chars)
    return float(distance) / float(len(reference_chars))


def _percentile(values, p):
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = int(round((float(p) / 100.0) * (len(ordered) - 1)))
    return float(ordered[index])


def _build_setup_summary(setup, rows, latency_budget_ms):
    setup_id = str((setup or {}).get("id") or "").strip()
    wer_raw_values = [float(row["wer_raw"]) for row in rows]
    wer_norm_values = [float(row["wer_normalized"]) for row in rows]
    cer_raw_values = [float(row["cer_raw"]) for row in rows]
    cer_norm_values = [float(row["cer_normalized"]) for row in rows]
    latency_values = [float(row["latency_ms"]) for row in rows]
    error_count = sum(1 for row in rows if str(row.get("error") or "").strip())

    avg_wer_raw = (sum(wer_raw_values) / float(len(wer_raw_values))) if wer_raw_values else 1.0
    avg_wer_normalized = (sum(wer_norm_values) / float(len(wer_norm_values))) if wer_norm_values else 1.0
    avg_cer_raw = (sum(cer_raw_values) / float(len(cer_raw_values))) if cer_raw_values else 1.0
    avg_cer_normalized = (sum(cer_norm_values) / float(len(cer_norm_values))) if cer_norm_values else 1.0
    p95_latency_ms = _percentile(latency_values, 95)

    quality_score = max(0.0, 1.0 - ((0.70 * avg_wer_normalized) + (0.30 * avg_cer_normalized)))
    latency_penalty = max(0.0, (float(p95_latency_ms) - float(latency_budget_ms)) / max(1.0, float(latency_budget_ms)))
    balance_score = quality_score - (0.35 * latency_penalty)

    return {
        "id": setup_id,
        "label": str((setup or {}).get("label") or setup_id),
        "backend": str((setup or {}).get("backend") or ""),
        "model": str((setup or {}).get("model") or ""),
        "mode": str((setup or {}).get("mode") or "").strip(),
        "scenario_count": len(rows),
        "avg_wer_raw": float(avg_wer_raw),
        "avg_wer_normalized": float(avg_wer_normalized),
        "avg_cer_raw": float(avg_cer_raw),
        "avg_cer_normalized": float(avg_cer_normalized),
        "normalization_wer_gain": float(avg_wer_raw - avg_wer_normalized),
        "normalization_cer_gain": float(avg_cer_raw - avg_cer_normalized),
        "p50_latency_ms": _percentile(latency_values, 50),
        "p95_latency_ms": float(p95_latency_ms),
        "acceptable_latency": bool(float(p95_latency_ms) <= float(latency_budget_ms)),
        "error_count": int(error_count),
        "success_rate": ((float(len(rows) - error_count) / float(len(rows))) if rows else 0.0),
        "quality_score": float(quality_score),
        "balance_score": float(balance_score),
        "results": rows,
    }


def _evaluate_setup(pack, setup, latency_budget_ms):
    setup_id = str((setup or {}).get("id") or "").strip()
    rows = []

    for scenario in list(pack.get("scenarios") or []):
        expected_text = str(scenario.get("expected_text") or "").strip()
        if not expected_text:
            continue

        setup_predictions = dict(scenario.get("setup_predictions") or {})
        prediction = dict(setup_predictions.get(setup_id) or {})

        transcript_raw = str(prediction.get("transcript") or "").strip()
        transcript_normalized = normalize_arabic_post_transcript(transcript_raw)
        latency_ms = _coerce_float(prediction.get("latency_ms"), default=0.0, minimum=0.0)

        wer_raw = _wer(expected_text, transcript_raw)
        wer_normalized = _wer(expected_text, transcript_normalized)
        cer_raw = _cer(expected_text, transcript_raw)
        cer_normalized = _cer(expected_text, transcript_normalized)

        rows.append(
            {
                "name": str(scenario.get("name") or "egy_case"),
                "domain": str(scenario.get("domain") or "general"),
                "expected_text": expected_text,
                "transcript_raw": transcript_raw,
                "transcript_normalized": transcript_normalized,
                "wer_raw": float(wer_raw),
                "wer_normalized": float(wer_normalized),
                "cer_raw": float(cer_raw),
                "cer_normalized": float(cer_normalized),
                "latency_ms": float(latency_ms),
                "error": "",
            }
        )

    return _build_setup_summary(setup, rows, latency_budget_ms)


def _recommend_setups(setup_summaries, *, baseline_setup, latency_budget_ms):
    by_id = {str(item.get("id") or ""): item for item in setup_summaries}
    baseline = by_id.get(str(baseline_setup or "").strip()) or (setup_summaries[0] if setup_summaries else {})

    acceptable_setups = [item for item in setup_summaries if bool(item.get("acceptable_latency"))]
    reliable_all = [
        item
        for item in setup_summaries
        if int(item.get("error_count") or 0) == 0 and float(item.get("success_rate") or 0.0) > 0.0
    ]
    reliable_acceptable = [
        item
        for item in acceptable_setups
        if int(item.get("error_count") or 0) == 0 and float(item.get("success_rate") or 0.0) > 0.0
    ]

    # Candidate priority:
    # 1) reliable + latency acceptable
    # 2) reliable (even if latency is above budget)
    # 3) latency acceptable (if no reliable setup exists)
    # 4) all setups (last resort)
    if reliable_acceptable:
        candidate_pool = reliable_acceptable
    elif reliable_all:
        candidate_pool = reliable_all
    elif acceptable_setups:
        candidate_pool = acceptable_setups
    else:
        candidate_pool = setup_summaries

    recommended = {}
    if candidate_pool:
        recommended = max(
            candidate_pool,
            key=lambda item: (
                float(item.get("balance_score") or 0.0),
                float(item.get("quality_score") or 0.0),
                -float(item.get("p95_latency_ms") or 0.0),
            ),
        )

    baseline_wer = _coerce_float((baseline or {}).get("avg_wer_normalized"), default=1.0, minimum=0.0)
    recommended_wer = _coerce_float((recommended or {}).get("avg_wer_normalized"), default=1.0, minimum=0.0)
    wer_gain_abs = max(0.0, baseline_wer - recommended_wer)
    wer_gain_rel = (wer_gain_abs / baseline_wer) if baseline_wer > 0 else 0.0

    latency_acceptable = bool((recommended or {}).get("acceptable_latency"))
    clearly_better_quality = bool((wer_gain_abs >= 0.03) or (wer_gain_rel >= 0.20))
    stable_runtime = bool(float((recommended or {}).get("success_rate") or 0.0) >= 0.95)

    return {
        "baseline": baseline,
        "recommended": recommended,
        "baseline_vs_recommended": {
            "baseline_setup": str((baseline or {}).get("id") or baseline_setup),
            "recommended_setup": str((recommended or {}).get("id") or ""),
            "baseline_avg_wer_normalized": float(baseline_wer),
            "recommended_avg_wer_normalized": float(recommended_wer),
            "wer_gain_abs": float(wer_gain_abs),
            "wer_gain_rel": float(wer_gain_rel),
            "baseline_p95_latency_ms": float((baseline or {}).get("p95_latency_ms") or 0.0),
            "recommended_p95_latency_ms": float((recommended or {}).get("p95_latency_ms") or 0.0),
        },
        "recommendation": {
            "setup_id": str((recommended or {}).get("id") or ""),
            "label": str((recommended or {}).get("label") or ""),
            "quality_score": float((recommended or {}).get("quality_score") or 0.0),
            "balance_score": float((recommended or {}).get("balance_score") or 0.0),
            "avg_wer_normalized": float((recommended or {}).get("avg_wer_normalized") or 1.0),
            "p95_latency_ms": float((recommended or {}).get("p95_latency_ms") or 0.0),
            "acceptable_latency": bool((recommended or {}).get("acceptable_latency")),
            "success_rate": float((recommended or {}).get("success_rate") or 0.0),
        },
        "done_gate": {
            "quality_clearly_better_than_baseline": clearly_better_quality,
            "latency_acceptable_low_mid_cpu": latency_acceptable,
            "runtime_stable": stable_runtime,
            "passed": bool(clearly_better_quality and latency_acceptable and stable_runtime),
        },
    }


def _normalize_language_hint(value):
    raw = str(value or "").strip().lower()
    if raw in {"ar", "arabic"}:
        return "ar"
    if raw in {"en", "english"}:
        return "en"
    return None


def _resolve_audio_path(base_dir, audio_file):
    raw_path = str(audio_file or "").strip()
    if not raw_path:
        return None
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return candidate


def _normalize_runtime_backends(value):
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",") if str(item or "").strip()]
    elif value is None:
        parts = ["faster_whisper", "huggingface"]
    else:
        parts = [str(item or "").strip() for item in list(value) if str(item or "").strip()]

    normalized = []
    seen = set()
    for part in parts:
        backend = stt_runtime._normalize_stt_backend(part)
        if backend in seen:
            continue
        seen.add(backend)
        normalized.append(backend)

    if not normalized:
        normalized = ["faster_whisper"]
    return normalized


def _runtime_setup_descriptor(backend):
    backend = stt_runtime._normalize_stt_backend(backend)
    if backend == "huggingface":
        hf_settings = stt_runtime.get_runtime_hf_settings()
        model = str(hf_settings.get("model") or "")
        mode = str(hf_settings.get("mode") or "")
        return {
            "id": "runtime_huggingface",
            "label": f"runtime huggingface ({mode or 'auto'})",
            "backend": "huggingface",
            "model": model,
            "mode": mode,
        }
    return {
        "id": "runtime_faster_whisper",
        "label": "runtime faster-whisper",
        "backend": "faster_whisper",
        "model": "runtime_configured",
        "mode": "direct",
    }


def _run_runtime_ab(pack, *, base_dir, latency_budget_ms, runtime_backends=None, runtime_max_cases=None):
    scenarios = list(pack.get("scenarios") or [])
    selected_cases = []
    missing_audio = []
    for scenario in scenarios:
        expected_text = str(scenario.get("expected_text") or "").strip()
        if not expected_text:
            continue

        audio_file = str(scenario.get("audio_file") or "").strip()
        if not audio_file:
            continue

        audio_path = _resolve_audio_path(base_dir, audio_file)
        if audio_path is None or not audio_path.exists() or not audio_path.is_file():
            missing_audio.append(audio_file)
            continue

        selected_cases.append(
            {
                "name": str(scenario.get("name") or "egy_case"),
                "domain": str(scenario.get("domain") or "general"),
                "expected_text": expected_text,
                "language_hint": _normalize_language_hint(scenario.get("language")),
                "audio_path": audio_path,
                "audio_file": audio_file,
            }
        )

    runtime_limit = int(runtime_max_cases or 0)
    if runtime_limit > 0:
        selected_cases = selected_cases[:runtime_limit]

    backends = _normalize_runtime_backends(runtime_backends)
    if not selected_cases:
        return {
            "enabled": True,
            "executed": False,
            "reason": "no_audio_scenarios_available",
            "requested_backends": backends,
            "audio_scenario_count": 0,
            "missing_audio_file_count": len(missing_audio),
            "missing_audio_files": missing_audio[:10],
        }

    setup_summaries = []
    for backend in backends:
        setup = _runtime_setup_descriptor(backend)
        rows = []
        for case in selected_cases:
            started = time.perf_counter()
            transcript_raw = ""
            detected_language = ""
            error = ""
            try:
                result = stt_runtime.transcribe_backend_direct_with_meta(
                    str(case["audio_path"]),
                    backend=backend,
                    language_hint=case.get("language_hint"),
                )
                transcript_raw = str((result or {}).get("text") or "").strip()
                detected_language = str((result or {}).get("language") or "").strip()
            except Exception as exc:
                error = f"transcription_error:{exc}"
            latency_ms = max(0.0, (time.perf_counter() - started) * 1000.0)

            transcript_normalized = normalize_arabic_post_transcript(transcript_raw)
            expected_text = str(case.get("expected_text") or "")

            rows.append(
                {
                    "name": str(case.get("name") or "egy_case"),
                    "domain": str(case.get("domain") or "general"),
                    "expected_text": expected_text,
                    "audio_file": str(case.get("audio_file") or ""),
                    "language": detected_language,
                    "transcript_raw": transcript_raw,
                    "transcript_normalized": transcript_normalized,
                    "wer_raw": float(_wer(expected_text, transcript_raw)),
                    "wer_normalized": float(_wer(expected_text, transcript_normalized)),
                    "cer_raw": float(_cer(expected_text, transcript_raw)),
                    "cer_normalized": float(_cer(expected_text, transcript_normalized)),
                    "latency_ms": float(latency_ms),
                    "error": str(error or ""),
                }
            )

        summary = _build_setup_summary(setup, rows, latency_budget_ms)
        summary["runtime_direct"] = True
        summary["requested_backend"] = backend
        setup_summaries.append(summary)

    baseline_setup_id = ""
    for item in setup_summaries:
        if str(item.get("backend") or "") == "faster_whisper":
            baseline_setup_id = str(item.get("id") or "")
            break
    if not baseline_setup_id and setup_summaries:
        baseline_setup_id = str(setup_summaries[0].get("id") or "")

    recommendation_block = _recommend_setups(
        setup_summaries,
        baseline_setup=baseline_setup_id,
        latency_budget_ms=latency_budget_ms,
    )

    return {
        "enabled": True,
        "executed": True,
        "requested_backends": backends,
        "audio_scenario_count": len(selected_cases),
        "missing_audio_file_count": len(missing_audio),
        "missing_audio_files": missing_audio[:10],
        "setups": sorted(
            setup_summaries,
            key=lambda item: (
                float(item.get("balance_score") or 0.0),
                float(item.get("quality_score") or 0.0),
            ),
            reverse=True,
        ),
        "baseline_setup": str((recommendation_block.get("baseline") or {}).get("id") or baseline_setup_id),
        "recommendation": recommendation_block.get("recommendation") or {},
        "baseline_vs_recommended": recommendation_block.get("baseline_vs_recommended") or {},
        "done_gate": recommendation_block.get("done_gate") or {},
    }


def run_stt_egyptian_benchmark(
    *,
    corpus_path=None,
    include_runtime_ab=False,
    runtime_backends=None,
    runtime_max_cases=None,
):
    pack, base_dir, source_path = _load_pack(corpus_path=corpus_path)

    setups = list(pack.get("setups") or [])
    latency_budget_ms = _coerce_float(pack.get("latency_budget_ms_low_mid_cpu"), default=800.0, minimum=200.0)
    baseline_setup = str(pack.get("baseline_setup") or (setups[0].get("id") if setups else "")).strip()

    setup_summaries = [_evaluate_setup(pack, setup, latency_budget_ms=latency_budget_ms) for setup in setups]
    recommendation_block = _recommend_setups(
        setup_summaries,
        baseline_setup=baseline_setup,
        latency_budget_ms=latency_budget_ms,
    )
    baseline = recommendation_block.get("baseline") or {}
    recommended = recommendation_block.get("recommended") or {}

    payload = {
        "timestamp": time.time(),
        "corpus": {
            "name": str(pack.get("name") or "stt_egyptian_dialect_pack"),
            "version": str(pack.get("version") or "unknown"),
            "source": source_path,
            "scenario_count": len(list(pack.get("scenarios") or [])),
        },
        "latency_budget_ms_low_mid_cpu": float(latency_budget_ms),
        "baseline_setup": str((baseline or {}).get("id") or baseline_setup),
        "setups": sorted(
            setup_summaries,
            key=lambda item: (
                float(item.get("balance_score") or 0.0),
                float(item.get("quality_score") or 0.0),
            ),
            reverse=True,
        ),
        "recommendation": recommendation_block.get("recommendation") or {},
        "baseline_vs_recommended": recommendation_block.get("baseline_vs_recommended") or {},
        "done_gate": recommendation_block.get("done_gate") or {},
    }

    if include_runtime_ab:
        payload["runtime_ab"] = _run_runtime_ab(
            pack,
            base_dir=base_dir,
            latency_budget_ms=latency_budget_ms,
            runtime_backends=runtime_backends,
            runtime_max_cases=runtime_max_cases,
        )

    return payload
