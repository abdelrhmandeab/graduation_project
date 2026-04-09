from core.benchmark import (
    run_quick_benchmark,
    run_resilience_demo,
    run_stt_reliability_benchmark,
    run_tts_quality_benchmark,
    run_wake_reliability_benchmark,
)


def handle(parsed, route_command_fn):
    action = parsed.action
    if action == "run":
        payload = run_quick_benchmark(route_command_fn)
        sla = payload.get("sla") or {}
        history = payload.get("history") or {}
        lines = [
            "Benchmark Report",
            f"scenarios: {payload['scenario_count']}",
            f"success_rate: {payload['success_rate']:.2%}",
            f"p50_latency_ms: {payload['p50_latency_ms']:.1f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"sla_passed: {sla.get('passed')}",
        ]
        if history:
            lines.extend(
                [
                    f"history_file: {history.get('history_file')}",
                    f"history_runs: {history.get('run_count')}",
                    f"history_daily_points: {history.get('daily_points')}",
                    f"history_weekly_points: {history.get('weekly_points')}",
                ]
            )
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            lines.append(
                (
                    f"- {row['name']}: ok={row['ok']}, latency_ms={row['latency_ms']:.1f}, "
                    f"command={row['command']}"
                )
            )
        return True, "\n".join(lines), {"benchmark_success_rate": payload["success_rate"]}

    if action == "resilience_demo":
        payload = run_resilience_demo(route_command_fn)
        sla = payload.get("sla") or {}
        history = payload.get("history") or {}
        lines = [
            "Resilience Report",
            f"scenarios: {payload['scenario_count']}",
            f"success_rate: {payload['success_rate']:.2%}",
            f"p50_latency_ms: {payload['p50_latency_ms']:.1f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"sla_passed: {sla.get('passed')}",
        ]
        if history:
            lines.extend(
                [
                    f"history_file: {history.get('history_file')}",
                    f"history_runs: {history.get('run_count')}",
                    f"history_daily_points: {history.get('daily_points')}",
                    f"history_weekly_points: {history.get('weekly_points')}",
                ]
            )
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            line = f"- {row['name']}: ok={row['ok']}, latency_ms={row['latency_ms']:.1f}"
            if row.get("error"):
                line += f", error={row['error']}"
            lines.append(line)
        return True, "\n".join(lines), {"resilience_success_rate": payload["success_rate"]}

    if action == "wake_reliability":
        payload = run_wake_reliability_benchmark()
        sla = payload.get("sla") or {}
        history = payload.get("history") or {}
        lines = [
            "Wake Reliability Report",
            f"scenarios: {payload['scenario_count']}",
            f"scenario_success_rate: {payload['success_rate']:.2%}",
            f"detection_rate: {payload['detection_rate']:.2%}",
            f"false_positive_rate: {payload['false_positive_rate']:.2%}",
            f"p50_latency_ms: {payload['p50_latency_ms']:.1f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"sla_passed: {sla.get('passed')}",
        ]
        if history:
            lines.extend(
                [
                    f"history_file: {history.get('history_file')}",
                    f"history_runs: {history.get('run_count')}",
                    f"history_daily_points: {history.get('daily_points')}",
                    f"history_weekly_points: {history.get('weekly_points')}",
                ]
            )
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            line = (
                f"- {row['name']}: ok={row['ok']}, expected={row['expected_detection']}, "
                f"detected={row['detected']}, source={row.get('detected_source') or 'none'}"
            )
            if row.get("latency_ms") is not None:
                line += f", latency_ms={float(row['latency_ms']):.1f}"
            if row.get("error"):
                line += f", error={row['error']}"
            lines.append(line)
        return True, "\n".join(lines), {
            "wake_detection_rate": payload["detection_rate"],
            "wake_false_positive_rate": payload["false_positive_rate"],
        }

    if action == "stt_reliability":
        payload = run_stt_reliability_benchmark()
        sla = payload.get("sla") or {}
        history = payload.get("history") or {}
        corpus = payload.get("corpus") or {}
        lines = [
            "STT Reliability Report",
            f"corpus: {(corpus.get('name') or 'unknown')} ({(corpus.get('version') or 'unknown')})",
            f"mode: {(corpus.get('mode') or 'auto')}",
            f"scenarios: {payload['scenario_count']}",
            f"evaluated: {payload.get('evaluated_count', 0)}",
            f"success_rate: {payload['success_rate']:.2%}",
            f"avg_wer: {float(payload.get('avg_wer') or 0.0):.4f}",
            f"p95_wer: {float(payload.get('p95_wer') or 0.0):.4f}",
            f"p50_latency_ms: {payload['p50_latency_ms']:.1f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"sla_passed: {sla.get('passed')}",
        ]
        if history:
            lines.extend(
                [
                    f"history_file: {history.get('history_file')}",
                    f"history_runs: {history.get('run_count')}",
                    f"history_daily_points: {history.get('daily_points')}",
                    f"history_weekly_points: {history.get('weekly_points')}",
                ]
            )
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            line = (
                f"- {row['name']}: ok={row['ok']}, source={row.get('source')}, "
                f"wer={float(row.get('wer') or 0.0):.4f}, latency_ms={float(row.get('latency_ms') or 0.0):.1f}"
            )
            if row.get("error"):
                line += f", error={row['error']}"
            lines.append(line)
        return True, "\n".join(lines), {
            "stt_avg_wer": payload["avg_wer"],
            "stt_success_rate": payload["success_rate"],
        }

    if action == "tts_quality":
        payload = run_tts_quality_benchmark()
        sla = payload.get("sla") or {}
        history = payload.get("history") or {}
        corpus = payload.get("corpus") or {}
        lines = [
            "TTS Quality Report",
            f"corpus: {(corpus.get('name') or 'unknown')} ({(corpus.get('version') or 'unknown')})",
            f"mode: {(corpus.get('mode') or 'auto')}",
            f"backend: {(corpus.get('backend') or 'auto')}",
            f"scenarios: {payload['scenario_count']}",
            f"success_rate: {payload['success_rate']:.2%}",
            f"avg_quality_score: {float(payload.get('avg_quality_score') or 0.0):.4f}",
            f"avg_rtf: {float(payload.get('avg_rtf') or 0.0):.4f}",
            f"p95_latency_ms: {payload['p95_latency_ms']:.1f}",
            f"real_scenarios: {int(payload.get('real_scenario_count') or 0)}",
            f"mock_scenarios: {int(payload.get('mock_scenario_count') or 0)}",
            f"sla_passed: {sla.get('passed')}",
        ]
        if history:
            lines.extend(
                [
                    f"history_file: {history.get('history_file')}",
                    f"history_runs: {history.get('run_count')}",
                    f"history_daily_points: {history.get('daily_points')}",
                    f"history_weekly_points: {history.get('weekly_points')}",
                ]
            )
        for check in sla.get("checks", []):
            lines.append(
                f"- sla {check.get('name')}: actual={check.get('actual')}, "
                f"expected {check.get('operator')} {check.get('threshold')}, passed={check.get('passed')}"
            )
        for row in payload["results"]:
            line = (
                f"- {row['name']}: ok={row['ok']}, source={row.get('source')}, backend={row.get('backend_used')}, "
                f"quality_score={float(row.get('quality_score') or 0.0):.4f}, rtf={float(row.get('rtf') or 0.0):.4f}, "
                f"latency_ms={float(row.get('latency_ms') or 0.0):.1f}"
            )
            if row.get("error"):
                line += f", error={row['error']}"
            lines.append(line)
        return True, "\n".join(lines), {
            "tts_avg_quality_score": payload["avg_quality_score"],
            "tts_success_rate": payload["success_rate"],
        }

    return False, "Unsupported benchmark command.", {}
