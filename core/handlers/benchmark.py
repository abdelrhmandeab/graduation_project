from core.benchmark import run_quick_benchmark, run_resilience_demo


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

    return False, "Unsupported benchmark command.", {}
