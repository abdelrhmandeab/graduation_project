import threading
import time


def _percentile(values, p):
    if not values:
        return None
    ordered = sorted(values)
    index = int(round((p / 100) * (len(ordered) - 1)))
    return ordered[index]


def _bucket_summary(bucket):
    count = bucket["count"]
    success = bucket["success_count"]
    lats = bucket["latencies"]
    return {
        "count": count,
        "success_rate": (success / count) if count else 0.0,
        "p50_ms": (_percentile(lats, 50) or 0.0) * 1000,
        "p95_ms": (_percentile(lats, 95) or 0.0) * 1000,
    }


def _new_bucket():
    return {"count": 0, "success_count": 0, "latencies": []}


def _normalize_language(language):
    value = str(language or "").strip().lower()
    if not value:
        return "unknown"
    if value in {"ar", "en", "unsupported", "unknown"}:
        return value
    return "other"


def _resource_snapshot():
    try:
        import psutil  # type: ignore
    except Exception:
        return {
            "cpu_percent": None,
            "rss_mb": None,
            "backend": "none",
        }

    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "rss_mb": float(memory_info.rss) / (1024 * 1024),
        "backend": "psutil",
    }


class Metrics:
    def __init__(self):
        self.start_times = {}
        self.command_stats = {}
        self.language_stats = {}
        self.intent_language_stats = {}
        self.stage_stats = {}
        self.diagnostic_stats = {}
        self._lock = threading.Lock()

    def _record_bucket(self, container, key, success, latency_seconds):
        bucket = container.setdefault(key, _new_bucket())
        bucket["count"] += 1
        if success:
            bucket["success_count"] += 1
        bucket["latencies"].append(float(latency_seconds))
        return bucket

    def start(self, key):
        with self._lock:
            self.start_times[key] = time.time()

    def end(self, key):
        with self._lock:
            start_time = self.start_times.pop(key, None)
        if start_time is None:
            return None
        return time.time() - start_time

    def record_command(self, command_type, success, latency_seconds, language=None):
        with self._lock:
            self._record_bucket(self.command_stats, command_type, success, latency_seconds)
            normalized_language = _normalize_language(language)
            self._record_bucket(self.language_stats, normalized_language, success, latency_seconds)
            per_intent = self.intent_language_stats.setdefault(command_type, {})
            self._record_bucket(per_intent, normalized_language, success, latency_seconds)

    def record_stage(self, stage_name, latency_seconds, success=True):
        with self._lock:
            self._record_bucket(self.stage_stats, stage_name, success, latency_seconds)

    def record_diagnostic(self, check_name, success, latency_seconds):
        with self._lock:
            self._record_bucket(self.diagnostic_stats, check_name, success, latency_seconds)

    def reset(self):
        with self._lock:
            self.start_times = {}
            self.command_stats = {}
            self.language_stats = {}
            self.intent_language_stats = {}
            self.stage_stats = {}
            self.diagnostic_stats = {}

    def snapshot(self):
        with self._lock:
            command_data = {
                k: {
                    "count": v["count"],
                    "success_count": v["success_count"],
                    "latencies": list(v["latencies"]),
                }
                for k, v in self.command_stats.items()
            }
            language_data = {
                k: {
                    "count": v["count"],
                    "success_count": v["success_count"],
                    "latencies": list(v["latencies"]),
                }
                for k, v in self.language_stats.items()
            }
            intent_language_data = {
                intent: {
                    language: {
                        "count": bucket["count"],
                        "success_count": bucket["success_count"],
                        "latencies": list(bucket["latencies"]),
                    }
                    for language, bucket in per_language.items()
                }
                for intent, per_language in self.intent_language_stats.items()
            }
            stage_data = {
                k: {
                    "count": v["count"],
                    "success_count": v["success_count"],
                    "latencies": list(v["latencies"]),
                }
                for k, v in self.stage_stats.items()
            }
            diagnostic_data = {
                k: {
                    "count": v["count"],
                    "success_count": v["success_count"],
                    "latencies": list(v["latencies"]),
                }
                for k, v in self.diagnostic_stats.items()
            }

        total_count = sum(v["count"] for v in command_data.values())
        total_success = sum(v["success_count"] for v in command_data.values())
        overall_success_rate = (total_success / total_count) if total_count else 0.0

        commands = {key: _bucket_summary(value) for key, value in command_data.items()}
        languages = {key: _bucket_summary(value) for key, value in language_data.items()}
        intent_language = {
            intent: {language: _bucket_summary(bucket) for language, bucket in per_language.items()}
            for intent, per_language in intent_language_data.items()
        }
        stages = {key: _bucket_summary(value) for key, value in stage_data.items()}
        diagnostics = {key: _bucket_summary(value) for key, value in diagnostic_data.items()}
        rollback = commands.get("OS_ROLLBACK", {"count": 0, "success_rate": 0.0})
        resources = _resource_snapshot()

        return {
            "overall": {
                "count": total_count,
                "success_rate": overall_success_rate,
            },
            "rollback": {
                "count": rollback["count"],
                "success_rate": rollback["success_rate"],
            },
            "commands": commands,
            "languages": languages,
            "intent_language": intent_language,
            "stages": stages,
            "diagnostics": diagnostics,
            "resources": resources,
        }

    def format_report(self):
        snap = self.snapshot()
        lines = [
            "Metrics Report",
            f"Overall commands: {snap['overall']['count']}",
            f"Overall success rate: {snap['overall']['success_rate']:.2%}",
            f"Rollback success rate: {snap['rollback']['success_rate']:.2%}",
            "",
            "Per-command latency and success:",
        ]

        for command_type in sorted(snap["commands"]):
            stat = snap["commands"][command_type]
            lines.append(
                (
                    f"- {command_type}: count={stat['count']}, "
                    f"success={stat['success_rate']:.2%}, "
                    f"p50={stat['p50_ms']:.1f}ms, p95={stat['p95_ms']:.1f}ms"
                )
            )

        return "\n".join(lines)

    def format_observability_report(self):
        snap = self.snapshot()
        lines = [
            "Observability Dashboard",
            f"Overall commands: {snap['overall']['count']}",
            f"Overall success rate: {snap['overall']['success_rate']:.2%}",
            "",
            "Command Metrics:",
        ]
        for key in sorted(snap["commands"]):
            stat = snap["commands"][key]
            lines.append(
                (
                    f"- {key}: count={stat['count']}, success={stat['success_rate']:.2%}, "
                    f"p50={stat['p50_ms']:.1f}ms, p95={stat['p95_ms']:.1f}ms"
                )
            )

        lines.append("")
        lines.append("Language Metrics:")
        if not snap["languages"]:
            lines.append("- no language data yet")
        else:
            for key in sorted(snap["languages"]):
                stat = snap["languages"][key]
                lines.append(
                    (
                        f"- {key}: count={stat['count']}, success={stat['success_rate']:.2%}, "
                        f"p50={stat['p50_ms']:.1f}ms, p95={stat['p95_ms']:.1f}ms"
                    )
                )

        lines.append("")
        lines.append("Intent/Language Metrics:")
        if not snap["intent_language"]:
            lines.append("- no per-intent/per-language data yet")
        else:
            for intent in sorted(snap["intent_language"]):
                per_language = snap["intent_language"][intent]
                for language in sorted(per_language):
                    stat = per_language[language]
                    lines.append(
                        (
                            f"- {intent} [{language}]: count={stat['count']}, success={stat['success_rate']:.2%}, "
                            f"p50={stat['p50_ms']:.1f}ms, p95={stat['p95_ms']:.1f}ms"
                        )
                    )

        lines.append("")
        lines.append("Pipeline Stage Metrics:")
        if not snap["stages"]:
            lines.append("- no stage data yet")
        else:
            for key in sorted(snap["stages"]):
                stat = snap["stages"][key]
                lines.append(
                    (
                        f"- {key}: count={stat['count']}, success={stat['success_rate']:.2%}, "
                        f"p50={stat['p50_ms']:.1f}ms, p95={stat['p95_ms']:.1f}ms"
                    )
                )

        lines.append("")
        lines.append("Diagnostics Metrics:")
        if not snap["diagnostics"]:
            lines.append("- no diagnostics data yet")
        else:
            for key in sorted(snap["diagnostics"]):
                stat = snap["diagnostics"][key]
                lines.append(
                    (
                        f"- {key}: count={stat['count']}, success={stat['success_rate']:.2%}, "
                        f"p50={stat['p50_ms']:.1f}ms, p95={stat['p95_ms']:.1f}ms"
                    )
                )

        lines.append("")
        resources = snap["resources"]
        lines.append("Resource Snapshot:")
        lines.append(f"- backend: {resources['backend']}")
        lines.append(f"- cpu_percent: {resources['cpu_percent']}")
        lines.append(f"- rss_mb: {resources['rss_mb']}")
        return "\n".join(lines)


metrics = Metrics()
