import logging
import threading
import time
import re
from collections import OrderedDict

_logger = logging.getLogger("jarvis")


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


def _new_clarification_bucket():
    return {"requested": 0, "resolved": 0, "cancelled": 0, "reprompt": 0}


def _new_quality_bucket():
    return {
        "count": 0,
        "human_likeness_sum": 0.0,
        "coherence_sum": 0.0,
        "lexical_diversity_sum": 0.0,
    }


def _normalize_language(language):
    value = str(language or "").strip().lower()
    if not value:
        return "unknown"
    if value in {"ar", "en", "unsupported", "unknown"}:
        return value
    return "other"


def _normalize_quality_text(text):
    return " ".join(str(text or "").strip().split())


def _tokenize_quality_text(text):
    return re.findall(r"[a-z0-9\u0600-\u06FF]+", str(text or "").lower())


def _lexical_diversity(tokens):
    if not tokens:
        return 0.0
    return max(0.0, min(1.0, len(set(tokens)) / float(len(tokens))))


def _human_likeness_score(text):
    value = _normalize_quality_text(text)
    tokens = _tokenize_quality_text(value)
    if not tokens:
        return 0.0

    punctuation_count = len(re.findall(r"[.!?؟,;:]", value))
    sentence_count = max(1, len(re.findall(r"[.!?؟]", value)) or 1)
    word_count = len(tokens)
    avg_sentence_len = word_count / float(sentence_count)
    diversity = _lexical_diversity(tokens)

    score = 0.28
    if 4 <= avg_sentence_len <= 24:
        score += 0.24
    elif avg_sentence_len < 3:
        score -= 0.12
    else:
        score -= 0.06

    score += min(0.25, diversity * 0.35)
    if punctuation_count > 0:
        score += min(0.15, punctuation_count * 0.03)

    repeated_sequence = bool(re.search(r"\b(\w+)\b(?:\s+\1\b){2,}", value.lower()))
    if repeated_sequence:
        score -= 0.18
    if word_count <= 2:
        score -= 0.15

    return max(0.0, min(1.0, score))


def _coherence_score(response_text, user_text="", previous_response=""):
    response_tokens = set(_tokenize_quality_text(response_text))
    if not response_tokens:
        return 0.0

    user_tokens = set(_tokenize_quality_text(user_text))
    prev_tokens = set(_tokenize_quality_text(previous_response))

    overlap_with_user = (len(response_tokens & user_tokens) / float(len(response_tokens))) if user_tokens else 0.0
    overlap_with_prev = (len(response_tokens & prev_tokens) / float(len(response_tokens))) if prev_tokens else 0.0

    score = 0.34
    score += min(0.34, overlap_with_user * 1.7)
    score += min(0.20, overlap_with_prev * 1.2)

    if user_tokens and overlap_with_user == 0.0:
        score -= 0.08
    if len(response_tokens) < 3:
        score -= 0.12
    if len(response_tokens) > 64:
        score -= 0.05

    return max(0.0, min(1.0, score))


def _quality_summary(bucket):
    count = int(bucket.get("count") or 0)
    if count <= 0:
        return {
            "count": 0,
            "human_likeness": 0.0,
            "coherence": 0.0,
            "lexical_diversity": 0.0,
        }
    return {
        "count": count,
        "human_likeness": float(bucket.get("human_likeness_sum") or 0.0) / float(count),
        "coherence": float(bucket.get("coherence_sum") or 0.0) / float(count),
        "lexical_diversity": float(bucket.get("lexical_diversity_sum") or 0.0) / float(count),
    }


def _normalize_ambiguous_token(source_text):
    text = str(source_text or "").strip().lower()
    if not text:
        return ""

    text = re.sub(
        r"^(?:open app|open|close app|close|launch|start|find file|search file|find|search)\s+",
        "",
        text,
    )
    text = re.sub(
        r"^(?:افتحلي برنامج|افتح|شغللي برنامج|شغل|اقفللي برنامج|اقفل برنامج|سكر برنامج|دور على ملف|دوّر على ملف|دور)\s+",
        "",
        text,
    )
    text = text.replace("\\", "/")
    if "/" in text:
        text = text.split("/")[-1]

    text = re.sub(r"[^a-z0-9\u0600-\u06FF._\s-]", " ", text)
    text = " ".join(text.split())
    if not text:
        return ""

    stop_words = {
        "the",
        "app",
        "application",
        "folder",
        "file",
        "path",
        "option",
        "one",
        "برنامج",
        "ملف",
        "مجلد",
        "مسار",
        "الاختيار",
    }
    parts = [part for part in text.split() if part not in stop_words]
    if not parts:
        return ""
    return " ".join(parts[:2])


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


def _llm_cache_snapshot():
    payload = {}
    available = False
    try:
        from core.command_router import get_llm_response_cache_stats

        payload = dict(get_llm_response_cache_stats() or {})
        available = True
    except Exception:
        payload = {}

    hits = max(0, int(payload.get("hits") or 0))
    misses = max(0, int(payload.get("misses") or 0))
    lookups = hits + misses
    hit_rate = (hits / float(lookups)) if lookups else 0.0

    return {
        "available": bool(available),
        "enabled": bool(payload.get("enabled", False)),
        "size": max(0, int(payload.get("size") or 0)),
        "hits": hits,
        "misses": misses,
        "lookups": lookups,
        "hit_rate": hit_rate,
        "stores": max(0, int(payload.get("stores") or 0)),
        "evictions": max(0, int(payload.get("evictions") or 0)),
        "ttl_seconds": max(0, int(payload.get("ttl_seconds") or 0)),
        "max_size": max(0, int(payload.get("max_size") or 0)),
    }


class Metrics:
    def __init__(self):
        self.start_times = {}
        self.command_stats = {}
        self.language_stats = {}
        self.intent_language_stats = {}
        self.stage_stats = {}
        self.diagnostic_stats = {}
        self.clarification_stats = {
            "requested": 0,
            "resolved": 0,
            "resolved_without_retry": 0,
            "resolved_on_first_retry": 0,
            "resolved_after_many_retries": 0,
            "resolved_execution_failed": 0,
            "cancelled": 0,
            "reprompt": 0,
            "wrong_action_prevented": 0,
            "post_resolution_corrections": 0,
            "likely_false_clarification": 0,
        }
        self.clarification_reason_counts = {}
        self.clarification_intent_language = {}
        self.ambiguous_token_counts = {}
        self.response_quality_stats = _new_quality_bucket()
        self.response_quality_language = {}
        self.response_quality_persona = {}
        self.response_quality_mode = {}
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

    def record_clarification_event(
        self,
        event_type,
        *,
        intent="",
        language="unknown",
        reason="",
        source_text="",
        retry_count=0,
        wrong_action_prevented=False,
    ):
        event_key = str(event_type or "").strip().lower()
        if not event_key:
            return

        normalized_language = _normalize_language(language)
        intent_key = str(intent or "").strip().upper() or "UNKNOWN"
        reason_key = str(reason or "").strip().lower()
        retry_count_value = max(0, int(retry_count or 0))

        with self._lock:
            bucket = self.clarification_intent_language.setdefault(intent_key, {})
            counters = bucket.setdefault(normalized_language, _new_clarification_bucket())

            if event_key == "requested":
                self.clarification_stats["requested"] += 1
                counters["requested"] += 1
                if wrong_action_prevented:
                    self.clarification_stats["wrong_action_prevented"] += 1
                if reason_key:
                    self.clarification_reason_counts[reason_key] = int(
                        self.clarification_reason_counts.get(reason_key) or 0
                    ) + 1
                if reason_key in {
                    "app_name_ambiguous",
                    "app_close_ambiguous",
                    "file_search_multiple_matches",
                    "open_target_ambiguous",
                }:
                    token = _normalize_ambiguous_token(source_text)
                    if token:
                        self.ambiguous_token_counts[token] = int(self.ambiguous_token_counts.get(token) or 0) + 1
                return

            if event_key == "resolved":
                self.clarification_stats["resolved"] += 1
                counters["resolved"] += 1
                if retry_count_value <= 0:
                    self.clarification_stats["resolved_without_retry"] += 1
                elif retry_count_value == 1:
                    self.clarification_stats["resolved_on_first_retry"] += 1
                else:
                    self.clarification_stats["resolved_after_many_retries"] += 1
                return

            if event_key in {"resolved_failed", "resolved_execution_failed"}:
                self.clarification_stats["resolved"] += 1
                self.clarification_stats["resolved_execution_failed"] += 1
                self.clarification_stats["likely_false_clarification"] += 1
                counters["resolved"] += 1
                return

            if event_key == "cancelled":
                self.clarification_stats["cancelled"] += 1
                counters["cancelled"] += 1
                return

            if event_key in {"reprompt", "needs_clarification", "not_a_reply"}:
                self.clarification_stats["reprompt"] += 1
                counters["reprompt"] += 1
                return

            if event_key in {"post_correction", "post_resolution_correction"}:
                self.clarification_stats["post_resolution_corrections"] += 1
                self.clarification_stats["likely_false_clarification"] += 1
                return

    def record_response_quality(
        self,
        response_text,
        *,
        language="unknown",
        user_text="",
        previous_response="",
        persona="assistant",
        response_mode="default",
    ):
        response_value = _normalize_quality_text(response_text)
        if not response_value:
            return

        human_likeness = _human_likeness_score(response_value)
        coherence = _coherence_score(response_value, user_text=user_text, previous_response=previous_response)
        diversity = _lexical_diversity(_tokenize_quality_text(response_value))

        language_key = _normalize_language(language)
        persona_key = str(persona or "assistant").strip().lower() or "assistant"
        mode_key = str(response_mode or "default").strip().lower() or "default"

        with self._lock:
            for bucket in (
                self.response_quality_stats,
                self.response_quality_language.setdefault(language_key, _new_quality_bucket()),
                self.response_quality_persona.setdefault(persona_key, _new_quality_bucket()),
                self.response_quality_mode.setdefault(mode_key, _new_quality_bucket()),
            ):
                bucket["count"] += 1
                bucket["human_likeness_sum"] += float(human_likeness)
                bucket["coherence_sum"] += float(coherence)
                bucket["lexical_diversity_sum"] += float(diversity)

    def reset(self):
        with self._lock:
            self.start_times = {}
            self.command_stats = {}
            self.language_stats = {}
            self.intent_language_stats = {}
            self.stage_stats = {}
            self.diagnostic_stats = {}
            self.clarification_stats = {
                "requested": 0,
                "resolved": 0,
                "resolved_without_retry": 0,
                "resolved_on_first_retry": 0,
                "resolved_after_many_retries": 0,
                "resolved_execution_failed": 0,
                "cancelled": 0,
                "reprompt": 0,
                "wrong_action_prevented": 0,
                "post_resolution_corrections": 0,
                "likely_false_clarification": 0,
            }
            self.clarification_reason_counts = {}
            self.clarification_intent_language = {}
            self.ambiguous_token_counts = {}
            self.response_quality_stats = _new_quality_bucket()
            self.response_quality_language = {}
            self.response_quality_persona = {}
            self.response_quality_mode = {}

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
            clarification_counts = dict(self.clarification_stats)
            clarification_reason_data = dict(self.clarification_reason_counts)
            clarification_intent_language_data = {
                intent: {language: dict(bucket) for language, bucket in per_language.items()}
                for intent, per_language in self.clarification_intent_language.items()
            }
            ambiguous_token_data = dict(self.ambiguous_token_counts)
            response_quality_totals = dict(self.response_quality_stats)
            response_quality_language_data = {
                key: dict(value) for key, value in self.response_quality_language.items()
            }
            response_quality_persona_data = {
                key: dict(value) for key, value in self.response_quality_persona.items()
            }
            response_quality_mode_data = {
                key: dict(value) for key, value in self.response_quality_mode.items()
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
        llm_cache = _llm_cache_snapshot()

        tts_prewarm_bucket = stage_data.get("tts_prewarm")
        tts_prewarm_count = int((tts_prewarm_bucket or {}).get("count") or 0)
        tts_prewarm_success_count = int((tts_prewarm_bucket or {}).get("success_count") or 0)
        tts_prewarm_hit_rate = (
            tts_prewarm_success_count / float(tts_prewarm_count)
            if tts_prewarm_count
            else 0.0
        )
        tts_prewarm_latency = _bucket_summary(tts_prewarm_bucket) if tts_prewarm_bucket else {
            "count": 0,
            "success_rate": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
        }

        requested = int(clarification_counts.get("requested") or 0)
        resolved = int(clarification_counts.get("resolved") or 0)
        resolved_on_first_retry = int(clarification_counts.get("resolved_on_first_retry") or 0)
        resolved_execution_failed = int(clarification_counts.get("resolved_execution_failed") or 0)
        post_resolution_corrections = int(clarification_counts.get("post_resolution_corrections") or 0)
        likely_false_count = int(clarification_counts.get("likely_false_clarification") or 0)
        clarification_success_rate = (resolved / requested) if requested else 0.0
        first_retry_success_rate = (resolved_on_first_retry / resolved) if resolved else 0.0
        post_resolution_failure_rate = (resolved_execution_failed / resolved) if resolved else 0.0
        post_correction_rate = (post_resolution_corrections / resolved) if resolved else 0.0
        likely_false_rate = (likely_false_count / requested) if requested else 0.0

        clarification_intent_language = {}
        for intent, per_language in clarification_intent_language_data.items():
            rows = {}
            for language, bucket in per_language.items():
                command_count = int(
                    ((intent_language_data.get(intent) or {}).get(language) or {}).get("count") or 0
                )
                requested_count = int(bucket.get("requested") or 0)
                rows[language] = {
                    "requested": requested_count,
                    "resolved": int(bucket.get("resolved") or 0),
                    "cancelled": int(bucket.get("cancelled") or 0),
                    "reprompt": int(bucket.get("reprompt") or 0),
                    "command_count": command_count,
                    "clarification_rate": (requested_count / command_count) if command_count else 0.0,
                }
            clarification_intent_language[intent] = rows

        top_ambiguous_tokens = sorted(
            (
                {"token": token, "count": count}
                for token, count in ambiguous_token_data.items()
                if token
            ),
            key=lambda row: row["count"],
            reverse=True,
        )[:10]

        response_quality = {
            "overall": _quality_summary(response_quality_totals),
            "by_language": {
                key: _quality_summary(value)
                for key, value in sorted(response_quality_language_data.items(), key=lambda item: item[0])
            },
            "by_persona": {
                key: _quality_summary(value)
                for key, value in sorted(response_quality_persona_data.items(), key=lambda item: item[0])
            },
            "by_mode": {
                key: _quality_summary(value)
                for key, value in sorted(response_quality_mode_data.items(), key=lambda item: item[0])
            },
        }

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
            "clarification": {
                "requested": requested,
                "resolved": resolved,
                "cancelled": int(clarification_counts.get("cancelled") or 0),
                "reprompt": int(clarification_counts.get("reprompt") or 0),
                "resolved_without_retry": int(clarification_counts.get("resolved_without_retry") or 0),
                "resolved_on_first_retry": resolved_on_first_retry,
                "resolved_after_many_retries": int(
                    clarification_counts.get("resolved_after_many_retries") or 0
                ),
                "resolved_execution_failed": resolved_execution_failed,
                "success_rate": clarification_success_rate,
                "success_on_first_retry_rate": first_retry_success_rate,
                "post_resolution_failure_rate": post_resolution_failure_rate,
                "post_resolution_corrections": post_resolution_corrections,
                "post_correction_rate": post_correction_rate,
                "likely_false_clarification": likely_false_count,
                "likely_false_rate": likely_false_rate,
                "wrong_action_prevented": int(clarification_counts.get("wrong_action_prevented") or 0),
                "reason_counts": clarification_reason_data,
                "intent_language": clarification_intent_language,
                "top_ambiguous_tokens": top_ambiguous_tokens,
            },
            "response_quality": response_quality,
            "llm_cache": llm_cache,
            "tts_prewarm": {
                "count": tts_prewarm_count,
                "warm_hits": tts_prewarm_success_count,
                "hit_rate": tts_prewarm_hit_rate,
                "p50_ms": float(tts_prewarm_latency.get("p50_ms") or 0.0),
                "p95_ms": float(tts_prewarm_latency.get("p95_ms") or 0.0),
            },
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
        clarification = snap.get("clarification") or {}
        lines.append("Clarification Metrics:")
        lines.append(
            (
                f"- requested={clarification.get('requested', 0)}, resolved={clarification.get('resolved', 0)}, "
                f"success={float(clarification.get('success_rate') or 0.0):.2%}, "
                f"first_retry_success={float(clarification.get('success_on_first_retry_rate') or 0.0):.2%}"
            )
        )
        lines.append(
            (
                f"- post_resolution_failure_rate={float(clarification.get('post_resolution_failure_rate') or 0.0):.2%}, "
                f"post_correction_rate={float(clarification.get('post_correction_rate') or 0.0):.2%}, "
                f"likely_false_rate={float(clarification.get('likely_false_rate') or 0.0):.2%}"
            )
        )
        lines.append(f"- wrong_action_prevented={clarification.get('wrong_action_prevented', 0)}")

        reason_counts = dict(clarification.get("reason_counts") or {})
        if reason_counts:
            lines.append("- by_reason:")
            for reason, count in sorted(reason_counts.items(), key=lambda row: row[1], reverse=True):
                lines.append(f"  - {reason}: {count}")

        intent_language_rows = dict(clarification.get("intent_language") or {})
        if intent_language_rows:
            lines.append("- by_intent_language:")
            for intent in sorted(intent_language_rows):
                per_language = dict(intent_language_rows[intent] or {})
                for language in sorted(per_language):
                    row = per_language[language]
                    lines.append(
                        (
                            f"  - {intent} [{language}]: requested={row.get('requested', 0)}, "
                            f"resolved={row.get('resolved', 0)}, rate={float(row.get('clarification_rate') or 0.0):.2%}"
                        )
                    )

        top_tokens = list(clarification.get("top_ambiguous_tokens") or [])
        if top_tokens:
            lines.append("- top_ambiguous_tokens:")
            for row in top_tokens:
                lines.append(f"  - {row.get('token')}: {row.get('count')}")

        lines.append("")
        quality = snap.get("response_quality") or {}
        quality_overall = dict(quality.get("overall") or {})
        lines.append("Response Quality Metrics:")
        lines.append(
            (
                f"- count={int(quality_overall.get('count') or 0)}, "
                f"human_likeness={float(quality_overall.get('human_likeness') or 0.0):.2f}, "
                f"coherence={float(quality_overall.get('coherence') or 0.0):.2f}, "
                f"lexical_diversity={float(quality_overall.get('lexical_diversity') or 0.0):.2f}"
            )
        )

        by_language = dict(quality.get("by_language") or {})
        if by_language:
            lines.append("- by_language:")
            for language in sorted(by_language):
                row = dict(by_language[language] or {})
                lines.append(
                    (
                        f"  - {language}: human_likeness={float(row.get('human_likeness') or 0.0):.2f}, "
                        f"coherence={float(row.get('coherence') or 0.0):.2f}"
                    )
                )

        by_mode = dict(quality.get("by_mode") or {})
        if by_mode:
            lines.append("- by_mode:")
            for mode in sorted(by_mode):
                row = dict(by_mode[mode] or {})
                lines.append(
                    (
                        f"  - {mode}: count={int(row.get('count') or 0)}, "
                        f"human_likeness={float(row.get('human_likeness') or 0.0):.2f}, "
                        f"coherence={float(row.get('coherence') or 0.0):.2f}"
                    )
                )

        lines.append("")
        cache = dict(snap.get("llm_cache") or {})
        prewarm = dict(snap.get("tts_prewarm") or {})
        lines.append("Latency Cache + Prewarm Metrics:")
        lines.append(
            (
                f"- llm_cache: enabled={cache.get('enabled', False)}, available={cache.get('available', False)}, "
                f"size={int(cache.get('size') or 0)}/{int(cache.get('max_size') or 0)}, "
                f"lookups={int(cache.get('lookups') or 0)}, hits={int(cache.get('hits') or 0)}, "
                f"misses={int(cache.get('misses') or 0)}, hit_rate={float(cache.get('hit_rate') or 0.0):.2%}"
            )
        )
        lines.append(
            (
                f"- tts_prewarm: attempts={int(prewarm.get('count') or 0)}, "
                f"warm_hits={int(prewarm.get('warm_hits') or 0)}, "
                f"hit_rate={float(prewarm.get('hit_rate') or 0.0):.2%}, "
                f"p50={float(prewarm.get('p50_ms') or 0.0):.1f}ms, "
                f"p95={float(prewarm.get('p95_ms') or 0.0):.1f}ms"
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


class LatencyTracker:
    """Per-stage latency recorder with budget enforcement.

    Records durations for named pipeline stages and warns when a stage
    exceeds its budget.  Thread-safe.  Module-level singleton: ``latency_tracker``.
    """

    _budgets: dict = {
        "wake_to_stt_start": 0.1,
        "stt_partial_latency": 0.5,
        "stt_total": 1.0,
        "intent_detection": 0.02,
        "action_execution": 0.1,
        "llm_first_token": 1.5,
        "tts_first_word": 0.3,
        "e2e_command": 1.5,
        "e2e_llm_query": 4.0,
    }

    def __init__(self) -> None:
        self._stages: OrderedDict = OrderedDict()
        self._lock = threading.Lock()

    def record(self, stage: str, duration_seconds: float) -> None:
        """Append a duration for *stage*; warn if it exceeds the stage budget."""
        key = str(stage or "").strip()
        if not key:
            return
        duration = max(0.0, float(duration_seconds))
        with self._lock:
            if key not in self._stages:
                self._stages[key] = []
            self._stages[key].append(duration)
        budget = self._budgets.get(key)
        if budget is not None and duration > budget:
            _logger.warning(
                "LatencyTracker: stage '%s' took %.0fms (budget: %.0fms)",
                key,
                duration * 1000,
                budget * 1000,
            )

    def report(self) -> dict:
        """Return per-stage stats: count/avg_ms/p50_ms/p95_ms/max_ms/budget_ms."""
        with self._lock:
            snapshot = {k: list(v) for k, v in self._stages.items()}
        result: dict = {}
        for stage, durations in snapshot.items():
            if not durations:
                continue
            sorted_d = sorted(durations)
            count = len(sorted_d)
            avg_ms = (sum(sorted_d) / count) * 1000
            p50_ms = (_percentile(sorted_d, 50) or 0.0) * 1000
            p95_ms = (_percentile(sorted_d, 95) or 0.0) * 1000
            max_ms = sorted_d[-1] * 1000
            budget_ms = (self._budgets.get(stage) or 0.0) * 1000
            result[stage] = {
                "count": count,
                "avg_ms": round(avg_ms, 1),
                "p50_ms": round(p50_ms, 1),
                "p95_ms": round(p95_ms, 1),
                "max_ms": round(max_ms, 1),
                "budget_ms": round(budget_ms, 1),
            }
        return result

    def reset(self) -> None:
        with self._lock:
            self._stages = OrderedDict()


latency_tracker = LatencyTracker()
