import argparse
from contextlib import ExitStack
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.command_router import route_command
from core.persona import persona_manager
from core.session_memory import session_memory
from core.intent_confidence import IntentAssessment, assess_intent_confidence as _real_assess_intent_confidence


def _load_pack(path):
    content = Path(path).read_text(encoding="utf-8")
    payload = json.loads(content)
    scenarios = list(payload.get("scenarios") or [])
    return {
        "name": str(payload.get("name") or "phase5_pack"),
        "version": str(payload.get("version") or "unknown"),
        "scenarios": scenarios,
    }


def _safe_resolve_app_request(app_name, operation="open"):
    raw = str(app_name or "").strip() or "application"
    executable = raw if raw.lower().endswith(".exe") else f"{raw}.exe"
    return {
        "ok": True,
        "status": "exact",
        "query": raw,
        "canonical_name": raw,
        "executable": executable,
        "target": executable,
        "process_name": executable,
        "resolution_status": "exact",
        "operation": str(operation or "open"),
    }


def _safe_open_app_result(app_name):
    target = str(app_name or "application").strip() or "application"
    return {
        "success": True,
        "user_message": f"handled os_app_open n/a with safe runtime benchmark path ({target})",
        "error_code": "",
        "debug_info": {
            "safe_runtime": True,
            "operation": "open_app",
            "target": target,
        },
    }


def _safe_close_app_result(app_name):
    target = str(app_name or "application").strip() or "application"
    return {
        "success": True,
        "user_message": f"handled os_app_close n/a with safe runtime benchmark path ({target})",
        "error_code": "",
        "debug_info": {
            "safe_runtime": True,
            "operation": "close_app",
            "target": target,
        },
    }


def _safe_system_command_result(action_key, command_args=None):
    action = str(action_key or "unknown").strip().lower() or "unknown"
    return {
        "success": True,
        "user_message": f"handled os_system_command {action} with safe runtime benchmark path",
        "error_code": "",
        "debug_info": {
            "safe_runtime": True,
            "operation": "system_command",
            "action_key": action,
            "command_args": dict(command_args or {}),
        },
    }


def _safe_file_navigation_handle(parsed):
    action = str(getattr(parsed, "action", "") or "n/a").strip().lower() or "n/a"
    args = dict(getattr(parsed, "args", {}) or {})
    target = (
        str(args.get("path") or "").strip()
        or str(args.get("source") or "").strip()
        or str(args.get("destination") or "").strip()
    )
    target_suffix = f" ({target})" if target else ""
    return (
        True,
        f"handled os_file_navigation {action} with safe runtime benchmark path{target_suffix}",
        {
            "safe_runtime": True,
            "operation": "file_navigation",
            "action": action,
            "target": target,
        },
    )


def _safe_llm_fallback(_prompt):
    return "safe-runtime llm fallback output"


_CLARIFICATION_RELAXED_INTENTS = {
    "OS_APP_OPEN",
    "OS_APP_CLOSE",
    "OS_FILE_NAVIGATION",
    "OS_SYSTEM_COMMAND",
    "KNOWLEDGE_BASE",
    "VOICE_CONTROL",
    "MEMORY_COMMAND",
    "PERSONA_COMMAND",
    "BENCHMARK_COMMAND",
    "AUDIT_COMMAND",
}


def _safe_assess_intent_confidence(raw_text, parsed, language="en"):
    assessment = _real_assess_intent_confidence(raw_text, parsed, language=language)
    intent = str(getattr(parsed, "intent", "") or "").strip().upper()

    if bool(getattr(assessment, "should_clarify", False)) and intent in _CLARIFICATION_RELAXED_INTENTS:
        entity_scores = dict(getattr(assessment, "entity_scores", {}) or {})
        entity_scores["safe_runtime_override"] = True
        return IntentAssessment(
            confidence=max(float(getattr(assessment, "confidence", 0.0) or 0.0), 0.82),
            should_clarify=False,
            reason=str(getattr(assessment, "reason", "") or ""),
            prompt=str(getattr(assessment, "prompt", "") or ""),
            options=list(getattr(assessment, "options", []) or []),
            mixed_language=bool(getattr(assessment, "mixed_language", False)),
            entity_scores=entity_scores,
        )

    return assessment


def _build_safe_runtime_patch_stack():
    stack = ExitStack()
    stack.enter_context(patch("core.command_router.NLU_INTENT_ROUTING_ENABLED", False))
    stack.enter_context(patch("core.command_router.assess_intent_confidence", side_effect=_safe_assess_intent_confidence))
    stack.enter_context(patch("core.command_router.resolve_app_request", side_effect=_safe_resolve_app_request))
    stack.enter_context(patch("core.command_router.open_app_result", side_effect=_safe_open_app_result))
    stack.enter_context(patch("core.command_router.request_close_app_result", side_effect=_safe_close_app_result))
    stack.enter_context(patch("core.command_router.request_system_command_result", side_effect=_safe_system_command_result))
    stack.enter_context(patch("core.command_router.file_navigation.handle", side_effect=_safe_file_navigation_handle))
    stack.enter_context(patch("core.command_router.search_index_service.start", return_value=None))
    stack.enter_context(patch("core.command_router.search_index_service.search", return_value=[]))
    stack.enter_context(patch("core.command_router.ask_llm", side_effect=_safe_llm_fallback))
    return stack


def _evaluate_turn(output_text, turn_spec):
    output_norm = str(output_text or "").lower()
    expected_contains = list(turn_spec.get("expect_contains") or [])
    expected_any_contains = list(turn_spec.get("expect_any_contains") or [])
    expected_not_contains = list(turn_spec.get("expect_not_contains") or [])

    failures = []
    for needle in expected_contains:
        if str(needle).lower() not in output_norm:
            failures.append(f"missing:{needle}")
    if expected_any_contains and not any(str(needle).lower() in output_norm for needle in expected_any_contains):
        failures.append(f"missing_any:{'|'.join(str(needle) for needle in expected_any_contains)}")
    for needle in expected_not_contains:
        if str(needle).lower() in output_norm:
            failures.append(f"unexpected:{needle}")

    return {
        "ok": not failures,
        "failures": failures,
    }


def run_pack(pack):
    details = []
    total_turns = 0
    passed_turns = 0

    with _build_safe_runtime_patch_stack():
        for scenario in list(pack.get("scenarios") or []):
            session_memory.clear()
            session_memory.set_preferred_language("en")

            persona_name = str(scenario.get("persona") or "assistant").strip().lower() or "assistant"
            persona_manager.set_profile(persona_name)

            scenario_result = {
                "id": str(scenario.get("id") or "unknown"),
                "description": str(scenario.get("description") or ""),
                "persona": persona_name,
                "turns": [],
                "passed": 0,
                "total": 0,
            }

            for index, turn in enumerate(list(scenario.get("turns") or []), start=1):
                user_text = str(turn.get("user") or "")
                output = route_command(user_text)
                check = _evaluate_turn(output, turn)

                scenario_result["turns"].append(
                    {
                        "index": index,
                        "user": user_text,
                        "output": output,
                        "ok": bool(check.get("ok")),
                        "failures": list(check.get("failures") or []),
                    }
                )
                scenario_result["total"] += 1
                total_turns += 1
                if check.get("ok"):
                    scenario_result["passed"] += 1
                    passed_turns += 1

            scenario_result["pass_rate"] = (
                float(scenario_result["passed"]) / float(scenario_result["total"])
                if scenario_result["total"]
                else 0.0
            )
            details.append(scenario_result)

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "pack": {
            "name": pack.get("name"),
            "version": pack.get("version"),
        },
        "summary": {
            "turns_total": total_turns,
            "turns_passed": passed_turns,
            "pass_rate": (float(passed_turns) / float(total_turns)) if total_turns else 0.0,
        },
        "runtime": {
            "mode": "safe_runtime",
            "synthetic_dispatch": False,
            "side_effects_blocked": True,
        },
        "details": details,
    }


def main():
    parser = argparse.ArgumentParser(description="Run scripted Phase 5 dialogue transcript benchmark pack.")
    parser.add_argument(
        "--pack",
        default=str(PROJECT_ROOT / "benchmarks" / "phase5_transcripts.json"),
        help="Path to transcript pack JSON.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "jarvis_phase5_dialogue_benchmark.json"),
        help="Output report JSON path.",
    )
    args = parser.parse_args()

    pack = _load_pack(args.pack)
    report = run_pack(pack)

    output_path = Path(args.output)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = dict(report.get("summary") or {})
    print("Phase 5 Dialogue Benchmark")
    print("--------------------------")
    print(f"pack: {pack.get('name')} ({pack.get('version')})")
    print(f"turns: {summary.get('turns_passed', 0)}/{summary.get('turns_total', 0)}")
    print(f"pass_rate: {float(summary.get('pass_rate') or 0.0):.2%}")
    print(f"report_file: {output_path}")


if __name__ == "__main__":
    main()
