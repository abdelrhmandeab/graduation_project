import importlib.util
import traceback
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_MODULES = (
    "numpy",
    "sounddevice",
    "openwakeword",
    "faster_whisper",
    "pyttsx3",
)

OPTIONAL_MODULES = (
    "edge_tts",
)


def _check_module(name):
    ok = importlib.util.find_spec(name) is not None
    return ok, "installed" if ok else "missing"


def _print_check(name, ok, details):
    state = "OK" if ok else "FAIL"
    print(f"[{state}] {name}: {details}")


def collect_diagnostics(*, include_model_load_checks=False):
    checks = []

    for module_name in REQUIRED_MODULES:
        ok, details = _check_module(module_name)
        checks.append(
            {
                "name": f"python_module:{module_name}",
                "ok": bool(ok),
                "details": details,
                "required": True,
            }
        )

    for module_name in OPTIONAL_MODULES:
        ok, details = _check_module(module_name)
        checks.append(
            {
                "name": f"python_module_optional:{module_name}",
                "ok": True,
                "details": details if ok else "missing (optional)",
                "required": False,
            }
        )

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default = sd.default.device
        checks.append(
            {
                "name": "audio_devices",
                "ok": bool(len(devices) > 0),
                "details": f"count={len(devices)} default={default}",
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "audio_devices",
                "ok": False,
                "details": str(exc),
            }
        )

    try:
        from audio.wake_word import _get_model

        if include_model_load_checks:
            model = _get_model()
            details = f"loaded={list(getattr(model, 'models', {}).keys())}"
        else:
            details = "check_skipped(model_load=False)"
        checks.append(
            {
                "name": "wake_word_model",
                "ok": True,
                "details": details,
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "wake_word_model",
                "ok": False,
                "details": str(exc),
            }
        )

    try:
        from audio import stt as stt_runtime

        backend = stt_runtime.get_runtime_stt_backend()
        if include_model_load_checks and backend == "faster_whisper":
            _ = stt_runtime._get_whisper_model()
            details = f"backend={backend} model_load=ok"
        else:
            details = f"backend={backend} model_load_skipped={not include_model_load_checks}"
        checks.append(
            {
                "name": "stt_runtime",
                "ok": True,
                "details": details,
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "stt_runtime",
                "ok": False,
                "details": str(exc),
            }
        )

    try:
        import subprocess

        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        ok = result.returncode == 0
        details = (result.stdout or result.stderr or "").strip() or f"return_code={result.returncode}"
        checks.append(
            {
                "name": "ollama_cli",
                "ok": bool(ok),
                "details": details,
            }
        )
    except Exception as exc:
        checks.append(
            {
                "name": "ollama_cli",
                "ok": False,
                "details": str(exc),
            }
        )

    ok_count = sum(1 for row in checks if row.get("ok"))
    required_checks = [row for row in checks if row.get("required", True)]
    required_ok_count = sum(1 for row in required_checks if row.get("ok"))
    return {
        "timestamp": time.time(),
        "ok": bool(required_ok_count == len(required_checks)),
        "check_count": len(checks),
        "ok_count": ok_count,
        "required_check_count": len(required_checks),
        "required_ok_count": required_ok_count,
        "checks": checks,
    }


def format_diagnostics_report(payload):
    lines = [
        "Jarvis Realtime Doctor",
        "----------------------",
        f"overall_ok: {payload.get('ok')}",
        (
            f"required_checks: {payload.get('required_ok_count')}/"
            f"{payload.get('required_check_count')}"
        ),
        f"all_checks: {payload.get('ok_count')}/{payload.get('check_count')}",
        "",
    ]
    for row in payload.get("checks", []):
        state = "OK" if row.get("ok") else "FAIL"
        lines.append(f"[{state}] {row.get('name')}: {row.get('details')}")
    lines.append("")
    lines.append("If all checks are OK, run: python core\\orchestrator.py")
    return "\n".join(lines)


def run():
    payload = collect_diagnostics(include_model_load_checks=True)
    print(format_diagnostics_report(payload))


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
