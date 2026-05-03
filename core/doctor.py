import importlib.util
import traceback
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import (
    ELEVENLABS_API_KEY,
    LLM_MODEL,
    STT_BACKEND,
    TTS_DEFAULT_BACKEND,
    TTS_ELEVENLABS_ARABIC_ENABLED,
)


REQUIRED_MODULES = (
    "numpy",
    "sounddevice",
    "openwakeword",
    "psutil",
    "httpx",
    "rapidfuzz",
)

OPTIONAL_MODULES = (
    "edge_tts",
    "faster_whisper",
    # Phase 2: live data
    "duckduckgo_search",
    "ddgs",
    # Phase 3: desktop commands
    "pyperclip",
    "screen_brightness_control",
    "win32com",
    # Phase 4: semantic intent routing
    "sentence_transformers",
)


def _check_module(name):
    try:
        ok = importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError):
        ok = False
    return ok, "installed" if ok else "missing"


def _print_check(name, ok, details):
    state = "OK" if ok else "FAIL"
    print(f"[{state}] {name}: {details}")


def _probe_ollama_models():
    try:
        import subprocess

        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=12)
        if result.returncode != 0:
            return False, f"ollama list failed: {(result.stderr or '').strip() or 'unknown error'}"

        lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if len(lines) <= 1:
            return True, "no local models listed"

        models = []
        for line in lines[1:]:
            model_name = line.split()[0].strip() if line.split() else ""
            if model_name:
                models.append(model_name)

        configured = str(LLM_MODEL or "").strip()
        configured_ok = True
        if configured:
            configured_ok = any(configured in item for item in models)

        summary = f"models={models[:6]} total={len(models)} configured_present={configured_ok}"
        return bool(configured_ok), summary
    except Exception as exc:
        return False, f"ollama list probe failed: {exc}"


def _probe_vram_status():
    try:
        import subprocess

        cmd = [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
        if result.returncode != 0:
            return True, "nvidia-smi unavailable (CPU-only or non-NVIDIA)"

        rows = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
        if not rows:
            return True, "nvidia-smi returned no GPU rows"

        gpu_summaries = []
        for row in rows[:2]:
            parts = [part.strip() for part in row.split(",")]
            if len(parts) < 4:
                continue
            name = parts[0]
            total_mb = parts[1]
            free_mb = parts[2]
            used_mb = parts[3]
            gpu_summaries.append(f"{name}: total={total_mb}MB free={free_mb}MB used={used_mb}MB")

        return True, "; ".join(gpu_summaries) if gpu_summaries else "GPU info parse failed"
    except Exception as exc:
        return True, f"VRAM probe skipped: {exc}"


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

    elevenlabs_key_configured = bool(str(ELEVENLABS_API_KEY or "").strip())
    elevenlabs_required = bool(
        str(STT_BACKEND or "").strip().lower() == "hybrid_elevenlabs"
        or (
            str(TTS_DEFAULT_BACKEND or "").strip().lower() == "hybrid"
            and bool(TTS_ELEVENLABS_ARABIC_ENABLED)
        )
    )
    checks.append(
        {
            "name": "elevenlabs_api_key",
            "ok": bool(elevenlabs_key_configured or not elevenlabs_required),
            "details": (
                f"configured={elevenlabs_key_configured} required={elevenlabs_required}"
            ),
            "required": bool(elevenlabs_required),
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
        if include_model_load_checks:
            preload_snapshot = stt_runtime.preload_runtime_models()
            details = f"backend={backend} preload={preload_snapshot}"
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

    models_ok, models_details = _probe_ollama_models()
    checks.append(
        {
            "name": "ollama_models",
            "ok": bool(models_ok),
            "details": models_details,
            "required": False,
        }
    )

    vram_ok, vram_details = _probe_vram_status()
    checks.append(
        {
            "name": "gpu_vram",
            "ok": bool(vram_ok),
            "details": vram_details,
            "required": False,
        }
    )

    feature_tiers = _summarize_feature_tiers(checks)
    checks.extend(feature_tiers)

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


def _summarize_feature_tiers(checks):
    """Build feature-availability rows so users can see what's degraded.

    Each row marks ok=True since these are informational; required=False so
    they never trip the overall ok flag.
    """
    by_module = {}
    for row in checks:
        name = str(row.get("name") or "")
        if name.startswith("python_module_optional:") or name.startswith("python_module:"):
            mod = name.split(":", 1)[1]
            details = str(row.get("details") or "")
            by_module[mod] = "installed" in details

    tiers = []
    web_search_ok = bool(by_module.get("ddgs") or by_module.get("duckduckgo_search"))
    tiers.append({
        "name": "feature:web_search",
        "ok": True,
        "details": "available" if web_search_ok else "degraded (no DDGS package)",
        "required": False,
    })
    tiers.append({
        "name": "feature:clipboard",
        "ok": True,
        "details": "available" if by_module.get("pyperclip") else "degraded (pyperclip missing)",
        "required": False,
    })
    tiers.append({
        "name": "feature:brightness_python",
        "ok": True,
        "details": (
            "available"
            if by_module.get("screen_brightness_control")
            else "degraded (PowerShell fallback only)"
        ),
        "required": False,
    })
    tiers.append({
        "name": "feature:outlook_com",
        "ok": True,
        "details": (
            "available"
            if by_module.get("win32com")
            else "degraded (email/calendar/Windows Search Index unavailable)"
        ),
        "required": False,
    })
    tiers.append({
        "name": "feature:semantic_router",
        "ok": True,
        "details": (
            "available"
            if by_module.get("sentence_transformers")
            else "degraded (regex + keyword NLP only)"
        ),
        "required": False,
    })
    return tiers


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
    lines.append("If all checks are OK, run: python main.py")
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
