import importlib.util
import traceback
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _check_module(name):
    ok = importlib.util.find_spec(name) is not None
    return ok, "installed" if ok else "missing"


def _print_check(name, ok, details):
    state = "OK" if ok else "FAIL"
    print(f"[{state}] {name}: {details}")


def run():
    print("Jarvis Realtime Doctor")
    print("----------------------")

    modules = [
        "numpy",
        "sounddevice",
        "openwakeword",
        "onnxruntime",
        "faster_whisper",
        "pyttsx3",
    ]
    for mod in modules:
        ok, details = _check_module(mod)
        _print_check(f"python module '{mod}'", ok, details)

    try:
        import sounddevice as sd

        devices = sd.query_devices()
        default = sd.default.device
        ok = len(devices) > 0
        _print_check("audio devices", ok, f"count={len(devices)} default={default}")
        if ok:
            print("Input devices:")
            shown = 0
            for idx, dev in enumerate(devices):
                if int(dev.get("max_input_channels", 0)) <= 0:
                    continue
                print(f"- {idx}: {dev.get('name')}")
                shown += 1
                if shown >= 12:
                    break
    except Exception as exc:
        _print_check("audio devices", False, str(exc))

    try:
        from audio.wake_word import _get_model

        model = _get_model()
        _print_check("wake-word model", True, f"loaded={list(model.models.keys())}")
    except Exception as exc:
        _print_check("wake-word model", False, str(exc))

    try:
        from audio.stt import _get_model

        model = _get_model()
        _print_check("stt model", True, f"loaded={type(model).__name__}")
    except Exception as exc:
        _print_check("stt model", False, str(exc))

    try:
        import subprocess

        result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            _print_check("ollama cli", True, result.stdout.strip() or result.stderr.strip())
        else:
            _print_check("ollama cli", False, result.stderr.strip() or f"return_code={result.returncode}")
    except Exception as exc:
        _print_check("ollama cli", False, str(exc))

    print("")
    print("If all checks are OK, run: python core\\orchestrator.py")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise
