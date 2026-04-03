import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Safety + clarification focused regression checks for Phase 4.
TEST_SCRIPTS = [
    "tests/safety_suite.py",
    "tests/intent_clarification_smoke.py",
    "tests/language_gate_smoke.py",
    "tests/arabic_commands_smoke.py",
]


def _run_script(script_path):
    script = PROJECT_ROOT / script_path
    command = [sys.executable, str(script)]

    print(f"==> Running {script_path}", flush=True)
    start = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - start
    status = "PASS" if completed.returncode == 0 else "FAIL"
    print(f"<== {status} {script_path} ({elapsed:.2f}s)", flush=True)
    return completed.returncode, elapsed


def main():
    failures = []
    durations = {}
    suite_start = time.perf_counter()

    for script_path in TEST_SCRIPTS:
        return_code, elapsed = _run_script(script_path)
        durations[script_path] = elapsed
        if return_code != 0:
            failures.append((script_path, return_code))

    total_elapsed = time.perf_counter() - suite_start
    print("\nPhase 4 Regression Summary")
    for script_path in TEST_SCRIPTS:
        print(f"- {script_path}: {durations[script_path]:.2f}s")
    print(f"- total: {total_elapsed:.2f}s")

    if failures:
        print("\nPhase 4 regression failed:")
        for script_path, return_code in failures:
            print(f"- {script_path} exited with code {return_code}")
        return 1

    print("\nPhase 4 regression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
