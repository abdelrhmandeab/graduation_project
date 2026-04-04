import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Phase 8 regression combines bilingual parsing depth, adversarial safety,
# end-to-end route coverage, and explicit SLA gate checks.
TEST_SCRIPTS = [
    "tests/english_commands_smoke.py",
    "tests/arabic_commands_smoke.py",
    "tests/adversarial_safety_phase8.py",
    "tests/e2e_regression_phase8.py",
    "tests/performance_gates_phase8.py",
    "tests/phase7_smoke.py",
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
    print("\nPhase 8 Regression Summary")
    for script_path in TEST_SCRIPTS:
        print(f"- {script_path}: {durations[script_path]:.2f}s")
    print(f"- total: {total_elapsed:.2f}s")

    if failures:
        print("\nPhase 8 regression failed:")
        for script_path, return_code in failures:
            print(f"- {script_path} exited with code {return_code}")
        return 1

    print("\nPhase 8 regression passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
