import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def run_once(model, prompt, timeout_seconds, num_ctx, latency_budget_ms):
    payload = {
        "model": str(model),
        "prompt": str(prompt),
        "stream": False,
        "options": {
            "num_ctx": int(num_ctx),
        },
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        print(f"STATUS=ERROR")
        print(f"ERROR=connection_failed:{exc}")
        return 3
    except Exception as exc:
        print("STATUS=ERROR")
        print(f"ERROR=request_failed:{exc}")
        return 3

    elapsed_ms = int((time.perf_counter() - started) * 1000)

    try:
        parsed = json.loads(body)
    except Exception:
        print("STATUS=ERROR")
        print("ERROR=invalid_json_response")
        print(f"ELAPSED_MS={elapsed_ms}")
        return 3

    response_text = str(parsed.get("response") or "").strip()
    done = bool(parsed.get("done"))

    print(f"MODEL={parsed.get('model')}")
    print(f"ELAPSED_MS={elapsed_ms}")
    print(f"LATENCY_BUDGET_MS={int(latency_budget_ms)}")
    print(f"DONE={done}")
    print(f"RESPONSE_CHARS={len(response_text)}")

    if not response_text:
        print("STATUS=FAIL")
        print("REASON=empty_response")
        return 2

    if elapsed_ms > int(latency_budget_ms):
        print("STATUS=FAIL")
        print("REASON=latency_budget_exceeded")
        return 2

    print("STATUS=PASS")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Phase 1 latency smoke check (single Ollama call)")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument(
        "--prompt",
        default="Give a short one-line summary about Egypt today.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=45.0)
    parser.add_argument("--num-ctx", type=int, default=2048)
    parser.add_argument("--latency-budget-ms", type=int, default=35000)
    args = parser.parse_args()

    exit_code = run_once(
        model=args.model,
        prompt=args.prompt,
        timeout_seconds=args.timeout_seconds,
        num_ctx=args.num_ctx,
        latency_budget_ms=args.latency_budget_ms,
    )
    return int(exit_code)


if __name__ == "__main__":
    sys.exit(main())
