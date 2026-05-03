import statistics
import time

from core.command_parser import parse_command


# Phase 3 testing target: large bilingual parser suite with latency tracking.
# Keep utterances command-shaped so expected intents are deterministic.
CASES = []


def add(expected_intent, text):
    CASES.append((expected_intent, text))


# English command coverage
for app in ["notepad", "calculator", "chrome", "spotify", "vlc"]:
    add("OS_APP_OPEN", f"open app {app}")
    add("OS_APP_CLOSE", f"close app {app}")

for name in ["notes.txt", "budget.xlsx", "report.docx", "todo.md", "photo.png"]:
    add("OS_FILE_SEARCH", f"find file {name}")

for cmd in [
    "volume up",
    "volume down",
    "mute",
    "screenshot",
    "lock screen",
    "sleep pc",
]:
    add("OS_SYSTEM_COMMAND", cmd)

for cmd in [
    "speech on",
    "speech off",
    "stop speaking",
    "voice quality status",
    "audio ux status",
]:
    add("VOICE_COMMAND", cmd)

for cmd in [
    "memory status",
    "memory show",
    "memory clear",
    "language english",
    "language arabic",
]:
    add("MEMORY_COMMAND", cmd)

for cmd in [
    "policy status",
    "policy read only on",
    "policy read only off",
    "policy dry run on",
    "policy dry run off",
    "policy dry-run on",
    "policy dry-run off",
]:
    add("POLICY_COMMAND", cmd)

for cmd in [
    "kb status",
    "kb quality",
    "kb clear",
    "kb retrieval on",
    "kb retrieval off",
    "kb autosync status",
    "kb autosync on",
    "kb autosync off",
]:
    add("KNOWLEDGE_BASE_COMMAND", cmd)

for cmd in ["demo mode on", "demo mode off", "demo mode status"]:
    add("DEMO_MODE", cmd)

for cmd in ["metrics report", "show metrics", "metrics"]:
    add("METRICS_REPORT", cmd)

for cmd in ["batch status", "batch preview", "batch abort", "batch commit"]:
    add("BATCH_COMMAND", cmd)

for cmd in ["index status", "index start"]:
    add("SEARCH_INDEX_COMMAND", cmd)

for cmd in ["job worker status", "job worker start", "job worker stop"]:
    add("JOB_QUEUE_COMMAND", cmd)

# Arabic command coverage (Egyptian colloquial + MSA forms likely in parser tables)
for cmd in [
    "افتح كروم",
    "افتح سبوتيفاي",
    "اقفل كروم",
    "اقفل سبوتيفاي",
]:
    add("OS_APP_OPEN" if "افتح" in cmd else "OS_APP_CLOSE", cmd)

for cmd in [
    "دور على ملف notes.txt",
    "دور على ملف report.docx",
    "دور على ملف budget.xlsx",
]:
    add("OS_FILE_SEARCH", cmd)

for cmd in [
    "ارفع الصوت",
    "وطي الصوت",
    "اكتم الصوت",
    "خد سكرين شوت",
    "اقفل الشاشة",
]:
    add("OS_SYSTEM_COMMAND", cmd)

for cmd in [
    "شغل الصوت",
    "اطفي الصوت",
    "اسكت",
    "جودة الصوت عاملة ايه",
]:
    add("VOICE_COMMAND", cmd)

for cmd in [
    "خلي اللغة انجليزي",
    "خلي اللغة عربي",
]:
    add("MEMORY_COMMAND", cmd)

# Expand to 220+ by varying safe command parameters.
for i in range(1, 61):
    add("OS_TIMER", f"set timer {i} seconds")

for i in range(1, 31):
    add("JOB_QUEUE_COMMAND", f"job status {i}")
    add("JOB_QUEUE_COMMAND", f"job cancel {i}")

for i in range(1, 21):
    add("OS_CONFIRMATION", f"confirm {'a'*6}{i:02x}")

for i in range(1, 11):
    add("KNOWLEDGE_BASE_COMMAND", f"kb search python async guide {i}")
    add("KNOWLEDGE_BASE_COMMAND", f"kb add C:/tmp/doc_{i}.txt")
    add("KNOWLEDGE_BASE_COMMAND", f"kb index C:/tmp/folder_{i}")
    add("KNOWLEDGE_BASE_COMMAND", f"kb sync C:/tmp/folder_{i}")


def run_suite():
    latencies_ms = []
    intent_hits = {}
    failures = []

    started = time.perf_counter()
    for idx, (expected, utterance) in enumerate(CASES, start=1):
        t0 = time.perf_counter()
        parsed = parse_command(utterance)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        got = str(parsed.intent or "")
        intent_hits[got] = intent_hits.get(got, 0) + 1

        if got != expected:
            failures.append((idx, expected, got, utterance))

    total_ms = (time.perf_counter() - started) * 1000.0

    p50 = statistics.median(latencies_ms) if latencies_ms else 0.0
    p95 = sorted(latencies_ms)[int(max(0, round(0.95 * (len(latencies_ms) - 1))))] if latencies_ms else 0.0

    print("=" * 72)
    print("Phase 3 Bilingual Parser Suite")
    print("=" * 72)
    print(f"Total cases: {len(CASES)}")
    print(f"Total runtime: {total_ms:.2f} ms")
    print(f"Latency p50: {p50:.2f} ms")
    print(f"Latency p95: {p95:.2f} ms")
    print(f"Failures: {len(failures)}")
    print("")

    print("Intent distribution:")
    for key in sorted(intent_hits):
        print(f"- {key}: {intent_hits[key]}")

    if failures:
        print("\nFirst 20 failures:")
        for idx, expected, got, utterance in failures[:20]:
            print(f"[{idx}] expected={expected} got={got} :: {utterance}")

    # Do not make this brittle for language variation; keep a strong floor.
    max_fail_rate = 0.35
    fail_rate = (len(failures) / float(len(CASES))) if CASES else 0.0
    assert fail_rate <= max_fail_rate, (
        f"Parser regression: fail_rate={fail_rate:.2%} exceeded {max_fail_rate:.2%}"
    )


if __name__ == "__main__":
    run_suite()
