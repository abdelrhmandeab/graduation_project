import random
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_parser import ParsedCommand, parse_command


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _random_text():
    size = random.randint(0, 240)
    alphabet = string.ascii_letters + string.digits + string.punctuation + " \t"
    return "".join(random.choice(alphabet) for _ in range(size))


def test_parser_fuzz_stability():
    random.seed(1337)
    for _ in range(2000):
        text = _random_text()
        parsed = parse_command(text)
        _assert(isinstance(parsed, ParsedCommand), "Parser returned an unexpected type")
        _assert(isinstance(parsed.intent, str) and parsed.intent != "", "Intent must be non-empty string")
        _assert(parsed.raw == (text or ""), "Raw text should be preserved")
        _assert(parsed.normalized == " ".join(parsed.raw.lower().split()).strip(), "Normalization mismatch")


def test_parser_known_commands():
    samples = {
        "confirm abc123 2468": "OS_CONFIRMATION",
        "verify audit log": "AUDIT_VERIFY",
        "policy profile strict": "POLICY_COMMAND",
        "batch add create folder demo": "BATCH_COMMAND",
        "index status": "SEARCH_INDEX_COMMAND",
        "queue job in 2 create folder x": "JOB_QUEUE_COMMAND",
        "job list": "JOB_QUEUE_COMMAND",
        "persona set formal": "PERSONA_COMMAND",
        "voice clone on": "VOICE_COMMAND",
        "stop speaking": "VOICE_COMMAND",
        "kb search privacy model": "KNOWLEDGE_BASE_COMMAND",
        "kb quality": "KNOWLEDGE_BASE_COMMAND",
        "kb sync c:\\docs": "KNOWLEDGE_BASE_COMMAND",
        "memory status": "MEMORY_COMMAND",
        "observability": "OBSERVABILITY_REPORT",
        "benchmark run": "BENCHMARK_COMMAND",
        "resilience demo": "BENCHMARK_COMMAND",
        "audit reseal": "AUDIT_RESEAL",
        "persona voice status": "PERSONA_COMMAND",
        "find file notes.txt": "OS_FILE_SEARCH",
        "can you open for me the c partition on my pc?": "OS_FILE_NAVIGATION",
        "hey jarvis can you open for me the sea partition on my pc?": "OS_FILE_NAVIGATION",
        "i want you to open the desktop file for me": "OS_FILE_NAVIGATION",
        "please shut down computer.": "OS_SYSTEM_COMMAND",
        "jarvis can you open app calculator": "OS_APP_OPEN",
        "افتح تطبيق المفكرة": "OS_APP_OPEN",
        "اعرض الاقراص": "OS_FILE_NAVIGATION",
        "اذهب الى سطح المكتب": "OS_FILE_NAVIGATION",
        "انشئ مجلد تجربة": "OS_FILE_NAVIGATION",
        "احذف ملف.txt": "OS_FILE_NAVIGATION",
        "انقل a.txt الى b.txt": "OS_FILE_NAVIGATION",
        "اعد تسمية old.txt الى new.txt": "OS_FILE_NAVIGATION",
        "تاكيد abc123 2468": "OS_CONFIRMATION",
        "ابحث عن ملف notes.txt في desktop": "OS_FILE_SEARCH",
        "اطفي الكمبيوتر": "OS_SYSTEM_COMMAND",
    }
    for text, expected_intent in samples.items():
        parsed = parse_command(text)
        _assert(
            parsed.intent == expected_intent,
            f"Unexpected intent for '{text}': {parsed.intent}",
        )


if __name__ == "__main__":
    test_parser_fuzz_stability()
    test_parser_known_commands()
    print("Parser fuzz tests passed.")
