import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_parser import parse_command


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def _safe_text(text):
    return (text or "").encode("unicode_escape").decode("ascii")


def _check_case(utterance, expected_intent, expected_action="", expected_args_subset=None):
    parsed = parse_command(utterance)
    label = _safe_text(utterance)
    _assert(
        parsed.intent == expected_intent,
        f"Unexpected intent for {label}: got={parsed.intent} expected={expected_intent}",
    )
    _assert(
        parsed.action == expected_action,
        f"Unexpected action for {label}: got={parsed.action} expected={expected_action}",
    )
    if expected_args_subset:
        for key, value in expected_args_subset.items():
            actual = parsed.args.get(key)
            _assert(
                actual == value,
                f"Unexpected arg for {label}: key={key} got={actual} expected={value}",
            )


def test_english_utterance_intent_action_mapping():
    cases = [
        # App open
        {"utterance": "open app notepad", "intent": "OS_APP_OPEN", "action": "", "args": {"app_name": "notepad"}},
        {"utterance": "launch calculator", "intent": "OS_APP_OPEN", "action": ""},
        {"utterance": "start camera", "intent": "OS_APP_OPEN", "action": ""},
        {"utterance": "jarvis open app settings", "intent": "OS_APP_OPEN", "action": ""},
        # App close
        {"utterance": "close app notepad", "intent": "OS_APP_CLOSE", "action": ""},
        {"utterance": "please close application calc", "intent": "OS_APP_CLOSE", "action": ""},
        # Navigation and listing
        {"utterance": "current directory", "intent": "OS_FILE_NAVIGATION", "action": "pwd"},
        {"utterance": "pwd", "intent": "OS_FILE_NAVIGATION", "action": "pwd"},
        {"utterance": "list drives", "intent": "OS_FILE_NAVIGATION", "action": "list_drives"},
        {"utterance": "drive list", "intent": "OS_FILE_NAVIGATION", "action": "list_drives"},
        {"utterance": "list files", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "show directory", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "go to desktop", "intent": "OS_FILE_NAVIGATION", "action": "cd"},
        {"utterance": "change directory downloads", "intent": "OS_FILE_NAVIGATION", "action": "cd"},
        {"utterance": "cd C:\\Temp", "intent": "OS_FILE_NAVIGATION", "action": "cd"},
        {"utterance": "open desktop", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "open downloads", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "open documents", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        # File ops
        {"utterance": "file info notes.txt", "intent": "OS_FILE_NAVIGATION", "action": "file_info"},
        {"utterance": "metadata C:\\temp\\a.txt", "intent": "OS_FILE_NAVIGATION", "action": "file_info"},
        {"utterance": "create folder qa_pack", "intent": "OS_FILE_NAVIGATION", "action": "create_directory"},
        {"utterance": "mkdir qa_pack", "intent": "OS_FILE_NAVIGATION", "action": "create_directory"},
        {"utterance": "delete notes.txt", "intent": "OS_FILE_NAVIGATION", "action": "delete_item"},
        {"utterance": "remove old.log", "intent": "OS_FILE_NAVIGATION", "action": "delete_item"},
        {"utterance": "delete permanently danger.txt", "intent": "OS_FILE_NAVIGATION", "action": "delete_item_permanent"},
        {"utterance": "move a.txt to b.txt", "intent": "OS_FILE_NAVIGATION", "action": "move_item"},
        {"utterance": "rename old.txt to new.txt", "intent": "OS_FILE_NAVIGATION", "action": "rename_item"},
        # Search
        {"utterance": "find file notes.txt", "intent": "OS_FILE_SEARCH", "action": ""},
        {"utterance": "search file report.docx in desktop", "intent": "OS_FILE_SEARCH", "action": ""},
        {"utterance": "i want to find file budget in downloads", "intent": "OS_FILE_SEARCH", "action": ""},
        {"utterance": "look for release_notes inside documents", "intent": "OS_FILE_SEARCH", "action": ""},
        # Confirmation and rollback
        {"utterance": "confirm abc123", "intent": "OS_CONFIRMATION", "action": "", "args": {"token": "abc123"}},
        {
            "utterance": "confirm abc123 2468",
            "intent": "OS_CONFIRMATION",
            "action": "",
            "args": {"token": "abc123", "second_factor": "2468"},
        },
        {"utterance": "undo", "intent": "OS_ROLLBACK", "action": ""},
        {"utterance": "rollback", "intent": "OS_ROLLBACK", "action": ""},
        # System
        {"utterance": "shutdown computer", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        {"utterance": "restart computer", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        {"utterance": "lock computer", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        # Voice
        {"utterance": "voice status", "intent": "VOICE_COMMAND", "action": "status"},
        {"utterance": "audio ux status", "intent": "VOICE_COMMAND", "action": "audio_ux_status"},
        {"utterance": "audio ux profiles", "intent": "VOICE_COMMAND", "action": "audio_ux_profiles"},
        {"utterance": "audio ux profile responsive", "intent": "VOICE_COMMAND", "action": "audio_ux_profile_set", "args": {"profile": "responsive"}},
        {"utterance": "voice quality natural", "intent": "VOICE_COMMAND", "action": "voice_quality_set", "args": {"mode": "natural"}},
        {"utterance": "stt profile noisy", "intent": "VOICE_COMMAND", "action": "stt_profile_set", "args": {"profile": "noisy"}},
        {"utterance": "stop speaking", "intent": "VOICE_COMMAND", "action": "interrupt"},
        # Memory / observability / benchmark
        {"utterance": "memory status", "intent": "MEMORY_COMMAND", "action": "status"},
        {"utterance": "memory clear", "intent": "MEMORY_COMMAND", "action": "clear"},
        {"utterance": "show metrics", "intent": "METRICS_REPORT", "action": ""},
        {"utterance": "observability", "intent": "OBSERVABILITY_REPORT", "action": ""},
        {"utterance": "benchmark run", "intent": "BENCHMARK_COMMAND", "action": "run"},
        {"utterance": "resilience demo", "intent": "BENCHMARK_COMMAND", "action": "resilience_demo"},
        # Policy and audit
        {"utterance": "policy status", "intent": "POLICY_COMMAND", "action": "status"},
        {"utterance": "policy profile strict", "intent": "POLICY_COMMAND", "action": "set_profile", "args": {"profile": "strict"}},
        {"utterance": "verify audit log", "intent": "AUDIT_VERIFY", "action": ""},
        {"utterance": "audit reseal", "intent": "AUDIT_RESEAL", "action": ""},
    ]

    _assert(len(cases) >= 40, f"Expected at least 40 cases, found {len(cases)}")
    for case in cases:
        _check_case(
            utterance=case["utterance"],
            expected_intent=case["intent"],
            expected_action=case.get("action", ""),
            expected_args_subset=case.get("args"),
        )


if __name__ == "__main__":
    test_english_utterance_intent_action_mapping()
    print("English commands smoke tests passed.")
