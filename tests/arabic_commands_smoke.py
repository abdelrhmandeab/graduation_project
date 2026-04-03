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


def test_arabic_utterance_intent_action_mapping():
    cases = [
        # App open
        {
            "utterance": "افتح تطبيق المفكرة",
            "intent": "OS_APP_OPEN",
            "action": "",
            "args": {"app_name": "المفكرة"},
        },
        {
            "utterance": "شغل تطبيق الآلة الحاسبة",
            "intent": "OS_APP_OPEN",
            "action": "",
            "args": {"app_name": "الآلة الحاسبة"},
        },
        {
            "utterance": "يا جارفيس افتح تطبيق الكاميرا",
            "intent": "OS_APP_OPEN",
            "action": "",
        },
        # Open filesystem targets via Arabic phrases
        {
            "utterance": "افتح سطح المكتب",
            "intent": "OS_FILE_NAVIGATION",
            "action": "list_directory",
        },
        {
            "utterance": "افتح التحميلات",
            "intent": "OS_FILE_NAVIGATION",
            "action": "list_directory",
        },
        {
            "utterance": "افتح المستندات",
            "intent": "OS_FILE_NAVIGATION",
            "action": "list_directory",
        },
        # Rollback
        {"utterance": "تراجع", "intent": "OS_ROLLBACK", "action": ""},
        {"utterance": "الغاء اخر عملية", "intent": "OS_ROLLBACK", "action": ""},
        {"utterance": "الغاء اخر امر", "intent": "OS_ROLLBACK", "action": ""},
        # PWD / drive listing
        {"utterance": "المجلد الحالي", "intent": "OS_FILE_NAVIGATION", "action": "pwd"},
        {"utterance": "اين انا", "intent": "OS_FILE_NAVIGATION", "action": "pwd"},
        {"utterance": "اعرض الاقراص", "intent": "OS_FILE_NAVIGATION", "action": "list_drives"},
        {"utterance": "اظهر الاقراص", "intent": "OS_FILE_NAVIGATION", "action": "list_drives"},
        {"utterance": "قائمة الاقراص", "intent": "OS_FILE_NAVIGATION", "action": "list_drives"},
        # Confirmation
        {"utterance": "تاكيد abc123", "intent": "OS_CONFIRMATION", "action": "", "args": {"token": "abc123"}},
        {
            "utterance": "تأكيد abc123 2468",
            "intent": "OS_CONFIRMATION",
            "action": "",
            "args": {"token": "abc123", "second_factor": "2468"},
        },
        # File search
        {"utterance": "ابحث عن ملف notes.txt", "intent": "OS_FILE_SEARCH", "action": ""},
        {"utterance": "ابحث ملف report.docx في desktop", "intent": "OS_FILE_SEARCH", "action": ""},
        {"utterance": "دور على ملف budget.xlsx في downloads", "intent": "OS_FILE_SEARCH", "action": ""},
        # List directory
        {"utterance": "اعرض الملفات", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "اظهر الملفات في desktop", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "اعرض المجلد في downloads", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        {"utterance": "اظهر المجلد", "intent": "OS_FILE_NAVIGATION", "action": "list_directory"},
        # File info
        {"utterance": "معلومات ملف notes.txt", "intent": "OS_FILE_NAVIGATION", "action": "file_info"},
        {"utterance": "بيانات ملف C:\\temp\\a.txt", "intent": "OS_FILE_NAVIGATION", "action": "file_info"},
        # Create folder
        {"utterance": "انشئ مجلد تجارب", "intent": "OS_FILE_NAVIGATION", "action": "create_directory"},
        {"utterance": "اعمل مجلد مشروع", "intent": "OS_FILE_NAVIGATION", "action": "create_directory"},
        {"utterance": "اصنع مجلد archive", "intent": "OS_FILE_NAVIGATION", "action": "create_directory"},
        # Delete
        {"utterance": "احذف notes.txt", "intent": "OS_FILE_NAVIGATION", "action": "delete_item"},
        {"utterance": "امسح old.log", "intent": "OS_FILE_NAVIGATION", "action": "delete_item"},
        {"utterance": "ازل temp.txt", "intent": "OS_FILE_NAVIGATION", "action": "delete_item"},
        {"utterance": "من فضلك احذف temp2.txt", "intent": "OS_FILE_NAVIGATION", "action": "delete_item"},
        # Move
        {"utterance": "انقل a.txt الى b.txt", "intent": "OS_FILE_NAVIGATION", "action": "move_item"},
        {"utterance": "انقل file1.txt إلى backup\\file1.txt", "intent": "OS_FILE_NAVIGATION", "action": "move_item"},
        {"utterance": "حرك report.docx الى archive\\report.docx", "intent": "OS_FILE_NAVIGATION", "action": "move_item"},
        # Rename
        {"utterance": "اعد تسمية old.txt الى new.txt", "intent": "OS_FILE_NAVIGATION", "action": "rename_item"},
        {"utterance": "غير اسم draft.txt إلى final.txt", "intent": "OS_FILE_NAVIGATION", "action": "rename_item"},
        {"utterance": "غيّر اسم a.txt الى b.txt", "intent": "OS_FILE_NAVIGATION", "action": "rename_item"},
        # CD commands in Arabic
        {"utterance": "اذهب الى desktop", "intent": "OS_FILE_NAVIGATION", "action": "cd"},
        {"utterance": "روح الى downloads", "intent": "OS_FILE_NAVIGATION", "action": "cd"},
        {"utterance": "انتقل إلى documents", "intent": "OS_FILE_NAVIGATION", "action": "cd"},
        # System actions in Arabic
        {"utterance": "اطفي الكمبيوتر", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        {"utterance": "اعادة تشغيل", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        {"utterance": "قفل الكمبيوتر", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        {"utterance": "سجل خروج", "intent": "OS_SYSTEM_COMMAND", "action": ""},
        {"utterance": "من فضلك اغلق الكمبيوتر", "intent": "OS_SYSTEM_COMMAND", "action": ""},
    ]

    _assert(len(cases) >= 30, f"Expected at least 30 cases, found {len(cases)}")
    for case in cases:
        _check_case(
            utterance=case["utterance"],
            expected_intent=case["intent"],
            expected_action=case.get("action", ""),
            expected_args_subset=case.get("args"),
        )


if __name__ == "__main__":
    test_arabic_utterance_intent_action_mapping()
    print("Arabic commands smoke tests passed.")
