import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.command_router import route_command
from core.language_gate import UNSUPPORTED_LANGUAGE_MESSAGE, detect_supported_language
from core.session_memory import SessionMemory, session_memory


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


def test_detect_supported_languages():
    en = detect_supported_language("open calculator")
    _assert(en.supported, "English text should be supported")
    _assert(en.language == "en", f"Expected en, got {en.language}")

    ar = detect_supported_language("أهلا وسهلا")
    _assert(ar.supported, "Arabic text should be supported")
    _assert(ar.language == "ar", f"Expected ar, got {ar.language}")
    _assert(ar.normalized_text == "اهلا وسهلا", f"Unexpected Arabic normalization: {ar.normalized_text}")


def test_detect_unsupported_script():
    unsupported = detect_supported_language("привет как дела")
    _assert(not unsupported.supported, "Cyrillic-only text should be blocked")
    _assert(unsupported.language == "unsupported", f"Unexpected language tag: {unsupported.language}")


def test_detect_unsupported_script_dominant_mixed_text():
    unsupported = detect_supported_language("привет мир привет open")
    _assert(not unsupported.supported, "Unsupported-script dominant text should be blocked")
    _assert(unsupported.reason == "unsupported_script_dominant", f"Unexpected reason: {unsupported.reason}")


def test_mixed_script_tie_uses_previous_language():
    tie_ar = detect_supported_language("abc ابت", previous_language="ar")
    _assert(tie_ar.supported, "Mixed tie should still be supported")
    _assert(tie_ar.language == "ar", f"Expected previous language ar, got {tie_ar.language}")

    tie_en = detect_supported_language("abc ابت", previous_language="en")
    _assert(tie_en.supported, "Mixed tie should still be supported")
    _assert(tie_en.language == "en", f"Expected previous language en, got {tie_en.language}")


def test_route_blocks_unsupported_script():
    response = route_command("привет как дела")
    _assert(
        response == UNSUPPORTED_LANGUAGE_MESSAGE,
        f"Unexpected gate response: {response}",
    )


def test_preferred_language_persistence():
    original = session_memory.get_preferred_language()
    try:
        session_memory.set_preferred_language("ar")
        reloaded = SessionMemory()
        _assert(
            reloaded.get_preferred_language() == "ar",
            "Preferred language should persist to disk as 'ar'",
        )

        session_memory.set_preferred_language("en")
        reloaded = SessionMemory()
        _assert(
            reloaded.get_preferred_language() == "en",
            "Preferred language should persist to disk as 'en'",
        )
    finally:
        session_memory.set_preferred_language(original)


if __name__ == "__main__":
    test_detect_supported_languages()
    test_detect_unsupported_script()
    test_detect_unsupported_script_dominant_mixed_text()
    test_mixed_script_tie_uses_previous_language()
    test_route_blocks_unsupported_script()
    test_preferred_language_persistence()
    print("Language gate smoke tests passed.")
