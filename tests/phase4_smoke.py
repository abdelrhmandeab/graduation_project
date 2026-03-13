import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from audio.tts import speech_engine
from core.command_router import route_command
from llm.prompt_builder import build_prompt
from os_control.policy import policy_engine


def _assert(condition, message):
    if not condition:
        raise AssertionError(message)


@contextmanager
def _workspace_tempdir():
    base = Path(__file__).resolve().parents[1] / ".tmp_tests"
    base.mkdir(parents=True, exist_ok=True)
    temp_path = base / f"case_{uuid.uuid4().hex}"
    temp_path.mkdir(parents=True, exist_ok=True)
    try:
        yield temp_path
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)


def test_persona_and_voice_controls():
    policy_engine.set_profile("normal")

    response = route_command("persona set formal")
    _assert("Persona set to: formal" in response, f"Unexpected response: {response}")

    response = route_command("persona status")
    _assert("active_profile: formal" in response, f"Unexpected response: {response}")

    with _workspace_tempdir() as tmp:
        sample = tmp / "voice_sample.wav"
        sample.write_bytes(b"RIFF0000WAVEfmt ")

        response = route_command(f"voice clone reference {sample}")
        _assert("Voice clone reference audio set" in response, f"Unexpected response: {response}")

    response = route_command("voice clone provider xtts")
    _assert("provider set to: xtts" in response.lower(), f"Unexpected response: {response}")

    response = route_command("voice clone on")
    _assert("enabled" in response.lower(), f"Unexpected response: {response}")

    response = route_command("voice status")
    _assert("clone_enabled: True" in response, f"Unexpected response: {response}")

    route_command("voice clone off")
    route_command("persona set assistant")


def test_interruptibility():
    policy_engine.set_profile("normal")
    route_command("speech on")

    long_text = "interruptibility test " * 80
    ok, message = speech_engine.speak_async(long_text)
    _assert(ok, f"Speech did not start: {message}")
    time.sleep(0.2)

    response = route_command("stop speaking")
    _assert("interrupted" in response.lower(), f"Unexpected response: {response}")
    _assert(not speech_engine.is_speaking(), "Speech should stop after interruption")


def test_knowledge_base_flow():
    policy_engine.set_profile("normal")
    route_command("kb clear")

    with _workspace_tempdir() as tmp:
        doc = tmp / "notes.txt"
        doc.write_text(
            (
                "Jarvis phase four introduces persona profiles, interruptible speech, "
                "and offline vector retrieval.\n"
                "The codename for this test document is nebula-lattice."
            ),
            encoding="utf-8",
        )

        response = route_command(f"kb add {doc}")
        _assert("Indexed" in response, f"Unexpected response: {response}")

        response = route_command("kb search nebula-lattice")
        _assert("Knowledge Search Results" in response, f"Unexpected response: {response}")
        _assert(str(doc) in response, f"Search should include source path: {response}")

        route_command("kb retrieval on")
        prompt = build_prompt("What is nebula-lattice?")
        _assert("LOCAL KNOWLEDGE BASE CONTEXT:" in prompt, "Prompt should include KB context")
        _assert("nebula-lattice" in prompt.lower(), "Prompt context should include indexed phrase")


if __name__ == "__main__":
    test_persona_and_voice_controls()
    test_interruptibility()
    test_knowledge_base_flow()
    print("Phase 4 smoke tests passed.")
