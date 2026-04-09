import unittest
from unittest.mock import patch
import sys
import logging
import os
import numpy as np

import audio.tts as tts_runtime
from audio.tts import SpeechEngine


class _Voice:
    def __init__(self, voice_id, name, languages=None):
        self.id = voice_id
        self.name = name
        self.languages = languages or []


class _Engine:
    def __init__(self, voices):
        self._voices = list(voices)

    def getProperty(self, key):
        if key == "voices":
            return self._voices
        return None


class TtsVoiceUnificationTests(unittest.TestCase):
    def test_run_async_propagates_coroutine_runtime_error(self):
        speech = SpeechEngine()

        async def _explode():
            raise RuntimeError("boom")

        with self.assertRaises(RuntimeError):
            speech._run_async(_explode())

    def test_prepare_text_for_speech_rewrites_formal_arabic(self):
        speech = SpeechEngine()

        with patch("audio.tts.TTS_ARABIC_SPOKEN_DIALECT", "egyptian"), patch(
            "audio.tts.TTS_EGYPTIAN_COLLOQUIAL_REWRITE", True
        ):
            rewritten = speech._prepare_text_for_speech("يمكنني مساعدتك الآن")

        self.assertIn("اقدر", rewritten)
        self.assertIn("دلوقتي", rewritten)

    def test_prepare_text_for_speech_keeps_english_unchanged(self):
        speech = SpeechEngine()
        text = "I can help you now"

        with patch("audio.tts.TTS_ARABIC_SPOKEN_DIALECT", "egyptian"), patch(
            "audio.tts.TTS_EGYPTIAN_COLLOQUIAL_REWRITE", True
        ):
            rewritten = speech._prepare_text_for_speech(text)

        self.assertEqual(rewritten, text)

    def test_prepare_text_for_speech_rewrites_arabic_dominant_mixed_text(self):
        speech = SpeechEngine()
        text = "يمكنني مساعدتك الآن in app"

        with patch("audio.tts.TTS_ARABIC_SPOKEN_DIALECT", "egyptian"), patch(
            "audio.tts.TTS_EGYPTIAN_COLLOQUIAL_REWRITE", True
        ):
            rewritten = speech._prepare_text_for_speech(text)

        self.assertIn("اقدر", rewritten)
        self.assertIn("دلوقتي", rewritten)
        self.assertIn("in app", rewritten)

    def test_edge_tts_applies_arabic_pitch_and_volume_when_supported(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, _text, **kwargs):
                calls.append(dict(kwargs))

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_ARABIC_PITCH", "-8Hz"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOLUME", "+4%"
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("محتاج مساعدة دلوقتي")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0].get("pitch"), "-8Hz")
        self.assertEqual(calls[0].get("volume"), "+4%")

    def test_install_transformers_filter_sets_generation_loggers_to_error(self):
        speech = SpeechEngine()
        logger_names = (
            "transformers.generation.utils",
            "transformers.generation.configuration_utils",
        )
        original_levels = {}

        try:
            for name in logger_names:
                generation_logger = logging.getLogger(name)
                original_levels[name] = generation_logger.level
                generation_logger.setLevel(logging.WARNING)

            speech._install_transformers_generation_warning_filter()

            for name in logger_names:
                self.assertEqual(logging.getLogger(name).level, logging.ERROR)
        finally:
            for name, level in original_levels.items():
                logging.getLogger(name).setLevel(level)

    def test_configure_hf_runtime_noise_sets_progress_env_flags(self):
        speech = SpeechEngine()
        with patch.dict(os.environ, {}, clear=True):
            speech._configure_hf_runtime_noise()

            self.assertEqual(os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"), "1")
            self.assertEqual(os.environ.get("TRANSFORMERS_NO_ADVISORY_WARNINGS"), "1")

    def test_hf_component_loader_prefers_local_cache(self):
        calls = []

        def _loader(local_files_only=False):
            calls.append(bool(local_files_only))
            return "loaded-local"

        result = tts_runtime._load_hf_component_with_local_cache(
            _loader,
            "suno/bark-small",
            "Bark model",
        )

        self.assertEqual(result, "loaded-local")
        self.assertEqual(calls, [True])

    def test_hf_component_loader_falls_back_to_online(self):
        calls = []

        def _loader(local_files_only=False):
            calls.append(bool(local_files_only))
            if local_files_only:
                raise OSError("local_files_only cache miss")
            return "loaded-online"

        result = tts_runtime._load_hf_component_with_local_cache(
            _loader,
            "suno/bark-small",
            "Bark model",
        )

        self.assertEqual(result, "loaded-online")
        self.assertEqual(calls, [True, False])

    def test_transformers_generation_filter_suppresses_known_noise(self):
        warning_filter = tts_runtime._SuppressTransformersGenerationWarnings()
        noisy_record = logging.LogRecord(
            name="transformers.generation.utils",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Both `max_new_tokens` (=60) and `max_length`(=20) seem to have been set.",
            args=(),
            exc_info=None,
        )
        useful_record = logging.LogRecord(
            name="transformers.generation.utils",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg="Meaningful generation warning",
            args=(),
            exc_info=None,
        )

        self.assertFalse(warning_filter.filter(noisy_record))
        self.assertTrue(warning_filter.filter(useful_record))

    def test_effectively_silent_detector_flags_tiny_waveform(self):
        speech = SpeechEngine()
        waveform = np.array([0.0, 1.0e-5, -2.0e-5, 0.0], dtype=np.float32)

        self.assertTrue(speech._is_effectively_silent(waveform))

    def test_effectively_silent_detector_allows_normal_waveform(self):
        speech = SpeechEngine()
        waveform = np.array([0.0, 0.08, -0.06, 0.02], dtype=np.float32)

        self.assertFalse(speech._is_effectively_silent(waveform))

    def test_edge_tts_missing_dependency_warning_logged_once(self):
        speech = SpeechEngine()

        with patch.dict(sys.modules, {"edge_tts": None}), patch("audio.tts.logger.warning") as warning_mock:
            first = speech._speak_edge_tts("hello")
            second = speech._speak_edge_tts("hello again")

        self.assertFalse(first)
        self.assertFalse(second)
        self.assertEqual(warning_mock.call_count, 1)

    def test_decode_edge_audio_bytes_uses_soundfile_for_compressed_stream(self):
        speech = SpeechEngine()
        expected_waveform = np.array([0.0, 0.2, -0.1], dtype=np.float32)

        class _SoundFileModule:
            @staticmethod
            def read(_buffer, dtype="float32"):
                _ = dtype
                return expected_waveform, 24000

        compressed_stream = b"\xff\xf3d\xc4" + (b"\x00" * 16)
        with patch.dict(sys.modules, {"soundfile": _SoundFileModule()}):
            sample_rate, waveform = speech._decode_edge_audio_bytes(compressed_stream)

        self.assertEqual(sample_rate, 24000)
        self.assertTrue(np.array_equal(waveform, expected_waveform))

    def test_choose_voice_prefers_last_voice_when_requested(self):
        speech = SpeechEngine()
        speech._last_pyttsx3_voice_id = "en_voice"

        voices = [
            _Voice("ar_voice", "Arabic Voice", ["ar-SA"]),
            _Voice("en_voice", "English Voice", ["en-US"]),
        ]
        engine = _Engine(voices)

        selected_id, selected_name, language_match = speech._choose_pyttsx3_voice(
            engine,
            "مرحبا بك",
            prefer_last_voice=True,
            force_english_voice=False,
        )

        self.assertEqual(selected_id, "en_voice")
        self.assertEqual(selected_name, "English Voice")
        self.assertTrue(language_match)

    def test_hf_natural_uses_system_voice_for_arabic_with_unified_flag(self):
        speech = SpeechEngine()

        with patch("audio.tts.TTS_FORCE_ENGLISH_VOICE_FOR_ARABIC", True), patch(
            "audio.tts.persona_manager.get_clone_settings",
            return_value={"enabled": False, "provider": "voicecraft", "reference_audio": ""},
        ), patch("audio.tts.persona_manager.get_speech_style", return_value="neutral"), patch(
            "audio.tts.metrics.record_stage"
        ), patch.object(
            speech, "_resolve_backend", return_value="huggingface"
        ), patch.object(
            speech, "get_quality_mode", return_value="natural"
        ), patch.object(
            speech,
            "get_hf_runtime_settings",
            return_value={"model": "facebook/mms-tts-ara", "sample_rate": 0},
        ), patch.object(
            speech,
            "_speak_pyttsx3",
            return_value=True,
        ) as pyttsx3_mock:
            speech._run_speech("مرحبا بك")

        self.assertEqual(pyttsx3_mock.call_count, 1)
        kwargs = pyttsx3_mock.call_args.kwargs
        self.assertFalse(kwargs.get("require_language_match", True))
        self.assertTrue(kwargs.get("prefer_last_voice", False))
        self.assertTrue(kwargs.get("force_english_voice", False))

    def test_hf_natural_arabic_requires_match_when_unified_flag_disabled(self):
        speech = SpeechEngine()

        with patch("audio.tts.TTS_FORCE_ENGLISH_VOICE_FOR_ARABIC", False), patch(
            "audio.tts.persona_manager.get_clone_settings",
            return_value={"enabled": False, "provider": "voicecraft", "reference_audio": ""},
        ), patch("audio.tts.persona_manager.get_speech_style", return_value="neutral"), patch(
            "audio.tts.metrics.record_stage"
        ), patch.object(
            speech, "_resolve_backend", return_value="huggingface"
        ), patch.object(
            speech, "get_quality_mode", return_value="natural"
        ), patch.object(
            speech,
            "get_hf_runtime_settings",
            return_value={"model": "facebook/mms-tts-ara", "sample_rate": 0},
        ), patch.object(
            speech,
            "_speak_pyttsx3",
            return_value=False,
        ) as pyttsx3_mock, patch.object(
            speech,
            "_speak_edge_tts",
            return_value=False,
        ) as edge_mock, patch.object(
            speech,
            "_speak_huggingface",
            return_value=True,
        ) as hf_mock:
            speech._run_speech("مرحبا بك")

        self.assertEqual(pyttsx3_mock.call_count, 0)
        self.assertEqual(edge_mock.call_count, 1)
        self.assertEqual(hf_mock.call_count, 1)

    def test_hf_bark_model_prefers_multilingual_hf_before_system_fallbacks(self):
        speech = SpeechEngine()

        with patch("audio.tts.persona_manager.get_clone_settings", return_value={"enabled": False, "provider": "voicecraft", "reference_audio": ""}), patch(
            "audio.tts.persona_manager.get_speech_style", return_value="neutral"
        ), patch("audio.tts.metrics.record_stage"), patch.object(
            speech, "_resolve_backend", return_value="huggingface"
        ), patch.object(
            speech, "get_quality_mode", return_value="natural"
        ), patch.object(
            speech,
            "get_hf_runtime_settings",
            return_value={"model": "suno/bark-small", "sample_rate": 24000},
        ), patch.object(
            speech,
            "_speak_pyttsx3",
            return_value=True,
        ) as pyttsx3_mock, patch.object(
            speech,
            "_speak_edge_tts",
            return_value=True,
        ) as edge_mock, patch.object(
            speech,
            "_speak_huggingface",
            return_value=True,
        ) as hf_mock:
            speech._run_speech("مرحبا")

        self.assertEqual(hf_mock.call_count, 1)
        self.assertEqual(pyttsx3_mock.call_count, 0)
        self.assertEqual(edge_mock.call_count, 0)

    def test_edge_tts_keeps_english_voice_for_english_text(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                calls.append({"voice": voice, "rate": rate, "output_format": output_format})

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("hello there")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0]["voice"], "en-US-AriaNeural")
        self.assertEqual(calls[0]["rate"], "+0%")

    def test_edge_tts_arabic_uses_egyptian_voice_profile(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                calls.append({"voice": voice, "rate": rate, "output_format": output_format})

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE_FALLBACKS", ("ar-EG-ShakirNeural", "ar-SA-HamedNeural")
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_RATE", "-4%"
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("حالة الطقس اليوم")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0]["voice"], "ar-EG-SalmaNeural")
        self.assertEqual(calls[0]["rate"], "-4%")

    def test_edge_tts_mixed_arabic_dominant_text_prefers_arabic_voice(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                calls.append({"voice": voice, "rate": rate, "output_format": output_format})

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE_FALLBACKS", ("ar-EG-ShakirNeural", "ar-SA-HamedNeural")
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_RATE", "-4%"
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("درجة الحرارة 25 في Cairo")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0]["voice"], "ar-EG-SalmaNeural")
        self.assertEqual(calls[0]["rate"], "-4%")

    def test_edge_tts_mixed_english_dominant_text_keeps_english_voice(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                calls.append({"voice": voice, "rate": rate, "output_format": output_format})

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural"
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("Temperature is 25 in القاهرة")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0]["voice"], "en-US-AriaNeural")
        self.assertEqual(calls[0]["rate"], "+0%")

    def test_edge_tts_mixed_text_uses_english_voice_for_english_chunk(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, text, voice=None, rate=None, output_format=None):
                calls.append({"text": text, "voice": voice, "rate": rate, "output_format": output_format})

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_MIXED_SCRIPT_CHUNKING", True
        ), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE_FALLBACKS", ("ar-EG-ShakirNeural",)
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("درجة الحرارة in Cairo اليوم")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 2)

        english_chunk_calls = [
            item for item in calls if "in" in str(item.get("text") or "").lower() or "cairo" in str(item.get("text") or "").lower()
        ]
        self.assertTrue(english_chunk_calls)
        self.assertTrue(all(item.get("voice") == "en-US-AriaNeural" for item in english_chunk_calls))

    def test_edge_tts_mixed_chunk_normalizes_arabic_prefix_before_english_words(self):
        speech = SpeechEngine()

        chunks = speech._edge_tts_text_chunks("مثل الHTML وJavaScript اليوم")
        latin_chunks = [item for item in chunks if str(item.get("script") or "") == "latin"]

        self.assertTrue(latin_chunks)
        latin_text = " ".join(str(item.get("text") or "") for item in latin_chunks)
        self.assertIn("HTML", latin_text)
        self.assertIn("JavaScript", latin_text)
        self.assertNotIn("الHTML", latin_text)
        self.assertNotIn("وJavaScript", latin_text)

    def test_edge_tts_mixed_chunking_skips_when_chunk_count_exceeds_cap(self):
        speech = SpeechEngine()

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                _ = voice, rate, output_format

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch("audio.tts.TTS_EDGE_MIXED_SCRIPT_MAX_CHUNKS", 2):
            ok = speech._speak_edge_tts_mixed_chunks(
                "درجة الحرارة in Cairo اليوم",
                _EdgeModule(),
                supports_output_format=False,
                supports_pitch=False,
                supports_volume=False,
            )

        self.assertFalse(ok)

    def test_edge_tts_mixed_chunking_skips_when_text_exceeds_length_cap(self):
        speech = SpeechEngine()

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                _ = voice, rate, output_format

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        long_mixed = (
            "درجة الحرارة in Cairo اليوم مستقرة والرياح خفيفة مع توقعات "
            "humidity around fifty percent and visibility is good جدا"
        )

        with patch("audio.tts.TTS_EDGE_MIXED_SCRIPT_MAX_TEXT_LENGTH", 40):
            ok = speech._speak_edge_tts_mixed_chunks(
                long_mixed,
                _EdgeModule(),
                supports_output_format=False,
                supports_pitch=False,
                supports_volume=False,
            )

        self.assertFalse(ok)

    def test_edge_tts_arabic_fallback_skips_unsupported_primary_in_long_session(self):
        speech = SpeechEngine()
        calls = []

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                calls.append(str(voice or ""))
                if str(voice or "") == "ar-EG-SalmaNeural":
                    raise RuntimeError("voice not found")

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE_FALLBACKS", ("ar-EG-ShakirNeural", "ar-SA-HamedNeural")
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            first_ok = speech._speak_edge_tts("اختبار الصوت الاول")
            second_ok = speech._speak_edge_tts("اختبار الصوت الثاني")

        self.assertTrue(first_ok)
        self.assertTrue(second_ok)
        self.assertGreaterEqual(len(calls), 3)
        self.assertEqual(calls[0], "ar-EG-SalmaNeural")
        self.assertEqual(calls[1], "ar-EG-ShakirNeural")
        self.assertEqual(calls[2], "ar-EG-ShakirNeural")

    def test_edge_tts_arabic_does_not_lock_to_non_egyptian_voice(self):
        speech = SpeechEngine()
        calls = []
        speech._last_edge_arabic_voice = "ar-SA-HamedNeural"

        class _Communicate:
            def __init__(self, _text, voice=None, rate=None, output_format=None):
                calls.append(str(voice or ""))

            async def stream(self):
                yield {"type": "audio", "data": b"edge-bytes"}

        class _EdgeModule:
            Communicate = _Communicate

        with patch.dict(sys.modules, {"edge_tts": _EdgeModule()}), patch(
            "audio.tts.TTS_EDGE_VOICE", "en-US-AriaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE", "ar-EG-SalmaNeural"
        ), patch(
            "audio.tts.TTS_EDGE_ARABIC_VOICE_FALLBACKS", ("ar-EG-ShakirNeural", "ar-SA-HamedNeural")
        ), patch.object(
            speech,
            "_decode_edge_audio_bytes",
            return_value=(24000, np.array([0.1, -0.1], dtype=np.float32)),
        ), patch.object(
            speech,
            "_play_waveform",
            return_value=True,
        ):
            ok = speech._speak_edge_tts("عايز اعرف اخبار الطقس")

        self.assertTrue(ok)
        self.assertGreaterEqual(len(calls), 1)
        self.assertEqual(calls[0], "ar-EG-SalmaNeural")


if __name__ == "__main__":
    unittest.main()
