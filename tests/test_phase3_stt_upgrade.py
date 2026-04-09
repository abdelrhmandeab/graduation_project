import time
import logging
import unittest
from unittest.mock import patch

import numpy as np

from audio import stt as stt_runtime
from core import orchestrator


class Phase3SttUpgradeTests(unittest.TestCase):
    def setUp(self):
        self._saved_stt_settings = stt_runtime.get_runtime_stt_settings()
        self._saved_hf_cpu_heavy_warned_models = set(stt_runtime._hf_cpu_heavy_warned_models)

    def tearDown(self):
        stt_runtime.set_runtime_stt_settings(
            beam_size=self._saved_stt_settings.get("beam_size"),
            vad_filter=self._saved_stt_settings.get("vad_filter"),
            condition_on_previous_text=self._saved_stt_settings.get("condition_on_previous_text"),
            language_hint=self._saved_stt_settings.get("language_hint"),
            quality_retry_threshold=self._saved_stt_settings.get("quality_retry_threshold"),
            quality_retry_beam_size=self._saved_stt_settings.get("quality_retry_beam_size"),
        )
        stt_runtime._hf_cpu_heavy_warned_models.clear()
        stt_runtime._hf_cpu_heavy_warned_models.update(self._saved_hf_cpu_heavy_warned_models)

    def test_language_hint_normalization(self):
        stt_runtime.set_runtime_stt_settings(language_hint="arabic")
        self.assertEqual(stt_runtime.get_runtime_stt_settings().get("language_hint"), "ar")

        stt_runtime.set_runtime_stt_settings(language_hint="ENGLISH")
        self.assertEqual(stt_runtime.get_runtime_stt_settings().get("language_hint"), "en")

        stt_runtime.set_runtime_stt_settings(language_hint="unknown")
        self.assertEqual(stt_runtime.get_runtime_stt_settings().get("language_hint"), "auto")

    def test_load_hf_component_prefers_local_cache_first(self):
        calls = []

        def _loader(local_files_only=False):
            calls.append(bool(local_files_only))
            return "ok"

        result = stt_runtime._load_hf_component_with_local_cache(_loader, "openai/whisper-small", "STT model")
        self.assertEqual(result, "ok")
        self.assertEqual(calls, [True])

    def test_load_hf_component_falls_back_to_online_on_cache_miss(self):
        calls = []

        def _loader(local_files_only=False):
            calls.append(bool(local_files_only))
            if local_files_only:
                raise OSError("local_files_only cache miss")
            return "downloaded"

        result = stt_runtime._load_hf_component_with_local_cache(_loader, "openai/whisper-small", "STT model")
        self.assertEqual(result, "downloaded")
        self.assertEqual(calls, [True, False])

    def test_duplicate_transformers_warning_filter_suppresses_known_noise(self):
        warning_filter = stt_runtime._SuppressDuplicateTransformersProcessorWarnings()
        record = logging.LogRecord(
            name="transformers.generation.utils",
            level=logging.WARNING,
            pathname=__file__,
            lineno=1,
            msg=(
                "A custom logits processor of type "
                "<class 'transformers.generation.logits_process.SuppressTokensLogitsProcessor'> has been passed "
                "to `.generate()`, but it was also created in `.generate()`, given its parameterization."
            ),
            args=(),
            exc_info=None,
        )
        self.assertFalse(warning_filter.filter(record))

    def test_transcribe_streaming_forwards_language_hint(self):
        captured = {}

        def fake_transcribe_once_with_meta(
            model,
            audio_file,
            *,
            language,
            beam_size,
            vad_filter,
            condition_on_previous_text,
            on_partial=None,
        ):
            captured["language"] = language
            return "this transcript has enough words", "ar", 0.95

        with patch("audio.stt._resolve_stt_backend", return_value="faster_whisper"), patch(
            "audio.stt._get_whisper_model", return_value=object()
        ), patch("audio.stt._transcribe_once_with_meta", side_effect=fake_transcribe_once_with_meta):
            text = stt_runtime.transcribe_streaming("dummy.wav", language_hint="ar")

        self.assertEqual(text, "this transcript has enough words")
        self.assertEqual(captured.get("language"), "ar")

    def test_coerce_supported_language_prefers_script_over_wrong_detected_language(self):
        self.assertEqual(stt_runtime._coerce_supported_language("ar", "open notepad", fallback=""), "en")
        self.assertEqual(stt_runtime._coerce_supported_language("en", "افتح المفكرة", fallback=""), "ar")

    def test_faster_whisper_arabic_refinement_uses_explicit_ar_on_low_confidence(self):
        call_languages = []

        def fake_transcribe_once_with_meta(
            model,
            audio_file,
            *,
            language,
            beam_size,
            vad_filter,
            condition_on_previous_text,
            on_partial=None,
        ):
            _ = model, audio_file, beam_size, vad_filter, condition_on_previous_text, on_partial
            call_languages.append(language)
            if language is None:
                return "عايز تري بان نت ازاي", "ar", 0.41
            if language == "ar":
                return "عايز تبقى ازاي مهندس كمبيوتر", "ar", 0.93
            return "How can I become a computer engineer?", "en", 0.96

        with patch("audio.stt._get_whisper_model", return_value=object()), patch(
            "audio.stt._transcribe_once_with_meta", side_effect=fake_transcribe_once_with_meta
        ):
            text, language, confidence = stt_runtime._transcribe_faster_whisper_with_meta("dummy.wav")

        self.assertEqual(language, "ar")
        self.assertGreaterEqual(confidence, 0.90)
        self.assertEqual(call_languages[0], None)
        self.assertIn("ar", call_languages)

    def test_orchestrator_keeps_auto_stt_even_with_explicit_runtime_hint(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.session_memory.get_preferred_language", return_value="ar"
        ), patch(
            "core.orchestrator.stt_runtime.get_runtime_stt_settings", return_value={"language_hint": "ar"}
        ), patch("core.orchestrator.transcribe_streaming", return_value="open notepad") as transcribe_mock, patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch("core.orchestrator.metrics.record_stage"), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance("dummy.wav", pipeline_started=time.perf_counter())

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_wake_source_does_not_bias_stt_language_hint(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.session_memory.get_preferred_language", return_value="ar"
        ), patch(
            "core.orchestrator.stt_runtime.get_runtime_stt_settings", return_value={"language_hint": "auto"}
        ), patch("core.orchestrator.transcribe_streaming", return_value="open notepad") as transcribe_mock, patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch("core.orchestrator.metrics.record_stage"), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_auto_runtime_hint_without_wake_source_keeps_auto(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.session_memory.get_preferred_language", return_value="ar"
        ), patch(
            "core.orchestrator.stt_runtime.get_runtime_stt_settings", return_value={"language_hint": "auto"}
        ), patch("core.orchestrator.transcribe_streaming", return_value="open notepad") as transcribe_mock, patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch("core.orchestrator.metrics.record_stage"), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source=None,
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_ignores_runtime_hint_when_wake_source_is_arabic(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.session_memory.get_preferred_language", return_value="en"
        ), patch(
            "core.orchestrator.stt_runtime.get_runtime_stt_settings", return_value={"language_hint": "en"}
        ), patch("core.orchestrator.transcribe_streaming", return_value="افتح المفكرة") as transcribe_mock, patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch("core.orchestrator.metrics.record_stage"), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="arabic",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_does_not_retry_with_english_hint_for_unclear_auto_transcript(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="جربز",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "ar"},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ) as route_mock, patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))
        self.assertEqual(route_mock.call_args.args[0], "جربز")
        self.assertEqual(route_mock.call_args.kwargs.get("detected_language"), "ar")

    def test_orchestrator_does_not_retry_even_when_wake_source_changes(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="جربز",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "ar"},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="arabic",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_does_not_retry_when_arabic_detect_confidence_is_low(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="عايز تري بان نت ازاي اباماندس كومبيوتر",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "ar", "language_confidence": 0.41},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ) as route_mock, patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))
        self.assertEqual(route_mock.call_args.args[0], "عايز تري بان نت ازاي اباماندس كومبيوتر")
        self.assertEqual(route_mock.call_args.kwargs.get("detected_language"), "ar")

    def test_retry_decision_is_wake_source_independent_for_low_confidence_arabic(self):
        text = "عايز تري بان نت ازاي اباماندس كومبيوتر"
        decision_english_wake = orchestrator._should_retry_with_english_hint(
            text,
            "ar",
            wake_source="english",
            language_confidence=0.41,
        )
        decision_arabic_wake = orchestrator._should_retry_with_english_hint(
            text,
            "ar",
            wake_source="arabic",
            language_confidence=0.41,
        )
        self.assertFalse(decision_english_wake)
        self.assertEqual(decision_english_wake, decision_arabic_wake)

    def test_orchestrator_does_not_retry_even_for_low_confidence_arabic_wake(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="عايز تري بان نت ازاي اباماندس كومبيوتر",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "ar", "language_confidence": 0.41},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="arabic",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_keeps_coherent_arabic_after_english_wake(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="عايزك ان تري عن ميت اخبار التكسر النار",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "ar"},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ) as route_mock, patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))
        self.assertEqual(route_mock.call_args.args[0], "عايزك ان تري عن ميت اخبار التكسر النار")
        self.assertEqual(route_mock.call_args.kwargs.get("detected_language"), "ar")

    def test_orchestrator_keeps_single_pass_for_arabic_turn_after_arabic_wake(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="عايزك ان تري عن ميت اخبار التكسر النار",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "ar"},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="arabic",
            )

        self.assertEqual(transcribe_mock.call_count, 1)

    def test_orchestrator_skips_english_retry_for_unsupported_script_noise(self):
        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming",
            return_value="И лента в тушилом.",
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": "en"},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )

        self.assertEqual(transcribe_mock.call_count, 1)
        self.assertIsNone(transcribe_mock.call_args.kwargs.get("language_hint"))

    def test_orchestrator_switches_detected_language_across_consecutive_turns(self):
        transcripts = [
            "open notepad",
            "افتح المفكرة",
            "open calculator",
        ]

        with patch("core.orchestrator.is_speech", return_value=True), patch(
            "core.orchestrator.transcribe_streaming", side_effect=transcripts
        ) as transcribe_mock, patch(
            "core.orchestrator.stt_runtime.get_last_transcription_meta",
            return_value={"language": ""},
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ) as route_mock, patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="english",
            )
            orchestrator._process_utterance(
                "dummy.wav",
                pipeline_started=time.perf_counter(),
                wake_source="arabic",
            )

        self.assertEqual(transcribe_mock.call_count, 3)
        self.assertTrue(all(call.kwargs.get("language_hint") is None for call in transcribe_mock.call_args_list))

        self.assertEqual(route_mock.call_count, 3)
        detected = [call.kwargs.get("detected_language") for call in route_mock.call_args_list]
        self.assertEqual(detected, ["en", "ar", "en"])

    def test_safe_log_text_keeps_unicode_text_readable(self):
        value = orchestrator._safe_log_text("أريدك أن تخبرني عن الأخبار")
        self.assertIn("أريدك", value)
        self.assertNotIn("\\u", value)

    def test_orchestrator_skips_post_capture_speech_guard_for_responsive_audio_profile(self):
        with patch("core.orchestrator.session_memory.get_audio_ux_profile", return_value="responsive"), patch(
            "core.orchestrator.session_memory.get_preferred_language", return_value="en"
        ), patch("core.orchestrator.is_speech") as speech_guard_mock, patch(
            "core.orchestrator.transcribe_streaming", return_value="open notepad"
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance("dummy.wav", pipeline_started=time.perf_counter())

        self.assertEqual(speech_guard_mock.call_count, 0)

    def test_orchestrator_uses_post_capture_speech_guard_for_balanced_audio_profile(self):
        with patch("core.orchestrator.session_memory.get_audio_ux_profile", return_value="balanced"), patch(
            "core.orchestrator.session_memory.get_preferred_language", return_value="en"
        ), patch("core.orchestrator.is_speech", return_value=True) as speech_guard_mock, patch(
            "core.orchestrator.transcribe_streaming", return_value="open notepad"
        ), patch(
            "core.orchestrator.route_command", return_value="ok"
        ), patch(
            "core.orchestrator.metrics.record_stage"
        ), patch(
            "core.orchestrator._safe_remove"
        ), patch(
            "core.orchestrator.speech_engine.speak_async"
        ):
            orchestrator._process_utterance("dummy.wav", pipeline_started=time.perf_counter())

        self.assertEqual(speech_guard_mock.call_count, 1)

    def test_hf_manual_decode_prefers_language_task_and_attention_mask(self):
        if stt_runtime.torch is None:
            self.skipTest("torch unavailable")

        class _FakeTensor:
            def to(self, _device):
                return self

        class _FakeProcessor:
            class _Extractor:
                sampling_rate = 16000

            feature_extractor = _Extractor()

            def __call__(self, _audio, **_kwargs):
                return {
                    "input_features": _FakeTensor(),
                    "attention_mask": _FakeTensor(),
                }

            def batch_decode(self, _generated_ids, skip_special_tokens=True):
                _ = skip_special_tokens
                return ["decoded text"]

        class _FakeModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [[1, 2, 3]]

        fake_processor = _FakeProcessor()
        fake_model = _FakeModel()

        with patch("audio.stt._get_hf_manual_components", return_value=(fake_processor, fake_model, "cpu")), patch(
            "audio.stt._read_wav_mono_float", return_value=(np.array([0.1, 0.2, 0.1], dtype=np.float32), 16000)
        ):
            text = stt_runtime._transcribe_hf_manual_whisper(
                "dummy.wav",
                generate_kwargs={"task": "transcribe", "language": "ar"},
            )

        self.assertEqual(text, "decoded text")
        self.assertEqual(len(fake_model.calls), 1)
        kwargs = dict(fake_model.calls[0])
        self.assertEqual(kwargs.get("task"), "transcribe")
        self.assertEqual(kwargs.get("language"), "ar")
        self.assertIn("attention_mask", kwargs)
        self.assertNotIn("forced_decoder_ids", kwargs)

    def test_hf_manual_decode_falls_back_to_decoder_prompt_ids_for_legacy_generate(self):
        if stt_runtime.torch is None:
            self.skipTest("torch unavailable")

        class _FakeTensor:
            def to(self, _device):
                return self

        class _FakeProcessor:
            class _Extractor:
                sampling_rate = 16000

            feature_extractor = _Extractor()

            def __call__(self, _audio, **_kwargs):
                return {
                    "input_features": _FakeTensor(),
                    "attention_mask": _FakeTensor(),
                }

            def get_decoder_prompt_ids(self, language, task):
                _ = language, task
                return [[1, 10], [2, 11]]

            def batch_decode(self, _generated_ids, skip_special_tokens=True):
                _ = skip_special_tokens
                return ["legacy decoded"]

        class _LegacyModel:
            def __init__(self):
                self.calls = []

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                if len(self.calls) == 1:
                    raise TypeError("unexpected keyword argument 'language'")
                return [[4, 5, 6]]

        fake_processor = _FakeProcessor()
        fake_model = _LegacyModel()

        with patch("audio.stt._get_hf_manual_components", return_value=(fake_processor, fake_model, "cpu")), patch(
            "audio.stt._read_wav_mono_float", return_value=(np.array([0.1, 0.2, 0.1], dtype=np.float32), 16000)
        ):
            text = stt_runtime._transcribe_hf_manual_whisper(
                "dummy.wav",
                generate_kwargs={"task": "transcribe", "language": "ar"},
            )

        self.assertEqual(text, "legacy decoded")
        self.assertEqual(len(fake_model.calls), 2)
        first_kwargs = dict(fake_model.calls[0])
        second_kwargs = dict(fake_model.calls[1])
        self.assertEqual(first_kwargs.get("language"), "ar")
        self.assertIn("forced_decoder_ids", second_kwargs)
        self.assertNotIn("language", second_kwargs)

    def test_hf_manual_decode_aligns_input_features_dtype_with_model(self):
        if stt_runtime.torch is None:
            self.skipTest("torch unavailable")

        class _FakeTensor:
            def __init__(self, dtype):
                self.dtype = dtype

            def to(self, *args, **kwargs):
                if "dtype" in kwargs and kwargs.get("dtype") is not None:
                    self.dtype = kwargs.get("dtype")
                    return self
                if args and args[0] is not None and not isinstance(args[0], str):
                    self.dtype = args[0]
                return self

        class _FakeParameter:
            def __init__(self, dtype):
                self.dtype = dtype

        class _FakeProcessor:
            class _Extractor:
                sampling_rate = 16000

            feature_extractor = _Extractor()

            def __call__(self, _audio, **_kwargs):
                return {
                    "input_features": _FakeTensor(stt_runtime.torch.float32),
                    "attention_mask": _FakeTensor(stt_runtime.torch.int64),
                }

            def batch_decode(self, _generated_ids, skip_special_tokens=True):
                _ = skip_special_tokens
                return ["dtype aligned"]

        class _FakeModel:
            def __init__(self):
                self.calls = []

            def parameters(self):
                return iter([_FakeParameter(stt_runtime.torch.float16)])

            def generate(self, **kwargs):
                self.calls.append(kwargs)
                return [[1, 2, 3]]

        fake_processor = _FakeProcessor()
        fake_model = _FakeModel()

        with patch("audio.stt._get_hf_manual_components", return_value=(fake_processor, fake_model, "cpu")), patch(
            "audio.stt._read_wav_mono_float", return_value=(np.array([0.1, 0.2, 0.1], dtype=np.float32), 16000)
        ):
            text = stt_runtime._transcribe_hf_manual_whisper(
                "dummy.wav",
                generate_kwargs={"task": "transcribe", "language": "en"},
            )

        self.assertEqual(text, "dtype aligned")
        self.assertEqual(len(fake_model.calls), 1)
        kwargs = dict(fake_model.calls[0])
        self.assertEqual(kwargs["input_features"].dtype, stt_runtime.torch.float16)
        self.assertEqual(kwargs["attention_mask"].dtype, stt_runtime.torch.int64)

    def test_hf_cpu_heavy_model_detection(self):
        self.assertTrue(stt_runtime._is_hf_cpu_heavy_model("openai/whisper-large-v3"))
        self.assertTrue(stt_runtime._is_hf_cpu_heavy_model("openai/whisper-large-v2"))
        self.assertFalse(stt_runtime._is_hf_cpu_heavy_model("openai/whisper-small"))

    def test_transcribe_huggingface_short_circuits_on_cpu_heavy_model(self):
        with patch("audio.stt._resolve_hf_mode", return_value="manual"), patch(
            "audio.stt.get_runtime_hf_settings",
            return_value={"model": "openai/whisper-large-v3", "mode": "manual", "chunk_length_s": 20.0, "batch_size": 4},
        ), patch("audio.stt._should_skip_hf_for_realtime", return_value=True), patch(
            "audio.stt._transcribe_hf_manual_whisper"
        ) as manual_decode_mock:
            text = stt_runtime._transcribe_huggingface("dummy.wav", language_hint="en")

        self.assertEqual(text, "")
        self.assertEqual(manual_decode_mock.call_count, 0)

    def test_hf_cpu_heavy_warning_is_logged_once_per_model(self):
        stt_runtime._hf_cpu_heavy_warned_models.clear()
        with patch("audio.stt._resolve_hf_mode", return_value="manual"), patch(
            "audio.stt.get_runtime_hf_settings",
            return_value={"model": "openai/whisper-large-v3", "mode": "manual", "chunk_length_s": 20.0, "batch_size": 4},
        ), patch("audio.stt._should_skip_hf_for_realtime", return_value=True), patch(
            "audio.stt.logger.warning"
        ) as warning_mock:
            first = stt_runtime._transcribe_huggingface("dummy.wav", language_hint="en")
            second = stt_runtime._transcribe_huggingface("dummy.wav", language_hint="en")

        self.assertEqual(first, "")
        self.assertEqual(second, "")
        self.assertEqual(warning_mock.call_count, 1)

    def test_transcribe_streaming_falls_back_when_hf_short_circuits(self):
        with patch("audio.stt._resolve_stt_backend", return_value="huggingface"), patch(
            "audio.stt._transcribe_huggingface", return_value=""
        ), patch(
            "audio.stt._transcribe_faster_whisper_with_meta", return_value=("fallback transcript", "en")
        ) as fallback_mock:
            text = stt_runtime.transcribe_streaming("dummy.wav", language_hint="en")

        self.assertEqual(text, "fallback transcript")
        self.assertEqual(fallback_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()
