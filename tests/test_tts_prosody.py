"""Tests for core.tts_prosody.polish_for_voice."""

import unittest


class TestProsodyPolisher(unittest.TestCase):

    def _polish(self, text, language="en", **kw):
        from core.tts_prosody import polish_for_voice
        return polish_for_voice(text, language=language, **kw)

    # ── Bilingual: punctuation dedup ──

    def test_double_period_compressed(self):
        self.assertEqual(self._polish("hello.. world"), "hello. world")

    def test_double_question_mark_ar(self):
        result = self._polish("ليه؟؟", language="ar")
        self.assertEqual(result, "ليه؟")

    def test_double_exclamation(self):
        self.assertEqual(self._polish("wow!!"), "wow!")

    def test_double_comma(self):
        self.assertEqual(self._polish("a,, b"), "a, b")

    # ── Bilingual: em-dash / ellipsis ──

    def test_em_dash_to_comma_en(self):
        result = self._polish("something — else")
        self.assertIn(",", result)
        self.assertNotIn("—", result)

    def test_em_dash_to_arabic_comma(self):
        result = self._polish("حاجة — تانية", language="ar")
        self.assertIn("،", result)
        self.assertNotIn("—", result)

    def test_ellipsis_to_comma(self):
        result = self._polish("well… you know")
        self.assertIn(",", result)
        self.assertNotIn("…", result)

    def test_triple_dots_to_comma(self):
        result = self._polish("well... you know")
        self.assertIn(",", result)

    # ── Bilingual: markdown strip ──

    def test_bold_stripped(self):
        self.assertEqual(self._polish("**hello** world"), "hello world")

    def test_italic_stripped(self):
        self.assertEqual(self._polish("_hello_ world"), "hello world")

    def test_backtick_stripped(self):
        self.assertEqual(self._polish("`code`"), "code")

    # ── EN: hyphen spacing ──

    def test_hyphenated_collapsed(self):
        self.assertEqual(self._polish("built - in"), "built-in")

    def test_proper_hyphen_kept(self):
        self.assertEqual(self._polish("built-in"), "built-in")

    # ── EN: compound sentence comma ──

    def test_compound_comma_inserted(self):
        result = self._polish("the weather is nice today and I want to go outside")
        self.assertIn(", and", result)

    def test_short_clause_no_comma(self):
        result = self._polish("hi and bye")
        self.assertNotIn(",", result)

    # ── AR: discourse particle comma ──

    def test_ar_discourse_comma_tab(self):
        result = self._polish("طب انت عامل ايه", language="ar")
        self.assertIn("طب،", result)

    def test_ar_discourse_comma_yani(self):
        result = self._polish("يعني الموضوع كبير", language="ar")
        self.assertIn("يعني،", result)

    def test_ar_discourse_already_punctuated(self):
        result = self._polish("يعني، الموضوع كبير", language="ar")
        self.assertEqual(result.count("يعني،"), 1)

    # ── AR: formal connector rewrite ──

    def test_formal_connector_addition(self):
        result = self._polish("بالإضافة إلى ذلك", language="ar")
        self.assertIn("وكمان", result)

    def test_formal_connector_batali(self):
        result = self._polish("بالتالي هنعمل كده", language="ar")
        self.assertIn("وبكده", result)

    # ── Empty / no-op ──

    def test_empty_string(self):
        self.assertEqual(self._polish(""), "")

    def test_none_returns_empty(self):
        self.assertEqual(self._polish(None), "")

    def test_clean_text_unchanged(self):
        text = "Hello, how are you?"
        self.assertEqual(self._polish(text), text)


if __name__ == "__main__":
    unittest.main()
