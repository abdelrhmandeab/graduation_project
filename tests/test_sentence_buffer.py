import unittest

from llm.sentence_buffer import SentenceBuffer


class SentenceBufferTests(unittest.TestCase):
    def test_english_flushes_on_boundary_after_soft_words(self):
        buf = SentenceBuffer(is_arabic=False, en_soft_words=3, en_hard_words=8)
        self.assertIsNone(buf.add_token("This is "))
        self.assertEqual(buf.add_token("ready now."), "This is ready now.")

    def test_english_hard_flushes_without_punctuation(self):
        buf = SentenceBuffer(is_arabic=False, en_soft_words=7, en_hard_words=4)
        self.assertEqual(buf.add_token("one two three four five"), "one two three four")
        self.assertEqual(buf.flush(), "five")

    def test_arabic_flushes_on_arabic_boundary_after_soft_words(self):
        buf = SentenceBuffer(is_arabic=True, ar_soft_words=3, ar_hard_words=10)
        text = "\u062f\u0647 \u0631\u062f \u0637\u0628\u064a\u0639\u064a\u060c"
        self.assertEqual(buf.add_token(text), text)

    def test_arabic_holds_connector(self):
        buf = SentenceBuffer(is_arabic=True, ar_soft_words=3, ar_hard_words=4, hold_connectors=True)
        self.assertIsNone(buf.add_token("\u0648\u0627\u062d\u062f \u0627\u062a\u0646\u064a\u0646 \u062a\u0644\u0627\u062a\u0629 \u0648"))
        self.assertEqual(buf.add_token(" \u0623\u0631\u0628\u0639\u0629"), "\u0648\u0627\u062d\u062f \u0627\u062a\u0646\u064a\u0646 \u062a\u0644\u0627\u062a\u0629 \u0648 \u0623\u0631\u0628\u0639\u0629")


if __name__ == "__main__":
    unittest.main()
