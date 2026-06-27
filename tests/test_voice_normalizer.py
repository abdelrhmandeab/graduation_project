import unittest

from core.voice_normalizer import normalize_for_voice, normalize_search_block, normalize_weather_block


class VoiceNormalizerTests(unittest.TestCase):
    def test_weather_english(self):
        raw = "Weather in Cairo: Mainly clear, 33.1?C, humidity 33%, wind 10.0 km/h"
        out = normalize_weather_block(raw, "en")
        self.assertIn("Temperature 33.1 degrees", out)
        self.assertIn("33 percent", out)
        self.assertIn("10 kilometers per hour", out)

    def test_weather_arabic(self):
        raw = "Weather in Cairo: Mainly clear, 33.1?C, humidity 33%, wind 10.0 km/h"
        out = normalize_weather_block(raw, "ar")
        self.assertIn("الطقس في القاهرة", out)
        self.assertIn("درجة", out)
        self.assertIn("في المية", out)
        self.assertIn("كيلومتر في الساعة", out)

    def test_general_units(self):
        out = normalize_for_voice("27?C 45% 12 km/h 15:30 2026-06-22", "ar")
        self.assertIn("درجة", out)
        self.assertIn("في المية", out)
        self.assertIn("كيلومتر في الساعة", out)
        self.assertIn("بعد الظهر", out)
        self.assertIn("يونيو", out)

    def test_search_strips_urls(self):
        raw = "- Title | BBC: Snippet https://example.com/page with 45% data. [bbc.com]"
        out = normalize_search_block(raw, "en")
        self.assertNotIn("https://", out)
        self.assertIn("45 percent", out)


if __name__ == "__main__":
    unittest.main()
