import unittest

from core.command_parser import parse_command
from core.command_router import (
    _repair_low_value_llm_response,
    _looks_live_data_trigger_query,
    _looks_search_worthy_query,
    _strip_repeated_user_question,
)
from llm.prompt_builder import build_lightweight_prompt


class LLMRoutingGuardTests(unittest.TestCase):
    def test_arabic_advice_question_stays_llm_query(self):
        text = "ممكن تقول لي إزاي أكون مهندس كمبيوتر ناجح؟ أنا مش عارف أعمل كده."
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_codeswitched_advice_question_not_command_chain(self):
        text = "عاوزك تقولي إزاي أكون مهندس computer ناجح؟ أنا بحاول أعمل كده ومش عارف."
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_step_by_step_advice_question_not_timer(self):
        text = "عاوزك تقولي إزاي أكون مهندس كمبيوتر ناجح؟ قولي الخطوات خطوة خطوة"
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_advice_question_does_not_trigger_live_search(self):
        text = "عاوزك تقولي إزاي أكون مهندس كمبيوتر ناجح؟ قولي الخطوات خطوة خطوة"
        self.assertFalse(_looks_search_worthy_query(text))
        self.assertFalse(_looks_live_data_trigger_query(text))

    def test_repeated_question_prefix_is_removed(self):
        question = "ممكن تقول لي إزاي أكون مهندس كمبيوتر ناجح؟"
        response = f"{question} ركز على الأساسيات واعمل مشاريع صغيرة باستمرار."
        cleaned = _strip_repeated_user_question(response, question)
        self.assertEqual(cleaned, "ركز على الأساسيات واعمل مشاريع صغيرة باستمرار.")

    def test_question_answer_label_prefix_is_removed(self):
        question = "How can I become a successful computer engineer?"
        response = f"Question: {question}\nAnswer: Build fundamentals, ship projects, and get feedback."
        cleaned = _strip_repeated_user_question(response, question)
        self.assertEqual(cleaned, "Build fundamentals, ship projects, and get feedback.")

    def test_arabic_multi_question_not_command_chain(self):
        text = "هو ليه الأخبار كده؟ وإيه أخبار الجو في القاهرة وإسكندرية؟"
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_stt_corrupted_arabic_career_sentence_not_command_chain(self):
        text = "أريد أن أقول إزيك مهندس كمبيوتر شاطر وناجح."
        parsed = parse_command(text)
        self.assertEqual(parsed.intent, "LLM_QUERY")

    def test_arabic_prompt_examples_do_not_mix_english(self):
        package = build_lightweight_prompt("عايز أكون مهندس شاطر، إزاي أعمل كده؟", response_language="ar")
        prompt = package["prompt"]
        self.assertIn("عايز أكون مهندس", prompt)
        self.assertNotIn("what is machine learning", prompt.lower())

    def test_low_value_career_answer_gets_useful_fallback(self):
        question = "عايز أكون مهندس شاطر، إزاي أعمل كده؟"
        bad = "مش هقدر أساعدك في ده، قولّي هدفك."
        cleaned = _repair_low_value_llm_response(bad, parse_command(question), "ar", question)
        self.assertIn("الأساسيات", cleaned)
        self.assertIn("مشاريع", cleaned)


if __name__ == "__main__":
    unittest.main()
