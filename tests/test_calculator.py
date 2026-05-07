"""Quick-calc tests — 15 cases.

Tests tools/calculator.py quick_calc() with English expressions, Arabic
word operators, and Arabic-Indic numerals.  Covers percent-of, square root,
the four arithmetic operators, and rejection of non-math input.
"""

from __future__ import annotations

import pytest

from tools.calculator import quick_calc, to_arabic_numerals


# ---------------------------------------------------------------------------
# Group 1 — English expressions (5 tests)
# ---------------------------------------------------------------------------

class TestEnglishExpressions:

    def test_simple_addition(self):
        assert quick_calc("50 + 30") == "80"

    def test_simple_subtraction(self):
        assert quick_calc("100 - 25") == "75"

    def test_simple_multiplication(self):
        assert quick_calc("12 * 12") == "144"

    def test_simple_division(self):
        assert quick_calc("250 / 5") == "50"

    def test_square_root_english(self):
        assert quick_calc("square root of 144") == "12"


# ---------------------------------------------------------------------------
# Group 2 — Arabic word operators + Arabic-Indic numerals (7 tests)
# ---------------------------------------------------------------------------

class TestArabicOperators:

    def test_addition_arabic_word(self):
        assert quick_calc("٥٠ زائد ٣٠") == "80"

    def test_subtraction_arabic_word(self):
        assert quick_calc("١٠٠ ناقص ٢٥") == "75"

    def test_division_arabic_word(self):
        assert quick_calc("٢٥٠ قسمة ٥") == "50"

    def test_multiplication_fi_between_digits(self):
        assert quick_calc("١٢ في ١٢") == "144"

    def test_percent_of_arabic(self):
        assert quick_calc("٢٥ في المية من ٢٠٠") == "50"

    def test_percent_of_arabic_kaam(self):
        assert quick_calc("كام ١٥ في المية من ٢٣٠") == "34.5"

    def test_square_root_arabic(self):
        assert quick_calc("الجذر التربيعي من ١٤٤") == "12"


# ---------------------------------------------------------------------------
# Group 3 — Edge cases + to_arabic_numerals (3 tests)
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_non_math_returns_none(self):
        assert quick_calc("open chrome") is None

    def test_empty_input_returns_none(self):
        assert quick_calc("") is None

    def test_to_arabic_numerals_roundtrip(self):
        en = "144"
        ar = to_arabic_numerals(en)
        assert ar == "١٤٤"
