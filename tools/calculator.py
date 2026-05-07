"""Quick calculator — instant safe eval for math expressions.

Intercepts queries like "15% of 230" or "square root of 144" BEFORE the LLM
tier, returning the result in under 1ms with zero network overhead.
"""

from __future__ import annotations

import math
import re
from typing import Optional

# Arabic-Indic → ASCII digits, ٪ → %
_ARABIC_TO_LATIN = str.maketrans("٠١٢٣٤٥٦٧٨٩٪", "0123456789%")
# ASCII digits → Arabic-Indic (for Arabic language responses)
_LATIN_TO_ARABIC_DIGITS = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# Phrase substitutions applied in order before eval.
# Tuples of (compiled_re, replacement_str).  Order is load-bearing.
_PHRASE_SUBS: list = [
    # Arabic "percent of" — must run before the generic "من"→" " replacement
    (re.compile(r"في\s+(?:الميه|المية|المئه|المائه|المائة)\b", re.IGNORECASE), "% of "),
    # Strip leading Arabic question word
    (re.compile(r"^كام\s+", re.IGNORECASE), ""),
    # Arabic basic operators (written form)
    (re.compile(r"\bناقص\b", re.IGNORECASE), " - "),
    (re.compile(r"\bزائد\b|\bجمع\b", re.IGNORECASE), " + "),
    (re.compile(r"\bقسمة\b", re.IGNORECASE), " / "),
    # Arabic multiplication "في" only when flanked by digits (not a preposition)
    (re.compile(r"(?<=\d)\s+في\s+(?=\d)"), " * "),
    # Arabic square root
    (re.compile(r"\bالجذر\s+التربيعي\s+(?:من\s+|ل\s+|لـ\s*)?", re.IGNORECASE), "math.sqrt("),
    (re.compile(r"\bجذر\s+تربيعي\s+(?:من\s+|ل\s+|لـ\s*)?", re.IGNORECASE), "math.sqrt("),
    # English: strip question/command words
    (re.compile(r"\bwhat(?:\'?s)?\s+(?:is\s+)?", re.IGNORECASE), ""),
    (re.compile(r"\bcalculate\s+|\bcompute\s+|\beval(?:uate)?\s+", re.IGNORECASE), ""),
    # English square root / powers
    (re.compile(r"\bsquare\s+root\s+of\b", re.IGNORECASE), "math.sqrt("),
    (re.compile(r"\bsquared\b", re.IGNORECASE), " **2"),
    (re.compile(r"\bcubed\b", re.IGNORECASE), " **3"),
    (re.compile(r"\bto\s+the\s+power\s+of\b", re.IGNORECASE), " ** "),
    # "% of" / "percent of" → "* 0.01 *"  (must precede lone-% rules)
    (re.compile(r"%\s*of\b", re.IGNORECASE), " * 0.01 * "),
    (re.compile(r"\bpercent(?:age)?\s+of\b|\bpct\s+of\b", re.IGNORECASE), " * 0.01 * "),
    # lone % / percent
    (re.compile(r"%"), " * 0.01"),
    (re.compile(r"\bpercent(?:age)?\b|\bpct\b", re.IGNORECASE), " * 0.01"),
    # "of" remaining after percent patterns is noise; "من" is Arabic "of/from"
    (re.compile(r"\bof\b|\bمن\b"), " "),
    # English word operators
    (re.compile(r"\btimes\b|\bmultiplied\s+by\b", re.IGNORECASE), " * "),
    (re.compile(r"\bdivided\s+by\b", re.IGNORECASE), " / "),
    (re.compile(r"\bplus\b|\badded\s+to\b", re.IGNORECASE), " + "),
    (re.compile(r"\bminus\b|\bsubtracted\s+from\b", re.IGNORECASE), " - "),
    # Strip trailing punctuation
    (re.compile(r"[?؟.!,،]+$"), ""),
]

_SQRT_OPEN_RE = re.compile(r"math\.sqrt\(")
# After phrase subs we allow digits, operators, parens, dot, comma, space, and
# ASCII letters so that "math.sqrt" survives the strip.  _DANGER_RE is the real
# security gate — it rejects anything that could call builtins or import modules.
_UNSAFE_CHARS_RE = re.compile(r"[^0-9a-zA-Z_.+\-*/().,\^ ]")
_DANGER_RE = re.compile(r"__|\bimport\b|\bexec\b|\beval\b|\bopen\b|\bos\b|\bsys\b")

# Pre-check — must look like a math expression before we spend time normalizing.
_LOOKS_LIKE_MATH_RE = re.compile(
    r"""
    (?:
        \d[\d\s]*[+\-*/^%×÷]                        # digit then operator
      | [+\-*/^%×÷]\s*\d                             # operator then digit
      | \d+\s*%\s*of\b                               # "15% of"
      | \bpercent(?:age)?\s+of\b                     # "percent of"
      | \bsquare\s+root\b                            # "square root"
      | الجذر\s+التربيعي                              # Arabic square root
      | [٠-٩]+\s*[+\-*/×÷]                          # Arabic-Indic digits + symbol operator
      | [٠-٩]+\s*٪                                   # Arabic-Indic + percent sign
      | \d+\s*٪                                      # Latin digit + ٪
      | في\s+(?:الميه|المية|المئة|المائة)            # "percent of" in Egyptian Arabic
      | كام\s+\S+\s+في\s+ال                          # "كام X في المية"
      | (?:[٠-٩]+|\d+)\s*(?:زائد|جمع|ناقص|قسمة)    # Arabic-Indic/ASCII + word operator
      | (?:زائد|جمع|ناقص|قسمة)\s*(?:[٠-٩]+|\d+)    # word operator + digits
      | (?:[٠-٩]+|\d+)\s+في\s+(?:[٠-٩]+|\d+)       # "X في Y" multiplication
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

def _format_number(value: float) -> str:
    """Format a float: strip trailing zeros, add thousands commas."""
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    formatted = f"{value:.10g}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    if "." in formatted:
        int_part, dec_part = formatted.split(".", 1)
        try:
            int_part = f"{int(int_part):,}"
        except ValueError:
            pass
        return f"{int_part}.{dec_part}"
    try:
        return f"{int(formatted):,}"
    except ValueError:
        return formatted


def to_arabic_numerals(text: str) -> str:
    """Convert ASCII digits to Arabic-Indic for Arabic language responses."""
    return text.translate(_LATIN_TO_ARABIC_DIGITS)


def quick_calc(expression: str) -> Optional[str]:
    """Evaluate a math expression safely and return a formatted result.

    Returns None if the text does not look like a math expression, so the
    caller can fall through to the LLM tier unchanged.
    """
    if not expression:
        return None
    if not _LOOKS_LIKE_MATH_RE.search(expression):
        return None

    # Step 1: normalise Arabic-Indic digits and ٪ → %
    expr = str(expression).translate(_ARABIC_TO_LATIN)

    # Step 2: apply phrase substitutions in order
    for pattern, replacement in _PHRASE_SUBS:
        expr = pattern.sub(replacement, expr)

    # Step 3: strip characters not in the allowed set.
    # Letters are allowed so "math.sqrt" survives; _DANGER_RE is the security gate.
    expr = _UNSAFE_CHARS_RE.sub(" ", expr)

    # Step 4: balance unmatched "math.sqrt(" parens
    open_count = len(_SQRT_OPEN_RE.findall(expr))
    close_count = expr.count(")")
    if open_count > close_count:
        expr += ")" * (open_count - close_count)

    # Step 5: collapse whitespace; bail if empty
    expr = " ".join(expr.split())
    if not expr:
        return None

    # Step 6: safety gate — no dunder names or dangerous builtins
    if _DANGER_RE.search(expr):
        return None

    # Step 7: eval with a minimal namespace (math module only)
    try:
        result = eval(expr, {"math": math, "__builtins__": {}})  # noqa: S307
    except Exception:
        return None

    if not isinstance(result, (int, float)) or not math.isfinite(result):
        return None

    return _format_number(float(result))
