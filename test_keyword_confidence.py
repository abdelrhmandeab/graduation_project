#!/usr/bin/env python
from nlp.intent_classifier import classify_intent

test_cases = [
    "raise the volume",
    "turn up the sound",
    "make it louder",
    "lower the volume",
    "take a screenshot",
]

for text in test_cases:
    result = classify_intent(text)
    intent = result.get('intent')
    confidence = result.get('confidence')
    keywords = result.get('matched_keywords')
    print(f"{text:<30} → {intent:<20} conf={confidence:.2f} keywords={keywords}")
