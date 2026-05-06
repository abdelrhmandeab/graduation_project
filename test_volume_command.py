#!/usr/bin/env python
from core.command_parser import parse_command
from nlp.semantic_router import classify_semantic

text = "raise the volume"

# Test parser
parsed = parse_command(text)
print(f"Parser: {parsed.intent} / {parsed.action} / {parsed.args}")

# Test semantic router
result = classify_semantic(text)
if result:
    intent, confidence = result
    print(f"Semantic: {intent} / confidence={confidence:.3f}")
else:
    print("Semantic: None")
