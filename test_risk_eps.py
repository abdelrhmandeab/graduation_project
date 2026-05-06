#!/usr/bin/env python
import sys
sys.path.insert(0, '.')

# Test what the parser returns for "RISK AND EPS"
from core.command_parser import parse_command

parsed = parse_command("RISK AND EPS")
print(f"Parser Intent: {parsed.intent}")
print(f"Parser Action: {parsed.action}")
print(f"Parser Args: {parsed.args}")
print(f"Parser Normalized: {parsed.normalized}")

# Test what semantic router returns
from nlp.semantic_router import classify_semantic
result = classify_semantic("RISK AND EPS")
if result:
    intent, confidence = result
    print(f"\nSemantic Intent: {intent}")
    print(f"Semantic Confidence: {confidence}")
else:
    print("\nSemantic Router returned None")
