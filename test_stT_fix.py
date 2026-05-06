#!/usr/bin/env python
from core.command_router import route_command

test_cases = [
    ("RISK AND EPS", "Should not trigger volume"),
    ("raise the volume", "Should trigger volume_up"),
    ("turn up the sound", "Should trigger volume_up"),
    ("make it louder", "Should trigger volume_up"),
    ("lower the volume", "Should trigger volume_down"),
    ("unlock my files", "Should NOT trigger screenshot"),
    ("take a screenshot", "Should trigger screenshot"),
]

for text, expected in test_cases:
    result = route_command(text, detected_language='en')
    print(f"\nInput: {text}")
    print(f"Expected: {expected}")
    print(f"Response: {result[:60]}...")
