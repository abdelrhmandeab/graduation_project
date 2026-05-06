from nlp.intent_classifier import classify_intent


def test_noisy_english_text_does_not_become_search_or_volume_command():
    assert classify_intent("Set up the timer for 5 minutes on Windows clock. Spot my bunker.") == {
        "intent": "unknown",
        "confidence": 0.0,
        "matched_keywords": [],
    }
    assert classify_intent("RISK AND EPS") == {
        "intent": "unknown",
        "confidence": 0.0,
        "matched_keywords": [],
    }


def test_clear_volume_command_still_routes():
    result = classify_intent("raise the volume")
    assert result["intent"] == "volume_up"
    assert result["confidence"] >= 1.0 or result["confidence"] >= 0.9
