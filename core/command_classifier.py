from core.command_parser import parse_command
from core.intent_confidence import assess_intent_confidence


def classify(text: str):
    return parse_command(text).intent


def classify_with_confidence(text: str, language: str = "en"):
    parsed = parse_command(text)
    assessment = assess_intent_confidence(text, parsed, language=language)
    return {
        "intent": parsed.intent,
        "action": parsed.action,
        "args": dict(parsed.args or {}),
        "confidence": float(assessment.confidence),
        "entity_scores": dict(assessment.entity_scores or {}),
        "should_clarify": bool(assessment.should_clarify),
        "reason": assessment.reason,
    }
