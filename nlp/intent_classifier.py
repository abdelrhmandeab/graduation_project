"""Rule-based bilingual intent classifier tolerant to noisy STT text."""

from __future__ import annotations

from typing import Dict, List

from nlp.fuzzy_matcher import find_keyword_matches, normalize_text
from nlp.keyword_engine import INTENTS

ACTION_WEIGHT = 2
TARGET_WEIGHT = 1
MATCH_THRESHOLD = 70
MIN_INTENT_SCORE = 1
MIN_CONFIDENCE = 0.35
SUGGEST_THRESHOLD = 0.45


def _unique_sorted_keywords(matches: List[tuple[str, int]]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for keyword, _score in matches:
        if keyword in seen:
            continue
        seen.add(keyword)
        ordered.append(keyword)
    return ordered


def _evaluate_intent(text: str, intent_name: str, payload: Dict[str, List[str]]) -> Dict[str, object]:
    actions = list(payload.get("actions", []))
    targets = list(payload.get("targets", []))

    action_matches = find_keyword_matches(text, actions, threshold=MATCH_THRESHOLD)
    target_matches = find_keyword_matches(text, targets, threshold=MATCH_THRESHOLD)

    has_action = len(action_matches) > 0
    has_target = len(target_matches) > 0

    score = 0
    if has_action:
        score += ACTION_WEIGHT
    if has_target:
        score += TARGET_WEIGHT

    if score == 0:
        return {
            "intent": intent_name,
            "score": 0,
            "confidence": 0.0,
            "matched_keywords": [],
        }

    max_points = ACTION_WEIGHT + (TARGET_WEIGHT if targets else 0)
    quality_parts: List[float] = []
    if has_action:
        quality_parts.append(float(action_matches[0][1]) / 100.0)
    if has_target:
        quality_parts.append(float(target_matches[0][1]) / 100.0)

    quality = sum(quality_parts) / len(quality_parts) if quality_parts else 0.0
    base_ratio = float(score) / float(max_points or 1)
    confidence = base_ratio * (0.65 + 0.35 * quality)

    # Missing-word tolerance: a strong target hit (e.g., "يوتيوب") can be enough.
    if has_target and not has_action:
        target_quality = float(target_matches[0][1]) / 100.0
        confidence = max(confidence, 0.30 + (0.50 * target_quality))

    all_matches = action_matches + target_matches
    return {
        "intent": intent_name,
        "score": score,
        "confidence": float(round(confidence, 3)),
        "matched_keywords": _unique_sorted_keywords(all_matches)[:6],
    }


def classify_intent(text: str) -> Dict[str, object]:
    """Classify noisy bilingual text into one intent using weighted keyword/fuzzy scoring."""
    normalized = normalize_text(text)
    if not normalized:
        return {"intent": "unknown", "confidence": 0.0, "matched_keywords": []}

    candidates: List[Dict[str, object]] = []
    for intent_name, payload in INTENTS.items():
        result = _evaluate_intent(normalized, intent_name, payload)
        if int(result["score"]) > 0:
            candidates.append(result)

    if not candidates:
        return {"intent": "unknown", "confidence": 0.0, "matched_keywords": []}

    candidates.sort(key=lambda item: (int(item["score"]), float(item["confidence"])), reverse=True)
    best = candidates[0]

    # If two intents are nearly tied, avoid false confident classification.
    if len(candidates) > 1:
        second = candidates[1]
        if int(best["score"]) == int(second["score"]):
            if abs(float(best["confidence"]) - float(second["confidence"])) < 0.08:
                return {"intent": "unknown", "confidence": 0.0, "matched_keywords": []}

    if int(best["score"]) < MIN_INTENT_SCORE or float(best["confidence"]) < MIN_CONFIDENCE:
        return {"intent": "unknown", "confidence": 0.0, "matched_keywords": []}

    return {
        "intent": str(best["intent"]),
        "confidence": float(best["confidence"]),
        "matched_keywords": list(best["matched_keywords"]),
    }


def suggest_intent(text: str) -> Dict[str, object]:
    """Return the closest intent candidate using fuzzy keyword proximity."""
    normalized = normalize_text(text)
    if not normalized:
        return {"intent": "unknown", "confidence": 0.0, "matched_keywords": []}

    best_intent = "unknown"
    best_confidence = 0.0
    best_keyword = ""

    for intent_name, payload in INTENTS.items():
        keywords = list(payload.get("actions", [])) + list(payload.get("targets", []))
        matches = find_keyword_matches(normalized, keywords, threshold=1)
        if not matches:
            continue

        top_keyword, top_score = matches[0]
        confidence = float(top_score) / 100.0
        if confidence > best_confidence:
            best_intent = intent_name
            best_confidence = confidence
            best_keyword = top_keyword

    if best_confidence < SUGGEST_THRESHOLD:
        return {"intent": "unknown", "confidence": 0.0, "matched_keywords": []}

    return {
        "intent": best_intent,
        "confidence": float(round(best_confidence, 3)),
        "matched_keywords": [best_keyword] if best_keyword else [],
    }
