from __future__ import annotations
import re
from dataclasses import dataclass, field

# Longer phrases first so they match before substrings
_SIGNALS: list[tuple[str, int, str]] = [
    ("account will be blocked", 25, "Urgency pressure"),
    ("account blocked", 25, "Urgency pressure"),
    ("sim stopped working", 35, "SIM-swap risk"),
    ("sim swap", 35, "SIM-swap risk"),
    ("do not tell anyone", 30, "Secrecy pressure"),
    ("don't tell anyone", 30, "Secrecy pressure"),
    ("customer care", 20, "Fake authority"),
    ("send money", 30, "Financial manipulation"),
    ("password", 35, "Account takeover"),
    ("otp", 40, "OTP theft"),
    ("pin", 40, "PIN theft"),
]

_RISK_TABLE: list[tuple[int, str, str]] = [
    (81, "Critical", "Block immediately and contact official support."),
    (51, "High", "Do not comply. End the call and verify through official channels."),
    (21, "Medium", "Be cautious. Verify the caller's identity before sharing anything."),
    (0,  "Low",     "Allow and log. No immediate action required."),
]


@dataclass
class FraudResult:
    detected_signals: list[dict] = field(default_factory=list)
    risk_score: int = 0
    risk_level: str = "Low"
    fraud_category: str = "Normal Request"
    recommended_action: str = "Allow and log. No immediate action required."


def analyze_fraud(transcript: str) -> FraudResult:
    text = transcript.lower()
    result = FraudResult()
    seen: set[str] = set()
    category_scores: dict[str, int] = {}

    for phrase, points, category in _SIGNALS:
        if phrase in seen:
            continue
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            seen.add(phrase)
            result.detected_signals.append({"keyword": phrase, "points": points, "category": category})
            result.risk_score += points
            category_scores[category] = category_scores.get(category, 0) + points

    for threshold, level, action in _RISK_TABLE:
        if result.risk_score >= threshold:
            result.risk_level = level
            result.recommended_action = action
            break

    if category_scores:
        result.fraud_category = max(category_scores, key=lambda k: category_scores[k])

    return result
