import pytest
from fraud.engine import analyze_fraud


def test_clean_transcript():
    r = analyze_fraud("I want to check my transaction history.")
    assert r.risk_score == 0
    assert r.risk_level == "Low"
    assert r.detected_signals == []
    assert r.fraud_category == "Normal Request"


def test_otp_detected():
    r = analyze_fraud("Someone from customer care asked me for my OTP urgently.")
    keywords = [s["keyword"] for s in r.detected_signals]
    assert "otp" in keywords
    assert "customer care" in keywords
    assert r.risk_score >= 60
    assert r.risk_level in ("High", "Critical")


def test_pin_theft():
    r = analyze_fraud("A person called me and asked for my mobile money PIN.")
    assert r.risk_score >= 40
    assert any(s["keyword"] == "pin" for s in r.detected_signals)
    assert r.fraud_category == "PIN theft"


def test_risk_levels():
    assert analyze_fraud("").risk_level == "Low"
    assert analyze_fraud("send money to this account").risk_level == "Medium"
    assert analyze_fraud("my otp and password and send money now").risk_level in ("High", "Critical")


def test_recommended_action_high():
    r = analyze_fraud("They want my OTP and PIN right now or my account will be blocked.")
    assert r.risk_level in ("High", "Critical")
    assert "End the call" in r.recommended_action or "Block" in r.recommended_action
