"""
test_validators.py
------------------
Tests for Pydantic validators and data models.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from src.validators import (
    CybersecurityFactors,
    RiskResult,
    validate_factors,
    _score_to_level,
)


class TestCybersecurityFactors:
    def test_default_construction(self):
        f = CybersecurityFactors()
        assert f.likelihood == 50.0
        assert f.impact == 50.0
        assert f.exposure == 50.0
        assert f.data_sensitivity == 50.0
        assert f.threat_type == "Unknown"

    def test_valid_construction(self):
        f = CybersecurityFactors(
            threat_type="Phishing",
            attack_vector="Email",
            affected_asset="User Credentials",
            suspicious_activity="Unusual logins",
            sensitive_data_involved="Yes",
            likelihood=80.0,
            impact=75.0,
            exposure=60.0,
            data_sensitivity=90.0,
        )
        assert f.likelihood == 80.0
        assert f.threat_type == "Phishing"

    def test_clamp_above_100(self):
        f = CybersecurityFactors(likelihood=150.0, impact=200.0)
        assert f.likelihood == 100.0
        assert f.impact == 100.0

    def test_clamp_below_0(self):
        f = CybersecurityFactors(likelihood=-10.0, exposure=-50.0)
        assert f.likelihood == 0.0
        assert f.exposure == 0.0

    def test_invalid_numeric_falls_back_to_default(self):
        f = CybersecurityFactors(likelihood="not_a_number")
        assert f.likelihood == 50.0

    def test_empty_string_field_falls_back_to_unknown(self):
        f = CybersecurityFactors(threat_type="")
        assert f.threat_type == "Unknown"

    def test_none_string_field_falls_back_to_unknown(self):
        f = CybersecurityFactors(attack_vector=None)
        assert f.attack_vector == "Unknown"

    def test_whitespace_string_field_falls_back_to_unknown(self):
        f = CybersecurityFactors(affected_asset="   ")
        assert f.affected_asset == "Unknown"

    def test_boundary_values_exact(self):
        f = CybersecurityFactors(likelihood=0.0, impact=100.0)
        assert f.likelihood == 0.0
        assert f.impact == 100.0


class TestValidateFactors:
    def test_valid_dict(self):
        raw = {
            "threat_type": "Malware",
            "attack_vector": "USB",
            "likelihood": 70,
            "impact": 60,
            "exposure": 55,
            "data_sensitivity": 40,
        }
        f = validate_factors(raw)
        assert f.threat_type == "Malware"
        assert f.likelihood == 70.0

    def test_empty_dict_returns_defaults(self):
        f = validate_factors({})
        assert f.likelihood == 50.0

    def test_malformed_values_use_defaults(self):
        f = validate_factors({"likelihood": "bad", "impact": None})
        assert f.likelihood == 50.0
        assert f.impact == 50.0

    def test_extra_keys_ignored(self):
        raw = {"likelihood": 60, "unknown_key": "ignored"}
        f = validate_factors(raw)
        assert f.likelihood == 60.0

    def test_partial_dict(self):
        raw = {"likelihood": 30, "threat_type": "Ransomware"}
        f = validate_factors(raw)
        assert f.likelihood == 30.0
        assert f.threat_type == "Ransomware"
        assert f.impact == 50.0  # default


class TestRiskResult:
    def test_valid_construction(self):
        r = RiskResult(risk_score=75.0, risk_level="CRITICAL")
        assert r.risk_score == 75.0
        assert r.risk_level == "CRITICAL"

    def test_invalid_level_corrected(self):
        r = RiskResult(risk_score=30.0, risk_level="UNKNOWN_LEVEL")
        assert r.risk_level == "MEDIUM"  # corrected from score

    def test_recommendations_default_empty(self):
        r = RiskResult(risk_score=50.0, risk_level="HIGH")
        assert r.recommendations == []

    def test_recommendations_stored(self):
        r = RiskResult(
            risk_score=50.0,
            risk_level="HIGH",
            recommendations=["Change password", "Enable MFA"],
        )
        assert len(r.recommendations) == 2

    def test_score_boundary_zero(self):
        r = RiskResult(risk_score=0.0, risk_level="LOW")
        assert r.risk_score == 0.0

    def test_score_boundary_hundred(self):
        r = RiskResult(risk_score=100.0, risk_level="CRITICAL")
        assert r.risk_score == 100.0


class TestScoreToLevel:
    @pytest.mark.parametrize("score,expected", [
        (0,   "LOW"),
        (24,  "LOW"),
        (25,  "MEDIUM"),
        (49,  "MEDIUM"),
        (50,  "HIGH"),
        (74,  "HIGH"),
        (75,  "CRITICAL"),
        (100, "CRITICAL"),
    ])
    def test_mapping(self, score, expected):
        assert _score_to_level(score) == expected
