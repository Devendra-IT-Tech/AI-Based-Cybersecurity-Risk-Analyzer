"""
test_fuzzy_engine.py
--------------------
Automated tests for the fuzzy inference engine.

Tests:
  - Membership function construction
  - Fuzzification
  - Risk score output range
  - Risk categories (LOW / MEDIUM / HIGH / CRITICAL)
  - Boundary values
  - Monotonicity (higher inputs → higher scores)
  - Score-to-level mapping
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np

from src.fuzzy_engine import (
    compute_risk_score,
    get_membership_functions,
    get_universe,
    score_to_level,
    _fuzzify,
    _UNIVERSE,
    _build_mf_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mf_params():
    return _build_mf_params()


# ---------------------------------------------------------------------------
# Universe tests
# ---------------------------------------------------------------------------

class TestUniverse:
    def test_universe_range(self):
        u = get_universe()
        assert u[0] == 0.0
        assert u[-1] == 100.0

    def test_universe_length(self):
        u = get_universe()
        assert len(u) == 101  # 0 to 100 inclusive


# ---------------------------------------------------------------------------
# Membership function tests
# ---------------------------------------------------------------------------

class TestMembershipFunctions:
    def test_all_variables_present(self, mf_params):
        expected = {"likelihood", "impact", "exposure", "data_sensitivity", "risk"}
        assert set(mf_params.keys()) == expected

    def test_input_variables_have_three_terms(self, mf_params):
        for var in ["likelihood", "impact", "exposure", "data_sensitivity"]:
            assert set(mf_params[var].keys()) == {"Low", "Medium", "High"}

    def test_output_variable_has_four_terms(self, mf_params):
        assert set(mf_params["risk"].keys()) == {"Low", "Medium", "High", "Critical"}

    def test_mf_values_in_range(self):
        mfs = get_membership_functions()
        for var, terms in mfs.items():
            for term, arr in terms.items():
                assert arr.min() >= -0.01, f"{var}.{term} has values below 0"
                assert arr.max() <= 1.01, f"{var}.{term} has values above 1"

    def test_membership_at_extremes(self, mf_params):
        """Low MF should have high membership near 0; High MF near 100."""
        u = _UNIVERSE
        for var in ["likelihood", "impact", "exposure", "data_sensitivity"]:
            from src.fuzzy_engine import _mf_value
            low_mf  = _mf_value(u, mf_params[var]["Low"])
            high_mf = _mf_value(u, mf_params[var]["High"])
            assert low_mf[0]   > 0.5, f"{var} Low MF should be > 0.5 at x=0"
            assert high_mf[-1] > 0.5, f"{var} High MF should be > 0.5 at x=100"


# ---------------------------------------------------------------------------
# Fuzzification tests
# ---------------------------------------------------------------------------

class TestFuzzification:
    def test_low_value_activates_low_term(self, mf_params):
        deg = _fuzzify(_UNIVERSE, mf_params["likelihood"], 10.0)
        assert deg["Low"] > 0.5, "Likelihood=10 should strongly activate Low"

    def test_high_value_activates_high_term(self, mf_params):
        deg = _fuzzify(_UNIVERSE, mf_params["likelihood"], 90.0)
        assert deg["High"] > 0.5, "Likelihood=90 should strongly activate High"

    def test_medium_value_activates_medium_term(self, mf_params):
        deg = _fuzzify(_UNIVERSE, mf_params["likelihood"], 50.0)
        assert deg["Medium"] > 0.0, "Likelihood=50 should activate Medium"

    def test_all_degrees_in_unit_interval(self, mf_params):
        for val in [0, 25, 50, 75, 100]:
            deg = _fuzzify(_UNIVERSE, mf_params["likelihood"], float(val))
            for term, d in deg.items():
                assert 0.0 <= d <= 1.0, f"Degree out of [0,1]: {term}={d} at x={val}"


# ---------------------------------------------------------------------------
# Risk score computation tests
# ---------------------------------------------------------------------------

class TestComputeRiskScore:
    def test_output_in_range_low_scenario(self):
        score, _ = compute_risk_score(10, 10, 10, 10)
        assert 0.0 <= score <= 100.0, f"Score out of range: {score}"

    def test_output_in_range_medium_scenario(self):
        score, _ = compute_risk_score(50, 50, 50, 50)
        assert 0.0 <= score <= 100.0

    def test_output_in_range_high_scenario(self):
        score, _ = compute_risk_score(75, 80, 70, 60)
        assert 0.0 <= score <= 100.0

    def test_output_in_range_critical_scenario(self):
        score, _ = compute_risk_score(90, 95, 90, 90)
        assert 0.0 <= score <= 100.0

    def test_low_inputs_produce_low_score(self):
        score, _ = compute_risk_score(5, 5, 5, 5)
        assert score < 40.0, f"Very low inputs should yield low score, got {score}"

    def test_high_inputs_produce_high_score(self):
        score, _ = compute_risk_score(90, 90, 90, 90)
        assert score > 60.0, f"Very high inputs should yield high score, got {score}"

    def test_boundary_zero(self):
        score, _ = compute_risk_score(0, 0, 0, 0)
        assert 0.0 <= score <= 100.0

    def test_boundary_hundred(self):
        score, _ = compute_risk_score(100, 100, 100, 100)
        assert 0.0 <= score <= 100.0

    def test_monotonicity_likelihood(self):
        """Higher likelihood with equal other factors should yield higher or equal score."""
        score_low, _  = compute_risk_score(10, 50, 50, 50)
        score_high, _ = compute_risk_score(90, 50, 50, 50)
        assert score_high >= score_low, (
            f"Higher likelihood should not decrease risk: {score_low} -> {score_high}"
        )

    def test_monotonicity_impact(self):
        score_low, _  = compute_risk_score(50, 10, 50, 50)
        score_high, _ = compute_risk_score(50, 90, 50, 50)
        assert score_high >= score_low

    def test_debug_info_returned(self):
        _, debug = compute_risk_score(50, 50, 50, 50)
        assert isinstance(debug, dict)
        assert "fuzzification" in debug

    def test_clamping_above_100(self):
        score, _ = compute_risk_score(150, 200, 300, 999)
        assert 0.0 <= score <= 100.0

    def test_clamping_below_0(self):
        score, _ = compute_risk_score(-10, -50, -100, -5)
        assert 0.0 <= score <= 100.0


# ---------------------------------------------------------------------------
# Score-to-level mapping tests
# ---------------------------------------------------------------------------

class TestScoreToLevel:
    @pytest.mark.parametrize("score,expected", [
        (0,   "LOW"),
        (10,  "LOW"),
        (24,  "LOW"),
        (25,  "MEDIUM"),
        (40,  "MEDIUM"),
        (49,  "MEDIUM"),
        (50,  "HIGH"),
        (65,  "HIGH"),
        (74,  "HIGH"),
        (75,  "CRITICAL"),
        (90,  "CRITICAL"),
        (100, "CRITICAL"),
    ])
    def test_level_mapping(self, score, expected):
        assert score_to_level(score) == expected

    def test_score_to_level_boundary(self):
        assert score_to_level(0.0) == "LOW"
        assert score_to_level(100.0) == "CRITICAL"


# ---------------------------------------------------------------------------
# Scenario-based integration tests
# ---------------------------------------------------------------------------

class TestScenarios:
    def test_phishing_scenario(self):
        """High-likelihood phishing with sensitive data → HIGH or CRITICAL."""
        score, _ = compute_risk_score(
            likelihood=85,
            impact=80,
            exposure=70,
            data_sensitivity=85,
        )
        level = score_to_level(score)
        assert level in ("HIGH", "CRITICAL"), f"Got {level} (score={score:.1f})"

    def test_low_risk_scenario(self):
        """Low indicators → LOW."""
        score, _ = compute_risk_score(8, 8, 8, 8)
        level = score_to_level(score)
        assert level == "LOW", f"Got {level} (score={score:.1f})"

    def test_medium_risk_scenario(self):
        """Moderate indicators → LOW or MEDIUM."""
        score, _ = compute_risk_score(40, 40, 40, 40)
        level = score_to_level(score)
        assert level in ("LOW", "MEDIUM"), f"Got {level} (score={score:.1f})"

    def test_critical_scenario(self):
        """Maximum indicators → CRITICAL."""
        score, _ = compute_risk_score(95, 95, 95, 95)
        level = score_to_level(score)
        assert level == "CRITICAL", f"Got {level} (score={score:.1f})"
