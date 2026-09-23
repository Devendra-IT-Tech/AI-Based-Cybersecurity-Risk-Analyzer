"""
fuzzy_engine.py
---------------
Genuine Fuzzy Inference System using scikit-fuzzy (skfuzzy).

Pipeline:
  1. Universe of Discourse definition
  2. Membership function creation (triangular / trapezoidal)
  3. Fuzzification
  4. Fuzzy rule evaluation (Mamdani-style)
  5. Aggregation (maximum)
  6. Defuzzification (centroid)
  7. Risk score output + membership function data for visualisation
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
import skfuzzy as fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Universe ranges — all inputs/output on [0, 100]
# ---------------------------------------------------------------------------
_UNIVERSE = np.arange(0, 101, 1, dtype=float)

# ---------------------------------------------------------------------------
# Membership function definitions (for visualisation & engine)
# ---------------------------------------------------------------------------

def _build_mf_params() -> Dict[str, Dict[str, list]]:
    """
    Returns the membership-function parameters used by the fuzzy engine.
    Format: {variable: {term: [a, b, c] or [a, b, c, d]}}
    Triangular MFs use [a, b, c]; Trapezoidal use [a, b, c, d].
    """
    return {
        "likelihood": {
            "Low":    [0,  0,  35, 50],   # trapezoid
            "Medium": [30, 50, 70],         # triangle
            "High":   [55, 75, 100, 100],   # trapezoid
        },
        "impact": {
            "Low":    [0,  0,  30, 45],
            "Medium": [30, 50, 70],
            "High":   [55, 75, 100, 100],
        },
        "exposure": {
            "Low":    [0,  0,  30, 45],
            "Medium": [30, 50, 70],
            "High":   [55, 75, 100, 100],
        },
        "data_sensitivity": {
            "Low":    [0,  0,  30, 45],
            "Medium": [30, 50, 70],
            "High":   [55, 75, 100, 100],
        },
        "risk": {
            "Low":      [0,  0,  20, 30],
            "Medium":   [20, 35, 55],
            "High":     [45, 65, 80],
            "Critical": [70, 85, 100, 100],
        },
    }


def _mf_value(universe: np.ndarray, params: list) -> np.ndarray:
    """Apply triangular or trapezoidal MF based on param count."""
    if len(params) == 3:
        return fuzz.trimf(universe, params)
    elif len(params) == 4:
        return fuzz.trapmf(universe, params)
    else:
        raise ValueError(f"Unexpected MF param count: {len(params)}")


def _fuzzify(universe: np.ndarray, mf_params: dict, crisp_value: float) -> Dict[str, float]:
    """
    Fuzzification step.
    Returns a dict {term: membership_degree} for a given crisp input value.
    """
    result = {}
    for term, params in mf_params.items():
        mf = _mf_value(universe, params)
        result[term] = float(fuzz.interp_membership(universe, mf, crisp_value))
    return result


# ---------------------------------------------------------------------------
# Main fuzzy inference function
# ---------------------------------------------------------------------------

def compute_risk_score(
    likelihood: float,
    impact: float,
    exposure: float,
    data_sensitivity: float,
) -> Tuple[float, Dict]:
    """
    Run the complete Fuzzy Inference System.

    Parameters
    ----------
    likelihood        : float in [0, 100]
    impact            : float in [0, 100]
    exposure          : float in [0, 100]
    data_sensitivity  : float in [0, 100]

    Returns
    -------
    (risk_score, debug_info)
    risk_score  : defuzzified value in [0, 100]
    debug_info  : dict with fuzzification results and rule activations
    """
    # Clamp inputs
    likelihood       = float(np.clip(likelihood,       0, 100))
    impact           = float(np.clip(impact,           0, 100))
    exposure         = float(np.clip(exposure,         0, 100))
    data_sensitivity = float(np.clip(data_sensitivity, 0, 100))

    mf_params = _build_mf_params()

    # ------------------------------------------------------------------ #
    # Step 1 – Fuzzification
    # ------------------------------------------------------------------ #
    lik_deg  = _fuzzify(_UNIVERSE, mf_params["likelihood"],       likelihood)
    imp_deg  = _fuzzify(_UNIVERSE, mf_params["impact"],           impact)
    exp_deg  = _fuzzify(_UNIVERSE, mf_params["exposure"],         exposure)
    dat_deg  = _fuzzify(_UNIVERSE, mf_params["data_sensitivity"], data_sensitivity)

    # ------------------------------------------------------------------ #
    # Step 2 – Fuzzy Rule Base (Mamdani-style)
    #   Each rule produces a clipped output MF for the risk variable.
    # ------------------------------------------------------------------ #
    risk_mfs = mf_params["risk"]

    # Helper: clip output MF to activation strength
    def _clip(term: str, strength: float) -> np.ndarray:
        mf = _mf_value(_UNIVERSE, risk_mfs[term])
        return np.fmin(strength, mf)

    # --- Rules ---
    # R1: All Low → Low
    r1 = _clip("Low",
        np.fmin(np.fmin(lik_deg["Low"], imp_deg["Low"]), exp_deg["Low"]))

    # R2: All Medium → Medium
    r2 = _clip("Medium",
        np.fmin(np.fmin(lik_deg["Medium"], imp_deg["Medium"]), exp_deg["Medium"]))

    # R3: All High → High
    r3 = _clip("High",
        np.fmin(np.fmin(lik_deg["High"], imp_deg["High"]), exp_deg["High"]))

    # R4: Likelihood High + Impact High + Exposure High → Critical
    r4 = _clip("Critical",
        np.fmin(np.fmin(lik_deg["High"], imp_deg["High"]), exp_deg["High"]))

    # R5: Impact High + Exposure High → High
    r5 = _clip("High",
        np.fmin(imp_deg["High"], exp_deg["High"]))

    # R6: Likelihood High + Exposure Medium → High
    r6 = _clip("High",
        np.fmin(lik_deg["High"], exp_deg["Medium"]))

    # R7: Likelihood Medium + Impact High + Exposure High → High
    r7 = _clip("High",
        np.fmin(np.fmin(lik_deg["Medium"], imp_deg["High"]), exp_deg["High"]))

    # R8: Data Sensitivity High + Impact High → High
    r8 = _clip("High",
        np.fmin(dat_deg["High"], imp_deg["High"]))

    # R9: Data Sensitivity High + Likelihood High → Critical
    r9 = _clip("Critical",
        np.fmin(dat_deg["High"], lik_deg["High"]))

    # R10: Likelihood Low + Impact Low → Low
    r10 = _clip("Low",
        np.fmin(lik_deg["Low"], imp_deg["Low"]))

    # R11: Likelihood Low + Impact Medium → Low
    r11 = _clip("Low",
        np.fmin(lik_deg["Low"], imp_deg["Medium"]))

    # R12: Likelihood Medium + Impact Low → Low
    r12 = _clip("Low",
        np.fmin(lik_deg["Medium"], imp_deg["Low"]))

    # R13: Likelihood Medium + Impact Medium → Medium
    r13 = _clip("Medium",
        np.fmin(lik_deg["Medium"], imp_deg["Medium"]))

    # R14: Likelihood High + Impact Low → Medium
    r14 = _clip("Medium",
        np.fmin(lik_deg["High"], imp_deg["Low"]))

    # R15: Likelihood High + Impact Medium → High
    r15 = _clip("High",
        np.fmin(lik_deg["High"], imp_deg["Medium"]))

    # R16: Data Sensitivity High + Exposure High → High
    r16 = _clip("High",
        np.fmin(dat_deg["High"], exp_deg["High"]))

    # R17: Likelihood Low + Data Sensitivity High → Medium
    r17 = _clip("Medium",
        np.fmin(lik_deg["Low"], dat_deg["High"]))

    # ------------------------------------------------------------------ #
    # Step 3 – Aggregation (union = element-wise maximum)
    # ------------------------------------------------------------------ #
    # Aggregate Low outputs
    agg_low = np.fmax(r1, np.fmax(r10, np.fmax(r11, r12)))

    # Aggregate Medium outputs
    agg_medium = np.fmax(r2, np.fmax(r13, np.fmax(r14, r17)))

    # Aggregate High outputs
    agg_high = np.fmax(r3, np.fmax(r5, np.fmax(r6, np.fmax(r7, np.fmax(r8, np.fmax(r15, r16))))))

    # Aggregate Critical outputs
    agg_critical = np.fmax(r4, r9)

    # Final aggregated output (union of all output MFs)
    aggregated = np.fmax(agg_low, np.fmax(agg_medium, np.fmax(agg_high, agg_critical)))

    # ------------------------------------------------------------------ #
    # Step 4 – Defuzzification (centroid method)
    # ------------------------------------------------------------------ #
    # Guard: if aggregated surface is all zeros, return a default score
    if aggregated.sum() == 0:
        risk_score = 50.0
        logger.warning("Aggregated fuzzy output was zero; defaulting risk score to 50.")
    else:
        risk_score = float(fuzz.defuzz(_UNIVERSE, aggregated, "centroid"))
        risk_score = float(np.clip(risk_score, 0.0, 100.0))

    # ------------------------------------------------------------------ #
    # Debug / transparency info
    # ------------------------------------------------------------------ #
    debug_info = {
        "fuzzification": {
            "likelihood": lik_deg,
            "impact": imp_deg,
            "exposure": exp_deg,
            "data_sensitivity": dat_deg,
        },
        "rule_activations": {
            "R1_AllLow_Low":                  float(lik_deg["Low"]),
            "R2_AllMed_Medium":               float(lik_deg["Medium"]),
            "R3_AllHigh_High":                float(lik_deg["High"]),
            "R4_AllHigh_Critical":            float(np.fmin(np.fmin(lik_deg["High"], imp_deg["High"]), exp_deg["High"])),
            "R5_ImpHighExpHigh_High":         float(np.fmin(imp_deg["High"], exp_deg["High"])),
            "R6_LikHighExpMed_High":          float(np.fmin(lik_deg["High"], exp_deg["Medium"])),
            "R7_LikMedImpHighExpHigh_High":   float(np.fmin(np.fmin(lik_deg["Medium"], imp_deg["High"]), exp_deg["High"])),
        },
        "aggregated_output": aggregated.tolist(),
    }

    return risk_score, debug_info


# ---------------------------------------------------------------------------
# Membership function data for visualisation
# ---------------------------------------------------------------------------

def get_membership_functions() -> Dict[str, Dict[str, np.ndarray]]:
    """
    Return computed membership function arrays for all variables.
    Used by the Streamlit UI to plot fuzzy sets.

    Returns
    -------
    {variable_name: {term_name: mf_array_over_universe}}
    """
    mf_params = _build_mf_params()
    result = {}
    for variable, terms in mf_params.items():
        result[variable] = {}
        for term, params in terms.items():
            result[variable][term] = _mf_value(_UNIVERSE, params)
    return result


def get_universe() -> np.ndarray:
    """Return the shared universe of discourse array."""
    return _UNIVERSE.copy()


def score_to_level(score: float) -> str:
    """Map a defuzzified score to a human-readable risk level."""
    score = float(np.clip(score, 0, 100))
    if score < 25:
        return "LOW"
    elif score < 50:
        return "MEDIUM"
    elif score < 75:
        return "HIGH"
    else:
        return "CRITICAL"
