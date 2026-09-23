"""
risk_analyzer.py
----------------
Orchestration layer that ties the LLM service and the fuzzy engine together.

Flow:
  incident text
    -> LLM: extract cybersecurity factors
    -> Fuzzy Engine: compute risk score
    -> LLM: generate explanation
    -> LLM: generate recommendations
    -> return RiskResult
"""

from __future__ import annotations

import logging
from typing import Tuple

from src.validators import CybersecurityFactors, RiskResult
from src.fuzzy_engine import compute_risk_score, score_to_level
from src.llm_service import (
    extract_cybersecurity_factors,
    generate_explanation,
    generate_recommendations,
)

logger = logging.getLogger(__name__)


def analyze_incident(incident: str) -> Tuple[RiskResult, CybersecurityFactors, dict, list[str]]:
    """
    Full pipeline:
      1. Extract factors with LLM
      2. Run fuzzy inference
      3. Generate explanation with LLM
      4. Generate recommendations with LLM

    Returns
    -------
    (result, factors, errors, warnings)
      result   : RiskResult
      factors  : CybersecurityFactors (extracted values)
      debug    : dict (fuzzy debug info)
      messages : list[str] (non-fatal warning/info messages)
    """
    messages: list[str] = []

    # ------------------------------------------------------------------
    # Step 1: LLM factor extraction
    # ------------------------------------------------------------------
    factors, extraction_error = extract_cybersecurity_factors(incident)
    if extraction_error:
        messages.append(f"⚠️ Factor extraction: {extraction_error}")

    # ------------------------------------------------------------------
    # Step 2: Fuzzy inference (always runs — even with default factors)
    # ------------------------------------------------------------------
    try:
        risk_score, debug_info = compute_risk_score(
            likelihood=factors.likelihood,
            impact=factors.impact,
            exposure=factors.exposure,
            data_sensitivity=factors.data_sensitivity,
        )
    except Exception as exc:
        logger.error("Fuzzy engine error: %s", exc)
        risk_score = 50.0
        debug_info = {}
        messages.append(f"⚠️ Fuzzy engine error: {exc}. Default score used.")

    risk_level = score_to_level(risk_score)

    # ------------------------------------------------------------------
    # Step 3: LLM explanation
    # ------------------------------------------------------------------
    explanation, explanation_error = generate_explanation(
        incident=incident,
        factors=factors,
        risk_score=risk_score,
        risk_level=risk_level,
    )
    if explanation_error:
        messages.append(f"⚠️ Explanation: {explanation_error}")
        explanation = (
            f"This incident has been assessed with a risk score of "
            f"{risk_score:.1f}/100 ({risk_level}). "
            f"Please review the extracted factors for more detail."
        )

    # ------------------------------------------------------------------
    # Step 4: LLM recommendations
    # ------------------------------------------------------------------
    recommendations, recs_error = generate_recommendations(
        incident=incident,
        factors=factors,
        risk_score=risk_score,
        risk_level=risk_level,
    )
    if recs_error:
        messages.append(f"ℹ️ {recs_error}")

    # ------------------------------------------------------------------
    # Step 5: Assemble result
    # ------------------------------------------------------------------
    result = RiskResult(
        risk_score=round(risk_score, 2),
        risk_level=risk_level,
        explanation=explanation,
        recommendations=recommendations,
    )

    return result, factors, debug_info, messages
