"""
llm_service.py
--------------
LangChain integration using Google Gemini.

Responsibilities:
  1. Build the LangChain pipeline.
  2. Call the LLM for factor extraction.
  3. Call the LLM for risk explanation.
  4. Call the LLM for defensive recommendations.
  5. Validate / sanitise all LLM outputs.
  6. Never raise — always return a safe result with an error message.
"""

from __future__ import annotations

import json
import os
import re
import logging
from typing import Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from src.prompts import extraction_prompt, explanation_prompt, recommendations_prompt
from src.validators import CybersecurityFactors, validate_factors

logger = logging.getLogger(__name__)

def _get_llm(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    """
    Create and return a Google Gemini chat model.
    Raises a clear ValueError if GOOGLE_API_KEY is not set.
    Model is configurable via GEMINI_MODEL environment variable (default: gemini-3.8-flash).
    """
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Please add your Gemini API key to the .env file, sidebar, or Streamlit Secrets."
        )
    model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        convert_system_message_to_human=True,
    )


# ---------------------------------------------------------------------------
# Helper: extract JSON from LLM response string
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Attempt to parse a JSON object from a raw LLM response string.
    Handles markdown code fences and leading/trailing text.
    """
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`")

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object using regex
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:300]}")


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from a raw LLM response string."""
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`")

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON array from LLM response: {text[:300]}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cybersecurity_factors(incident: str) -> Tuple[CybersecurityFactors, str]:
    """
    Use LangChain + Gemini to extract structured cybersecurity factors
    from a natural-language incident description.

    Returns
    -------
    (factors, error_message)
    error_message is empty string on success.
    """
    if not incident or not incident.strip():
        return CybersecurityFactors(), "Incident description is empty."

    try:
        llm = _get_llm(temperature=0.1)
        chain = extraction_prompt | llm | StrOutputParser()
        raw_response = chain.invoke({"incident": incident.strip()})
        raw_dict = _extract_json(raw_response)
        factors = validate_factors(raw_dict)
        return factors, ""

    except ValueError as ve:
        # Missing API key or JSON parse error
        logger.warning("LLM extraction failed: %s", ve)
        return CybersecurityFactors(), str(ve)

    except Exception as exc:
        logger.error("Unexpected error in extract_cybersecurity_factors: %s", exc)
        return (
            CybersecurityFactors(),
            f"LLM extraction failed: {type(exc).__name__}: {exc}",
        )


def generate_explanation(
    incident: str,
    factors: CybersecurityFactors,
    risk_score: float,
    risk_level: str,
) -> Tuple[str, str]:
    """
    Generate a plain-English explanation of the risk assessment.

    Returns
    -------
    (explanation_text, error_message)
    """
    try:
        llm = _get_llm(temperature=0.3)
        chain = explanation_prompt | llm | StrOutputParser()
        explanation = chain.invoke(
            {
                "incident": incident.strip(),
                "threat_type": factors.threat_type,
                "attack_vector": factors.attack_vector,
                "affected_asset": factors.affected_asset,
                "suspicious_activity": factors.suspicious_activity,
                "sensitive_data_involved": factors.sensitive_data_involved,
                "likelihood": round(factors.likelihood, 1),
                "impact": round(factors.impact, 1),
                "exposure": round(factors.exposure, 1),
                "data_sensitivity": round(factors.data_sensitivity, 1),
                "risk_score": round(risk_score, 1),
                "risk_level": risk_level,
            }
        )
        return explanation.strip(), ""

    except Exception as exc:
        logger.error("Explanation generation failed: %s", exc)
        return "", f"Explanation generation failed: {type(exc).__name__}: {exc}"


def generate_recommendations(
    incident: str,
    factors: CybersecurityFactors,
    risk_score: float,
    risk_level: str,
) -> Tuple[list[str], str]:
    """
    Generate a list of safe defensive recommendations.

    Returns
    -------
    (recommendations_list, error_message)
    """
    try:
        llm = _get_llm(temperature=0.3)
        chain = recommendations_prompt | llm | StrOutputParser()
        raw = chain.invoke(
            {
                "incident": incident.strip(),
                "threat_type": factors.threat_type,
                "risk_level": risk_level,
                "risk_score": round(risk_score, 1),
                "affected_asset": factors.affected_asset,
            }
        )
        recs = _extract_json_array(raw)
        # Ensure all items are strings
        recs = [str(r).strip() for r in recs if r]
        return recs, ""

    except Exception as exc:
        logger.error("Recommendations generation failed: %s", exc)
        # Return safe fallback recommendations
        fallback = [
            "Report the incident to your IT/security team immediately.",
            "Change passwords for any potentially compromised accounts.",
            "Enable multi-factor authentication where available.",
            "Review recent account activity for unauthorised access.",
            "Do not share credentials and be cautious of suspicious emails.",
        ]
        return fallback, f"Note: Using fallback recommendations. ({type(exc).__name__})"
