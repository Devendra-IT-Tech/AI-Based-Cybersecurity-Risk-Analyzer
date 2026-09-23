"""
llm_service.py
--------------
LangChain integration using Groq.

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

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser

from src.prompts import (
    extraction_prompt,
    explanation_prompt,
    recommendations_prompt,
)
from src.validators import CybersecurityFactors, validate_factors

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0.2) -> ChatGroq:
    """
    Create and return a Groq chat model.

    API key is read from GROQ_API_KEY.
    Model is configurable through GROQ_MODEL.
    """

    api_key = os.environ.get("GROQ_API_KEY", "").strip()

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable is not set. "
            "Please add your Groq API key to the .env file "
            "or Streamlit Secrets."
        )

    model = (
        os.environ.get(
            "GROQ_MODEL",
            "openai/gpt-oss-20b",
        ).strip()
        or "openai/gpt-oss-20b"
    )

    return ChatGroq(
        model=model,
        groq_api_key=api_key,
        temperature=temperature,
    )


# ---------------------------------------------------------------------------
# Helper: extract JSON from LLM response string
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """
    Attempt to parse a JSON object from a raw LLM response string.
    Handles markdown code fences and leading/trailing text.
    """

    text = re.sub(
        r"```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = text.strip().strip("`")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from LLM response: {text[:300]}"
    )


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from a raw LLM response string."""

    text = re.sub(
        r"```(?:json)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

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

    raise ValueError(
        f"Could not parse JSON array from LLM response: {text[:300]}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_cybersecurity_factors(
    incident: str,
) -> Tuple[CybersecurityFactors, str]:
    """
    Use LangChain + Groq to extract structured cybersecurity
    factors from a natural-language incident.

    Returns:
        (factors, error_message)
    """

    if not incident or not incident.strip():
        return (
            CybersecurityFactors(),
            "Incident description is empty.",
        )

    try:
        llm = _get_llm(temperature=0.1)

        chain = extraction_prompt | llm | StrOutputParser()

        raw_response = chain.invoke(
            {
                "incident": incident.strip()
            }
        )

        raw_dict = _extract_json(raw_response)

        factors = validate_factors(raw_dict)

        return factors, ""

    except ValueError as ve:

        logger.warning(
            "LLM extraction failed: %s",
            ve,
        )

        return (
            CybersecurityFactors(),
            str(ve),
        )

    except Exception as exc:

        logger.error(
            "Unexpected error in extract_cybersecurity_factors: %s",
            exc,
        )

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
                "data_sensitivity": round(
                    factors.data_sensitivity,
                    1,
                ),
                "risk_score": round(risk_score, 1),
                "risk_level": risk_level,
            }
        )

        return explanation.strip(), ""

    except Exception as exc:

        logger.error(
            "Explanation generation failed: %s",
            exc,
        )

        return (
            "",
            f"Explanation generation failed: "
            f"{type(exc).__name__}: {exc}",
        )


def generate_recommendations(
    incident: str,
    factors: CybersecurityFactors,
    risk_score: float,
    risk_level: str,
) -> Tuple[list[str], str]:
    """
    Generate a list of safe defensive recommendations.
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

        recs = [
            str(r).strip()
            for r in recs
            if r
        ]

        return recs, ""

    except Exception as exc:

        logger.error(
            "Recommendations generation failed: %s",
            exc,
        )

        fallback = [
            "Report the incident to your IT/security team immediately.",
            "Change passwords for potentially compromised accounts.",
            "Enable multi-factor authentication where available.",
            "Review recent account activity for unauthorised access.",
            "Do not share credentials and be cautious of suspicious emails.",
        ]

        return (
            fallback,
            "Note: Using fallback recommendations. "
            f"({type(exc).__name__})",
        )
