"""
validators.py
-------------
Pydantic models for validating LLM-extracted cybersecurity factors
and other application data structures.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class CybersecurityFactors(BaseModel):
    """
    Structured cybersecurity factors extracted by the LLM from
    a natural-language incident description.

    All numeric fields are normalised to [0, 100].
    """

    threat_type: str = Field(
        default="Unknown",
        description="Category of threat (e.g. Phishing, Malware, Insider Threat)",
    )
    attack_vector: str = Field(
        default="Unknown",
        description="How the attack was carried out (e.g. Email, Web, USB)",
    )
    affected_asset: str = Field(
        default="Unknown",
        description="The resource targeted (e.g. User credentials, Database, Workstation)",
    )
    suspicious_activity: str = Field(
        default="Unknown",
        description="Observed suspicious behaviour described in the incident",
    )
    sensitive_data_involved: str = Field(
        default="Unknown",
        description="Whether sensitive data is potentially exposed (Yes / No / Unknown)",
    )

    # --- Numeric fuzzy inputs (0-100) ---
    likelihood: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Estimated probability that the threat is genuine (0-100)",
    )
    impact: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Potential damage if the threat materialises (0-100)",
    )
    exposure: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="How exposed the affected systems/data are (0-100)",
    )
    data_sensitivity: float = Field(
        default=50.0,
        ge=0.0,
        le=100.0,
        description="Sensitivity classification of the data at risk (0-100)",
    )

    @field_validator("likelihood", "impact", "exposure", "data_sensitivity", mode="before")
    @classmethod
    def clamp_to_range(cls, v: object) -> float:
        """Clamp any numeric value to [0, 100]."""
        try:
            val = float(v)
        except (TypeError, ValueError):
            return 50.0  # safe default
        return max(0.0, min(100.0, val))

    @field_validator(
        "threat_type", "attack_vector", "affected_asset",
        "suspicious_activity", "sensitive_data_involved",
        mode="before",
    )
    @classmethod
    def sanitise_string(cls, v: object) -> str:
        """Ensure string fields are non-empty strings."""
        if not isinstance(v, str) or not v.strip():
            return "Unknown"
        return v.strip()


class RiskResult(BaseModel):
    """Final risk assessment output after fuzzy inference."""

    risk_score: float = Field(ge=0.0, le=100.0, description="Defuzzified risk score")
    risk_level: str = Field(description="LOW | MEDIUM | HIGH | CRITICAL")
    explanation: str = Field(default="", description="LLM-generated explanation")
    recommendations: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_risk_level(self) -> "RiskResult":
        allowed = {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if self.risk_level not in allowed:
            self.risk_level = _score_to_level(self.risk_score)
        return self


def _score_to_level(score: float) -> str:
    """Map a numeric score to a risk level string."""
    if score < 25:
        return "LOW"
    elif score < 50:
        return "MEDIUM"
    elif score < 75:
        return "HIGH"
    else:
        return "CRITICAL"


def validate_factors(raw: dict) -> CybersecurityFactors:
    """
    Validate and coerce a raw dictionary (e.g. from LLM JSON output)
    into a CybersecurityFactors object.

    Never raises -- always returns a valid object (with defaults on error).
    """
    try:
        return CybersecurityFactors(**raw)
    except Exception:
        # Fall back to safe defaults
        return CybersecurityFactors()
