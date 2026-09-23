"""
app.py
------
Main Streamlit application for the AI-Based Cybersecurity Threat Risk Analyzer.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys
import json
import logging
from datetime import datetime, timezone

import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.risk_analyzer import analyze_incident
from src.fuzzy_engine import get_membership_functions, get_universe, score_to_level
from utils.sample_cases import SAMPLE_CASES, get_sample_titles, get_sample_by_title

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="AI Cybersecurity Threat Risk Analyzer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a73e8;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #5f6368;
        margin-bottom: 1.5rem;
    }
    .risk-badge-low      { background:#34a853; color:white; padding:8px 20px;
                            border-radius:8px; font-size:1.4rem; font-weight:700; }
    .risk-badge-medium   { background:#fbbc04; color:#333;  padding:8px 20px;
                            border-radius:8px; font-size:1.4rem; font-weight:700; }
    .risk-badge-high     { background:#ea4335; color:white; padding:8px 20px;
                            border-radius:8px; font-size:1.4rem; font-weight:700; }
    .risk-badge-critical { background:#7b0014; color:white; padding:8px 20px;
                            border-radius:8px; font-size:1.4rem; font-weight:700; }
    .factor-card { background:#f8f9fa; border-radius:8px; padding:12px;
                   margin-bottom:8px; border-left:4px solid #1a73e8; }
    .section-header { font-size:1.2rem; font-weight:600; color:#202124;
                      margin-top:1rem; margin-bottom:0.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

def _risk_badge(level: str) -> str:
    css = f"risk-badge-{level.lower()}"
    return f'<span class="{css}">{level}</span>'

def _api_key_configured() -> bool:
    key = os.environ.get("GOOGLE_API_KEY", "").strip()
    # Also check Streamlit secrets
    if not key:
        try:
            key = st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            pass
    if key:
        os.environ["GOOGLE_API_KEY"] = key
        return True
    return False


def _plot_membership_functions() -> plt.Figure:
    """Create a 2x3 grid of fuzzy membership function plots."""
    universe = get_universe()
    mfs = get_membership_functions()

    colors = {
        "Low": "#34a853",
        "Medium": "#fbbc04",
        "High": "#ea4335",
        "Critical": "#7b0014",
    }

    variables = ["likelihood", "impact", "exposure", "data_sensitivity", "risk"]
    titles = {
        "likelihood": "Threat Likelihood",
        "impact": "Potential Impact",
        "exposure": "Exposure",
        "data_sensitivity": "Data Sensitivity",
        "risk": "Cybersecurity Risk (Output)",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    fig.patch.set_facecolor("#f8f9fa")
    axes_flat = axes.flatten()

    for idx, var in enumerate(variables):
        ax = axes_flat[idx]
        ax.set_facecolor("#ffffff")
        for term, mf_array in mfs[var].items():
            color = colors.get(term, "#1a73e8")
            ax.plot(universe, mf_array, label=term, color=color, linewidth=2.5)
            ax.fill_between(universe, mf_array, alpha=0.15, color=color)
        ax.set_title(titles[var], fontsize=11, fontweight="bold", color="#202124")
        ax.set_xlabel("Value (0-100)", fontsize=9, color="#5f6368")
        ax.set_ylabel("Membership Degree", fontsize=9, color="#5f6368")
        ax.set_ylim(-0.05, 1.1)
        ax.set_xlim(0, 100)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes_flat[5].set_visible(False)

    plt.tight_layout(pad=2.0)
    return fig


def _render_sidebar():
    with st.sidebar:
        st.markdown("## 🛡️ About")
        st.markdown(
            """
            This application uses **two AI components**:

            **1. LangChain + Gemini**
            Extracts structured cybersecurity factors from natural-language incident descriptions.

            **2. Fuzzy Inference System**
            Uses genuine fuzzy logic (membership functions, rules, defuzzification) to compute a risk score.

            ---
            **Architecture:**
            ```
            Incident Text
                ↓
            LangChain / Gemini
                ↓
            Structured Factors
                ↓
            Fuzzy Inference
                ↓
            Risk Score
                ↓
            LLM Explanation
            ```
            ---
            """
        )
        st.markdown("### 📋 Quick Demo Scenarios")
        st.markdown("Select a pre-built scenario to try the system instantly:")
        selected_sample = st.selectbox(
            "Choose scenario:",
            ["-- Select --"] + get_sample_titles(),
            label_visibility="collapsed",
        )
        if selected_sample != "-- Select --":
            case = get_sample_by_title(selected_sample)
            if case:
                st.session_state["sample_description"] = case.description
                st.caption(f"Expected risk: **{case.expected_risk}**")

        st.markdown("---")
        st.markdown("### ⚙️ AI Configuration")
        user_key = st.text_input(
            "Gemini API Key:",
            type="password",
            value=os.environ.get("GOOGLE_API_KEY", ""),
            placeholder="AIzaSy...",
            help="Enter your Gemini API key from Google AI Studio",
            key="input_gemini_api_key",
        )
        if user_key.strip():
            os.environ["GOOGLE_API_KEY"] = user_key.strip()

        # Configurable model
        current_model = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash").strip() or "gemini-3.8-flash"
        model_options = ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        default_idx = model_options.index(current_model) if current_model in model_options else 0
        selected_model = st.selectbox(
            "Gemini Model:",
            options=model_options,
            index=default_idx,
            help="Configurable Gemini model (default: gemini-3.8-flash)",
            key="selected_gemini_model",
        )
        os.environ["GEMINI_MODEL"] = selected_model

        if _api_key_configured():
            st.success(f"✅ AI Active ({selected_model})")
        else:
            st.warning("⚠️ API Key not detected")
            st.caption(
                "Add `GOOGLE_API_KEY` to `.env` or input above for live Gemini extraction. "
                "Baseline Fuzzy Evaluation is available offline."
            )

        st.markdown("---")
        st.markdown("*College Internal Assessment Project*")
        st.markdown("*AI-Based Cybersecurity Threat Risk Analyzer*")


def main():
    _render_sidebar()

    # Title
    st.markdown(
        '<div class="main-title">🛡️ AI-Based Cybersecurity Threat Risk Analyzer</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-title">Analyze cybersecurity incidents using LangChain (Gemini 3.8 Flash) + Fuzzy Logic</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    tab_analyze, tab_fuzzy_viz, tab_about = st.tabs(
        ["🔍 Analyze Incident", "📊 Fuzzy Logic Visualization", "ℹ️ How It Works"]
    )

    with tab_analyze:
        if not _api_key_configured():
            st.info(
                "ℹ️ **Configuration Notice:** `GOOGLE_API_KEY` is not detected in `.env`. "
                "You can enter an API key in the sidebar for live Gemini extraction, "
                "or click **Analyze Threat** to test with baseline factors and the genuine Mamdani Fuzzy Inference Engine."
            )
        default_text = st.session_state.get("sample_description", "")

        st.markdown("### 📝 Incident Description")
        incident_text = st.text_area(
            label="Describe the cybersecurity incident in plain English:",
            value=default_text,
            height=180,
            placeholder=(
                "Example: An employee received a suspicious email and clicked a link. "
                "They entered their company password and later several unusual login "
                "attempts appeared from an unknown location..."
            ),
            key="incident_input",
        )

        col_btn, col_clear = st.columns([2, 1])
        with col_btn:
            analyze_clicked = st.button(
                "🔍 Analyze Threat",
                type="primary",
                use_container_width=True,
                disabled=not incident_text.strip(),
            )
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state["sample_description"] = ""
                st.rerun()

        if analyze_clicked:
            if not incident_text.strip():
                st.warning("⚠️ Please enter an incident description before analyzing.")
                st.stop()

            with st.spinner("🤖 Extracting factors with LangChain... then running Fuzzy Inference..."):
                result, factors, debug_info, messages = analyze_incident(incident_text)

            # Show any non-fatal messages
            for msg in messages:
                if "⚠️" in msg:
                    st.warning(msg)
                else:
                    st.info(msg)

            st.markdown("---")

            col_left, col_right = st.columns([1, 1])

            with col_left:
                # --- AI-Extracted Information ---
                st.markdown(
                    '<div class="section-header">🤖 AI-Extracted Information</div>',
                    unsafe_allow_html=True,
                )
                info_data = {
                    "🏷️ Threat Type": factors.threat_type,
                    "🚀 Attack Vector": factors.attack_vector,
                    "🎯 Affected Asset": factors.affected_asset,
                    "👁️ Suspicious Activity": factors.suspicious_activity,
                    "🔒 Sensitive Data Involved": factors.sensitive_data_involved,
                }
                for label, value in info_data.items():
                    st.markdown(
                        f'<div class="factor-card"><strong>{label}:</strong> {value}</div>',
                        unsafe_allow_html=True,
                    )

                # --- Fuzzy Input Values ---
                st.markdown(
                    '<div class="section-header">📊 Fuzzy Input Values</div>',
                    unsafe_allow_html=True,
                )
                fuzzy_inputs = {
                    "⚡ Threat Likelihood": factors.likelihood,
                    "💥 Potential Impact": factors.impact,
                    "🌐 Exposure": factors.exposure,
                    "🔐 Data Sensitivity": factors.data_sensitivity,
                }
                for label, value in fuzzy_inputs.items():
                    col_name, col_val, col_bar = st.columns([2, 0.8, 3])
                    with col_name:
                        st.write(label)
                    with col_val:
                        st.write(f"**{value:.1f}**")
                    with col_bar:
                        st.progress(int(value))

            with col_right:
                st.markdown(
                    '<div class="section-header">🎯 Risk Assessment Result</div>',
                    unsafe_allow_html=True,
                )
                score_pct = result.risk_score / 100.0
                level_colors = {
                    "LOW": "#34a853",
                    "MEDIUM": "#fbbc04",
                    "HIGH": "#ea4335",
                    "CRITICAL": "#7b0014",
                }
                color = level_colors.get(result.risk_level, "#1a73e8")

                fig_gauge, ax_gauge = plt.subplots(figsize=(5, 3))
                fig_gauge.patch.set_facecolor("#f8f9fa")
                ax_gauge.set_facecolor("#f8f9fa")
                # Background bar
                ax_gauge.barh(0, 100, color="#e0e0e0", height=0.5)
                # Score bar
                ax_gauge.barh(0, result.risk_score, color=color, height=0.5)
                ax_gauge.set_xlim(0, 100)
                ax_gauge.set_ylim(-0.5, 0.5)
                ax_gauge.set_yticks([])
                ax_gauge.set_xlabel("Risk Score (0-100)", fontsize=10)
                ax_gauge.spines["top"].set_visible(False)
                ax_gauge.spines["right"].set_visible(False)
                ax_gauge.spines["left"].set_visible(False)
                ax_gauge.text(
                    result.risk_score / 2,
                    0,
                    f"{result.risk_score:.1f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )
                plt.tight_layout()
                st.pyplot(fig_gauge)
                plt.close(fig_gauge)

                st.markdown(
                    f'<div style="text-align:center; margin:10px 0;">'
                    f'Risk Level: {_risk_badge(result.risk_level)}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="section-header">🔢 Fuzzy Logic Contribution</div>',
                    unsafe_allow_html=True,
                )
                if debug_info.get("fuzzification"):
                    fuzz_data = debug_info["fuzzification"]
                    for var_name, deg_dict in fuzz_data.items():
                        dominant = max(deg_dict, key=deg_dict.get)
                        dominant_deg = deg_dict[dominant]
                        label = var_name.replace("_", " ").title()
                        st.write(
                            f"**{label}**: dominant term = **{dominant}** "
                            f"(degree: {dominant_deg:.2f})"
                        )

            st.markdown("---")

            st.markdown(
                '<div class="section-header">💬 AI Security Explanation</div>',
                unsafe_allow_html=True,
            )
            if result.explanation:
                st.info(result.explanation)
            else:
                st.info(
                    f"Risk score: {result.risk_score:.1f}/100 ({result.risk_level}). "
                    "Review the extracted factors above for details."
                )
            st.markdown(
                '<div class="section-header">🛡️ Recommended Defensive Actions</div>',
                unsafe_allow_html=True,
            )
            if result.recommendations:
                for i, rec in enumerate(result.recommendations, 1):
                    st.markdown(f"**{i}.** {rec}")
            else:
                st.markdown(
                    "- Report the incident to your IT/security team.\n"
                    "- Change any potentially compromised passwords immediately.\n"
                    "- Enable multi-factor authentication.\n"
                    "- Review account access logs."
                )

            st.markdown("---")
            st.markdown(
                '<div class="section-header">📥 Export Incident Assessment Report</div>',
                unsafe_allow_html=True,
            )
            col_exp_md, col_exp_json = st.columns(2)

            timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

            # Markdown Report
            md_lines = [
                "# 🛡️ Cybersecurity Incident Risk Assessment Report",
                f"**Generated:** {timestamp_str}  ",
                f"**Overall Risk Level:** {result.risk_level} ({result.risk_score:.1f}/100)  ",
                "",
                "## 1. Incident Description",
                incident_text.strip(),
                "",
                "## 2. AI-Extracted Threat Intelligence",
                f"- **Threat Type:** {factors.threat_type}",
                f"- **Attack Vector:** {factors.attack_vector}",
                f"- **Affected Asset:** {factors.affected_asset}",
                f"- **Suspicious Activity:** {factors.suspicious_activity}",
                f"- **Sensitive Data Involved:** {factors.sensitive_data_involved}",
                "",
                "## 3. Fuzzy Logic Parameters (0-100)",
                f"- **Likelihood:** {factors.likelihood:.1f}",
                f"- **Impact:** {factors.impact:.1f}",
                f"- **Exposure:** {factors.exposure:.1f}",
                f"- **Data Sensitivity:** {factors.data_sensitivity:.1f}",
                f"- **Computed Centroid Risk Score:** {result.risk_score:.2f}",
                "",
                "## 4. AI Security Explanation",
                result.explanation or "N/A",
                "",
                "## 5. Recommended Defensive Actions",
            ]
            if result.recommendations:
                for idx, r in enumerate(result.recommendations, 1):
                    md_lines.append(f"{idx}. {r}")
            else:
                md_lines.append("- Refer to organizational incident response playbook.")

            md_content = "\n".join(md_lines)

            # JSON Report
            json_dict = {
                "timestamp": timestamp_str,
                "incident_text": incident_text.strip(),
                "extracted_factors": {
                    "threat_type": factors.threat_type,
                    "attack_vector": factors.attack_vector,
                    "affected_asset": factors.affected_asset,
                    "suspicious_activity": factors.suspicious_activity,
                    "sensitive_data_involved": factors.sensitive_data_involved,
                    "likelihood": factors.likelihood,
                    "impact": factors.impact,
                    "exposure": factors.exposure,
                    "data_sensitivity": factors.data_sensitivity,
                },
                "risk_assessment": {
                    "risk_score": round(result.risk_score, 2),
                    "risk_level": result.risk_level,
                    "explanation": result.explanation,
                    "recommendations": result.recommendations,
                },
            }
            json_content = json.dumps(json_dict, indent=2)

            with col_exp_md:
                st.download_button(
                    label="📄 Download Markdown Report",
                    data=md_content,
                    file_name=f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with col_exp_json:
                st.download_button(
                    label="💾 Download JSON Data",
                    data=json_content,
                    file_name=f"incident_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True,
                )

    with tab_fuzzy_viz:
        st.markdown("### 📊 Fuzzy Membership Functions")
        st.markdown(
            """
            These graphs show the **membership functions** that define how crisp numeric values
            are translated into fuzzy degrees (fuzzification). Each variable has overlapping
            fuzzy sets (Low / Medium / High / Critical for the output).

            The fuzzy inference system uses these functions to evaluate the rules and compute
            the final risk score via **centroid defuzzification**.
            """
        )

        with st.spinner("Generating membership function plots..."):
            fig = _plot_membership_functions()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown("---")
        st.markdown("### 📋 Fuzzy Rule Base")
        rules = [
            ("R1",  "Likelihood=Low AND Impact=Low AND Exposure=Low",     "Risk=Low"),
            ("R2",  "Likelihood=Medium AND Impact=Medium AND Exposure=Medium", "Risk=Medium"),
            ("R3",  "Likelihood=High AND Impact=High AND Exposure=High",  "Risk=High"),
            ("R4",  "Likelihood=High AND Impact=High AND Exposure=High",  "Risk=Critical"),
            ("R5",  "Impact=High AND Exposure=High",                      "Risk=High"),
            ("R6",  "Likelihood=High AND Exposure=Medium",                "Risk=High"),
            ("R7",  "Likelihood=Medium AND Impact=High AND Exposure=High","Risk=High"),
            ("R8",  "DataSensitivity=High AND Impact=High",               "Risk=High"),
            ("R9",  "DataSensitivity=High AND Likelihood=High",           "Risk=Critical"),
            ("R10", "Likelihood=Low AND Impact=Low",                      "Risk=Low"),
            ("R11", "Likelihood=Low AND Impact=Medium",                   "Risk=Low"),
            ("R12", "Likelihood=Medium AND Impact=Low",                   "Risk=Low"),
            ("R13", "Likelihood=Medium AND Impact=Medium",                "Risk=Medium"),
            ("R14", "Likelihood=High AND Impact=Low",                     "Risk=Medium"),
            ("R15", "Likelihood=High AND Impact=Medium",                  "Risk=High"),
            ("R16", "DataSensitivity=High AND Exposure=High",             "Risk=High"),
            ("R17", "Likelihood=Low AND DataSensitivity=High",            "Risk=Medium"),
        ]

        import pandas as pd
        df_rules = pd.DataFrame(rules, columns=["Rule", "IF (Condition)", "THEN (Consequence)"])
        st.dataframe(df_rules, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("### 🧪 Interactive Fuzzy Tester")
        st.markdown("Adjust the sliders to see how the fuzzy engine calculates the risk score directly.")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            test_lik = st.slider("Threat Likelihood", 0, 100, 50, key="test_lik")
            test_imp = st.slider("Potential Impact",  0, 100, 50, key="test_imp")
        with col_s2:
            test_exp = st.slider("Exposure",          0, 100, 50, key="test_exp")
            test_dat = st.slider("Data Sensitivity",  0, 100, 50, key="test_dat")

        from src.fuzzy_engine import compute_risk_score as _crs
        test_score, _ = _crs(test_lik, test_imp, test_exp, test_dat)
        test_level = score_to_level(test_score)
        level_emoji = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴", "CRITICAL": "⚫"}
        st.metric(
            label="Fuzzy Risk Score",
            value=f"{test_score:.1f} / 100",
            delta=f"{level_emoji.get(test_level, '')} {test_level}",
        )

    with tab_about:
        st.markdown("### 🏗️ System Architecture")
        st.markdown(
            """
            ```
            User Input (Natural Language Incident)
                         ↓
              LangChain PromptTemplate
                         ↓
               Google Gemini 1.5 Flash
                         ↓
            Structured JSON (Validated by Pydantic)
                         ↓
            ┌────────────────────────────────────┐
            │     FUZZY INFERENCE SYSTEM         │
            │  1. Universe of Discourse [0,100]  │
            │  2. Membership Functions           │
            │     (Triangular + Trapezoidal)     │
            │  3. Fuzzification                  │
            │  4. 17 Fuzzy Rules (Mamdani)       │
            │  5. Rule Evaluation (min operator) │
            │  6. Aggregation (max operator)     │
            │  7. Defuzzification (Centroid)     │
            └────────────────────────────────────┘
                         ↓
               Numerical Risk Score (0-100)
                         ↓
              LangChain: Explanation Generator
                         ↓
              LangChain: Recommendations Generator
                         ↓
                Streamlit Result Display
# Make sure the project root is in sys.path for imports
            ```
            """
        )

        st.markdown("### 🧩 Technology Stack")
        tech_data = {
            "Component": ["UI", "AI/LLM", "LangChain", "Fuzzy Logic", "Validation", "Visualisation"],
            "Technology": ["Streamlit", "Google Gemini 1.5 Flash", "LangChain Core + Google GenAI", "scikit-fuzzy + NumPy", "Pydantic v2", "Matplotlib"],
        }
        import pandas as pd
        st.table(pd.DataFrame(tech_data))

        st.markdown("### 📐 Fuzzy Variables Summary")
        var_data = {
            "Variable": ["Threat Likelihood", "Potential Impact", "Exposure", "Data Sensitivity", "Cybersecurity Risk"],
            "Type": ["Input", "Input", "Input", "Input", "Output"],
            "Range": ["0-100", "0-100", "0-100", "0-100", "0-100"],
            "Fuzzy Sets": ["Low, Medium, High", "Low, Medium, High", "Low, Medium, High", "Low, Medium, High", "Low, Medium, High, Critical"],
        }
        st.table(pd.DataFrame(var_data))

        st.markdown("### 🔒 Security Notes")
        st.markdown(
            """
            - **No API keys are hard-coded** in the source code.
            - API keys are loaded from `.env` (local) or Streamlit Secrets (deployed).
            - All LLM outputs are validated with Pydantic before use.
            - The application performs **defensive analysis only** — no offensive security features.
            - No user data is stored or transmitted beyond the LLM API call.
            """
        )


if __name__ == "__main__":
    main()
