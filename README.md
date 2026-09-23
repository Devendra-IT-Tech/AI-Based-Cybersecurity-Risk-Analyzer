# 🛡️ AI-Based Cybersecurity Threat Risk Analyzer

> **College Internal Assessment Project**  
> Analyzing cybersecurity incidents using LangChain (Google Gemini) + Genuine Fuzzy Inference System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3.x-green.svg)](https://langchain.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem Statement

Modern organizations face increasing cybersecurity threats that are difficult to assess quickly. Security analysts often receive vague, natural-language incident reports and must rapidly evaluate the risk level to prioritize responses. This project automates that process using AI.

---

## 📋 Features

| Feature | Description |
|---------|-------------|
| 🤖 **LangChain AI** | Extracts structured cybersecurity factors from natural-language descriptions |
| 🔢 **Fuzzy Inference System** | Genuine fuzzy logic with membership functions, rules & defuzzification |
| 📊 **Risk Visualization** | Interactive membership function plots and score gauges |
| 🎭 **5 Demo Scenarios** | Pre-built sample cases for instant demonstration |
| 🧪 **Interactive Tester** | Adjust fuzzy inputs with sliders and see live results |
| 🛡️ **Defensive Only** | Educational and protective recommendations only |
| ☁️ **Cloud Ready** | One-click deployment to Streamlit Community Cloud |

---

## 🏗️ Architecture

```
User Input (Natural Language Incident)
             ↓
  LangChain PromptTemplate
             ↓
   Google Gemini 3.8 Flash
             ↓
  Structured JSON (Pydantic Validated)
             ↓
┌─────────────────────────────────────┐
│      FUZZY INFERENCE SYSTEM         │
│  1. Universe of Discourse [0-100]   │
│  2. Membership Functions            │
│     (Triangular + Trapezoidal)      │
│  3. Fuzzification                   │
│  4. 17 Fuzzy Rules (Mamdani)        │
│  5. Rule Evaluation (min)           │
│  6. Aggregation (max)               │
│  7. Defuzzification (Centroid)      │
└─────────────────────────────────────┘
             ↓
   Risk Score (0-100) + Level
             ↓
  LangChain: Explanation + Recommendations
             ↓
      Streamlit Result Display
```

---

## 🧩 Technology Stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| AI / LLM | Google Gemini 3.8 Flash |
| LLM Framework | LangChain Core + langchain-google-genai |
| Fuzzy Logic | scikit-fuzzy + NumPy |
| Validation | Pydantic v2 |
| Visualisation | Matplotlib |
| Environment | python-dotenv |

---

## 🤖 LangChain Implementation

Three LangChain chains are used:

### Chain 1: Factor Extraction
```
ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser → JSON parse → Pydantic validate
```
Extracts: `threat_type`, `attack_vector`, `affected_asset`, `likelihood (0-100)`, `impact (0-100)`, `exposure (0-100)`, `data_sensitivity (0-100)`

### Chain 2: Risk Explanation
```
ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser
```
Generates plain-English explanation of the fuzzy risk assessment.

### Chain 3: Recommendations
```
ChatPromptTemplate → ChatGoogleGenerativeAI → StrOutputParser → JSON array parse
```
Generates 4-6 safe defensive recommendations.

---

## 🔢 Fuzzy Logic Implementation

### Variables and Membership Functions

**Inputs (all 0-100):**

| Variable | Low | Medium | High |
|----------|-----|--------|------|
| Threat Likelihood | Trap[0,0,35,50] | Tri[30,50,70] | Trap[55,75,100,100] |
| Potential Impact | Trap[0,0,30,45] | Tri[30,50,70] | Trap[55,75,100,100] |
| Exposure | Trap[0,0,30,45] | Tri[30,50,70] | Trap[55,75,100,100] |
| Data Sensitivity | Trap[0,0,30,45] | Tri[30,50,70] | Trap[55,75,100,100] |

**Output:**

| Variable | Low | Medium | High | Critical |
|----------|-----|--------|------|---------|
| Risk | Trap[0,0,20,30] | Tri[20,35,55] | Tri[45,65,80] | Trap[70,85,100,100] |

### Fuzzy Rule Base (17 rules)

```
R1:  IF Likelihood=Low AND Impact=Low AND Exposure=Low     THEN Risk=Low
R2:  IF Likelihood=Med AND Impact=Med AND Exposure=Med     THEN Risk=Medium
R3:  IF Likelihood=High AND Impact=High AND Exposure=High  THEN Risk=High
R4:  IF Likelihood=High AND Impact=High AND Exposure=High  THEN Risk=Critical
R5:  IF Impact=High AND Exposure=High                      THEN Risk=High
R6:  IF Likelihood=High AND Exposure=Medium                THEN Risk=High
R7:  IF Likelihood=Med AND Impact=High AND Exposure=High   THEN Risk=High
R8:  IF DataSensitivity=High AND Impact=High               THEN Risk=High
R9:  IF DataSensitivity=High AND Likelihood=High           THEN Risk=Critical
R10: IF Likelihood=Low AND Impact=Low                      THEN Risk=Low
R11: IF Likelihood=Low AND Impact=Medium                   THEN Risk=Low
R12: IF Likelihood=Med AND Impact=Low                      THEN Risk=Low
R13: IF Likelihood=Med AND Impact=Med                      THEN Risk=Medium
R14: IF Likelihood=High AND Impact=Low                     THEN Risk=Medium
R15: IF Likelihood=High AND Impact=Med                     THEN Risk=High
R16: IF DataSensitivity=High AND Exposure=High             THEN Risk=High
R17: IF Likelihood=Low AND DataSensitivity=High            THEN Risk=Medium
```

**Defuzzification**: Centroid method

---

## 📁 Project Structure

```
AI-Cybersecurity-Threat-Risk-Analyzer/
│
├── app.py                    # Streamlit application entry point
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore                # Git ignore rules
├── .env.example              # Example environment variables
├── LICENSE                   # MIT License
│
├── src/
│   ├── __init__.py
│   ├── llm_service.py        # LangChain + Gemini integration
│   ├── fuzzy_engine.py       # Fuzzy Inference System
│   ├── risk_analyzer.py      # Orchestration pipeline
│   ├── validators.py         # Pydantic data models
│   └── prompts.py            # LangChain prompt templates
│
├── utils/
│   ├── __init__.py
│   └── sample_cases.py       # Demo scenarios
│
├── tests/
│   ├── __init__.py
│   ├── test_fuzzy_engine.py  # Fuzzy engine tests
│   ├── test_validators.py    # Validator tests
│   └── test_risk_analyzer.py # Integration tests
│
└── docs/
    └── project_report.md     # College project report
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- Google Gemini API key (free at [aistudio.google.com](https://aistudio.google.com))

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/AI-Cybersecurity-Threat-Risk-Analyzer.git
cd AI-Cybersecurity-Threat-Risk-Analyzer
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# GOOGLE_API_KEY=your_actual_key_here
```

### 5. Run the application
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🔑 Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Google Gemini API key | ✅ Yes |

Get your free key at: [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_fuzzy_engine.py -v

# Run with coverage
pytest tests/ -v --tb=short
```

Expected: **All tests pass** without an API key (LLM calls are mocked in integration tests).

---

## ☁️ Streamlit Community Cloud Deployment

1. Push the project to GitHub (ensure `.env` is in `.gitignore` ✅).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Click **"New app"** → Connect your GitHub repo.
4. Set **Main file path**: `app.py`.
5. Under **"Advanced settings" → "Secrets"**, add:
   ```toml
   GOOGLE_API_KEY = "your_actual_api_key_here"
   ```
6. Click **"Deploy"**.

---

## 🖼️ Screenshots

> *[Add screenshots here after running the application]*
>
> Suggested screenshots:
> - Main analysis page with results
> - Fuzzy membership function plots
> - Interactive fuzzy tester tab

---

## 🔗 Links

- **GitHub**: [https://github.com/YOUR_USERNAME/AI-Cybersecurity-Threat-Risk-Analyzer](https://github.com/YOUR_USERNAME/AI-Cybersecurity-Threat-Risk-Analyzer)
- **Live App**: [https://YOUR_APP.streamlit.app](https://YOUR_APP.streamlit.app)

---

## ⚠️ Limitations

- The LLM extraction quality depends on the API response (Gemini 3.8 Flash).
- Fuzzy scores are approximations — not clinical or legal assessments.
- The application is for **educational purposes only**.
- Internet connection required for LLM calls.
- The fuzzy rule base covers common scenarios but is not exhaustive.

---

## 🚀 Future Scope

- Support additional LLM providers (OpenAI, Anthropic).
- CVSS score integration.
- Historical incident tracking with a database.
- PDF report generation.
- Multi-language support.
- More sophisticated fuzzy rule bases with domain expert input.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built for college Internal Assessment — AI + Fuzzy Logic applied to Cybersecurity*
