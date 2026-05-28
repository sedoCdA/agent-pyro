# AgentPyro - Agentic AI Security Risk Predictor

A machine learning pipeline to predict access decisions for autonomous AI agent actions - classifying each action as **Allowed**, **Blocked**, or **Needs Human Approval** in real time.

---

## Problem Statement

As AI agents grow more powerful - able to call APIs, access enterprise systems, and act independently - traditional access control is insufficient. This project builds a classifier that predicts whether an agent action should be permitted, denied, or escalated for human review, based on simulated security log data.

---

## Dataset

**Source:** Agentic AI Security Risk Dataset  
**File:** `data/agent_security_risk_scores.csv`  
**Size:** 2,200 rows × 15 columns  

| Feature | Type | Description |
|---|---|---|
| `agent_role` | categorical | Role of the AI agent |
| `agent_autonomy_level` | numeric (1–5) | Agent's autonomy setting |
| `user_role` | categorical | Role of the user triggering the agent |
| `requested_action` | categorical | Action the agent wants to perform |
| `tool_requested` | categorical | Tool/API the agent is calling |
| `resource_type` | categorical | Type of resource being accessed |
| `resource_sensitivity` | numeric (1–5) | Sensitivity level of the resource |
| `permission_match` | binary | Whether agent permissions match the resource |
| `action_risk_score` | numeric (0–100) | Computed risk score of the action |
| `prompt_injection_detected` | binary | Whether a prompt injection was detected |
| `data_exfiltration_risk` | numeric (0–100) | Risk score for data exfiltration |
| `human_approval_required` | binary | Whether human approval was flagged |
| `previous_failed_attempts` | numeric | Number of prior failed access attempts |
| `audit_log_available` | binary | Whether an audit log exists |
| `access_decision` | target | Allowed / Blocked / Needs_Human_Approval |

---

## Pipeline Steps

| Step | File | Description |
|---|---|---|
| 1 | `src/eda.py` | Exploratory data analysis — distributions, correlations, class balance |
| 2 | `src/preprocessing.py` | Encoding, scaling, SMOTE, train/test split |
| 3 | `src/baseline_models.py` | Logistic Regression + Decision Tree benchmarks |
| 4 | `src/primary_model.py` | XGBoost / LightGBM with hyperparameter tuning |
| 5 | `src/evaluation.py` | Confusion matrix, macro F1, per-class precision/recall |
| 6 | `src/explainability.py` | SHAP feature importance — global + per-prediction |
| 7 | `src/inference.py` | Inference function: input log entry → decision + confidence + reasoning |

---

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/agent-pyro.git
cd agent-pyro

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Run each step in order:

```bash
python src/eda.py
python src/preprocessing.py
python src/baseline_models.py
python src/primary_model.py
python src/evaluation.py
python src/explainability.py
python src/inference.py
```

Outputs (plots, metrics, saved models) are written to `reports/` and `models/`.

---

## Project Structure

```
agent-pyro/
├── data/
│   └── agent_security_risk_scores.csv
├── src/
│   ├── eda.py
│   ├── preprocessing.py
│   ├── baseline_models.py
│   ├── primary_model.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── inference.py
├── models/           # saved .pkl model files
├── reports/          # plots and metric outputs
├── notebooks/        # optional Jupyter exploration
├── requirements.txt
└── README.md
```

---

## Target Metric

Primary: **Macro F1-score** (accounts for class imbalance)  
Secondary: Per-class precision, recall, and confidence calibration

---

## Tech Stack

Python · scikit-learn · XGBoost · LightGBM · SHAP · imbalanced-learn · pandas · matplotlib · seaborn