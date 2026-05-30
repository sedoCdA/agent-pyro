"""
=============================================================
Step 7 — Inference Function + Demo
=============================================================
Purpose:
    The final step. Wraps the trained LightGBM model into a clean,
    production-ready inference function that takes a raw agent
    action log entry and returns:
      - access_decision   : Allowed / Blocked / Needs_Human_Approval
      - confidence        : 0.0 – 1.0 (how certain the model is)
      - probabilities     : per-class breakdown
      - top_reasons       : top 5 SHAP-based feature contributions
                            explaining WHY this decision was made

    Also includes an interactive demo that runs several realistic
    test scenarios so you can see the system working end-to-end.

Usage:
    # Run the built-in demo
    python src/inference.py

    # Import the function in your own code
    from src.inference import predict_access_decision
    result = predict_access_decision({ ... })
=============================================================
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR = "models"

# ── Feature order (must match preprocessing.py exactly) ───────────────────────
CATEGORICAL_COLS = [
    "agent_role", "user_role", "requested_action",
    "tool_requested", "resource_type",
]
NUMERIC_COLS = [
    "agent_autonomy_level", "resource_sensitivity",
    "action_risk_score", "data_exfiltration_risk",
    "previous_failed_attempts",
]
BINARY_COLS = [
    "permission_match", "prompt_injection_detected",
    "human_approval_required", "audit_log_available",
]
FEATURE_NAMES = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS

CLASS_LABELS = {0: "Allowed", 1: "Blocked", 2: "Needs_Human_Approval"}
CLASS_EMOJI  = {"Allowed": "✅", "Blocked": "🚫", "Needs_Human_Approval": "⚠️ "}

ANSI = {
    "Allowed":              "\033[92m",
    "Blocked":              "\033[91m",
    "Needs_Human_Approval": "\033[93m",
    "reset":                "\033[0m",
    "bold":                 "\033[1m",
    "dim":                  "\033[2m",
}


# ── 1. Load all model artifacts (once at import time) ─────────────────────────
def _load_artifacts():
    model        = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    le           = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))
    explainer    = joblib.load(os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    return model, preprocessor, le, explainer

_model, _preprocessor, _le, _explainer = _load_artifacts()
print(f"[INF] Artifacts loaded — model ready.")


# ── 2. Input validation ────────────────────────────────────────────────────────
def _validate_input(log_entry: dict) -> None:
    required = set(FEATURE_NAMES)
    missing  = required - set(log_entry.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    range_checks = {
        "agent_autonomy_level":     (1, 5),
        "resource_sensitivity":     (1, 5),
        "action_risk_score":        (0, 100),
        "data_exfiltration_risk":   (0, 100),
        "previous_failed_attempts": (0, 99),
    }
    for field, (lo, hi) in range_checks.items():
        val = log_entry[field]
        if not (lo <= val <= hi):
            raise ValueError(f"Field '{field}' = {val} outside valid range [{lo}, {hi}]")

    for field in BINARY_COLS:
        if log_entry[field] not in (0, 1):
            raise ValueError(f"Field '{field}' must be 0 or 1, got {log_entry[field]}")


# ── 3. Core inference function ────────────────────────────────────────────────
def predict_access_decision(log_entry: dict) -> dict:
    """
    Takes a raw agent action log entry (dict) and returns a structured
    decision with confidence score and SHAP-based explanation.

    Returns
    -------
    dict:
        decision      — "Allowed", "Blocked", or "Needs_Human_Approval"
        confidence    — probability of predicted class (0.0–1.0)
        probabilities — per-class probability breakdown
        top_reasons   — top 5 SHAP feature contributions
        risk_level    — "LOW" / "MEDIUM" / "HIGH" / "CRITICAL"
    """
    _validate_input(log_entry)

    row           = pd.DataFrame([log_entry])[FEATURE_NAMES]
    X_transformed = _preprocessor.transform(row)

    y_pred  = _model.predict(X_transformed)[0]
    y_proba = _model.predict_proba(X_transformed)[0]

    decision   = CLASS_LABELS[y_pred]
    confidence = float(y_proba[y_pred])

    probabilities = {CLASS_LABELS[i]: round(float(p), 6)
                     for i, p in enumerate(y_proba)}

    # SHAP explanation — handle both 3D array and list formats
    shap_raw = _explainer.shap_values(X_transformed)
    if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        sv_for_class = shap_raw[0, :, y_pred]
    else:
        sv_for_class = shap_raw[y_pred][0]

    shap_pairs = sorted(
        zip(FEATURE_NAMES, sv_for_class, X_transformed[0]),
        key=lambda x: abs(x[1]),
        reverse=True
    )
    top_reasons = [
        {
            "feature":       feat,
            "shap_value":    round(float(sv), 6),
            "feature_value": round(float(fv), 4),
            "direction":     "increases risk" if sv > 0 else "decreases risk",
        }
        for feat, sv, fv in shap_pairs[:5]
    ]

    risk_level = _compute_risk_level(decision, confidence, log_entry)

    return {
        "decision":      decision,
        "confidence":    round(confidence, 6),
        "probabilities": probabilities,
        "top_reasons":   top_reasons,
        "risk_level":    risk_level,
    }


# ── 4. Risk level helper ───────────────────────────────────────────────────────
def _compute_risk_level(decision: str, confidence: float,
                        log_entry: dict) -> str:
    if decision == "Allowed" and confidence >= 0.95:
        return "LOW"
    if decision == "Needs_Human_Approval":
        return "MEDIUM"
    if decision == "Blocked" and confidence < 0.90:
        return "HIGH"
    if decision == "Blocked":
        if (log_entry.get("prompt_injection_detected", 0) == 1 or
                log_entry.get("data_exfiltration_risk", 0) >= 80 or
                log_entry.get("previous_failed_attempts", 0) >= 3):
            return "CRITICAL"
        return "HIGH"
    return "MEDIUM"


# ── 5. Pretty-print result ────────────────────────────────────────────────────
def print_result(result: dict, scenario_name: str = "") -> None:
    d   = result["decision"]
    c   = result["confidence"]
    rl  = result["risk_level"]
    col = ANSI.get(d, "")
    rst = ANSI["reset"]
    bld = ANSI["bold"]
    dim = ANSI["dim"]
    risk_color = {
        "LOW": "\033[92m", "MEDIUM": "\033[93m",
        "HIGH": "\033[91m", "CRITICAL": "\033[95m"
    }.get(rl, "")

    print(f"\n{'─'*56}")
    if scenario_name:
        print(f"{bld}  Scenario : {scenario_name}{rst}")
    print(f"{'─'*56}")
    print(f"  {bld}Decision   :{rst}  {col}{bld}{CLASS_EMOJI[d]} {d}{rst}")
    print(f"  {bld}Confidence :{rst}  {c:.2%}")
    print(f"  {bld}Risk Level :{rst}  {risk_color}{bld}{rl}{rst}")

    print(f"\n  {bld}Class Probabilities:{rst}")
    for cls, prob in result["probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"    {cls:<25} {prob:>6.2%}  {dim}{bar}{rst}")

    print(f"\n  {bld}Top Reasons (SHAP):{rst}")
    for i, r in enumerate(result["top_reasons"], 1):
        dcol  = "\033[91m" if r["shap_value"] > 0 else "\033[92m"
        arrow = "▲" if r["shap_value"] > 0 else "▼"
        print(f"    {i}. {r['feature']:<28} "
              f"{dcol}{arrow} {r['shap_value']:+.4f}{rst}  "
              f"{dim}(val={r['feature_value']:.2f}){rst}")
    print(f"{'─'*56}")


# ── 6. Demo scenarios ─────────────────────────────────────────────────────────
DEMO_SCENARIOS = [
    {
        "name": "Safe read — junior analyst, low sensitivity",
        "entry": {
            "agent_role": "customer_support_agent", "agent_autonomy_level": 2,
            "user_role": "analyst", "requested_action": "read_record",
            "tool_requested": "crm_api", "resource_type": "customer_profile",
            "resource_sensitivity": 2, "permission_match": 1,
            "action_risk_score": 12, "prompt_injection_detected": 0,
            "data_exfiltration_risk": 10, "human_approval_required": 0,
            "previous_failed_attempts": 0, "audit_log_available": 1,
        }
    },
    {
        "name": "Hard block — no permission + prompt injection",
        "entry": {
            "agent_role": "customer_support_agent", "agent_autonomy_level": 4,
            "user_role": "vendor", "requested_action": "export_report",
            "tool_requested": "file_storage_api", "resource_type": "api_key_secret",
            "resource_sensitivity": 5, "permission_match": 0,
            "action_risk_score": 95, "prompt_injection_detected": 1,
            "data_exfiltration_risk": 90, "human_approval_required": 0,
            "previous_failed_attempts": 3, "audit_log_available": 0,
        }
    },
    {
        "name": "Escalate — permission OK but elevated risk",
        "entry": {
            "agent_role": "finance_agent", "agent_autonomy_level": 3,
            "user_role": "manager", "requested_action": "change_permission",
            "tool_requested": "hris_api", "resource_type": "employee_record",
            "resource_sensitivity": 4, "permission_match": 1,
            "action_risk_score": 58, "prompt_injection_detected": 0,
            "data_exfiltration_risk": 55, "human_approval_required": 1,
            "previous_failed_attempts": 1, "audit_log_available": 1,
        }
    },
    {
        "name": "Borderline — high autonomy + suspicious pattern",
        "entry": {
            "agent_role": "hr_onboarding_agent", "agent_autonomy_level": 5,
            "user_role": "admin", "requested_action": "send_email",
            "tool_requested": "external_webhook", "resource_type": "sales_pipeline",
            "resource_sensitivity": 3, "permission_match": 1,
            "action_risk_score": 72, "prompt_injection_detected": 0,
            "data_exfiltration_risk": 68, "human_approval_required": 1,
            "previous_failed_attempts": 2, "audit_log_available": 1,
        }
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 56)
    print("  STEP 7 — Inference Function Demo")
    print("  AgentPyro — Agentic AI Security Risk Predictor")
    print("=" * 56)

    summary = []
    for scenario in DEMO_SCENARIOS:
        try:
            result = predict_access_decision(scenario["entry"])
            print_result(result, scenario_name=scenario["name"])
            summary.append({
                "scenario":   scenario["name"],
                "decision":   result["decision"],
                "confidence": result["confidence"],
                "risk_level": result["risk_level"],
            })
        except ValueError as e:
            print(f"\n[INF] Validation error in '{scenario['name']}': {e}")

    print(f"\n{'='*56}")
    print(f"  {'SCENARIO':<36} {'DECISION':<22} {'CONF':>6}  RISK")
    print(f"  {'─'*36} {'─'*22} {'─'*6}  {'─'*8}")
    for r in summary:
        d   = r["decision"]
        col = ANSI.get(d, "")
        rst = ANSI["reset"]
        print(f"  {r['scenario'][:36]:<36} "
              f"{col}{d:<22}{rst} "
              f"{r['confidence']:>6.2%}  {r['risk_level']}")
    print(f"{'='*56}")
    