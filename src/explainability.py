"""
=============================================================
Step 6 — Explainability (SHAP)  (v2 — leakage-free)
=============================================================
Purpose:
    Explain WHY the model makes each decision using SHAP
    (SHapley Additive exPlanations). This is critical for a
    security system — stakeholders must understand and trust
    the reasoning behind every access decision.

    Two levels of explanation are produced:
      1. Global  — which features matter most overall
      2. Local   — why a specific individual prediction was made

    SHAP assigns each feature a contribution value (positive =
    pushes toward Blocked, negative = pushes toward Allowed)
    for every single prediction, grounded in game theory.

    Feature set: 11 clean features (v2 — leakage-free)
      Categorical (5): agent_role, user_role, requested_action,
                       tool_requested, resource_type
      Numeric     (3): agent_autonomy_level, resource_sensitivity,
                       previous_failed_attempts
      Binary      (3): permission_match, prompt_injection_detected,
                       audit_log_available

Outputs (saved to reports/):
    - shap_global_importance.png     ← mean |SHAP| per feature
    - shap_summary_beeswarm.png      ← full distribution of SHAP values
    - shap_dependence_top2.png       ← how top features interact
    - shap_local_examples.png        ← 3 individual prediction explanations
    - shap_class_importance.png      ← feature importance per class

Outputs (saved to models/):
    - shap_explainer.pkl             ← used by inference.py (Step 7)
    - shap_values.pkl                ← raw SHAP values

Run:
    python src/explainability.py
=============================================================
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Feature names (v2 — must match ColumnTransformer order exactly) ───────────
# Order: categorical → numeric → binary  (same as preprocessing.py)
#
# Removed vs v1:
#   action_risk_score       (was in NUMERIC_COLS)
#   data_exfiltration_risk  (was in NUMERIC_COLS)
#   human_approval_required (was in BINARY_COLS)
FEATURE_NAMES = [
    # Categorical — OrdinalEncoded (5)
    "agent_role",
    "user_role",
    "requested_action",
    "tool_requested",
    "resource_type",
    # Numeric — StandardScaled (3)
    "agent_autonomy_level",
    "resource_sensitivity",
    "previous_failed_attempts",
    # Binary — passthrough (3)
    "permission_match",
    "prompt_injection_detected",
    "audit_log_available",
]

CLASS_NAMES  = ["Allowed", "Blocked", "Needs Approval"]
CLASS_COLORS = ["#1D9E75", "#E24B4A", "#EF9F27"]


# ── 1. Load artifacts ─────────────────────────────────────────────────────────
def load_artifacts():
    """Load best model and test set from previous steps."""
    model  = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    X_test = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    le     = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()

    print(f"[SHAP] Model     : {model_name}")
    print(f"[SHAP] Test set  : {X_test.shape[0]} samples × {X_test.shape[1]} features")

    assert X_test.shape[1] == 11, (
        f"Feature count mismatch! Got {X_test.shape[1]}, expected 11. "
        "Re-run preprocessing.py first."
    )
    assert len(FEATURE_NAMES) == X_test.shape[1], (
        f"FEATURE_NAMES length ({len(FEATURE_NAMES)}) does not match "
        f"X_test columns ({X_test.shape[1]}). Check FEATURE_NAMES."
    )
    return model, X_test, y_test, le, model_name


# ── 2. Compute SHAP values ────────────────────────────────────────────────────
def compute_shap_values(model, X_test: np.ndarray):
    """
    TreeExplainer is the fast, exact SHAP method for tree-based models.
    Returns a list of arrays — one per class — each shape (n_samples, n_features).
    shap_values[0] = contributions toward 'Allowed'
    shap_values[1] = contributions toward 'Blocked'
    shap_values[2] = contributions toward 'Needs_Human_Approval'
    """
    print("[SHAP] Computing SHAP values (TreeExplainer) ...")
    explainer = shap.TreeExplainer(model)
    shap_raw  = explainer.shap_values(X_test)

    # Newer SHAP versions return a 3D array (n_samples, n_features, n_classes)
    # Older versions return a list of (n_samples, n_features) arrays per class.
    # Normalise to list format so all downstream code works identically.
    if isinstance(shap_raw, np.ndarray) and shap_raw.ndim == 3:
        shap_values = [shap_raw[:, :, i] for i in range(shap_raw.shape[2])]
        print(f"[SHAP] 3D array detected — reshaped to list of {len(shap_values)} class arrays")
    else:
        shap_values = shap_raw

    X_df = pd.DataFrame(X_test, columns=FEATURE_NAMES)
    print(f"[SHAP] SHAP values computed — shape per class: {np.array(shap_values[0]).shape}")
    return explainer, shap_values, X_df


# ── 3. Plot: global feature importance ────────────────────────────────────────
def plot_global_importance(shap_values: list, X_df: pd.DataFrame) -> None:
    """
    Mean absolute SHAP value per feature, averaged across all classes.
    Answers: 'Which features have the biggest overall impact
    on access decisions?'
    With leaky features removed, this now shows genuine signal —
    permission_match and prompt_injection_detected should rank highly.
    """
    mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    importance_df = pd.DataFrame({
        "feature":    FEATURE_NAMES,
        "importance": mean_abs
    }).sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4A90D9" if imp < importance_df["importance"].median()
              else "#1D63B8" for imp in importance_df["importance"]]
    bars = ax.barh(importance_df["feature"], importance_df["importance"],
                   color=colors, edgecolor="none", height=0.65)

    for bar, val in zip(bars, importance_df["importance"]):
        ax.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="#333333")

    ax.set_title(
        "Global Feature Importance (SHAP)\n"
        "Mean |SHAP value| across all classes  |  v2 clean features",
        fontweight="bold", pad=12
    )
    ax.set_xlabel("Mean |SHAP value|")
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "shap_global_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {path}")

    print("\n[SHAP] Feature importance ranking:")
    for _, row in importance_df.sort_values("importance", ascending=False).iterrows():
        bar_len = int(row["importance"] / importance_df["importance"].max() * 30)
        print(f"  {row['feature']:<30} {'█' * bar_len}  {row['importance']:.4f}")


# ── 4. Plot: beeswarm summary ─────────────────────────────────────────────────
def plot_beeswarm(shap_values: list, X_df: pd.DataFrame) -> None:
    """
    Beeswarm plot for the 'Blocked' class (index 1) — the most critical class.
    Each dot = one test sample. X-axis = SHAP value (impact on Blocked prediction).
    Color = feature value (red=high, blue=low).
    max_display set to 11 to show all clean features.
    """
    print("[SHAP] Generating beeswarm plot (Blocked class) ...")
    plt.figure(figsize=(9, 6))
    shap.summary_plot(
        shap_values[1],      # index 1 = Blocked class
        X_df,
        feature_names=FEATURE_NAMES,
        show=False,
        plot_size=None,
        color_bar=True,
        max_display=11,      # show all 11 clean features
    )
    plt.title(
        "SHAP Beeswarm — Blocked Class\n"
        "Each point = one prediction. Right = pushes toward Blocked.  "
        "v2 clean features",
        fontweight="bold", pad=12
    )
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "shap_summary_beeswarm.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {path}")


# ── 5. Plot: per-class feature importance ─────────────────────────────────────
def plot_class_importance(shap_values: list) -> None:
    """
    Side-by-side bar charts showing top 8 features for each class.
    Reveals whether different features drive different decisions —
    e.g., prompt_injection_detected may matter more for Blocked than
    for Needs_Human_Approval.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, sv, name, color in zip(axes, shap_values, CLASS_NAMES, CLASS_COLORS):
        mean_abs = np.abs(sv).mean(axis=0)
        idx      = np.argsort(mean_abs)[-8:]     # top 8 of 11 features
        feats    = [FEATURE_NAMES[i] for i in idx]
        vals     = mean_abs[idx]

        ax.barh(feats, vals, color=color, edgecolor="none", height=0.6)
        ax.set_title(f"{name}", fontweight="bold", color=color)
        ax.set_xlabel("Mean |SHAP|")
        sns.despine(ax=ax)

    fig.suptitle(
        "Feature Importance by Class (SHAP)\n"
        "Top 8 of 11 features per decision outcome  |  v2 clean features",
        fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "shap_class_importance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {path}")


# ── 6. Plot: dependence plots for top 2 features ─────────────────────────────
def plot_dependence(shap_values: list, X_df: pd.DataFrame) -> None:
    """
    Dependence plot: shows how a single feature's value relates to its
    SHAP value, with a second feature colour-coded for interaction effects.
    Top 2 global features are selected automatically.
    """
    mean_abs  = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    top2_idx  = np.argsort(mean_abs)[-2:][::-1]
    top2_feat = [FEATURE_NAMES[i] for i in top2_idx]

    print(f"[SHAP] Dependence plots for top 2 features: {top2_feat}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, feat in zip(axes, top2_feat):
        shap.dependence_plot(
            feat,
            shap_values[1],      # Blocked class
            X_df,
            interaction_index="auto",
            show=False,
            ax=ax,
        )
        ax.set_title(
            f"Dependence: {feat}\n(Blocked class, colour = interaction feature)",
            fontweight="bold", fontsize=10
        )

    plt.suptitle(
        "SHAP Dependence Plots — Top 2 Features  |  v2 clean features",
        fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "shap_dependence_top2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {path}")


# ── 7. Plot: local explanations (3 individual predictions) ───────────────────
def plot_local_examples(explainer, shap_values: list,
                        X_df: pd.DataFrame,
                        y_test: np.ndarray,
                        y_pred: np.ndarray) -> None:
    """
    Waterfall plots for 3 individual predictions:
      - One clearly Allowed
      - One clearly Blocked
      - One Needs_Human_Approval

    Each plot shows exactly which features pushed the decision
    up or down, making the model's reasoning fully transparent.
    """
    samples = {}
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask = (y_test == cls_idx) & (y_pred == cls_idx)
        idxs = np.where(mask)[0]
        if len(idxs) > 0:
            samples[cls_name] = idxs[0]

    fig, axes = plt.subplots(1, len(samples), figsize=(6 * len(samples), 5))
    if len(samples) == 1:
        axes = [axes]

    for ax, (cls_name, sample_idx) in zip(axes, samples.items()):
        cls_idx   = CLASS_NAMES.index(cls_name)
        sv_sample = shap_values[cls_idx][sample_idx]

        ev = explainer.expected_value
        if hasattr(ev, '__len__') and len(ev) > cls_idx:
            base_value = ev[cls_idx]
        elif hasattr(ev, '__len__'):
            base_value = ev[0]
        else:
            base_value = float(ev)

        # Sort features by |SHAP| for this sample — show top 8 of 11
        order      = np.argsort(np.abs(sv_sample))[-8:]
        feat_names = [FEATURE_NAMES[i] for i in order]
        feat_vals  = sv_sample[order]
        feat_data  = X_df.iloc[sample_idx][FEATURE_NAMES].values[order]

        colors = ["#1D9E75" if v < 0 else "#E24B4A" for v in feat_vals]

        ax.barh(feat_names, feat_vals, color=colors, edgecolor="none", height=0.6)
        ax.axvline(0, color="#666666", linewidth=0.8)
        ax.set_title(
            f"Prediction: {cls_name}\n(sample #{sample_idx})",
            fontweight="bold",
            color=CLASS_COLORS[cls_idx],
            fontsize=10
        )
        ax.set_xlabel("SHAP value\n(+ = toward this class, − = away)")
        for i, (bar_val, data_val) in enumerate(zip(feat_vals, feat_data)):
            ax.text(
                bar_val + (0.002 if bar_val >= 0 else -0.002),
                i,
                f" val={data_val:.2f}" if isinstance(data_val, float)
                else f" val={data_val}",
                va="center",
                ha="left" if bar_val >= 0 else "right",
                fontsize=7.5,
                color="#444444"
            )
        sns.despine(ax=ax)

    fig.suptitle(
        "Local SHAP Explanations — One Sample Per Class\n"
        "Red = increases prediction probability, Green = decreases it  "
        "|  v2 clean features",
        fontweight="bold", y=1.02
    )
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "shap_local_examples.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SHAP] Saved → {path}")


# ── 8. Save raw SHAP values and explainer ────────────────────────────────────
def save_shap_artifacts(shap_values: list, explainer) -> None:
    """
    Persist SHAP values and explainer for use in Step 7 (inference).
    The inference function uses the explainer to explain each live prediction.
    """
    joblib.dump(shap_values, os.path.join(MODELS_DIR, "shap_values.pkl"))
    joblib.dump(explainer,   os.path.join(MODELS_DIR, "shap_explainer.pkl"))
    print(f"[SHAP] Saved → models/shap_values.pkl")
    print(f"[SHAP] Saved → models/shap_explainer.pkl")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 6 — Explainability (SHAP)  (v2 — leakage-free)")
    print("=" * 60)

    # Load
    model, X_test, y_test, le, model_name = load_artifacts()
    y_pred = model.predict(X_test)

    # Compute SHAP
    explainer, shap_values, X_df = compute_shap_values(model, X_test)

    # Global plots
    print("\n[SHAP] Generating plots ...")
    plot_global_importance(shap_values, X_df)
    plot_beeswarm(shap_values, X_df)
    plot_class_importance(shap_values)
    plot_dependence(shap_values, X_df)

    # Local plots
    plot_local_examples(explainer, shap_values, X_df, y_test, y_pred)

    # Save artifacts for Step 7
    save_shap_artifacts(shap_values, explainer)

    print(f"\n[SHAP] All outputs saved to reports/")
    print("[SHAP] Step 6 complete.")
    print("[SHAP] Next → Step 7: Update inference.py\n")