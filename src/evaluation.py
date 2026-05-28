"""
=============================================================
Step 5 — Evaluation & Confidence Scoring
=============================================================
Purpose:
    Deep evaluation of the best model (LightGBM) beyond simple
    accuracy. This step focuses on:
      - Detailed per-class metrics (precision, recall, F1)
      - Confidence score analysis (predicted probabilities)
      - Probability calibration check (are the scores trustworthy?)
      - Confidence threshold analysis (what if we only act on
        high-confidence predictions?)
      - Misclassification analysis (what does the model get wrong?)
      - ROC-AUC per class (one-vs-rest)

    High confidence scores are the goal stated at project start.
    This step measures exactly how confident the model is and
    whether those confidence scores are well-calibrated.

Outputs (saved to reports/):
    - eval_confidence_distribution.png
    - eval_calibration_curve.png
    - eval_confidence_by_class.png
    - eval_roc_curves.png
    - eval_threshold_analysis.png
    - eval_misclassifications.csv
    - eval_full_report.txt

Run:
    python src/evaluation.py
=============================================================
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    brier_score_loss,
)
from sklearn.calibration    import calibration_curve
from sklearn.preprocessing  import label_binarize

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
DATA_PATH   = os.path.join("data", "agent_security_risk_scores.csv")
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES  = ["Allowed", "Blocked", "Needs Approval"]
CLASS_COLORS = ["#1D9E75", "#E24B4A", "#EF9F27"]
N_CLASSES    = 3


# ── 1. Load artifacts ─────────────────────────────────────────────────────────
def load_artifacts():
    """Load the best model and test set from previous steps."""
    model   = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    X_test  = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test  = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    le      = joblib.load(os.path.join(MODELS_DIR, "label_encoder.pkl"))

    with open(os.path.join(MODELS_DIR, "best_model_name.txt")) as f:
        model_name = f.read().strip()

    print(f"[EVAL] Model loaded  : {model_name}")
    print(f"[EVAL] Test set      : {X_test.shape[0]} samples")
    return model, X_test, y_test, le, model_name


# ── 2. Core predictions ───────────────────────────────────────────────────────
def get_predictions(model, X_test, y_test):
    """
    Get hard predictions and soft probability scores.
    predict_proba returns a (n_samples, 3) array where each row
    sums to 1.0 — the model's confidence in each class.
    """
    y_pred      = model.predict(X_test)
    y_proba     = model.predict_proba(X_test)          # shape (440, 3)
    confidence  = y_proba.max(axis=1)                  # highest prob per sample

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="macro")
    print(f"[EVAL] Accuracy      : {acc:.4f}")
    print(f"[EVAL] Macro F1      : {f1:.4f}")
    print(f"[EVAL] Mean confidence  : {confidence.mean():.4f}")
    print(f"[EVAL] Median confidence: {np.median(confidence):.4f}")
    print(f"[EVAL] % above 0.90  : {(confidence >= 0.90).mean()*100:.1f}%")
    print(f"[EVAL] % above 0.95  : {(confidence >= 0.95).mean()*100:.1f}%")
    print(f"[EVAL] % above 0.99  : {(confidence >= 0.99).mean()*100:.1f}%")
    return y_pred, y_proba, confidence


# ── 3. Plot: confidence distribution ─────────────────────────────────────────
def plot_confidence_distribution(confidence: np.ndarray,
                                 y_pred: np.ndarray,
                                 y_test: np.ndarray) -> None:
    """
    Histogram of max predicted probability across all test samples.
    Colour-coded by whether the prediction was correct or wrong.
    A good model should have most mass above 0.90.
    """
    correct   = confidence[y_pred == y_test]
    incorrect = confidence[y_pred != y_test]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(correct,   bins=30, alpha=0.75, color="#1D9E75",
            label=f"Correct ({len(correct)})",   edgecolor="none")
    ax.hist(incorrect, bins=30, alpha=0.85, color="#E24B4A",
            label=f"Incorrect ({len(incorrect)})", edgecolor="none")

    ax.axvline(0.90, color="#333333", linestyle="--", linewidth=1,
               label="0.90 threshold")
    ax.axvline(0.95, color="#888888", linestyle=":",  linewidth=1,
               label="0.95 threshold")

    ax.set_xlabel("Confidence Score (max predicted probability)")
    ax.set_ylabel("Number of predictions")
    ax.set_title("Confidence Score Distribution\n(correct vs incorrect predictions)",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eval_confidence_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved → {path}")


# ── 4. Plot: confidence by class ──────────────────────────────────────────────
def plot_confidence_by_class(y_proba: np.ndarray,
                              y_test: np.ndarray) -> None:
    """
    Box plot of predicted probability for the TRUE class, split by class.
    High median = model is confident when it's right.
    Wide spread = model is uncertain for that class.
    """
    rows = []
    for i, (proba_row, true_label) in enumerate(zip(y_proba, y_test)):
        rows.append({
            "class":      CLASS_NAMES[true_label],
            "confidence": proba_row[true_label]
        })
    df_conf = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    palette = dict(zip(CLASS_NAMES, CLASS_COLORS))
    sns.boxplot(data=df_conf, x="class", y="confidence",
                palette=palette, width=0.45, linewidth=1.2,
                fliersize=3, ax=ax)

    ax.axhline(0.90, color="#333333", linestyle="--", linewidth=1,
               label="0.90 threshold")
    ax.set_title("Confidence Score for True Class\n(how sure the model is when correct)",
                 fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Predicted probability for true class")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.legend(frameon=False, fontsize=9)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eval_confidence_by_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved → {path}")


# ── 5. Plot: calibration curve ───────────────────────────────────────────────
def plot_calibration_curve(y_proba: np.ndarray, y_test: np.ndarray) -> None:
    """
    Reliability diagram — compares predicted confidence vs actual accuracy.
    A perfectly calibrated model lies on the diagonal.
    Above diagonal = underconfident. Below diagonal = overconfident.
    """
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        prob_true, prob_pred = calibration_curve(
            y_test_bin[:, i], y_proba[:, i], n_bins=8, strategy="quantile"
        )
        brier = brier_score_loss(y_test_bin[:, i], y_proba[:, i])
        ax.plot(prob_pred, prob_true, marker="o", markersize=5,
                color=color, linewidth=1.8,
                label=f"{name}  (Brier: {brier:.4f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration Curve (Reliability Diagram)\n"
                 "Brier score: lower is better",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eval_calibration_curve.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved → {path}")


# ── 6. Plot: ROC curves (one-vs-rest) ────────────────────────────────────────
def plot_roc_curves(y_proba: np.ndarray, y_test: np.ndarray) -> None:
    """
    ROC curve per class using one-vs-rest strategy.
    AUC close to 1.0 = near-perfect class separation.
    """
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    fig, ax    = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random classifier")

    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        auc         = roc_auc_score(y_test_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name}  (AUC = {auc:.4f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — One vs Rest\n(per class)",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eval_roc_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved → {path}")


# ── 7. Plot: confidence threshold analysis ────────────────────────────────────
def plot_threshold_analysis(confidence: np.ndarray,
                             y_pred: np.ndarray,
                             y_test: np.ndarray) -> None:
    """
    Shows how accuracy and coverage (% of predictions made) change
    as we raise the confidence threshold. In deployment, low-confidence
    predictions can be routed to a human reviewer instead of acted on.
    """
    thresholds = np.arange(0.50, 1.001, 0.01)
    accuracies, coverages = [], []

    for t in thresholds:
        mask     = confidence >= t
        coverage = mask.mean()
        if mask.sum() == 0:
            acc = np.nan
        else:
            acc = accuracy_score(y_test[mask], y_pred[mask])
        accuracies.append(acc)
        coverages.append(coverage)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, accuracies, color="#1D9E75", linewidth=2,
             label="Accuracy on retained predictions")
    ax2.plot(thresholds, [c * 100 for c in coverages],
             color="#4A90D9", linewidth=2, linestyle="--",
             label="Coverage (%)")

    ax1.axvline(0.90, color="#333333", linestyle=":", linewidth=1)
    ax1.axvline(0.95, color="#888888", linestyle=":", linewidth=1)

    ax1.set_xlabel("Confidence threshold")
    ax1.set_ylabel("Accuracy",  color="#1D9E75")
    ax2.set_ylabel("Coverage (%)", color="#4A90D9")
    ax1.set_title("Confidence Threshold Analysis\n"
                  "Trade-off between accuracy and coverage",
                  fontweight="bold")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1%}"))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, frameon=False, fontsize=9)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eval_threshold_analysis.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EVAL] Saved → {path}")


# ── 8. Save misclassification analysis ───────────────────────────────────────
def save_misclassifications(X_test: np.ndarray, y_test: np.ndarray,
                             y_pred: np.ndarray, y_proba: np.ndarray,
                             le) -> None:
    """
    Export every misclassified test sample with its true label,
    predicted label, and confidence score. Useful for understanding
    the model's failure modes.
    """
    mask       = y_pred != y_test
    wrong_idx  = np.where(mask)[0]

    rows = []
    for i in wrong_idx:
        rows.append({
            "sample_index":   i,
            "true_label":     le.classes_[y_test[i]],
            "predicted_label":le.classes_[y_pred[i]],
            "confidence":     round(y_proba[i].max(), 4),
            "prob_allowed":   round(y_proba[i][0], 4),
            "prob_blocked":   round(y_proba[i][1], 4),
            "prob_needs_appr":round(y_proba[i][2], 4),
        })

    df_wrong = pd.DataFrame(rows).sort_values("confidence", ascending=False)
    path = os.path.join(REPORTS_DIR, "eval_misclassifications.csv")
    df_wrong.to_csv(path, index=False)
    print(f"[EVAL] Misclassified samples : {len(df_wrong)}")
    print(f"[EVAL] Saved → {path}")


# ── 9. Save full evaluation report ───────────────────────────────────────────
def save_full_report(model_name: str, y_test: np.ndarray,
                     y_pred: np.ndarray, y_proba: np.ndarray,
                     confidence: np.ndarray) -> None:
    """Write a comprehensive text report summarising all evaluation metrics."""
    acc   = accuracy_score(y_test, y_pred)
    f1    = f1_score(y_test, y_pred, average="macro")
    y_bin = label_binarize(y_test, classes=[0, 1, 2])
    auc   = roc_auc_score(y_bin, y_proba, average="macro", multi_class="ovr")

    path = os.path.join(REPORTS_DIR, "eval_full_report.txt")
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"  STEP 5 — Evaluation Report  [{model_name}]\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"  Accuracy            : {acc:.4f}\n")
        f.write(f"  Macro F1            : {f1:.4f}\n")
        f.write(f"  Macro ROC-AUC       : {auc:.4f}\n\n")
        f.write(f"  Confidence scores (max predicted probability):\n")
        f.write(f"    Mean              : {confidence.mean():.4f}\n")
        f.write(f"    Median            : {np.median(confidence):.4f}\n")
        f.write(f"    Std dev           : {confidence.std():.4f}\n")
        f.write(f"    % above 0.90      : {(confidence >= 0.90).mean()*100:.1f}%\n")
        f.write(f"    % above 0.95      : {(confidence >= 0.95).mean()*100:.1f}%\n")
        f.write(f"    % above 0.99      : {(confidence >= 0.99).mean()*100:.1f}%\n\n")
        f.write("  Classification Report:\n")
        f.write(classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        f.write("\n  Confusion Matrix (raw counts):\n")
        cm = confusion_matrix(y_test, y_pred)
        df_cm = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
        f.write(df_cm.to_string())
        f.write("\n")
    print(f"[EVAL] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 5 — Evaluation & Confidence Scoring")
    print("=" * 60)

    # Load
    model, X_test, y_test, le, model_name = load_artifacts()

    # Predict
    y_pred, y_proba, confidence = get_predictions(model, X_test, y_test)

    # Plots
    print("\n[EVAL] Generating evaluation plots ...")
    plot_confidence_distribution(confidence, y_pred, y_test)
    plot_confidence_by_class(y_proba, y_test)
    plot_calibration_curve(y_proba, y_test)
    plot_roc_curves(y_proba, y_test)
    plot_threshold_analysis(confidence, y_pred, y_test)

    # Reports
    save_misclassifications(X_test, y_test, y_pred, y_proba, le)
    save_full_report(model_name, y_test, y_pred, y_proba, confidence)

    print(f"\n[EVAL] All outputs saved to reports/")
    