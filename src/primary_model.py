"""
=============================================================
Step 4 — Primary Model (XGBoost + LightGBM)
=============================================================
Purpose:
    Train two gradient-boosted tree models with hyperparameter
    tuning via RandomizedSearchCV. Compare against Step 3 baselines
    and select the best model for evaluation and explainability.

    Models:
      - XGBoost   (extreme gradient boosting — fast, regularised)
      - LightGBM  (leaf-wise growth — efficient on larger datasets)

    Both are trained on the SMOTE-resampled training set and
    evaluated on the untouched test set.

    Tuning strategy:
      RandomizedSearchCV with 5-fold stratified cross-validation.
      Optimises for macro F1 to treat all three classes equally.

Outputs (saved to reports/):
    - primary_confusion_matrices.png
    - primary_model_comparison.png
    - primary_classification_report.txt
    - primary_cv_results.csv

Outputs (saved to models/):
    - xgboost_best.pkl
    - lightgbm_best.pkl
    - best_model.pkl          ← the single best model (used in Steps 5-7)
    - best_model_name.txt     ← records which model won

Run:
    python src/primary_model.py
=============================================================
"""

import os
import joblib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics          import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES   = ["Allowed", "Blocked", "Needs Approval"]
RANDOM_STATE  = 42
CV_FOLDS      = 5
N_ITER        = 30          # number of random hyperparameter combinations to try


# ── 1. Load preprocessed data ─────────────────────────────────────────────────
def load_data():
    """Load SMOTE-resampled training data and the raw test set from Step 2."""
    X_train = joblib.load(os.path.join(MODELS_DIR, "X_train_resampled.pkl"))
    y_train = joblib.load(os.path.join(MODELS_DIR, "y_train_resampled.pkl"))
    X_test  = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test  = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    print(f"[PRIM] Train : {X_train.shape}  |  Test : {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ── 2. Hyperparameter search spaces ───────────────────────────────────────────
def xgb_param_grid() -> dict:
    """
    XGBoost search space.
    - n_estimators       : number of boosting rounds
    - max_depth          : tree depth (deeper = more complex)
    - learning_rate      : shrinkage — lower needs more trees
    - subsample          : fraction of rows per tree (reduces overfitting)
    - colsample_bytree   : fraction of features per tree
    - min_child_weight   : minimum leaf node weight (regularisation)
    - gamma              : min loss reduction to split (regularisation)
    """
    return {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [3, 4, 5, 6, 7],
        "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight":  [1, 3, 5, 7],
        "gamma":             [0, 0.1, 0.2, 0.3],
    }


def lgbm_param_grid() -> dict:
    """
    LightGBM search space.
    - n_estimators       : number of boosting rounds
    - max_depth          : tree depth (-1 = unlimited)
    - learning_rate      : shrinkage rate
    - num_leaves         : max leaves per tree (key LightGBM param)
    - min_child_samples  : min samples in a leaf node
    - subsample          : row subsampling ratio
    - colsample_bytree   : feature subsampling ratio
    - reg_alpha          : L1 regularisation
    - reg_lambda         : L2 regularisation
    """
    return {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [3, 4, 5, 6, 7, -1],
        "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
        "num_leaves":        [15, 31, 63, 127],
        "min_child_samples": [5, 10, 20, 30],
        "subsample":         [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree":  [0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha":         [0, 0.01, 0.1, 0.5],
        "reg_lambda":        [0, 0.01, 0.1, 1.0],
    }


# ── 3. Train with RandomizedSearchCV ──────────────────────────────────────────
def tune_model(base_model, param_grid: dict, X_train, y_train,
               model_name: str):
    """
    RandomizedSearchCV with stratified 5-fold CV.
    Scoring = macro F1 to penalise ignoring the minority class.
    n_iter=30 balances search thoroughness with runtime.
    """
    print(f"\n[PRIM] Tuning {model_name} ({N_ITER} iterations × {CV_FOLDS} folds) ...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_grid,
        n_iter=N_ITER,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=RANDOM_STATE,
        refit=True           # refit best params on full training set
    )
    search.fit(X_train, y_train)

    print(f"[PRIM] {model_name} best CV Macro F1 : {search.best_score_:.4f}")
    print(f"[PRIM] Best params : {search.best_params_}")
    return search


# ── 4. Evaluate a trained search object ───────────────────────────────────────
def evaluate_model(search, X_test, y_test, model_name: str) -> dict:
    """Evaluate the best estimator on the held-out test set."""
    y_pred  = search.best_estimator_.predict(X_test)
    acc     = accuracy_score(y_test, y_pred)
    f1      = f1_score(y_test, y_pred, average="macro")
    report  = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"\n── {model_name} (test set) ──────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n{report}")

    return {
        "name":     model_name,
        "model":    search.best_estimator_,
        "search":   search,
        "y_pred":   y_pred,
        "accuracy": acc,
        "macro_f1": f1,
    }


# ── 5. Plot: confusion matrices ───────────────────────────────────────────────
def plot_confusion_matrices(results: list, y_test: np.ndarray) -> None:
    """Side-by-side normalised confusion matrices for XGBoost and LightGBM."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, result in zip(axes, results):
        cm = confusion_matrix(y_test, result["y_pred"], normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax, annot_kws={"size": 11}
        )
        ax.set_title(
            f"{result['name']}\nAccuracy {result['accuracy']:.4f}  |  "
            f"Macro F1 {result['macro_f1']:.4f}",
            fontweight="bold", pad=10
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("Primary Models — Confusion Matrices (normalised)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "primary_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PRIM] Saved → {path}")


# ── 6. Plot: model comparison (vs baselines) ──────────────────────────────────
def plot_model_comparison(results: list) -> None:
    """
    Bar chart comparing all four models (baselines + primary).
    Baseline numbers are loaded from the classification report.
    """
    # Baseline numbers from Step 3
    all_results = [
        {"name": "Logistic\nRegression", "accuracy": 0.9523, "macro_f1": 0.9215},
        {"name": "Decision\nTree",       "accuracy": 0.9864, "macro_f1": 0.9736},
    ] + [{"name": r["name"].replace(" ", "\n"), 
          "accuracy": r["accuracy"], "macro_f1": r["macro_f1"]} for r in results]

    names  = [r["name"] for r in all_results]
    acc    = [r["accuracy"]  for r in all_results]
    f1     = [r["macro_f1"]  for r in all_results]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, acc, width, label="Accuracy",
                   color=["#AACDE8", "#AACDE8", "#4A90D9", "#4A90D9"],
                   edgecolor="none")
    bars2 = ax.bar(x + width/2, f1,  width, label="Macro F1",
                   color=["#A8DBBF", "#A8DBBF", "#1D9E75", "#1D9E75"],
                   edgecolor="none")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=9
            )

    ax.set_ylim(0.85, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Score")
    ax.set_title("All Models — Accuracy & Macro F1 Comparison\n(darker = primary models)",
                 fontweight="bold", pad=12)
    ax.legend(frameon=False)
    ax.axvline(1.5, color="#cccccc", linestyle="--", linewidth=0.8)
    ax.text(0.75, 1.03, "Baselines", ha="center", fontsize=9, color="#888888")
    ax.text(2.5,  1.03, "Primary",   ha="center", fontsize=9, color="#888888")
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "primary_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[PRIM] Saved → {path}")


# ── 7. Save CV results ────────────────────────────────────────────────────────
def save_cv_results(results: list) -> None:
    """Save the top 10 hyperparameter combinations for each model to CSV."""
    path = os.path.join(REPORTS_DIR, "primary_cv_results.csv")
    frames = []
    for result in results:
        cv_df = pd.DataFrame(result["search"].cv_results_)
        cv_df["model"] = result["name"]
        frames.append(cv_df.nlargest(10, "mean_test_score"))
    pd.concat(frames).to_csv(path, index=False)
    print(f"[PRIM] Saved → {path}")


# ── 8. Save classification report ─────────────────────────────────────────────
def save_classification_report(results: list, y_test: np.ndarray) -> None:
    path = os.path.join(REPORTS_DIR, "primary_classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  STEP 4 — Primary Models Classification Report\n")
        f.write("=" * 60 + "\n\n")
        f.write("  Baseline reference:\n")
        f.write("  Logistic Regression  Accuracy: 0.9523   Macro F1: 0.9215\n")
        f.write("  Decision Tree        Accuracy: 0.9864   Macro F1: 0.9736\n\n")
        for result in results:
            f.write(f"── {result['name']} ──────────────────────────────\n")
            f.write(f"  Accuracy  : {result['accuracy']:.4f}\n")
            f.write(f"  Macro F1  : {result['macro_f1']:.4f}\n\n")
            f.write(classification_report(
                y_test, result["y_pred"], target_names=CLASS_NAMES
            ))
            f.write("\n\n")
    print(f"[PRIM] Saved → {path}")


# ── 9. Save models + elect best ───────────────────────────────────────────────
def save_models(results: list) -> None:
    """
    Save both tuned models. Also save the single best model separately
    as best_model.pkl — this is what Steps 5, 6, and 7 will load.
    """
    name_map = {
        "XGBoost":  "xgboost_best.pkl",
        "LightGBM": "lightgbm_best.pkl",
    }
    for result in results:
        path = os.path.join(MODELS_DIR, name_map[result["name"]])
        joblib.dump(result["model"], path)
        print(f"[PRIM] Saved → {path}")

    # Select best model by macro F1
    best = max(results, key=lambda r: r["macro_f1"])
    best_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(best["model"], best_path)
    print(f"[PRIM] Best model  → {best['name']}  (Macro F1: {best['macro_f1']:.4f})")
    print(f"[PRIM] Saved       → {best_path}")

    # Record name for Steps 5–7 to reference
    name_path = os.path.join(MODELS_DIR, "best_model_name.txt")
    with open(name_path, "w", encoding="utf-8") as f:
        f.write(best["name"])


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 4 — Primary Model (XGBoost + LightGBM)")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # ── XGBoost ──
    xgb_base = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )
    xgb_search = tune_model(xgb_base, xgb_param_grid(), X_train, y_train, "XGBoost")
    xgb_result = evaluate_model(xgb_search, X_test, y_test, "XGBoost")

    # ── LightGBM ──
    lgbm_base = LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    lgbm_search = tune_model(lgbm_base, lgbm_param_grid(), X_train, y_train, "LightGBM")
    lgbm_result = evaluate_model(lgbm_search, X_test, y_test, "LightGBM")

    results = [xgb_result, lgbm_result]

    # Visualise
    plot_confusion_matrices(results, y_test)
    plot_model_comparison(results)

    # Save reports and models
    save_cv_results(results)
    save_classification_report(results, y_test)
    save_models(results)

    # Final summary
    print("\n── Summary ───────────────────────────────────────────")
    print(f"  {'Model':<12} {'Accuracy':>10}   {'Macro F1':>10}")
    print(f"  {'─'*12}   {'─'*10}   {'─'*10}")
    print(f"  {'LR (base)':<12} {'0.9523':>10}   {'0.9215':>10}")
    print(f"  {'DT (base)':<12} {'0.9864':>10}   {'0.9736':>10}")
    for r in results:
        print(f"  {r['name']:<12} {r['accuracy']:>10.4f}   {r['macro_f1']:>10.4f}")
    print("\n[PRIM] Step 4 complete. Ready to commit.")
    print("[PRIM] Next → Step 5: Evaluation & Confidence Scoring\n")