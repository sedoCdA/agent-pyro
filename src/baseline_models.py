"""
=============================================================
Step 3 — Baseline Models
=============================================================
Purpose:
    Establish a performance benchmark before training the primary
    model. Two simple, interpretable classifiers are trained and
    evaluated so we have a clear bar to beat in Step 4.

    Models:
      - Logistic Regression   (linear, fast, good probability estimates)
      - Decision Tree         (non-linear, fully interpretable, visual)

    Both are trained on the SMOTE-resampled training set and
    evaluated on the untouched test set.

Outputs (saved to reports/):
    - baseline_confusion_matrices.png
    - baseline_classification_report.txt
    - baseline_decision_tree.png

Outputs (saved to models/):
    - baseline_logistic_regression.pkl
    - baseline_decision_tree.pkl

Run:
    python src/baseline_models.py
=============================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model   import LogisticRegression
from sklearn.tree           import DecisionTreeClassifier, plot_tree
from sklearn.metrics        import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Styling ───────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.05)
CLASS_NAMES = ["Allowed", "Blocked", "Needs Approval"]


# ── 1. Load preprocessed data ─────────────────────────────────────────────────
def load_data():
    """Load the artifacts produced by Step 2."""
    X_train = joblib.load(os.path.join(MODELS_DIR, "X_train_resampled.pkl"))
    y_train = joblib.load(os.path.join(MODELS_DIR, "y_train_resampled.pkl"))
    X_test  = joblib.load(os.path.join(MODELS_DIR, "X_test.pkl"))
    y_test  = joblib.load(os.path.join(MODELS_DIR, "y_test.pkl"))
    print(f"[BASE] Train set : {X_train.shape}  |  Test set : {X_test.shape}")
    return X_train, y_train, X_test, y_test


# ── 2. Train models ───────────────────────────────────────────────────────────
def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Logistic Regression — multiclass via one-vs-rest.
    max_iter raised to 1000 to ensure convergence on this dataset.
    C=1.0 is the default regularisation strength (balanced bias/variance).
    """
    print("[BASE] Training Logistic Regression ...")
    model = LogisticRegression(
        multi_class="ovr",
        max_iter=1000,
        C=1.0,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Decision Tree — max_depth=6 to balance expressiveness and overfitting.
    min_samples_leaf=10 prevents tiny leaf nodes that memorise noise.
    class_weight='balanced' gives more weight to the minority class during
    training (complements SMOTE without double-counting).
    """
    print("[BASE] Training Decision Tree ...")
    model = DecisionTreeClassifier(
        max_depth=6,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


# ── 3. Evaluate a single model ────────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Returns accuracy, macro F1, and per-class metrics.
    Macro F1 is the primary metric — it treats all classes equally
    regardless of their size, which matters given our class imbalance.
    """
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="macro")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)

    print(f"\n── {model_name} ──────────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Macro F1  : {f1:.4f}")
    print(f"\n{report}")

    return {"name": model_name, "model": model,
            "y_pred": y_pred, "accuracy": acc, "macro_f1": f1}


# ── 4. Plot: confusion matrices ───────────────────────────────────────────────
def plot_confusion_matrices(results: list, y_test: np.ndarray) -> None:
    """Side-by-side normalised confusion matrices for both baseline models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, result in zip(axes, results):
        cm = confusion_matrix(y_test, result["y_pred"], normalize="true")
        sns.heatmap(
            cm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            linewidths=0.5, ax=ax, cbar=True,
            annot_kws={"size": 11}
        )
        ax.set_title(
            f"{result['name']}\nAccuracy {result['accuracy']:.3f}  |  "
            f"Macro F1 {result['macro_f1']:.3f}",
            fontweight="bold", pad=10
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual",    fontsize=11)
        ax.tick_params(axis="x", rotation=15)
        ax.tick_params(axis="y", rotation=0)

    plt.suptitle("Baseline Models — Confusion Matrices (normalised)", 
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "baseline_confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[BASE] Saved → {path}")


# ── 5. Plot: decision tree visualisation ──────────────────────────────────────
def plot_decision_tree(model: DecisionTreeClassifier,
                       feature_names: list) -> None:
    """
    Visual representation of the top 4 levels of the decision tree.
    Useful for understanding which features the model splits on first
    and validating that the logic makes intuitive sense.
    """
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        max_depth=4,          # show top 4 levels only — deeper gets unreadable
        fontsize=8,
        ax=ax
    )
    ax.set_title("Decision Tree — Top 4 Levels", fontweight="bold", pad=12)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "baseline_decision_tree.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[BASE] Saved → {path}")


# ── 6. Plot: model comparison bar chart ───────────────────────────────────────
def plot_model_comparison(results: list) -> None:
    """Bar chart comparing accuracy and macro F1 across baseline models."""
    names   = [r["name"] for r in results]
    acc     = [r["accuracy"]  for r in results]
    f1      = [r["macro_f1"]  for r in results]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
#  Fixed code (removed borderpad)
    bars1 = ax.bar(x - width/2, acc, width, label="Accuracy", color="#4A90D9", edgecolor="none")
    bars2 = ax.bar(x + width/2, f1,  width, label="Macro F1",  color="#1D9E75", edgecolor="none")

    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=10
            )

    ax.set_ylim(0, 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Score")
    ax.set_title("Baseline Model Comparison", fontweight="bold", pad=12)
    ax.legend(frameon=False)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "baseline_model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[BASE] Saved → {path}")


# ── 7. Save classification report to text ────────────────────────────────────
def save_classification_report(results: list, y_test: np.ndarray) -> None:
    """Write a combined classification report for both models to a .txt file."""
    path = os.path.join(REPORTS_DIR, "baseline_classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  STEP 3 — Baseline Models Classification Report\n")
        f.write("=" * 60 + "\n\n")
        for result in results:
            f.write(f"── {result['name']} ──────────────────────────────\n")
            f.write(f"  Accuracy : {result['accuracy']:.4f}\n")
            f.write(f"  Macro F1 : {result['macro_f1']:.4f}\n\n")
            f.write(classification_report(
                y_test, result["y_pred"], target_names=CLASS_NAMES
            ))
            f.write("\n\n")
    print(f"[BASE] Saved → {path}")


# ── 8. Save models ─────────────────────────────────────────────────────────────
def save_models(results: list) -> None:
    name_map = {
        "Logistic Regression": "baseline_logistic_regression.pkl",
        "Decision Tree":       "baseline_decision_tree.pkl",
    }
    for result in results:
        filename = name_map.get(result["name"], f"{result['name']}.pkl")
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(result["model"], path)
        print(f"[BASE] Saved → {path}")


# ── Feature name helper ────────────────────────────────────────────────────────
def get_feature_names() -> list:
    """
    Reconstruct the ordered feature names after ColumnTransformer.
    Must match the column order defined in preprocessing.py.
    """
    from preprocessing import CATEGORICAL_COLS, NUMERIC_COLS, BINARY_COLS
    return CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 3 — Baseline Models")
    print("=" * 60)

    # Load data
    X_train, y_train, X_test, y_test = load_data()

    # Train
    lr_model = train_logistic_regression(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)

    # Evaluate
    results = [
        evaluate_model(lr_model, X_test, y_test, "Logistic Regression"),
        evaluate_model(dt_model, X_test, y_test, "Decision Tree"),
    ]

    # Visualise
    plot_confusion_matrices(results, y_test)
    plot_model_comparison(results)

    feature_names = get_feature_names()
    plot_decision_tree(dt_model, feature_names)

    # Save reports and models
    save_classification_report(results, y_test)
    save_models(results)

    # Summary
    print("\n── Summary ───────────────────────────────────────────")
    for r in results:
        print(f"  {r['name']:<25} Accuracy: {r['accuracy']:.4f}   Macro F1: {r['macro_f1']:.4f}")
    