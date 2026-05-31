"""
=============================================================
Step 2 — Preprocessing Pipeline  (v2 — leakage-free)
=============================================================
Purpose:
    Transform the raw dataset into a clean, model-ready form.
    This script handles:
      - Encoding categorical features (OrdinalEncoder for tree models)
      - Scaling numeric features (StandardScaler)
      - Encoding the target label (LabelEncoder)
      - Stratified train/test split (80/20)
      - Handling class imbalance with SMOTE (on training set only)
      - Saving the processed splits and fitted encoders for reuse

Feature set (11 clean features — leakage-free):
    Categorical (5) : agent_role, user_role, requested_action,
                      tool_requested, resource_type
    Numeric     (3) : agent_autonomy_level, resource_sensitivity,
                      previous_failed_attempts
    Binary      (3) : permission_match, prompt_injection_detected,
                      audit_log_available

Removed vs v1 (feature leakage):
    - human_approval_required   ← restatement of the target label
    - action_risk_score         ← pre-computed aggregate risk score
    - data_exfiltration_risk    ← pre-computed risk score

Outputs (saved to models/):
    - preprocessor.pkl
    - label_encoder.pkl
    - X_train.pkl, y_train.pkl
    - X_test.pkl,  y_test.pkl
    - X_train_resampled.pkl, y_train_resampled.pkl  ← after SMOTE

Run:
    python src/preprocessing.py
=============================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose            import ColumnTransformer
from sklearn.pipeline           import Pipeline
from imblearn.over_sampling     import SMOTE

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join("data", "agent_security_risk_scores.csv")
MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Column definitions (v2 — leakage-free) ───────────────────────────────────
CATEGORICAL_COLS = [
    "agent_role",
    "user_role",
    "requested_action",
    "tool_requested",
    "resource_type",
]

# Removed from v1: action_risk_score, data_exfiltration_risk
# Both were pre-computed aggregate scores derived from the same
# labelling logic that produced the target — circular by design.
NUMERIC_COLS = [
    "agent_autonomy_level",
    "resource_sensitivity",
    "previous_failed_attempts",
]

# Removed from v1: human_approval_required
# That column is a direct restatement of the target label
# (Needs_Human_Approval ↔ human_approval_required=1). Keeping it
# would let the model trivially memorise the answer.
BINARY_COLS = [
    "permission_match",
    "prompt_injection_detected",
    "audit_log_available",
]

TARGET_COL = "access_decision"


# ── 1. Load data ───────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[PREP] Loaded dataset : {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 2. Split features and target ──────────────────────────────────────────────
def split_features_target(df: pd.DataFrame):
    """
    Select only the 11 clean features.
    Any extra columns in the CSV (including the 3 leaky ones) are
    silently ignored via this explicit allowlist.
    """
    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS + BINARY_COLS
    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()
    print(f"[PREP] Features used  : {X.shape[1]}  {feature_cols}")
    print(f"[PREP] Target         : {y.value_counts().to_dict()}")
    return X, y


# ── 3. Build the preprocessing transformer ────────────────────────────────────
def build_preprocessor() -> ColumnTransformer:
    """
    ColumnTransformer that applies:
      - OrdinalEncoder  → categorical columns
        (suitable for tree-based models; avoids sparse high-dim one-hot)
      - StandardScaler  → continuous numeric columns
      - passthrough      → binary 0/1 columns (already on a 0-1 scale)
    """
    categorical_transformer = Pipeline(steps=[
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1      # unseen categories at inference get -1
        ))
    ])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, CATEGORICAL_COLS),
            ("num", numeric_transformer,     NUMERIC_COLS),
            ("bin", "passthrough",           BINARY_COLS),
        ],
        remainder="drop"          # safely drops any unlisted columns
    )
    return preprocessor


# ── 4. Encode target labels ────────────────────────────────────────────────────
def encode_target(y_train: pd.Series, y_test: pd.Series):
    """
    Encode string class labels to integers.
    Alphabetical order:  Allowed=0, Blocked=1, Needs_Human_Approval=2
    """
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc  = le.transform(y_test)
    print(f"[PREP] Label encoding : {dict(zip(le.classes_, le.transform(le.classes_)))}")
    return y_train_enc, y_test_enc, le


# ── 5. Train / test split ──────────────────────────────────────────────────────
def split_data(X, y, test_size: float = 0.20, random_state: int = 42):
    """
    Stratified 80/20 split — preserves class proportions in both sets.
    random_state is fixed for full reproducibility.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"[PREP] Train size      : {len(X_train):,}  |  Test size : {len(X_test):,}")
    return X_train, X_test, y_train, y_test


# ── 6. Apply SMOTE to the training set ───────────────────────────────────────
def apply_smote(X_train_transformed: np.ndarray, y_train_enc: np.ndarray,
                random_state: int = 42):
    """
    SMOTE (Synthetic Minority Over-sampling Technique):
      - Generates synthetic examples for the minority classes
      - Applied ONLY to the training set — never the test set
      - Prevents the model from ignoring Needs_Human_Approval (~11%)
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train_enc)

    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"[PREP] After SMOTE    : {dict(zip(unique, counts))}")
    return X_resampled, y_resampled


# ── 7. Plot: before vs after SMOTE ───────────────────────────────────────────
def plot_smote_comparison(y_train_enc, y_resampled, label_encoder: LabelEncoder) -> None:
    """Side-by-side bar chart showing class balance before and after SMOTE."""
    def counts_dict(y):
        unique, counts = np.unique(y, return_counts=True)
        return {label_encoder.classes_[k]: v for k, v in zip(unique, counts)}

    before = counts_dict(y_train_enc)
    after  = counts_dict(y_resampled)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)
    colors = ["#1D9E75", "#E24B4A", "#EF9F27"]

    for ax, data, title in zip(axes, [before, after], ["Before SMOTE", "After SMOTE"]):
        bars = ax.bar(data.keys(), data.values(),
                      color=colors[:len(data)], edgecolor="none", width=0.5)
        for bar, val in zip(bars, data.values()):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 5, f"{val:,}",
                    ha="center", va="bottom", fontsize=10)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Count")
        ax.set_xticklabels(
            [k.replace("Needs_Human_Approval", "Needs\nApproval") for k in data.keys()],
            fontsize=9
        )
        sns.despine(ax=ax)

    fig.suptitle("Class Balance — Training Set (v2 clean features)",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "prep_smote_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[PREP] Saved → {path}")


# ── 8. Save artifacts ─────────────────────────────────────────────────────────
def save_artifacts(preprocessor, label_encoder,
                   X_train, X_test, y_train_enc, y_test_enc,
                   X_resampled, y_resampled) -> None:
    """Persist all fitted transformers and data splits for use in later steps."""
    artifacts = {
        "preprocessor.pkl":          preprocessor,
        "label_encoder.pkl":         label_encoder,
        "X_train.pkl":               X_train,
        "X_test.pkl":                X_test,
        "y_train.pkl":               y_train_enc,
        "y_test.pkl":                y_test_enc,
        "X_train_resampled.pkl":     X_resampled,
        "y_train_resampled.pkl":     y_resampled,
    }
    for filename, obj in artifacts.items():
        path = os.path.join(MODELS_DIR, filename)
        joblib.dump(obj, path)
        print(f"[PREP] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 2 — Preprocessing Pipeline  (v2 — leakage-free)")
    print("=" * 60)

    # Load
    df = load_data(DATA_PATH)

    # Separate features and target — leaky columns are excluded here
    X, y = split_features_target(df)

    # Stratified train/test split (before any transformation)
    X_train, X_test, y_train_raw, y_test_raw = split_data(X, y)

    # Build and fit the preprocessor on training data only
    preprocessor = build_preprocessor()
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed  = preprocessor.transform(X_test)
    print(f"[PREP] Transformed    → train: {X_train_transformed.shape}  "
          f"test: {X_test_transformed.shape}")

    # Encode target labels
    y_train_enc, y_test_enc, label_encoder = encode_target(y_train_raw, y_test_raw)

    # Apply SMOTE to balance training classes
    X_resampled, y_resampled = apply_smote(X_train_transformed, y_train_enc)

    # Visualise class balance change
    plot_smote_comparison(y_train_enc, y_resampled, label_encoder)

    # Save everything
    save_artifacts(
        preprocessor, label_encoder,
        X_train_transformed, X_test_transformed,
        y_train_enc, y_test_enc,
        X_resampled, y_resampled
    )

    print("\n[PREP] All artifacts saved to models/")
    print("[PREP] Step 2 complete — 11 clean features, no leakage.")
    print("[PREP] Next → Step 3: Baseline Models\n")