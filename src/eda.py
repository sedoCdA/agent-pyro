"""
=============================================================
Step 1 — Exploratory Data Analysis (EDA)
=============================================================
Purpose:
    Understand the dataset before any modelling. This script
    examines class distributions, feature statistics, correlations,
    and the separability of target classes across key features.

Outputs (saved to reports/):
    - eda_class_distribution.png
    - eda_risk_score_by_class.png
    - eda_feature_correlation.png
    - eda_permission_vs_class.png
    - eda_numeric_distributions.png

Run:
    python src/eda.py
=============================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_PATH    = os.path.join("data", "agent_security_risk_scores.csv")
REPORTS_DIR  = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# ── Plot style ────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
CLASS_COLORS = {
    "Allowed":              "#1D9E75",
    "Blocked":              "#E24B4A",
    "Needs_Human_Approval": "#EF9F27",
}

# ── 1. Load data ──────────────────────────────────────────────────────────────
def load_data(path: str) -> pd.DataFrame:
    """Load the CSV dataset and perform a basic sanity check."""
    df = pd.read_csv(path)
    print(f"[EDA] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[EDA] Missing values : {df.isnull().sum().sum()}")
    return df


# ── 2. Summary statistics ─────────────────────────────────────────────────────
def print_summary(df: pd.DataFrame) -> None:
    """Print class distribution and basic descriptive statistics."""
    print("\n── Target distribution ──────────────────────────")
    counts = df["access_decision"].value_counts()
    for label, count in counts.items():
        pct = count / len(df) * 100
        print(f"  {label:<25} {count:>5}  ({pct:.1f}%)")

    print("\n── Numeric feature summary ──────────────────────")
    print(df.describe(include=[np.number]).T.to_string())

    print("\n── Categorical feature cardinality ──────────────")
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "access_decision"]
    for col in cat_cols:
        print(f"  {col:<30} {df[col].nunique()} unique values")


# ── 3. Plot: class distribution ───────────────────────────────────────────────
def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart of the three target classes with count and percentage labels."""
    counts = df["access_decision"].value_counts()
    colors = [CLASS_COLORS[c] for c in counts.index]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="none", width=0.5)

    # Annotate each bar with count + percentage
    total = len(df)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 15,
            f"{count:,}\n({count/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10, color="#333333"
        )

    ax.set_title("Class Distribution — access_decision", fontweight="bold", pad=12)
    ax.set_ylabel("Count")
    ax.set_ylim(0, counts.max() * 1.2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("")
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eda_class_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] Saved → {path}")


# ── 4. Plot: risk score distribution by class ──────────────────────────────────
def plot_risk_score_by_class(df: pd.DataFrame) -> None:
    """Box + strip plot showing action_risk_score separation across classes."""
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ["Allowed", "Needs_Human_Approval", "Blocked"]
    palette = {k: v for k, v in CLASS_COLORS.items()}

    sns.boxplot(
        data=df, x="access_decision", y="action_risk_score",
        order=order, palette=palette,
        width=0.45, linewidth=1.2, fliersize=0, ax=ax
    )
    sns.stripplot(
        data=df.sample(min(400, len(df)), random_state=42),
        x="access_decision", y="action_risk_score",
        order=order, palette=palette,
        size=2.5, alpha=0.35, jitter=True, ax=ax
    )

    ax.set_title("Action Risk Score by Class", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("action_risk_score")
    ax.set_xticklabels(["Allowed", "Needs Approval", "Blocked"])
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eda_risk_score_by_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] Saved → {path}")


# ── 5. Plot: feature correlation with target ───────────────────────────────────
def plot_feature_correlation(df: pd.DataFrame) -> None:
    """
    Horizontal bar chart of Pearson correlation between each numeric
    feature and a numeric encoding of the target (Allowed=0, Needs=1, Blocked=2).
    """
    df_enc = df.copy()
    df_enc["target_num"] = df_enc["access_decision"].map(
        {"Allowed": 0, "Needs_Human_Approval": 1, "Blocked": 2}
    )
    corr = (
        df_enc.corr(numeric_only=True)["target_num"]
        .drop("target_num")
        .sort_values()
    )

    colors = ["#1D9E75" if v < 0 else "#E24B4A" for v in corr.values]
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(corr.index, corr.values, color=colors, edgecolor="none", height=0.6)

    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax.set_title("Feature Correlation with Target\n(negative = safer, positive = riskier)",
                 fontweight="bold", pad=12)
    ax.set_xlabel("Pearson r")
    for bar, val in zip(bars, corr.values):
        ax.text(
            val + (0.01 if val >= 0 else -0.01),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9, color="#333333"
        )
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eda_feature_correlation.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] Saved → {path}")


# ── 6. Plot: permission_match vs class ────────────────────────────────────────
def plot_permission_vs_class(df: pd.DataFrame) -> None:
    """
    Stacked bar showing how permission_match (0/1) distributes
    across the three target classes.
    """
    ct = (
        df.groupby(["access_decision", "permission_match"])
        .size()
        .unstack(fill_value=0)
        .reindex(["Allowed", "Needs_Human_Approval", "Blocked"])
    )
    ct.index = ["Allowed", "Needs Approval", "Blocked"]

    fig, ax = plt.subplots(figsize=(7, 4))
    ct.plot(
        kind="bar", stacked=True,
        color={"0": "#E24B4A", "1": "#1D9E75"} if "0" in ct.columns
              else {0: "#E24B4A", 1: "#1D9E75"},
        edgecolor="none", ax=ax, width=0.5
    )
    ax.set_title("Permission Match vs Class", fontweight="bold", pad=12)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    legend = ax.legend(title="permission_match", labels=["match = 0", "match = 1"],
                       frameon=False, fontsize=10)
    sns.despine()
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eda_permission_vs_class.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[EDA] Saved → {path}")


# ── 7. Plot: numeric feature distributions ────────────────────────────────────
def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    KDE plots for each continuous numeric feature, coloured by class.
    Reveals how well each feature separates the three outcomes.
    """
    numeric_cols = [
        "action_risk_score", "data_exfiltration_risk",
        "resource_sensitivity", "agent_autonomy_level",
        "previous_failed_attempts"
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        ax = axes[i]
        for label, color in CLASS_COLORS.items():
            subset = df[df["access_decision"] == label][col]
            subset.plot.kde(ax=ax, label=label.replace("_", " "), color=color, linewidth=1.8)
        ax.set_title(col, fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8, frameon=False)
        sns.despine(ax=ax)

    # hide unused subplot
    axes[-1].set_visible(False)
    fig.suptitle("Numeric Feature Distributions by Class", fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(REPORTS_DIR, "eda_numeric_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[EDA] Saved → {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  STEP 1 — Exploratory Data Analysis")
    print("=" * 60)

    df = load_data(DATA_PATH)
    print_summary(df)

    print("\n[EDA] Generating plots ...")
    plot_class_distribution(df)
    plot_risk_score_by_class(df)
    plot_feature_correlation(df)
    plot_permission_vs_class(df)
    plot_numeric_distributions(df)

    print("\n[EDA] All outputs saved to reports/")
    print("[EDA] Step 1 complete. Ready to commit.\n")