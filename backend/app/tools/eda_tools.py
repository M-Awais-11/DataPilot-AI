"""
EDA Tools — deterministic chart generation and insight extraction.
No LLM. Returns chart file paths and statistical insight strings.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

# High-resolution & clean styling for crystal-clear PDF reports
CHART_DPI = 250
CHART_FMT = "png"

# Setup high-contrast, clean theme optimal for both screen and printed PDF
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
})


def _save(fig, path: str) -> str:
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=CHART_DPI, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return path


# ── Target Distribution ───────────────────────────────────────────────────────
def plot_target_distribution(df: pd.DataFrame, target_col: str, output_dir: str) -> tuple[str, str]:
    """High-clarity bar chart with value counts for classification, or histogram with mean/median for regression."""
    path = str(Path(output_dir) / f"target_distribution.{CHART_FMT}")
    fig, ax = plt.subplots(figsize=(10, 5.8))

    if df[target_col].nunique() <= 20:
        vc = df[target_col].value_counts()
        colors = sns.color_palette("tab10", len(vc))
        bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="#1E293B", linewidth=1.2, alpha=0.9)
        ax.set_title(f"Target Distribution: {target_col}", fontsize=16, pad=15)
        ax.set_xlabel(target_col, labelpad=10)
        ax.set_ylabel("Count (Instances)", labelpad=10)
        
        # Explicit data labels above every bar
        max_val = max(vc.values) if len(vc) else 1
        ax.set_ylim(0, max_val * 1.18)
        pct = (vc / len(df) * 100).round(1)
        for bar, count, p in zip(bars, vc.values, pct.values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max_val * 0.02),
                f"{count:,}\n({p}%)",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#0F172A",
            )
        insight = f"Target '{target_col}' has {len(vc)} classes. Most frequent: '{vc.index[0]}' ({pct.iloc[0]}%). Least frequent: '{vc.index[-1]}' ({pct.iloc[-1]}%)."
    else:
        counts, bins, patches = ax.hist(df[target_col].dropna(), bins=30, color="#2563EB", edgecolor="white", linewidth=1.2, alpha=0.85)
        ax.set_title(f"Target Distribution: {target_col}", fontsize=16, pad=15)
        ax.set_xlabel(target_col, labelpad=10)
        ax.set_ylabel("Frequency", labelpad=10)
        
        mean_val = float(df[target_col].mean())
        median_val = float(df[target_col].median())
        std_val = float(df[target_col].std())
        
        ax.axvline(mean_val, color="#DC2626", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.2f}")
        ax.axvline(median_val, color="#059669", linestyle="-.", linewidth=2, label=f"Median: {median_val:.2f}")
        ax.legend(frameon=True, facecolor="white", edgecolor="#CBD5E1", fontsize=10)
        
        insight = f"Target '{target_col}' is numerical. Mean: {mean_val:.4g}, Std: {std_val:.4g}. Distribution appears {'skewed' if abs(df[target_col].skew()) > 1 else 'approximately normal'}."

    _save(fig, path)
    return path, insight


# ── Missing Values Heatmap ────────────────────────────────────────────────────
def plot_missing_values(df: pd.DataFrame, output_dir: str) -> tuple[str, str]:
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if len(missing) == 0:
        return "", "No missing values detected in the dataset."

    path = str(Path(output_dir) / f"missing_values.{CHART_FMT}")
    fig, ax = plt.subplots(figsize=(10, max(5, len(missing) * 0.48)))
    pct = (missing / len(df) * 100)
    bars = ax.barh(missing.index[::-1], pct.values[::-1], color="#DC2626", edgecolor="#991B1B", linewidth=1, alpha=0.88)
    ax.set_xlabel("Missing Percentage (%)", labelpad=10)
    ax.set_title("Missing Values by Column", fontsize=16, pad=15)
    ax.set_xlim(0, max(pct.max() * 1.18, 10))
    for bar, val, count in zip(bars, pct.values[::-1], missing.values[::-1]):
        ax.text(
            bar.get_width() + 0.6,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}% ({count:,} rows)",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1E293B",
        )
    _save(fig, path)
    worst = pct.idxmax()
    insight = f"{len(missing)} columns have missing values. Worst: '{worst}' ({pct.max():.1f}% missing)."
    return path, insight


# ── Correlation Heatmap ───────────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str, max_cols: int = 20) -> tuple[str, str]:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()[:max_cols]
    if len(num_cols) < 2:
        return "", "Insufficient numerical columns for correlation heatmap."

    path = str(Path(output_dir) / f"correlation_heatmap.{CHART_FMT}")
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    fig_w = max(9.5, len(num_cols) * 0.75)
    fig_h = max(7.5, len(num_cols) * 0.65)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    
    sns.heatmap(
        corr,
        mask=mask,
        annot=len(num_cols) <= 15,
        fmt=".2f",
        annot_kws={"size": 9, "weight": "bold"},
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=16, pad=15)
    plt.xticks(rotation=45, ha="right", fontsize=9, fontweight="semibold")
    plt.yticks(rotation=0, fontsize=9, fontweight="semibold")
    _save(fig, path)

    # Find highest correlation pair
    corr_abs = corr.abs()
    corr_arr = corr_abs.values.copy()          # writable copy (pandas >= 2 returns read-only)
    np.fill_diagonal(corr_arr, 0)
    corr_abs = pd.DataFrame(corr_arr, index=corr_abs.index, columns=corr_abs.columns)
    max_pair = corr_abs.stack().idxmax()
    max_val = corr.loc[max_pair]
    insight = f"Highest correlation: '{max_pair[0]}' and '{max_pair[1]}' ({max_val:.3f}). {'Potential multicollinearity concern.' if abs(max_val) > 0.9 else ''}"
    return path, insight


# ── Numerical Distributions ───────────────────────────────────────────────────
def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: list[str],
    output_dir: str,
    max_cols: int = 12,
) -> list[tuple[str, str]]:
    results = []
    cols = [c for c in numerical_cols if c in df.columns][:max_cols]
    if not cols:
        return results

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 4.0))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        data = df[col].dropna()
        ax.hist(data, bins=25, color="#2563EB", edgecolor="white", linewidth=1.2, alpha=0.85)
        
        # Add Mean and Median vertical lines for clarity
        mean_v = float(data.mean()) if len(data) else 0.0
        med_v = float(data.median()) if len(data) else 0.0
        ax.axvline(mean_v, color="#DC2626", linestyle="--", linewidth=1.8, label=f"Mean: {mean_v:.2g}")
        ax.axvline(med_v, color="#059669", linestyle="-.", linewidth=1.8, label=f"Median: {med_v:.2g}")
        
        ax.set_title(col, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Value", fontsize=10, labelpad=4)
        ax.set_ylabel("Count", fontsize=10, labelpad=4)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    fig.suptitle("Numerical Feature Distributions", fontsize=16, fontweight="bold", y=1.02)
    path = str(Path(output_dir) / f"numerical_distributions.{CHART_FMT}")
    _save(fig, path)

    skewed = [(c, df[c].skew()) for c in cols if abs(df[c].skew()) > 1.5]
    insight = f"Distributions plotted for {len(cols)} numerical columns."
    if skewed:
        insight += f" Highly skewed columns: {[f'{c} (skew={s:.2f})' for c, s in skewed[:3]]}. Log transformation may help."
    results.append((path, insight))
    return results


# ── Box Plots ─────────────────────────────────────────────────────────────────
def plot_boxplots(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target_col: Optional[str],
    output_dir: str,
    max_cols: int = 10,
) -> list[tuple[str, str]]:
    results = []
    cols = [c for c in numerical_cols if c in df.columns and c != target_col][:max_cols]
    if not cols:
        return results

    n = len(cols)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.2, nrows * 4.0))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        ax = axes[i]
        box = ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#93C5FD", color="#1E40AF", linewidth=1.5),
            medianprops=dict(color="#DC2626", linewidth=2),
            whiskerprops=dict(color="#1E40AF", linewidth=1.2),
            capprops=dict(color="#1E40AF", linewidth=1.2),
            flierprops=dict(marker="o", color="#DC2626", alpha=0.6, markersize=4),
        )
        ax.set_title(col, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel("Range", fontsize=10, labelpad=4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.subplots_adjust(hspace=0.45, wspace=0.35)
    fig.suptitle("Box Plots — Outlier & Spread Detection", fontsize=16, fontweight="bold", y=1.02)
    path = str(Path(output_dir) / f"boxplots.{CHART_FMT}")
    _save(fig, path)

    high_iqr = []
    for col in cols:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).mean() * 100
        if outliers > 5:
            high_iqr.append(f"{col} ({outliers:.1f}%)")

    insight = "Box plots show IQR spread for numerical features."
    if high_iqr:
        insight += f" High outlier rates: {high_iqr[:3]}."
    results.append((path, insight))
    return results


# ── Categorical Bar Charts ────────────────────────────────────────────────────
def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_cols: list[str],
    output_dir: str,
    max_cols: int = 8,
    max_bars: int = 15,
) -> list[tuple[str, str]]:
    results = []
    cols = [c for c in categorical_cols if c in df.columns and df[c].nunique() <= 50][:max_cols]
    if not cols:
        return results

    for col in cols:
        vc = df[col].value_counts().head(max_bars)
        fig_w = max(8.0, len(vc) * 0.75)
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        colors = sns.color_palette("tab10", len(vc))
        bars = ax.bar(vc.index.astype(str), vc.values, color=colors, edgecolor="#1E293B", linewidth=1.2, alpha=0.9)
        ax.set_title(f"Categorical Distribution: {col}", fontsize=15, fontweight="bold", pad=12)
        ax.set_xlabel(col, fontsize=11, labelpad=8)
        ax.set_ylabel("Count", fontsize=11, labelpad=8)
        plt.xticks(rotation=35, ha="right", fontsize=10, fontweight="semibold")
        
        # Direct data labels on bars
        max_v = max(vc.values) if len(vc) else 1
        ax.set_ylim(0, max_v * 1.18)
        for bar, val in zip(bars, vc.values):
            p = (val / len(df) * 100)
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + (max_v * 0.02),
                f"{val:,}\n({p:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
                color="#0F172A",
            )
            
        path = str(Path(output_dir) / f"cat_{col.replace('/', '_')}.{CHART_FMT}")
        _save(fig, path)
        top_val = vc.index[0]
        top_pct = vc.iloc[0] / len(df) * 100
        insight = f"Column '{col}' has {df[col].nunique()} unique values. Most common: '{top_val}' ({top_pct:.1f}%)."
        results.append((path, insight))

    return results


# ── Feature vs Target ─────────────────────────────────────────────────────────
def plot_feature_vs_target(
    df: pd.DataFrame,
    numerical_cols: list[str],
    target_col: str,
    problem_type: str,
    output_dir: str,
    max_cols: int = 6,
    top_corr_n: int = 6,
) -> list[tuple[str, str]]:
    """Plot top N most correlated features against target."""
    results = []
    if not target_col or target_col not in df.columns:
        return results

    if problem_type == "regression":
        # Scatter plots for top correlated numerical features
        num_features = [c for c in numerical_cols if c in df.columns and c != target_col]
        if not num_features:
            return results
        corrs = df[num_features + [target_col]].corr()[target_col].drop(target_col).abs().sort_values(ascending=False)
        top_features = corrs.head(top_corr_n).index.tolist()

        n = len(top_features)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.8, nrows * 4.0))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(top_features):
            ax = axes[i]
            ax.scatter(df[col], df[target_col], alpha=0.45, s=20, color="#2563EB", edgecolors="none")
            
            # Add linear trend line
            valid_mask = df[col].notnull() & df[target_col].notnull()
            if valid_mask.sum() > 2:
                m, b = np.polyfit(df.loc[valid_mask, col], df.loc[valid_mask, target_col], 1)
                ax.plot(df[col], m * df[col] + b, color="#DC2626", linewidth=1.8, label="Trend")
            
            ax.set_xlabel(col, fontsize=10, labelpad=4)
            ax.set_ylabel(target_col, fontsize=10, labelpad=4)
            corr_val = df[col].corr(df[target_col])
            ax.set_title(f"{col}\n(Correlation r = {corr_val:.2f})", fontsize=11, fontweight="bold")

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.subplots_adjust(hspace=0.45, wspace=0.35)
        fig.suptitle(f"Top Correlated Features vs Target ({target_col})", fontsize=16, fontweight="bold", y=1.02)
        path = str(Path(output_dir) / f"feature_vs_target.{CHART_FMT}")
        _save(fig, path)
        insight = f"Top {len(top_features)} correlated features with '{target_col}': {dict(zip(top_features, [round(float(corrs[f]), 3) for f in top_features]))}."
        results.append((path, insight))

    else:
        # Box plots split by target class for top features
        num_features = [c for c in numerical_cols if c in df.columns and c != target_col][:max_cols]
        if not num_features or df[target_col].nunique() > 20:
            return results

        n = len(num_features)
        ncols = min(3, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.0, nrows * 4.2))
        axes = np.array(axes).flatten() if n > 1 else [axes]

        for i, col in enumerate(num_features):
            ax = axes[i]
            sns.boxplot(
                data=df,
                x=target_col,
                y=col,
                hue=target_col,
                legend=False,
                ax=ax,
                palette="tab10",
                boxprops=dict(alpha=0.85, linewidth=1.2),
                medianprops=dict(color="#DC2626", linewidth=2),
            )
            ax.set_title(col, fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel(target_col, fontsize=10, labelpad=4)
            ax.set_ylabel(col, fontsize=10, labelpad=4)
            plt.sca(ax)
            plt.xticks(rotation=30, ha="right", fontsize=9)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.subplots_adjust(hspace=0.45, wspace=0.35)
        fig.suptitle(f"Feature Distribution Across Classes of {target_col}", fontsize=16, fontweight="bold", y=1.02)
        path = str(Path(output_dir) / f"feature_vs_target_class.{CHART_FMT}")
        _save(fig, path)
        insight = f"Box plots show how numerical features vary across classes of '{target_col}'."
        results.append((path, insight))

    return results
