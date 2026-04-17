import numpy as np
import pandas as pd
from typing import Tuple, Dict
from scipy import stats


# POWER & MDE


def calculate_mde(std: float,
                  n: int,
                  alpha: float = 0.05,
                  power: float = 0.8) -> float:
    """
    Calculates Minimum Detectable Effect (absolute).
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    mde = (z_alpha + z_beta) * std * np.sqrt(2 / n)
    return mde


def required_sample_size(std: float,
                         mde: float,
                         alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """
    Calculates required sample size per group.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)

    n = 2 * ((z_alpha + z_beta) * std / mde) ** 2
    return int(np.ceil(n))


# VARIANCE REDUCTION (CUPED HELPERS)


def calculate_theta(control_df: pd.DataFrame,
                     metric_col: str,
                     covariate_col: str) -> float:
    """
    Calculates CUPED theta using control group only.
    """
    cov_matrix = np.cov(
        control_df[covariate_col],
        control_df[metric_col]
    )

    return cov_matrix[0, 1] / cov_matrix[0, 0]


def calculate_variance_reduction(original: pd.Series,
                                 adjusted: pd.Series) -> float:
    """
    Computes % variance reduction after CUPED.
    """
    var_orig = original.var()
    var_adj = adjusted.var()

    return (1 - var_adj / var_orig) * 100


# NORMALITY & ROBUSTNESS


def check_normality(series: pd.Series,
                     sample_size: int = 5000) -> Dict[str, float]:
    """
    Shapiro-Wilk test on a sample (for large datasets).
    """
    if len(series) > sample_size:
        series = series.sample(sample_size, random_state=42)

    stat, p_value = stats.shapiro(series)

    return {
        "stat": stat,
        "p_value": p_value
    }


def check_variance_homogeneity(a: pd.Series,
                               b: pd.Series) -> Dict[str, float]:
    """
    Levene test for equal variances.
    """
    stat, p_value = stats.levene(a, b)

    return {
        "stat": stat,
        "p_value": p_value
    }


# ROBUST METRICS (HEAVY TAILS)


def trimmed_mean(series: pd.Series,
                  proportion_to_cut: float = 0.01) -> float:
    """
    Robust mean removing extreme values.
    """
    return stats.trim_mean(series, proportion_to_cut)


def log_transform(series: pd.Series) -> pd.Series:
    """
    Applies log transform to reduce skewness.
    """
    return np.log1p(series)


# DELTA METHOD


def delta_method_ratio(mean_num: float,
                       mean_den: float,
                       var_num: float,
                       var_den: float,
                       cov: float) -> float:
    """
    Variance approximation for ratio metrics.
    """
    return (
        (var_num / mean_den**2)
        + (mean_num**2 * var_den / mean_den**4)
        - (2 * mean_num * cov / mean_den**3)
    )


# =========================
# EFFECT SIZE
# =========================

def cohens_d(a: pd.Series,
              b: pd.Series) -> float:
    """
    Calculates Cohen's d effect size.
    """
    mean_diff = b.mean() - a.mean()

    pooled_std = np.sqrt(
        (a.var() + b.var()) / 2
    )

    return mean_diff / pooled_std if pooled_std != 0 else 0.0

# QUICK EXPERIMENT SUMMARY

def experiment_summary(df: pd.DataFrame,
                        metric: str = "revenue") -> Dict[str, float]:
    """
    Quick summary for A/B experiment.
    """
    A = df[df["group"] == "A"][metric]
    B = df[df["group"] == "B"][metric]

    uplift = B.mean() - A.mean()

    t_stat, p_value = stats.ttest_ind(B, A, equal_var=False)

    return {
        "mean_A": A.mean(),
        "mean_B": B.mean(),
        "uplift": uplift,
        "p_value": p_value
    }