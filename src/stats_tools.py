"""
statistics.py
=============

Statistical engine for:
    - Notebook 3 (03_power_analysis.ipynb): MDE, sample size, power curves
    - Notebook 4 (04_ab_testing.ipynb): Welch's t-test, CIs, decision rule

Design note
-----------
`statsmodels` gives convenient power-analysis helpers (TTestIndPower), but to
keep this module dependency-light and fully testable in restricted
environments, every function here has a pure NumPy/SciPy implementation of
the standard two-sample power/sample-size formulas. If statsmodels IS
installed, we use it for a cross-check / more precise noncentral-t solve;
otherwise we transparently fall back to the normal-approximation formulas,
which are the same formulas statsmodels itself is approximating for large n.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

try:
    from statsmodels.stats.power import TTestIndPower  # type: ignore

    _HAS_STATSMODELS = True
except ImportError:  # pragma: no cover
    _HAS_STATSMODELS = False


# --------------------------------------------------------------------------- #
# Power analysis / sample size planning
# --------------------------------------------------------------------------- #

def cohens_d(mean_diff: float, pooled_std: float) -> float:
    """Standardized effect size for a two-sample comparison."""
    if pooled_std == 0:
        raise ValueError("pooled_std must be > 0")
    return mean_diff / pooled_std


def required_sample_size(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.8,
    ratio: float = 1.0,
) -> int:
    """
    Sample size required PER GROUP to detect `effect_size` (Cohen's d) with
    the given significance level and power, for a two-sided two-sample test.

    Uses statsmodels' TTestIndPower solver if available (accounts for the
    t-distribution), else the standard normal-approximation formula:

        n = (z_alpha/2 + z_beta)^2 * (1 + 1/ratio) / effect_size^2
    """
    if effect_size <= 0:
        raise ValueError("effect_size must be > 0")

    if _HAS_STATSMODELS:
        analysis = TTestIndPower()
        n = analysis.solve_power(
            effect_size=effect_size, alpha=alpha, power=power, ratio=ratio
        )
        return int(np.ceil(n))

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    n = ((z_alpha + z_beta) ** 2) * (1 + 1 / ratio) / (effect_size ** 2)
    return int(np.ceil(n))


def power_given_n(
    effect_size: float, n_per_group: int, alpha: float = 0.05, ratio: float = 1.0
) -> float:
    """
    Statistical power achieved for a given per-group sample size and effect size.
    """
    if _HAS_STATSMODELS:
        analysis = TTestIndPower()
        return float(
            analysis.power(
                effect_size=effect_size, nobs1=n_per_group, alpha=alpha, ratio=ratio
            )
        )

    z_alpha = stats.norm.ppf(1 - alpha / 2)
    # noncentrality parameter under the normal approximation
    ncp = effect_size * np.sqrt(n_per_group / (1 + 1 / ratio))
    power = 1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    return float(power)


def mde_given_n(
    n_per_group: int, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0
) -> float:
    """
    Minimum detectable effect (as Cohen's d) achievable for a given
    per-group sample size, alpha, and power.
    """
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    d = (z_alpha + z_beta) * np.sqrt((1 + 1 / ratio) / n_per_group)
    return float(d)


@dataclass
class PowerAnalysisResult:
    baseline_mean: float
    baseline_std: float
    mde_relative: float          # e.g. 0.05 for "5% lift"
    mde_absolute: float          # in original units (e.g. dollars)
    effect_size: float           # Cohen's d
    alpha: float
    power: float
    required_n_per_group: int


def plan_experiment(
    baseline_mean: float,
    baseline_std: float,
    mde_relative: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> PowerAnalysisResult:
    """
    High-level planning helper matching Notebook 3's narrative:
    "we only care about improvements of at least X%" -> sample size needed.
    """
    mde_absolute = baseline_mean * mde_relative
    effect_size = cohens_d(mde_absolute, baseline_std)
    n = required_sample_size(effect_size, alpha=alpha, power=power)
    return PowerAnalysisResult(
        baseline_mean=baseline_mean,
        baseline_std=baseline_std,
        mde_relative=mde_relative,
        mde_absolute=mde_absolute,
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        required_n_per_group=n,
    )


# --------------------------------------------------------------------------- #
# Hypothesis testing / confidence intervals (Notebook 4)
# --------------------------------------------------------------------------- #

@dataclass
class TTestResult:
    control_mean: float
    treatment_mean: float
    diff: float
    diff_relative: float
    se_diff: float
    ci_low: float
    ci_high: float
    t_stat: float
    p_value: float
    alpha: float
    significant: bool
    recommendation: str


def welch_t_test(
    control: np.ndarray, treatment: np.ndarray, alpha: float = 0.05
) -> TTestResult:
    """
    Run Welch's t-test (unequal variances) comparing control vs treatment,
    returning means, difference, standard error, confidence interval,
    t-statistic, p-value, and a ship/don't-ship recommendation.
    """
    control = np.asarray(control, dtype=float)
    treatment = np.asarray(treatment, dtype=float)

    n1, n2 = len(control), len(treatment)
    m1, m2 = control.mean(), treatment.mean()
    v1, v2 = control.var(ddof=1), treatment.var(ddof=1)

    diff = m2 - m1
    diff_relative = diff / m1 if m1 != 0 else float("nan")

    se_diff = np.sqrt(v1 / n1 + v2 / n2)

    # Welch-Satterthwaite degrees of freedom
    df = (v1 / n1 + v2 / n2) ** 2 / (
        (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    )

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    significant = p_value < alpha
    if significant and diff > 0:
        recommendation = "Ship (statistically significant positive lift)"
    elif significant and diff < 0:
        recommendation = "Don't ship (statistically significant negative impact)"
    else:
        recommendation = "Don't ship / inconclusive (not statistically significant)"

    return TTestResult(
        control_mean=float(m1),
        treatment_mean=float(m2),
        diff=float(diff),
        diff_relative=float(diff_relative),
        se_diff=float(se_diff),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        t_stat=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(significant),
        recommendation=recommendation,
    )


if __name__ == "__main__":
    result = plan_experiment(baseline_mean=120, baseline_std=140, mde_relative=0.05)
    print(result)

    rng = np.random.default_rng(0)
    control = rng.normal(120, 140, 5000)
    treatment = rng.normal(126, 140, 5000)
    print(welch_t_test(control, treatment))
