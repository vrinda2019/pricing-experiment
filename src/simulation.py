"""
simulation.py
=============

Monte Carlo simulation engine for Notebook 6 (06_monte_carlo_simulation.ipynb)
-- the portfolio differentiator.

Instead of running ONE experiment, we run thousands of simulated experiments
under a known "true" data-generating process, so we can empirically answer
questions that a single p-value can't:

    - If the true lift is 5%, how often does our test actually detect it?
      (empirical power)
    - How often does the test falsely reject H0 when there is truly no
      effect? (empirical Type I error rate)
    - How often does it fail to detect a real effect? (Type II error rate)
    - Does our 95% CI actually contain the true effect ~95% of the time?
      (CI coverage)
    - What does the distribution of p-values / estimated lifts look like?

This complements the closed-form power analysis in statistics.py with an
empirical, simulation-based estimate -- useful because it makes no
asymptotic-normality assumptions and can be extended to arbitrary,
non-Gaussian revenue distributions (e.g. the zero-inflated revenue
distribution we actually use in this project).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SimulationConfig:
    n_simulations: int = 10_000
    n_per_group: int = 2_000
    baseline_mean: float = 120.0
    baseline_std: float = 140.0
    true_lift_pct: float = 0.05     # the TRUE underlying revenue lift, e.g. 5%
    alpha: float = 0.05
    seed: int = 2024
    # If True, draw from a zero-inflated distribution (more realistic for
    # subscription revenue, where many customers have $0 revenue in a period)
    zero_inflated: bool = True
    zero_prob: float = 0.30


def _draw_group(rng: np.random.Generator, n: int, mean: float, std: float, cfg: SimulationConfig) -> np.ndarray:
    """Draw one group's revenue values under the configured distribution."""
    if not cfg.zero_inflated:
        return rng.normal(mean, std, n)

    # Zero-inflated normal: a customer either has $0 revenue (didn't purchase)
    # or a positive revenue draw. We inflate the positive-branch mean so the
    # OVERALL mean (including zeros) still equals `mean`.
    purchased = rng.random(n) > cfg.zero_prob
    n_purchased = int(purchased.sum())
    positive_mean = mean / (1 - cfg.zero_prob)
    values = np.zeros(n)
    if n_purchased > 0:
        values[purchased] = np.clip(
            rng.normal(positive_mean, std, n_purchased), a_min=0, a_max=None
        )
    return values


def run_monte_carlo(config: SimulationConfig | None = None) -> pd.DataFrame:
    """
    Run `n_simulations` independent simulated A/B tests and return a
    dataframe with one row per simulation containing:
        control_mean, treatment_mean, estimated_lift_pct,
        p_value, ci_low, ci_high, ci_covers_truth, rejected_h0
    """
    config = config or SimulationConfig()
    rng = np.random.default_rng(config.seed)

    true_treatment_mean = config.baseline_mean * (1 + config.true_lift_pct)
    true_diff = true_treatment_mean - config.baseline_mean

    rows = []
    for _ in range(config.n_simulations):
        control = _draw_group(rng, config.n_per_group, config.baseline_mean, config.baseline_std, config)
        treatment = _draw_group(rng, config.n_per_group, true_treatment_mean, config.baseline_std, config)

        m1, m2 = control.mean(), treatment.mean()
        v1, v2 = control.var(ddof=1), treatment.var(ddof=1)
        n1, n2 = len(control), len(treatment)

        diff = m2 - m1
        se_diff = np.sqrt(v1 / n1 + v2 / n2)

        df_welch = (v1 / n1 + v2 / n2) ** 2 / (
            (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        )
        t_crit = stats.t.ppf(1 - config.alpha / 2, df_welch)
        ci_low = diff - t_crit * se_diff
        ci_high = diff + t_crit * se_diff

        t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=False)

        rows.append(
            {
                "control_mean": m1,
                "treatment_mean": m2,
                "diff": diff,
                "estimated_lift_pct": diff / m1 if m1 != 0 else np.nan,
                "se_diff": se_diff,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "ci_covers_truth": ci_low <= true_diff <= ci_high,
                "p_value": p_value,
                "rejected_h0": p_value < config.alpha,
            }
        )

    return pd.DataFrame(rows)


@dataclass
class SimulationSummary:
    n_simulations: int
    true_lift_pct: float
    empirical_power: float          # P(reject H0 | true effect exists), if true_lift_pct != 0
    mean_estimated_lift_pct: float
    ci_coverage: float              # should be ~ (1 - alpha) if everything is well calibrated
    mean_p_value: float


def summarize_simulation(results: pd.DataFrame, config: SimulationConfig) -> SimulationSummary:
    """Summarize a Monte Carlo run into the headline diagnostics."""
    return SimulationSummary(
        n_simulations=len(results),
        true_lift_pct=config.true_lift_pct,
        empirical_power=float(results["rejected_h0"].mean()),
        mean_estimated_lift_pct=float(results["estimated_lift_pct"].mean()),
        ci_coverage=float(results["ci_covers_truth"].mean()),
        mean_p_value=float(results["p_value"].mean()),
    )


def estimate_type1_error(config: SimulationConfig) -> float:
    """
    Run the simulation under H0 (true_lift_pct=0) and return the empirical
    false-positive rate -- should be close to `alpha` if the test is
    well-calibrated.
    """
    null_config = SimulationConfig(**{**config.__dict__, "true_lift_pct": 0.0})
    results = run_monte_carlo(null_config)
    return float(results["rejected_h0"].mean())


if __name__ == "__main__":
    cfg = SimulationConfig(n_simulations=500, n_per_group=1000, true_lift_pct=0.05)
    res = run_monte_carlo(cfg)
    summary = summarize_simulation(res, cfg)
    print(summary)

    type1_error = estimate_type1_error(SimulationConfig(n_simulations=500, n_per_group=1000))
    print("Empirical Type I error rate (should be close to alpha=0.05):", type1_error)
