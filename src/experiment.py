"""
experiment.py
=============

Experiment design mechanics for Notebook 2 (02_experiment_design.ipynb) and
Notebook 4 (04_ab_testing.ipynb):

    - randomize customers into Control / Treatment
    - apply a price increase to the Treatment group
    - model the demand response (price elasticity) so that a price increase
      trades off against a drop in purchase probability
    - roll everything up into the metric we actually test: revenue/customer

Hypotheses
----------
H0: price increase has no effect on revenue per customer
H1: price increase changes (we hope: increases) revenue per customer
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class TreatmentConfig:
    """
    Configuration for the pricing treatment.

    Parameters
    ----------
    price_increase_pct : float
        Fractional price increase applied to the treatment group, e.g. 0.10 for +10%.
    elasticity : float
        Price elasticity of demand. We use the convention:
            pct_change_in_purchase_prob = -elasticity * price_increase_pct
        e.g. elasticity=0.5 and price_increase_pct=0.10 -> 5% fewer purchases.
        (This mirrors the "10% price increase -> 5% fewer purchases" example.)
    treatment_fraction : float
        Fraction of customers assigned to treatment (default 50/50 split).
    seed : int
        RNG seed for reproducible randomization.
    """

    price_increase_pct: float = 0.10
    elasticity: float = 0.5
    treatment_fraction: float = 0.5
    seed: int = 123


def randomize_users(df: pd.DataFrame, config: TreatmentConfig | None = None) -> pd.DataFrame:
    """
    Randomly assign each customer to 'control' or 'treatment'.

    Adds a `group` column to a COPY of the input dataframe and returns it.
    """
    config = config or TreatmentConfig()
    rng = np.random.default_rng(config.seed)

    out = df.copy()
    is_treatment = rng.random(len(out)) < config.treatment_fraction
    out["group"] = np.where(is_treatment, "treatment", "control")
    return out


def apply_price_treatment(df: pd.DataFrame, config: TreatmentConfig | None = None) -> pd.DataFrame:
    """
    Apply the pricing treatment to a randomized dataframe (must already have
    a `group` column from `randomize_users`).

    For the treatment group:
        - order_value is scaled up by (1 + price_increase_pct)
        - each order has an independent probability of being "dropped"
          (i.e. the customer decides not to purchase) equal to
          elasticity * price_increase_pct, simulating reduced demand at
          the higher price.

    Returns a new dataframe with adjusted `order_value`, `orders`,
    `revenue`, and `converted` columns reflecting the post-treatment world.
    """
    config = config or TreatmentConfig()
    if "group" not in df.columns:
        raise ValueError("df must have a `group` column; call randomize_users first.")

    rng = np.random.default_rng(config.seed + 1)
    out = df.copy()

    is_treat = out["group"] == "treatment"
    n_treat = int(is_treat.sum())

    # 1) Price increase for treatment group
    out.loc[is_treat, "order_value"] = out.loc[is_treat, "order_value"] * (
        1 + config.price_increase_pct
    )

    # 2) Demand response: each order in the treatment group has probability
    #    `drop_prob` of not happening due to the higher price.
    drop_prob = config.elasticity * config.price_increase_pct
    drop_prob = float(np.clip(drop_prob, 0.0, 1.0))

    original_orders = out.loc[is_treat, "orders"].to_numpy()
    # For each existing order, simulate whether it survives the price increase
    surviving_orders = np.array(
        [rng.binomial(n=int(o), p=1 - drop_prob) if o > 0 else 0 for o in original_orders]
    )
    out.loc[is_treat, "orders"] = surviving_orders

    # 3) Recompute revenue and conversion after treatment effects
    out["revenue"] = (out["orders"] * out["order_value"]).round(2)
    out["converted"] = (out["orders"] > 0).astype(int)

    return out


def run_experiment(df: pd.DataFrame, config: TreatmentConfig | None = None) -> pd.DataFrame:
    """Convenience wrapper: randomize then apply treatment in one call."""
    config = config or TreatmentConfig()
    randomized = randomize_users(df, config)
    treated = apply_price_treatment(randomized, config)
    return treated


def summarize_groups(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize revenue/customer, conversion, and order count by group.
    Expects a `group` and `revenue` column (post-experiment dataframe).
    """
    summary = (
        df.groupby("group")
        .agg(
            n=("customer_id", "count"),
            revenue_per_customer_mean=("revenue", "mean"),
            revenue_per_customer_std=("revenue", "std"),
            conversion_rate=("converted", "mean"),
            avg_orders=("orders", "mean"),
            avg_order_value=("order_value", "mean"),
        )
        .reset_index()
    )
    return summary


if __name__ == "__main__":
    from data_processing import BusinessConfig, generate_synthetic_customers

    customers = generate_synthetic_customers(BusinessConfig())
    tconfig = TreatmentConfig(price_increase_pct=0.10, elasticity=0.5)
    experiment_df = run_experiment(customers, tconfig)
    print(summarize_groups(experiment_df))
