"""
data_processing.py
===================

Generates a realistic synthetic subscription-business dataset and computes
the "current state of the business" metrics used in Notebook 1
(01_business_understanding.ipynb).

The synthetic data models a subscription product where each customer has:
    - a base monthly subscription price
    - a purchase/renewal frequency
    - some churn / drop-off behaviour
    - a revenue history

This gives us a believable base population to randomize into control /
treatment groups in later notebooks.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Synthetic data generation
# --------------------------------------------------------------------------- #

@dataclass
class BusinessConfig:
    """Configuration for the synthetic business used across the project."""

    n_customers: int = 20_000
    base_price: float = 120.0          # current monthly revenue/user baseline
    price_std: float = 25.0            # spread in what customers actually pay (plans/discounts)
    monthly_orders_lambda: float = 1.15  # avg orders/purchases per customer per month
    seed: int = 42


def generate_synthetic_customers(config: BusinessConfig | None = None) -> pd.DataFrame:
    """
    Generate a synthetic customer-level dataset representing one month of
    activity for a subscription business.

    Returns
    -------
    pd.DataFrame with columns:
        customer_id, plan_tier, tenure_months, orders, order_value,
        revenue, converted
    """
    config = config or BusinessConfig()
    rng = np.random.default_rng(config.seed)
    n = config.n_customers

    customer_id = np.arange(1, n + 1)

    # Plan tier drives price heterogeneity (Basic / Standard / Premium)
    plan_tier = rng.choice(
        ["Basic", "Standard", "Premium"], size=n, p=[0.5, 0.35, 0.15]
    )
    tier_multiplier = pd.Series(plan_tier).map(
        {"Basic": 0.75, "Standard": 1.0, "Premium": 1.6}
    ).to_numpy()

    tenure_months = rng.integers(1, 48, size=n)

    # Orders per customer this month (0 = did not purchase / churned this cycle)
    orders = rng.poisson(lam=config.monthly_orders_lambda, size=n)

    # Average order value scales with plan tier, with individual noise
    order_value = np.clip(
        rng.normal(
            loc=config.base_price * tier_multiplier,
            scale=config.price_std * tier_multiplier,
        ),
        a_min=10,
        a_max=None,
    )

    revenue = orders * order_value
    converted = (orders > 0).astype(int)

    df = pd.DataFrame(
        {
            "customer_id": customer_id,
            "plan_tier": plan_tier,
            "tenure_months": tenure_months,
            "orders": orders,
            "order_value": order_value.round(2),
            "revenue": revenue.round(2),
            "converted": converted,
        }
    )
    return df


# --------------------------------------------------------------------------- #
# Business metrics ("what does our business look like today?")
# --------------------------------------------------------------------------- #

def compute_business_metrics(df: pd.DataFrame) -> dict:
    """
    Compute the headline business metrics referenced in Notebook 1:
        - number of customers
        - total revenue
        - average order value (AOV)
        - purchase frequency
        - simple CLV estimate
        - revenue distribution stats
        - high-value customer concentration (revenue share of top decile)
    """
    n_customers = df["customer_id"].nunique()
    total_revenue = df["revenue"].sum()
    conversion_rate = df["converted"].mean()

    purchasers = df[df["orders"] > 0]
    aov = purchasers["order_value"].mean() if len(purchasers) else 0.0
    purchase_frequency = df["orders"].mean()

    # Very simple CLV estimate: AOV * purchase frequency * average tenure (months)
    avg_tenure = df["tenure_months"].mean()
    simple_clv = aov * purchase_frequency * avg_tenure

    revenue_per_customer = df["revenue"]
    revenue_stats = {
        "mean": float(revenue_per_customer.mean()),
        "median": float(revenue_per_customer.median()),
        "std": float(revenue_per_customer.std()),
        "p10": float(revenue_per_customer.quantile(0.10)),
        "p90": float(revenue_per_customer.quantile(0.90)),
    }

    # High-value customer concentration: revenue share held by the top 10% of customers
    sorted_rev = revenue_per_customer.sort_values(ascending=False)
    top_decile_n = max(1, int(len(sorted_rev) * 0.10))
    top_decile_share = sorted_rev.iloc[:top_decile_n].sum() / sorted_rev.sum()

    return {
        "n_customers": int(n_customers),
        "total_revenue": float(total_revenue),
        "conversion_rate": float(conversion_rate),
        "average_order_value": float(aov),
        "purchase_frequency": float(purchase_frequency),
        "simple_clv": float(simple_clv),
        "revenue_per_customer": revenue_stats,
        "top_decile_revenue_share": float(top_decile_share),
    }


def save_processed(df: pd.DataFrame, path: str) -> None:
    """Save a processed dataframe to CSV, creating parent dirs if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def load_processed(path: str) -> pd.DataFrame:
    """Load a previously processed dataframe."""
    return pd.read_csv(path)


if __name__ == "__main__":
    cfg = BusinessConfig()
    customers = generate_synthetic_customers(cfg)
    #save_processed(customers,"/Users/vrindatibrewal/Desktop/pricing-experiment/data/processed/customers.csv" )
    metrics = compute_business_metrics(customers)
    print("Generated", len(customers), "synthetic customers")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
