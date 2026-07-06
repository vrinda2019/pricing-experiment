"""
streamlit_app.py
=================

Interactive dashboard for the pricing A/B testing simulator.

Run with:
    streamlit run dashboard/streamlit_app.py

A recruiter (or a PM) can move the sliders below and watch the whole
experimentation pipeline -- randomization, treatment, hypothesis test, and
ship/don't-ship recommendation -- update live, in under two minutes.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_processing import BusinessConfig, generate_synthetic_customers, compute_business_metrics
from experiment import TreatmentConfig, run_experiment, summarize_groups
from stats_tools import welch_t_test, plan_experiment
from simulation import SimulationConfig, run_monte_carlo, summarize_simulation

st.set_page_config(page_title="Pricing A/B Test Simulator", layout="wide")

st.title("Experimentation Framework for Pricing Decisions")
st.caption(
    "Should we increase the subscription price by 10%? "
    "Adjust the assumptions on the left and watch the full A/B testing pipeline update."
)

# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #

st.sidebar.header("Experiment assumptions")

price_increase_pct = st.sidebar.slider(
    "Price increase (%)", min_value=0, max_value=50, value=10, step=1
) / 100

elasticity = st.sidebar.slider(
    "Price elasticity of demand", min_value=0.0, max_value=2.0, value=0.5, step=0.05,
    help="A 10% price increase with elasticity=0.5 implies ~5% fewer purchases.",
)

sample_size = st.sidebar.slider(
    "Sample size per group", min_value=500, max_value=20_000, value=8_000, step=500
)

st.sidebar.header("Statistical settings")

confidence_level = st.sidebar.select_slider(
    "Confidence level", options=[0.90, 0.95, 0.99], value=0.95
)
alpha = 1 - confidence_level

mde_relative = st.sidebar.slider(
    "Minimum Detectable Effect (%)", min_value=1, max_value=20, value=5, step=1
) / 100

power_target = st.sidebar.slider(
    "Target power (%)", min_value=50, max_value=99, value=80, step=1
) / 100

seed = st.sidebar.number_input("Random seed", value=42, step=1)


# --------------------------------------------------------------------------- #
# Generate population + run experiment (cached so slider tweaks stay snappy)
# --------------------------------------------------------------------------- #

@st.cache_data(show_spinner=False)
def get_population(n_customers: int, seed: int) -> pd.DataFrame:
    return generate_synthetic_customers(BusinessConfig(n_customers=n_customers, seed=seed))


population = get_population(max(40_000, sample_size * 2 + 5000), seed)
metrics = compute_business_metrics(population)

config = TreatmentConfig(
    price_increase_pct=price_increase_pct,
    elasticity=elasticity,
    treatment_fraction=0.5,
    seed=seed,
)

# Subsample down to the requested sample size per group for a clean apples-to-apples run
subset = population.sample(n=min(len(population), sample_size * 2), random_state=seed).reset_index(drop=True)
experiment_df = run_experiment(subset, config)

control_rev = experiment_df.loc[experiment_df["group"] == "control", "revenue"].to_numpy()
treatment_rev = experiment_df.loc[experiment_df["group"] == "treatment", "revenue"].to_numpy()

result = welch_t_test(control_rev, treatment_rev, alpha=alpha)

plan = plan_experiment(
    baseline_mean=metrics["revenue_per_customer"]["mean"],
    baseline_std=metrics["revenue_per_customer"]["std"],
    mde_relative=mde_relative,
    alpha=alpha,
    power=power_target,
)

# --------------------------------------------------------------------------- #
# Top-line recommendation
# --------------------------------------------------------------------------- #

decision_color = {
    "Ship": "green",
    "Don't ship": "red",
}
decision_key = "Ship" if result.significant and result.diff > 0 else (
    "Don't ship" if result.significant and result.diff < 0 else "Borderline"
)
color = decision_color.get(decision_key, "orange")

st.markdown(
    f"### Recommendation: :{color}[{result.recommendation}]"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Expected revenue lift", f"{result.diff_relative:+.2%}", f"${result.diff:,.2f}/customer")
col2.metric("p-value", f"{result.p_value:.4f}")
col3.metric("95% CI", f"[${result.ci_low:,.1f}, ${result.ci_high:,.1f}]")
col4.metric("Conversion (control → treatment)",
            f"{experiment_df.loc[experiment_df.group=='control','converted'].mean():.1%} → "
            f"{experiment_df.loc[experiment_df.group=='treatment','converted'].mean():.1%}")

st.divider()

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #

left, right = st.columns(2)

with left:
    st.subheader("Revenue distribution: Control vs Treatment")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(control_rev, bins=50, alpha=0.6, label="Control", color="#4C72B0")
    ax.hist(treatment_rev, bins=50, alpha=0.6, label="Treatment", color="#C44E52")
    ax.axvline(control_rev.mean(), color="#4C72B0", linestyle="--")
    ax.axvline(treatment_rev.mean(), color="#C44E52", linestyle="--")
    ax.set_xlabel("Revenue per customer ($)")
    ax.legend()
    st.pyplot(fig)

with right:
    st.subheader("Guardrail metrics")
    guardrails = summarize_groups(experiment_df)
    st.dataframe(
        guardrails.set_index("group")[
            ["n", "revenue_per_customer_mean", "conversion_rate", "avg_orders", "avg_order_value"]
        ].style.format({
            "revenue_per_customer_mean": "${:,.2f}",
            "conversion_rate": "{:.1%}",
            "avg_orders": "{:.2f}",
            "avg_order_value": "${:,.2f}",
        }),
        use_container_width=True,
    )

st.divider()

# --------------------------------------------------------------------------- #
# Power analysis panel
# --------------------------------------------------------------------------- #

st.subheader("Power analysis for this configuration")
p1, p2, p3, p4 = st.columns(4)
p1.metric("MDE (relative)", f"{mde_relative:.0%}")
p2.metric("Required n / group", f"{plan.required_n_per_group:,}")
p3.metric("Your sample size / group", f"{sample_size:,}")
under_over = "sufficiently powered" if sample_size >= plan.required_n_per_group else "underpowered"
p4.metric("Status", under_over)

st.caption(
    "If your sample size is below the required n/group, a null result here is inconclusive rather "
    "than evidence of no effect -- see Notebook 3 for the full power analysis."
)

st.divider()

# --------------------------------------------------------------------------- #
# Optional: quick Monte Carlo check
# --------------------------------------------------------------------------- #

st.subheader("Quick Monte Carlo check (1,000 simulated experiments)")
run_mc = st.button("Run Monte Carlo simulation at these settings")

if run_mc:
    with st.spinner("Simulating 1,000 experiments..."):
        mc_config = SimulationConfig(
            n_simulations=1_000,
            n_per_group=sample_size,
            baseline_mean=metrics["revenue_per_customer"]["mean"],
            baseline_std=metrics["revenue_per_customer"]["std"],
            true_lift_pct=result.diff_relative if not np.isnan(result.diff_relative) else 0.0,
            alpha=alpha,
            seed=seed,
        )
        mc_results = run_monte_carlo(mc_config)
        mc_summary = summarize_simulation(mc_results, mc_config)

    m1, m2, m3 = st.columns(3)
    m1.metric("Empirical power", f"{mc_summary.empirical_power:.1%}")
    m2.metric("Mean estimated lift", f"{mc_summary.mean_estimated_lift_pct:.2%}")
    m3.metric("95% CI coverage", f"{mc_summary.ci_coverage:.1%}")

    fig2, ax2 = plt.subplots(figsize=(8, 3.5))
    ax2.hist(mc_results["p_value"], bins=40, color="#55A868", edgecolor="white")
    ax2.axvline(alpha, color="red", linestyle="--", label=f"alpha = {alpha:.2f}")
    ax2.set_xlabel("p-value")
    ax2.set_title("Distribution of p-values across 1,000 simulated experiments at your assumptions")
    ax2.legend()
    st.pyplot(fig2)

st.divider()
st.caption(
    "Built with the pricing-experiment framework — see `notebooks/` for the full "
    "business understanding → power analysis → A/B test → sensitivity → Monte Carlo walkthrough."
)
