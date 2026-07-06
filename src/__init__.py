"""
pricing-experiment
==================

An end-to-end A/B testing simulator for pricing decisions.

Modules
-------
data_processing : synthetic data generation + business metrics
experiment       : randomization, treatment assignment, price/elasticity model
statistics       : power analysis, hypothesis testing, confidence intervals
simulation        : Monte Carlo simulation of repeated experiments
"""

from . import data_processing
from . import experiment
from . import stats_tools
from . import simulation

__all__ = ["data_processing", "experiment", "stats_tools", "simulation"]
