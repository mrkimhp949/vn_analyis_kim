"""
Analytics Module

Provides advanced analytics and simulation tools:
- Monte Carlo simulation
- Risk analysis
- Performance metrics
"""

from .monte_carlo import MonteCarloSimulator, run_monte_carlo_analysis

__all__ = ["MonteCarloSimulator", "run_monte_carlo_analysis"]
