"""
Hybrid Intelligence Portfolio System -- Monte Carlo Simulator
================================================================
10,000-scenario forward simulation for portfolio risk analysis.

Features:
  1. Geometric Brownian Motion with correlated assets
  2. Cholesky decomposition for correlation structure
  3. 252-day (1-year) forward projection
  4. Percentile return distribution
  5. Max drawdown analysis across all paths
  6. Probability of loss and severe loss
  7. VaR and CVaR from simulation distribution
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class MonteCarloSimulator:
    """
    Monte Carlo portfolio simulation engine.

    Simulates N paths of portfolio returns using correlated
    Geometric Brownian Motion, then computes risk statistics.
    """

    DEFAULT_N_SIMULATIONS = 10_000
    DEFAULT_HORIZON = 252  # 1 year of trading days

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        tickers: list[str],
    ):
        self._mu = expected_returns       # Annualized expected returns
        self._cov = cov_matrix            # Annualized covariance matrix
        self._tickers = tickers
        self._n_assets = len(tickers)

    def simulate(
        self,
        weights: np.ndarray,
        n_simulations: int = DEFAULT_N_SIMULATIONS,
        horizon_days: int = DEFAULT_HORIZON,
        max_acceptable_drawdown: float = 0.15,
        seed: int = 42,
    ) -> dict:
        """
        Run Monte Carlo simulation.

        Args:
            weights: Portfolio weights (must sum to 1)
            n_simulations: Number of simulation paths
            horizon_days: Trading days to simulate
            max_acceptable_drawdown: Drawdown threshold for probability computation
            seed: Random seed for reproducibility

        Returns:
            MonteCarloResult-compatible dict
        """
        logger.info(
            f"Running Monte Carlo: {n_simulations} simulations x {horizon_days} days"
        )

        np.random.seed(seed)

        # ── Step 1: Cholesky decomposition ────────────────
        try:
            L = np.linalg.cholesky(self._cov)
        except np.linalg.LinAlgError:
            # Fallback: eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self._cov)
            eigvals = np.maximum(eigvals, 1e-8)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        # ── Step 2: Convert to daily parameters ───────────
        daily_mu = self._mu / horizon_days
        daily_cov_factor = L / np.sqrt(horizon_days)

        # ── Step 3: Simulate paths ────────────────────────
        # Each path: T steps of daily returns for all assets
        # Portfolio value evolves as: V_t = V_{t-1} * (1 + r_portfolio_t)

        terminal_returns = np.zeros(n_simulations)
        max_drawdowns = np.zeros(n_simulations)

        for i in range(n_simulations):
            # Generate correlated random returns for all days
            Z = np.random.randn(horizon_days, self._n_assets)
            daily_returns = Z @ daily_cov_factor.T + daily_mu

            # Portfolio daily returns (weighted sum)
            port_daily = daily_returns @ weights

            # Compute portfolio value path
            cumulative = np.cumprod(1 + port_daily)
            terminal_returns[i] = cumulative[-1] - 1.0  # Total return

            # Max drawdown along this path
            running_max = np.maximum.accumulate(cumulative)
            drawdown = (running_max - cumulative) / running_max
            max_drawdowns[i] = np.max(drawdown)

        # ── Step 4: Compute statistics ────────────────────
        percentiles = {
            "p1": round(float(np.percentile(terminal_returns, 1)), 4),
            "p5": round(float(np.percentile(terminal_returns, 5)), 4),
            "p10": round(float(np.percentile(terminal_returns, 10)), 4),
            "p25": round(float(np.percentile(terminal_returns, 25)), 4),
            "p50": round(float(np.percentile(terminal_returns, 50)), 4),
            "p75": round(float(np.percentile(terminal_returns, 75)), 4),
            "p90": round(float(np.percentile(terminal_returns, 90)), 4),
            "p95": round(float(np.percentile(terminal_returns, 95)), 4),
            "p99": round(float(np.percentile(terminal_returns, 99)), 4),
        }

        prob_loss = float(np.mean(terminal_returns < 0))
        prob_severe = float(np.mean(max_drawdowns > max_acceptable_drawdown))
        median_dd = float(np.median(max_drawdowns))

        # VaR and CVaR from simulation
        var_95 = float(-np.percentile(terminal_returns, 5))
        tail = terminal_returns[terminal_returns <= np.percentile(terminal_returns, 5)]
        cvar_95 = float(-np.mean(tail)) if len(tail) > 0 else var_95

        result = {
            "num_simulations": n_simulations,
            "time_horizon_days": horizon_days,
            "percentile_returns": percentiles,
            "probability_of_loss": round(prob_loss, 4),
            "probability_of_severe_loss": round(prob_severe, 4),
            "median_max_drawdown": round(median_dd, 4),
            "worst_case_return": round(float(np.percentile(terminal_returns, 1)), 4),
            "best_case_return": round(float(np.percentile(terminal_returns, 99)), 4),
            "simulation_var_95": round(var_95, 4),
            "simulation_cvar_95": round(cvar_95, 4),
            "mean_return": round(float(np.mean(terminal_returns)), 4),
            "std_return": round(float(np.std(terminal_returns)), 4),
        }

        logger.info(
            f"MC results: median={percentiles['p50']:.1%}, "
            f"P(loss)={prob_loss:.1%}, "
            f"P(severe DD)={prob_severe:.1%}, "
            f"median DD={median_dd:.1%}"
        )

        return result
