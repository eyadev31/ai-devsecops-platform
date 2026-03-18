"""
Hybrid Intelligence Portfolio System -- Portfolio Optimizer
==============================================================
Core quantitative optimization engine with 3 strategies:

  1. Mean-Variance (Markowitz): Maximize Sharpe ratio or minimize
     variance for a target return, with risk aversion parameter.

  2. Risk Parity: Equal risk contribution -- each asset contributes
     equal marginal risk to total portfolio volatility.

  3. CVaR-Constrained: Optimize expected return subject to a
     Conditional Value at Risk constraint (tail risk control).

All optimizations use scipy.optimize.minimize with SLSQP solver.
"""

import logging
import numpy as np
from scipy.optimize import minimize
from typing import Optional

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Multi-strategy portfolio optimizer.

    Takes expected returns, covariance matrix, and constraints,
    then produces optimal weights using multiple methodologies.
    """

    def __init__(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        tickers: list[str],
        risk_free_rate: float = 0.04,
    ):
        self._mu = expected_returns
        self._cov = cov_matrix
        self._tickers = tickers
        self._n = len(tickers)
        self._rf = risk_free_rate

        # Validate inputs
        assert len(expected_returns) == self._n, "Returns length mismatch"
        assert cov_matrix.shape == (self._n, self._n), "Covariance shape mismatch"

    # ════════════════════════════════════════════════════
    #  MEAN-VARIANCE OPTIMIZATION
    # ════════════════════════════════════════════════════

    def mean_variance(
        self,
        risk_aversion: float = 1.0,
        bounds: Optional[list[tuple]] = None,
    ) -> dict:
        """
        Mean-Variance optimization.

        Objective: max  w'μ - (λ/2) * w'Σw
        Subject to: Σw = 1, bounds on w

        Args:
            risk_aversion: Lambda parameter (higher = more conservative)
            bounds: Per-asset (min, max) weight bounds

        Returns:
            Dict with weights, expected return, volatility, Sharpe
        """
        if bounds is None:
            bounds = [(0.0, 1.0)] * self._n

        # Initial equal-weight guess
        w0 = np.ones(self._n) / self._n

        # Objective: minimize negative utility (maximize utility)
        def neg_utility(w):
            ret = w @ self._mu
            risk = w @ self._cov @ w
            return -(ret - (risk_aversion / 2) * risk)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        result = minimize(
            neg_utility,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x
        weights = np.maximum(weights, 0)  # Clip tiny negatives
        weights /= weights.sum()          # Re-normalize

        port_return = weights @ self._mu
        port_vol = np.sqrt(weights @ self._cov @ weights)
        sharpe = (port_return - self._rf) / port_vol if port_vol > 0 else 0

        return {
            "method": "mean_variance",
            "weights": {t: round(float(w), 4) for t, w in zip(self._tickers, weights)},
            "weights_array": weights,
            "expected_return": round(float(port_return), 4),
            "expected_volatility": round(float(port_vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "risk_aversion": risk_aversion,
            "converged": result.success,
        }

    # ════════════════════════════════════════════════════
    #  RISK PARITY OPTIMIZATION
    # ════════════════════════════════════════════════════

    def risk_parity(
        self,
        bounds: Optional[list[tuple]] = None,
    ) -> dict:
        """
        Risk Parity optimization.

        Objective: minimize Σ_i (RC_i - 1/N)^2
        where RC_i = w_i * (Σw)_i / (w'Σw) is the risk contribution.

        Each asset contributes equal risk to the portfolio.
        """
        if bounds is None:
            bounds = [(0.01, 1.0)] * self._n  # Minimum 1% to avoid division by zero

        w0 = np.ones(self._n) / self._n
        target_rc = 1.0 / self._n

        def risk_parity_obj(w):
            port_vol = np.sqrt(w @ self._cov @ w)
            if port_vol < 1e-10:
                return 1e10

            marginal_risk = self._cov @ w
            risk_contrib = w * marginal_risk / (port_vol ** 2)

            # Sum of squared deviations from equal risk contribution
            return np.sum((risk_contrib - target_rc) ** 2)

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        result = minimize(
            risk_parity_obj,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

        port_return = weights @ self._mu
        port_vol = np.sqrt(weights @ self._cov @ weights)
        sharpe = (port_return - self._rf) / port_vol if port_vol > 0 else 0

        # Compute actual risk contributions
        marginal = self._cov @ weights
        risk_contribs = weights * marginal / (port_vol ** 2) if port_vol > 0 else np.ones(self._n) / self._n

        return {
            "method": "risk_parity",
            "weights": {t: round(float(w), 4) for t, w in zip(self._tickers, weights)},
            "weights_array": weights,
            "expected_return": round(float(port_return), 4),
            "expected_volatility": round(float(port_vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "risk_contributions": {
                t: round(float(rc), 4) for t, rc in zip(self._tickers, risk_contribs)
            },
            "converged": result.success,
        }

    # ════════════════════════════════════════════════════
    #  CVaR-CONSTRAINED OPTIMIZATION
    # ════════════════════════════════════════════════════

    def cvar_constrained(
        self,
        max_cvar: float = 0.15,
        bounds: Optional[list[tuple]] = None,
        n_scenarios: int = 5000,
    ) -> dict:
        """
        CVaR-constrained optimization via scenario approximation.

        Maximize expected return subject to:
          - CVaR_95 <= max_cvar
          - Weights sum to 1

        Uses Monte Carlo scenarios to approximate CVaR.
        """
        if bounds is None:
            bounds = [(0.0, 1.0)] * self._n

        # Generate scenarios for CVaR estimation
        np.random.seed(42)
        try:
            L = np.linalg.cholesky(self._cov)
        except np.linalg.LinAlgError:
            # If not PD, use eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(self._cov)
            eigvals = np.maximum(eigvals, 1e-8)
            L = eigvecs @ np.diag(np.sqrt(eigvals))

        Z = np.random.randn(n_scenarios, self._n)
        scenarios = Z @ L.T + self._mu  # Annualized scenario returns

        w0 = np.ones(self._n) / self._n
        alpha = 0.05  # 95% CVaR

        def neg_return_with_cvar_penalty(w):
            port_returns = scenarios @ w
            var_threshold = np.percentile(port_returns, alpha * 100)
            tail_losses = port_returns[port_returns <= var_threshold]
            cvar = -np.mean(tail_losses) if len(tail_losses) > 0 else 0

            expected_ret = w @ self._mu

            # Penalty if CVaR exceeds limit
            cvar_penalty = max(0, cvar - max_cvar) * 100

            return -expected_ret + cvar_penalty

        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        ]

        result = minimize(
            neg_return_with_cvar_penalty,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-12},
        )

        weights = result.x
        weights = np.maximum(weights, 0)
        weights /= weights.sum()

        # Compute final metrics
        port_return = weights @ self._mu
        port_vol = np.sqrt(weights @ self._cov @ weights)
        sharpe = (port_return - self._rf) / port_vol if port_vol > 0 else 0

        # Final CVaR computation
        port_scenario_returns = scenarios @ weights
        var_95 = np.percentile(port_scenario_returns, 5)
        tail = port_scenario_returns[port_scenario_returns <= var_95]
        cvar_95 = -np.mean(tail) if len(tail) > 0 else 0

        return {
            "method": "cvar_constrained",
            "weights": {t: round(float(w), 4) for t, w in zip(self._tickers, weights)},
            "weights_array": weights,
            "expected_return": round(float(port_return), 4),
            "expected_volatility": round(float(port_vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "var_95": round(float(-var_95), 4),
            "cvar_95": round(float(cvar_95), 4),
            "max_cvar_target": max_cvar,
            "converged": result.success,
        }

    # ════════════════════════════════════════════════════
    #  STRATEGY SELECTOR
    # ════════════════════════════════════════════════════

    def optimize_for_profile(
        self,
        risk_score: float = 0.5,
        max_drawdown: float = 0.15,
        bounds: Optional[list[tuple]] = None,
    ) -> dict:
        """
        Run all 3 optimizations and select the best for the investor.

        Selection logic:
          - Conservative: CVaR-constrained (tail risk control)
          - Moderate: Blended MV + Risk Parity
          - Growth: Mean-Variance with moderate risk aversion
          - Aggressive: Mean-Variance with low risk aversion
        """
        # Map risk score to risk aversion parameter
        # Low risk score = high aversion, high risk score = low aversion
        risk_aversion = max(0.1, 3.0 - risk_score * 4.0)

        # Map max drawdown to CVaR constraint
        # Approximate: CVaR_95 ≈ max_drawdown * 0.6 (rough mapping)
        cvar_target = max_drawdown * 0.8

        logger.info(
            f"Optimizing: risk_score={risk_score:.2f}, "
            f"risk_aversion={risk_aversion:.2f}, "
            f"cvar_target={cvar_target:.2%}"
        )

        # Run all three strategies
        mv_result = self.mean_variance(risk_aversion=risk_aversion, bounds=bounds)
        rp_result = self.risk_parity(bounds=bounds)
        cvar_result = self.cvar_constrained(max_cvar=cvar_target, bounds=bounds)

        # Select strategy based on risk profile
        if risk_score < 0.30:
            # Conservative: prioritize tail risk control
            selected = cvar_result
            strategy = "defensive"
        elif risk_score < 0.45:
            # Moderate-conservative: blend CVaR and Risk Parity
            selected = self._blend_allocations(cvar_result, rp_result, alpha=0.6)
            strategy = "defensive_growth"
        elif risk_score < 0.60:
            # Moderate: blend MV and Risk Parity
            selected = self._blend_allocations(mv_result, rp_result, alpha=0.5)
            strategy = "balanced"
        elif risk_score < 0.75:
            # Growth: MV with some RP influence
            selected = self._blend_allocations(mv_result, rp_result, alpha=0.7)
            strategy = "growth"
        elif risk_score < 0.85:
            selected = mv_result
            strategy = "aggressive_growth"
        else:
            selected = mv_result
            strategy = "max_growth"

        selected["strategy_type"] = strategy
        selected["alternatives"] = {
            "mean_variance": {k: v for k, v in mv_result.items() if k != "weights_array"},
            "risk_parity": {k: v for k, v in rp_result.items() if k != "weights_array"},
            "cvar_constrained": {k: v for k, v in cvar_result.items() if k != "weights_array"},
        }

        logger.info(
            f"Selected strategy: {strategy} | "
            f"E[R]={selected['expected_return']:.1%} | "
            f"Vol={selected['expected_volatility']:.1%} | "
            f"Sharpe={selected['sharpe_ratio']:.2f}"
        )

        return selected

    # ────────────────────────────────────────────────────
    #  BLENDING
    # ────────────────────────────────────────────────────

    def _blend_allocations(
        self, alloc_a: dict, alloc_b: dict, alpha: float = 0.5
    ) -> dict:
        """
        Blend two allocations: result = alpha * A + (1-alpha) * B.
        """
        w_a = alloc_a["weights_array"]
        w_b = alloc_b["weights_array"]
        blended = alpha * w_a + (1 - alpha) * w_b
        blended = np.maximum(blended, 0)
        blended /= blended.sum()

        port_return = blended @ self._mu
        port_vol = np.sqrt(blended @ self._cov @ blended)
        sharpe = (port_return - self._rf) / port_vol if port_vol > 0 else 0

        return {
            "method": "blended",
            "weights": {t: round(float(w), 4) for t, w in zip(self._tickers, blended)},
            "weights_array": blended,
            "expected_return": round(float(port_return), 4),
            "expected_volatility": round(float(port_vol), 4),
            "sharpe_ratio": round(float(sharpe), 4),
            "blend_sources": [alloc_a["method"], alloc_b["method"]],
            "blend_ratio": alpha,
            "converged": True,
        }

    # ────────────────────────────────────────────────────
    #  RISK DECOMPOSITION
    # ────────────────────────────────────────────────────

    def compute_risk_contributions(self, weights: np.ndarray) -> dict:
        """Compute per-asset risk contribution for given weights."""
        port_vol = np.sqrt(weights @ self._cov @ weights)
        if port_vol < 1e-10:
            return {t: 1.0 / self._n for t in self._tickers}

        marginal = self._cov @ weights
        risk_contribs = weights * marginal / (port_vol ** 2)

        return {t: round(float(rc), 4) for t, rc in zip(self._tickers, risk_contribs)}

    def compute_diversification_ratio(self, weights: np.ndarray) -> float:
        """
        Diversification ratio = weighted avg individual vol / portfolio vol.
        Ratio > 1 means diversification is reducing risk.
        """
        individual_vols = np.sqrt(np.diag(self._cov))
        weighted_avg_vol = weights @ individual_vols
        port_vol = np.sqrt(weights @ self._cov @ weights)
        return round(float(weighted_avg_vol / port_vol), 4) if port_vol > 0 else 1.0
