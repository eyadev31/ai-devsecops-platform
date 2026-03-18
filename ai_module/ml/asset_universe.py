"""
Hybrid Intelligence Portfolio System -- Asset Universe Manager
=================================================================
Manages the investable asset universe for Agent 3.

Provides:
  1. Asset definitions (ticker, class, characteristics)
  2. Expected return estimation (historical + macro adjustment)
  3. Covariance matrix computation
  4. Synthetic data generation for mock mode
"""

import logging
import math
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)

# ================================================================
#  ASSET UNIVERSE DEFINITION
# ================================================================

ASSET_UNIVERSE = {
    "SPY": {
        "name": "S&P 500 ETF",
        "asset_class": "equity",
        "long_term_return": 0.10,     # ~10% historical annualized
        "long_term_vol": 0.16,        # ~16% annualized
        "description": "US large-cap equity broad market",
    },
    "BND": {
        "name": "Total Bond Market ETF",
        "asset_class": "bond",
        "long_term_return": 0.04,
        "long_term_vol": 0.05,
        "description": "US investment-grade bond aggregate",
    },
    "GLD": {
        "name": "Gold ETF",
        "asset_class": "commodity",
        "long_term_return": 0.06,
        "long_term_vol": 0.15,
        "description": "Gold bullion proxy -- inflation hedge & safe haven",
    },
    "BTC": {
        "name": "Bitcoin",
        "asset_class": "crypto",
        "long_term_return": 0.30,
        "long_term_vol": 0.65,
        "description": "Cryptocurrency -- high growth, extreme volatility",
    },
    "CASH": {
        "name": "Cash / Money Market",
        "asset_class": "cash",
        "long_term_return": 0.04,
        "long_term_vol": 0.001,
        "description": "Risk-free asset (T-bills / money market)",
    },
}

# Long-term correlation structure (stylized facts)
BASE_CORRELATION = np.array([
    #          SPY    BND    GLD    BTC   CASH
    [  1.00, -0.10,  0.05,  0.40,  0.00],   # SPY
    [ -0.10,  1.00,  0.15, -0.05,  0.10],   # BND
    [  0.05,  0.15,  1.00,  0.20,  0.00],   # GLD
    [  0.40, -0.05,  0.20,  1.00,  0.00],   # BTC
    [  0.00,  0.10,  0.00,  0.00,  1.00],   # CASH
])

TICKERS = list(ASSET_UNIVERSE.keys())


class AssetUniverseManager:
    """
    Manages asset data for portfolio optimization.

    Provides expected returns, covariance matrices, and constraints
    adjusted for the current market regime and investor profile.
    """

    def __init__(self):
        self._tickers = TICKERS
        self._n_assets = len(TICKERS)

    @property
    def tickers(self) -> list[str]:
        return self._tickers.copy()

    @property
    def n_assets(self) -> int:
        return self._n_assets

    # ────────────────────────────────────────────────────
    #  EXPECTED RETURNS
    # ────────────────────────────────────────────────────

    def get_expected_returns(
        self,
        agent1_output: Optional[dict] = None,
        risk_free_rate: float = 0.04,
    ) -> np.ndarray:
        """
        Compute expected annualized returns for each asset.

        Uses long-term historical returns adjusted by:
          1. Market regime (bull/bear shift)
          2. Macroeconomic environment
          3. Risk-free rate for cash
        """
        base_returns = np.array([
            ASSET_UNIVERSE[t]["long_term_return"] for t in self._tickers
        ])

        if agent1_output is None:
            return base_returns

        # ── Regime adjustment ─────────────────────────────
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "").lower()

        if "bear" in regime:
            # Reduce equity and crypto expectations, boost bonds and gold
            adjustments = np.array([
                -0.04,  # SPY: lower in bear
                +0.01,  # BND: flight to safety
                +0.02,  # GLD: safe haven demand
                -0.10,  # BTC: high beta, hit hardest
                 0.00,  # CASH
            ])
        elif "bull" in regime and "high_vol" in regime:
            adjustments = np.array([
                +0.02,
                -0.01,
                +0.01,
                +0.05,
                 0.00,
            ])
        elif "bull" in regime:
            adjustments = np.array([
                +0.03,
                -0.01,
                -0.01,
                +0.08,
                 0.00,
            ])
        else:
            adjustments = np.zeros(self._n_assets)

        # ── Macro adjustment ──────────────────────────────
        macro = agent1_output.get("macro_environment", {})
        inflation = macro.get("inflation_state", "")
        growth = macro.get("growth_state", "")

        if "above" in inflation:
            adjustments[2] += 0.02    # GLD benefits from high inflation
            adjustments[1] -= 0.01    # BND hurt by inflation
        if "slowing" in growth or "contraction" in growth:
            adjustments[0] -= 0.02    # SPY hurt by slowing growth

        # ── Risk-free rate for cash ───────────────────────
        rf_rate = risk_free_rate
        indicators = macro.get("key_indicators", {})
        if indicators.get("fed_funds_rate"):
            rf_rate = indicators["fed_funds_rate"] / 100.0  # Convert from percentage
            if rf_rate > 0.10:
                rf_rate = 0.04  # Sanity check

        adjusted = base_returns + adjustments
        adjusted[-1] = rf_rate  # Cash = risk-free rate

        logger.info(
            f"Expected returns (regime={regime}): "
            + ", ".join(f"{t}={r:.1%}" for t, r in zip(self._tickers, adjusted))
        )

        return adjusted

    # ────────────────────────────────────────────────────
    #  COVARIANCE MATRIX
    # ────────────────────────────────────────────────────

    def get_covariance_matrix(
        self,
        agent1_output: Optional[dict] = None,
    ) -> np.ndarray:
        """
        Compute annualized covariance matrix.

        Uses the base correlation structure adjusted by:
          1. Current volatility state
          2. Stress correlation increase (correlations rise in crisis)
        """
        # Base volatilities
        vols = np.array([
            ASSET_UNIVERSE[t]["long_term_vol"] for t in self._tickers
        ])

        # Start with base correlation
        corr = BASE_CORRELATION.copy()

        if agent1_output:
            # ── Volatility adjustment ─────────────────────
            vol_state = agent1_output.get("volatility_state", {}).get("current_state", "normal")
            vol_multipliers = {
                "extremely_low": 0.7,
                "low": 0.85,
                "normal": 1.0,
                "elevated": 1.3,
                "extreme": 1.6,
            }
            vol_mult = vol_multipliers.get(vol_state, 1.0)
            vols = vols * vol_mult
            vols[-1] = ASSET_UNIVERSE["CASH"]["long_term_vol"]  # Cash vol unchanged

            # ── Stress correlation increase ───────────────
            # In crises, correlations tend toward 1 (except safe havens)
            risk_level = agent1_output.get("systemic_risk", {}).get("overall_risk_level", 0.0)
            if risk_level > 0.3:
                stress_factor = min(1.0, (risk_level - 0.3) / 0.5)
                # Increase correlations between risky assets
                for i in range(self._n_assets):
                    for j in range(i + 1, self._n_assets):
                        asset_i = self._tickers[i]
                        asset_j = self._tickers[j]
                        # Only increase correlation between risky assets
                        risky = {"SPY", "BTC"}
                        if asset_i in risky and asset_j in risky:
                            corr[i, j] += stress_factor * 0.3 * (1 - corr[i, j])
                            corr[j, i] = corr[i, j]

            # Use real correlations from Agent 1 if available
            key_corrs = agent1_output.get("cross_asset_analysis", {}).get("key_correlations", {})
            if key_corrs.get("SPY_GLD") is not None:
                corr[0, 2] = key_corrs["SPY_GLD"]
                corr[2, 0] = key_corrs["SPY_GLD"]

        # Ensure correlation matrix is valid (positive semi-definite)
        corr = self._nearest_psd(corr)

        # Build covariance: Cov = D @ Corr @ D where D = diag(vols)
        D = np.diag(vols)
        cov = D @ corr @ D

        return cov

    # ────────────────────────────────────────────────────
    #  CONSTRAINTS
    # ────────────────────────────────────────────────────

    def get_weight_bounds(
        self,
        risk_score: float = 0.5,
        behavioral_type: str = "moderate_balanced",
    ) -> list[tuple[float, float]]:
        """
        Get per-asset weight bounds based on investor profile.

        Conservative investors get tighter equity/crypto caps.
        Aggressive investors get higher equity/crypto floors.
        """
        # Default bounds
        bounds = {
            "SPY": (0.10, 0.60),
            "BND": (0.05, 0.50),
            "GLD": (0.00, 0.25),
            "BTC": (0.00, 0.15),
            "CASH": (0.05, 0.40),
        }

        # Adjust based on risk score
        if risk_score < 0.25:  # Very conservative
            bounds["SPY"] = (0.10, 0.35)
            bounds["BTC"] = (0.00, 0.03)
            bounds["BND"] = (0.20, 0.50)
            bounds["CASH"] = (0.15, 0.40)
        elif risk_score < 0.45:  # Conservative-moderate
            bounds["SPY"] = (0.15, 0.45)
            bounds["BTC"] = (0.00, 0.07)
            bounds["BND"] = (0.15, 0.45)
            bounds["CASH"] = (0.10, 0.30)
        elif risk_score < 0.65:  # Moderate
            bounds["SPY"] = (0.20, 0.50)
            bounds["BTC"] = (0.00, 0.10)
            bounds["BND"] = (0.10, 0.35)
            bounds["CASH"] = (0.05, 0.25)
        elif risk_score < 0.80:  # Growth
            bounds["SPY"] = (0.25, 0.55)
            bounds["BTC"] = (0.02, 0.15)
            bounds["BND"] = (0.05, 0.25)
            bounds["CASH"] = (0.05, 0.15)
        else:  # Aggressive
            bounds["SPY"] = (0.30, 0.60)
            bounds["BTC"] = (0.05, 0.20)
            bounds["BND"] = (0.00, 0.20)
            bounds["GLD"] = (0.00, 0.15)
            bounds["CASH"] = (0.00, 0.10)

        return [bounds[t] for t in self._tickers]

    # ────────────────────────────────────────────────────
    #  UTILITIES
    # ────────────────────────────────────────────────────

    @staticmethod
    def _nearest_psd(matrix: np.ndarray) -> np.ndarray:
        """
        Find the nearest positive semi-definite matrix.
        Uses eigenvalue clipping approach.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, 1e-8)
        result = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        # Ensure diagonal is 1 for correlation matrix
        d = np.sqrt(np.diag(result))
        result = result / np.outer(d, d)
        np.fill_diagonal(result, 1.0)
        return result

    def get_asset_info(self) -> list[dict]:
        """Get asset universe information."""
        return [
            {
                "ticker": t,
                "name": ASSET_UNIVERSE[t]["name"],
                "asset_class": ASSET_UNIVERSE[t]["asset_class"],
                "long_term_return": ASSET_UNIVERSE[t]["long_term_return"],
                "long_term_vol": ASSET_UNIVERSE[t]["long_term_vol"],
            }
            for t in self._tickers
        ]
