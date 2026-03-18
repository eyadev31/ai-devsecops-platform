"""
Hybrid Intelligence Portfolio System -- Allocation Adjuster
==============================================================
Mathematical rebalancing engine for Agent 4.

When risk audits flag problems, this module modifies the allocation
to bring it within acceptable bounds while:
  1. Preserving investor intent as much as possible
  2. Ensuring weights sum to 1.0
  3. Respecting regime-specific limits
  4. Maintaining minimum diversification
"""

import logging
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


class AllocationAdjuster:
    """
    Adjusts portfolio allocations to satisfy risk constraints.

    Uses a priority-based rebalancing approach:
      1. Cap overweight assets at regime/profile limits
      2. Floor underweight defensive assets
      3. Redistribute excess weight proportionally
      4. Ensure sum = 1.0
    """

    TICKERS = ["SPY", "BND", "GLD", "BTC", "CASH"]

    @classmethod
    def adjust(
        cls,
        original_allocation: list[dict],
        audit_results: list[dict],
        agent1_output: dict,
        agent2_output: dict,
    ) -> list[dict]:
        """
        Adjust allocation based on audit findings.

        Args:
            original_allocation: Agent 3's proposed allocation
            audit_results: Results from all 5 risk audits
            agent1_output: Agent 1 market context
            agent2_output: Agent 2 investor profile

        Returns:
            List of AdjustedAllocation dicts
        """
        # Build current weights
        weights = {}
        for a in original_allocation:
            weights[a["ticker"]] = a["weight"]

        original = weights.copy()
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown").lower()
        vol_state = agent1_output.get("volatility_state", {}).get("current_state", "normal").lower()

        risk_cls = agent2_output.get("risk_classification", agent2_output)
        risk_score = risk_cls.get("risk_score", 0.5)
        max_dd = risk_cls.get("max_acceptable_drawdown", 0.15)

        # Compute regime-aware caps
        caps, floors = cls._compute_limits(regime, vol_state, risk_score, max_dd, agent1_output)

        # Step 1: Apply caps (reduce overweight assets)
        excess = 0.0
        for ticker in cls.TICKERS:
            w = weights.get(ticker, 0)
            cap = caps.get(ticker, 1.0)
            if w > cap:
                excess += w - cap
                weights[ticker] = cap

        # Step 2: Apply floors (increase underweight defensive assets)
        deficit = 0.0
        for ticker in cls.TICKERS:
            w = weights.get(ticker, 0)
            floor = floors.get(ticker, 0.0)
            if w < floor:
                deficit += floor - w
                weights[ticker] = floor

        # Step 3: Redistribute excess to defensive assets proportionally
        defensive_tickers = ["BND", "GLD", "CASH"]
        if excess > 0:
            available = []
            for t in defensive_tickers:
                room = caps.get(t, 1.0) - weights.get(t, 0)
                if room > 0:
                    available.append((t, room))

            if available:
                total_room = sum(r for _, r in available)
                for t, room in available:
                    share = (room / total_room) * excess if total_room > 0 else 0
                    weights[t] = weights.get(t, 0) + share

        # Step 4: If deficit created, reduce risky assets proportionally
        if deficit > 0:
            risky_tickers = ["SPY", "BTC"]
            available = []
            for t in risky_tickers:
                room = weights.get(t, 0) - floors.get(t, 0.0)
                if room > 0:
                    available.append((t, room))

            if available:
                total_room = sum(r for _, r in available)
                for t, room in available:
                    share = (room / total_room) * deficit if total_room > 0 else 0
                    weights[t] = weights.get(t, 0) - min(share, room)

        # Step 5: Normalize to sum = 1.0
        total = sum(weights.values())
        if total > 0 and abs(total - 1.0) > 0.001:
            for t in weights:
                weights[t] /= total

        # Ensure no negatives
        for t in weights:
            weights[t] = max(0, weights[t])

        # Final normalize
        total = sum(weights.values())
        if total > 0:
            for t in weights:
                weights[t] /= total

        # Build output
        adjusted = []
        for ticker in cls.TICKERS:
            orig_w = original.get(ticker, 0)
            adj_w = round(weights.get(ticker, 0), 4)
            change = round(adj_w - orig_w, 4)
            reason = cls._explain_change(ticker, orig_w, adj_w, regime, risk_score)
            adjusted.append({
                "ticker": ticker,
                "original_weight": round(orig_w, 4),
                "adjusted_weight": adj_w,
                "change": change,
                "reason": reason,
            })

        logger.info(
            "Allocation adjusted: " +
            ", ".join(f"{a['ticker']}: {a['original_weight']:.0%}→{a['adjusted_weight']:.0%}" for a in adjusted)
        )

        return adjusted

    @classmethod
    def _compute_limits(
        cls,
        regime: str,
        vol_state: str,
        risk_score: float,
        max_dd: float,
        agent1_output: dict = None,
    ) -> tuple[dict, dict]:
        """Compute per-asset caps and floors based on regime and profile."""
        
        # Base caps (loose)
        caps = {"SPY": 0.55, "BND": 0.50, "GLD": 0.30, "BTC": 0.15, "CASH": 0.40}
        floors = {"SPY": 0.10, "BND": 0.05, "GLD": 0.00, "BTC": 0.00, "CASH": 0.05}

        # Mathematical Effective Risk Tightening (Redesign)
        if agent1_output and "market_regime" in agent1_output:
            mr = agent1_output["market_regime"]
            eff_risk = mr.get("effective_risk_state", 0.5)
            adj_conf = mr.get("adjusted_confidence", 1.0)
            
            # If the market is highly risky (eff_risk > 0.60), scale down caps
            if eff_risk > 0.60:
                scalar = 1.0 - ((eff_risk - 0.60) * 1.5)  # Penalize high risk
                caps["SPY"] = min(caps["SPY"], 0.55 * max(0.5, scalar))
                caps["BTC"] = min(caps["BTC"], 0.15 * max(0.0, scalar - 0.2))
                floors["BND"] = max(floors["BND"], 0.05 + ((eff_risk - 0.60) * 0.5))
                floors["CASH"] = max(floors["CASH"], 0.05 + ((eff_risk - 0.60) * 0.4))
                
            # If regime label is uncertain, pull everything slightly towards center
            if adj_conf < 0.40:
                caps["SPY"] = min(caps["SPY"], 0.45)
                caps["BTC"] = min(caps["BTC"], 0.08)
                floors["BND"] = max(floors["BND"], 0.15)
                floors["CASH"] = max(floors["CASH"], 0.10)

        # Legacy Regime tightening (Fallback)
        if "bear_high_vol" in regime:
            caps["SPY"] = min(caps["SPY"], 0.35)
            caps["BTC"] = min(caps["BTC"], 0.05)
            floors["BND"] = max(floors["BND"], 0.15)
            floors["CASH"] = max(floors["CASH"], 0.10)
            floors["GLD"] = max(floors["GLD"], 0.10)
        elif "bear" in regime:
            caps["SPY"] = min(caps["SPY"], 0.40)
            caps["BTC"] = min(caps["BTC"], 0.08)
            floors["BND"] = max(floors["BND"], 0.10)
            floors["CASH"] = max(floors["CASH"], 0.08)
        elif "bull_high_vol" in regime:
            caps["BTC"] = min(caps["BTC"], 0.10)
            floors["CASH"] = max(floors["CASH"], 0.08)

        # Volatility tightening
        if vol_state in ("elevated", "extreme"):
            caps["BTC"] = min(caps["BTC"], 0.05)
            floors["CASH"] = max(floors["CASH"], 0.10)

        # Profile tightening
        if risk_score < 0.30:
            caps["SPY"] = min(caps["SPY"], 0.35)
            caps["BTC"] = min(caps["BTC"], 0.03)
            floors["BND"] = max(floors["BND"], 0.20)
            floors["CASH"] = max(floors["CASH"], 0.15)
        elif risk_score < 0.50:
            caps["BTC"] = min(caps["BTC"], 0.07)
            floors["BND"] = max(floors["BND"], 0.10)

        # Drawdown-based tightening
        if max_dd < 0.10:
            caps["BTC"] = min(caps["BTC"], 0.03)
            caps["SPY"] = min(caps["SPY"], 0.30)
            floors["BND"] = max(floors["BND"], 0.25)

        return caps, floors

    @staticmethod
    def _explain_change(
        ticker: str, orig: float, adj: float, regime: str, risk_score: float
    ) -> str:
        """Explain why an asset weight was changed."""
        if abs(adj - orig) < 0.005:
            return "No change required"

        if adj < orig:
            return (
                f"Reduced from {orig:.0%} to {adj:.0%} — "
                f"{'regime constraints' if 'bear' in regime else 'risk profile alignment'} "
                f"require lower exposure"
            )
        else:
            return (
                f"Increased from {orig:.0%} to {adj:.0%} — "
                f"{'defensive floor' if ticker in ('BND', 'CASH', 'GLD') else 'rebalancing'} "
                f"to maintain safety margins"
            )
