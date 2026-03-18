"""
Hybrid Intelligence Portfolio System -- Agent 3 Orchestrator
===============================================================
Strategic Allocation & Optimization Agent

Pipeline:
  1. Load upstream context (Agent 1 + Agent 2)
  2. Build asset universe (returns, covariance, constraints)
  3. Run portfolio optimization (MV, Risk Parity, CVaR)
  4. Select best strategy for investor profile
  5. Run Monte Carlo simulation (10K scenarios)
  6. Generate LLM explanation
  7. Assemble validated output
"""

import json
import logging
import time
import numpy as np
from datetime import datetime
from typing import Optional

from ml.asset_universe import AssetUniverseManager
from ml.portfolio_optimizer import PortfolioOptimizer
from ml.monte_carlo import MonteCarloSimulator
from llm.allocation_explainer import AllocationExplainer

logger = logging.getLogger(__name__)


class Agent3PortfolioStrategist:
    """
    Agent 3 -- Strategic Allocation & Optimization Agent.

    Consumes Agent 1 (market context) + Agent 2 (investor profile)
    to produce an optimal portfolio allocation with full quantitative
    analysis and institutional-quality explanation.
    """

    def __init__(self):
        self._universe = AssetUniverseManager()
        self._explainer = AllocationExplainer()
        self._execution_log = []

    def run(
        self,
        agent1_output: dict,
        agent2_output: dict,
        current_portfolio: Optional[dict] = None,
        bypass_llm: bool = False,
    ) -> dict:
        """
        Run the full Agent 3 pipeline.

        Args:
            agent1_output: Complete Agent 1 JSON output
            agent2_output: Complete Agent 2 JSON output (phase2_profile)

        Returns:
            Agent3Output-compatible dict
        """
        self._log_banner("AGENT 3 -- STRATEGIC ALLOCATION & OPTIMIZATION")
        self._execution_log.clear()
        start = time.time()

        # ── EXTRACT UPSTREAM CONTEXT ──────────────────────
        self._execution_log.append("Extracting upstream context...")
        
        # 1. Macro & Market Context (Agent 1)
        market_regime = agent1_output.get("market_regime", {})
        regime = market_regime.get("primary_regime", "unknown")
        
        # 2. Investor Profile (Agent 2)
        investor_profile = agent2_output.get("risk_classification", agent2_output)
        risk_score = investor_profile.get("risk_score", 0.5)
        max_drawdown = investor_profile.get("max_acceptable_drawdown", 0.15)
        behavioral_type = investor_profile.get("behavioral_type", "moderate_balanced")
        liquidity_pref = investor_profile.get("liquidity_preference", "medium")
        time_horizon = investor_profile.get("time_horizon", "medium")

        # ── FAIL-SAFE 1: MISSING DATA FALLBACK ────────────
        # If Agent 1 flagged data as missing/degraded, we MUST not take aggressive risk
        data_quality = agent1_output.get("agent_metadata", {}).get("data_quality", "full")
        if data_quality == "degraded":
            self._execution_log.append("FAIL-SAFE TRIGGERED: Missing data detected. Forcing conservative profile.")
            logger.warning("Agent 1 reported degraded data. Falling back to conservative allocation.")
            risk_score = min(risk_score, 0.30)
            max_drawdown = min(max_drawdown, 0.08)
            behavioral_type = "conservative_preservation"

        # ── MONSTER RULE: Agent 1 Confidence ──────────────
        a1_conf = market_regime.get("confidence", 1.0)
        adj_conf = market_regime.get("adjusted_confidence", a1_conf)
        self._execution_log.append(f"Agent 1 Macro Confidence: {a1_conf:.2f} (Adjusted: {adj_conf:.2f})")
        
        if a1_conf < 0.60:
            self._execution_log.append("MONSTER RULE TRIGGERED: Agent 1 Confidence < 0.60. Forcing defensive posture.")
            logger.warning(f"MONSTER RULE: Agent 1 Confidence ({a1_conf:.2f}) < 0.60. Automaticaly derisking.")
            risk_score = max(0.0, risk_score - 0.15)
            max_drawdown = max(0.02, max_drawdown * 0.70)  # Tighten max drawdown (CVaR) constraint by 30%

        # ── FAIL-SAFE 3: REGIME UNKNOWN MAX RISK CAP ──────
        # If adjusted confidence is terrible (<0.40) or regime is genuinely unknown
        if adj_conf < 0.40 or regime == "unknown":
            self._execution_log.append(f"FAIL-SAFE TRIGGERED: Regime highly uncertain (conf={adj_conf:.2f}). Cap max risk exposure at 30%.")
            logger.warning("Regime uncertain. Enforcing 30% maximum risk asset cap.")
            # We enforce this via Pydantic/Orchestrator limits, but we tell the optimizer
            # by severely cutting the permitted drawdown.
            max_drawdown = min(max_drawdown, 0.05)
            risk_score = min(risk_score, 0.30)

        # Risk-free rate from macro indicators
        rf_rate = 0.04
        indicators = agent1_output.get("macro_environment", {}).get("key_indicators", {})
        if indicators.get("fed_funds_rate"):
            rf_rate = min(0.08, indicators["fed_funds_rate"] / 100.0)

        # ── Step 1: Build Asset Universe ──────────────────
        self._log_step(1, 7, "Building asset universe...")
        step_start = time.time()

        expected_returns = self._universe.get_expected_returns(
            agent1_output=agent1_output,
            risk_free_rate=rf_rate,
        )
        cov_matrix = self._universe.get_covariance_matrix(
            agent1_output=agent1_output,
        )
        bounds = self._universe.get_weight_bounds(
            risk_score=risk_score,
            behavioral_type=behavioral_type,
        )

        self._log_execution("asset_universe", step_start)

        # ── Step 2: Portfolio Optimization ────────────────
        self._log_step(2, 7, "Running portfolio optimization (MV + RP + CVaR)...")
        step_start = time.time()

        optimizer = PortfolioOptimizer(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            tickers=self._universe.tickers,
            risk_free_rate=rf_rate,
        )

        try:
            opt_result = optimizer.optimize_for_profile(
                risk_score=risk_score,
                max_drawdown=max_drawdown,
                bounds=bounds,
            )
            if not opt_result.get("converged", False):
                raise RuntimeError("Optimization did not converge")

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            self._execution_log.append(f"OPTIMIZATION ERROR: {e}")
            
            # ── FAIL-SAFE 2: MODEL ERROR -> CAPITAL PRESERVATION ──
            self._execution_log.append("FAIL-SAFE TRIGGERED: Model optimization failed. Defaulting to Capital Preservation.")
            logger.error("Engine failed. Reverting to capital preservation fallback.")
            
            # Define fallback weights and metrics
            fallback_weights = {
                "SPY": 0.0,
                "BND": 0.20,
                "GLD": 0.10,
                "BTC": 0.0,
                "CASH": 0.70  # 70% cash in absolute failure
            }
            
            # Convert fallback_weights to an array matching the universe tickers
            weights_array = np.array([fallback_weights.get(t, 0.0) for t in self._universe.tickers])
            
            # Calculate basic metrics for fallback
            fallback_port_ret = np.dot(weights_array, expected_returns)
            fallback_port_vol = np.sqrt(np.dot(weights_array.T, np.dot(cov_matrix, weights_array)))
            fallback_sharpe = (fallback_port_ret - rf_rate) / fallback_port_vol if fallback_port_vol > 0 else 0.0

            opt_result = {
                "weights": fallback_weights,
                "weights_array": weights_array,
                "expected_return": fallback_port_ret,
                "expected_volatility": fallback_port_vol,
                "sharpe_ratio": fallback_sharpe,
                "method": "fallback_capital_preservation",
                "strategy_type": "defensive",
                "risk_aversion": 10.0,
                "converged": False,
                "alternatives": {},
            }

        self._log_execution("optimization", step_start)

        # ── Step 3: Risk Decomposition ────────────────────
        self._log_step(3, 7, "Computing risk decomposition...")
        step_start = time.time()

        weights_array = opt_result.get("weights_array", np.ones(self._universe.n_assets) / self._universe.n_assets)
        risk_contributions = optimizer.compute_risk_contributions(weights_array)
        div_ratio = optimizer.compute_diversification_ratio(weights_array)

        self._log_execution("risk_decomposition", step_start)

        # ── Step 4: Monte Carlo Simulation ────────────────
        self._log_step(4, 7, "Running Monte Carlo simulation (10K scenarios)...")
        step_start = time.time()

        mc_sim = MonteCarloSimulator(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            tickers=self._universe.tickers,
        )

        mc_result = mc_sim.simulate(
            weights=weights_array,
            n_simulations=10_000,
            horizon_days=252,
            max_acceptable_drawdown=max_drawdown,
        )

        self._log_execution("monte_carlo", step_start)

        # ── Step 5: Build Allocation Details ──────────────
        self._log_step(5, 7, "Assembling allocation details...")
        step_start = time.time()

        allocation_list = self._build_allocation_list(
            opt_result, expected_returns, cov_matrix,
            risk_contributions, self._universe.tickers
        )

        # Portfolio metrics
        port_vol = opt_result["expected_volatility"]
        port_ret = opt_result["expected_return"]
        sharpe = opt_result["sharpe_ratio"]

        # Sortino ratio approximation (assume downside vol ≈ 70% of total vol)
        downside_vol = port_vol * 0.7
        sortino = (port_ret - rf_rate) / downside_vol if downside_vol > 0 else 0

        # Max drawdown estimate: use MC or analytical approximation
        max_dd_est = mc_result.get("median_max_drawdown", max_drawdown)

        metrics = {
            "expected_annual_return": port_ret,
            "expected_annual_volatility": port_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": round(sortino, 4),
            "max_drawdown_estimate": round(max_dd_est, 4),
            "value_at_risk_95": mc_result.get("simulation_var_95", 0),
            "cvar_95": mc_result.get("simulation_cvar_95", 0),
            "risk_free_rate": round(rf_rate, 4),
            "diversification_ratio": div_ratio,
        }

        # ── Step 6: Portfolio Evolution Engine ────────────
        self._log_step(6, 7, "Executing Portfolio Evolution Engine (Drift/Turnover)...")
        step_start = time.time()
        
        evolution_metrics, final_allocation, final_metrics = self._apply_evolution_engine(
            current_portfolio=current_portfolio,
            target_allocation=allocation_list,
            target_metrics=metrics,
            regime=regime,
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            rf_rate=rf_rate
        )
        
        self._log_execution("evolution_engine", step_start)

        optimization_details = {
            "method_used": opt_result.get("method", "blended"),
            "strategy_type": opt_result.get("strategy_type", "balanced"),
            "risk_aversion_parameter": opt_result.get("risk_aversion", 1.0),
            "constraints_applied": [
                f"{t}: [{b[0]:.0%}, {b[1]:.0%}]"
                for t, b in zip(self._universe.tickers, bounds)
            ],
            "convergence_achieved": opt_result.get("converged", True),
            "alternatives_considered": opt_result.get("alternatives", {}),
        }

        self._log_execution("assembly", step_start)

        # ── Step 7: LLM Explanation ───────────────────────
        self._log_step(7, 7, "Generating LLM allocation explanation...")
        step_start = time.time()

        if bypass_llm:
            explanation = {
                "allocation_rationale": "MOCK EXPLANATION (LLM BYPASSED)",
                "regime_impact": "MOCK",
                "risk_profile_alignment": "MOCK",
                "trade_offs": ["MOCK"],
                "caveats": ["MOCK"],
                "rebalancing_triggers": ["MOCK"],
            }
            # We don't overwrite _explainer properties directly here, we just pass
        else:
            explanation = self._explainer.explain(
                allocation=final_allocation,
                metrics=final_metrics,
                monte_carlo=mc_result,
                optimization=optimization_details,
                agent1_output=agent1_output,
                agent2_output=agent2_output,
            )

        self._log_execution("llm_explanation", step_start)

        # ── Assemble Final Output ─────────────────────────
        total_ms = (time.time() - start) * 1000
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown")

        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": agent2_output.get("session_id", ""),
            "allocation": final_allocation,
            "portfolio_metrics": final_metrics,
            "monte_carlo": mc_result,
            "optimization": optimization_details,
            "evolution_metrics": evolution_metrics,
            "llm_explanation": explanation,
            "market_regime": regime,
            "investor_risk_score": risk_score,
            "investor_behavioral_type": behavioral_type,
            "investor_max_drawdown": max_drawdown,
            "agent_metadata": {
                "agent_id": "agent3_portfolio_strategist",
                "version": "1.0.0",
                "execution_time_ms": round(total_ms),
                "llm_calls": self._explainer.llm_calls,
                "llm_total_latency_ms": round(self._explainer.llm_latency_ms),
                "models_used": [
                    "asset_universe_v1",
                    "mean_variance_optimizer_v1",
                    "risk_parity_optimizer_v1",
                    "cvar_optimizer_v1",
                    "monte_carlo_simulator_10k",
                    "llm_core_model",
                ],
                "execution_log": self._execution_log.copy(),
            },
        }

        # Log summary
        alloc_str = ", ".join(
            f"{a['ticker']}={a['weight']:.0%}" for a in final_allocation
        )
        reb_str = "YES" if evolution_metrics["requires_rebalance"] else "NO (Hold)"
        
        logger.info(
            f"\n{'='*60}\n"
            f"AGENT 3 COMPLETE\n"
            f"  Strategy:     {opt_result.get('strategy_type', 'N/A')}\n"
            f"  Rebalance:    {reb_str} (Drift: {evolution_metrics['max_drift_detected']:.1%})\n"
            f"  Allocation:   {alloc_str}\n"
            f"  E[Return]:    {final_metrics['expected_annual_return']:.1%}\n"
            f"  Volatility:   {final_metrics['expected_annual_volatility']:.1%}\n"
            f"  Sharpe:       {final_metrics['sharpe_ratio']:.2f}\n"
            f"  Max DD Est:   {max_dd_est:.1%}\n"
            f"  P(Loss 1Y):   {mc_result.get('probability_of_loss', 0):.0%}\n"
            f"  Exec Time:    {total_ms:.0f}ms\n"
            f"{'='*60}"
        )

        return output

    # ════════════════════════════════════════════════════
    #  MOCK MODE
    # ════════════════════════════════════════════════════

    def run_mock(
        self,
        agent1_output: Optional[dict] = None,
        agent2_output: Optional[dict] = None,
        current_portfolio: Optional[dict] = None,
        bypass_llm: bool = False,
    ) -> dict:
        """
        Run Agent 3 with mock or provided upstream data.
        """
        if agent1_output is None:
            agent1_output = self._mock_agent1()
        if agent2_output is None:
            agent2_output = self._mock_agent2()

        # If agent2_output is the full combined output, extract phase2
        if "phase2_profile" in agent2_output:
            agent2_output = agent2_output["phase2_profile"]

        return self.run(
            agent1_output=agent1_output,
            agent2_output=agent2_output,
            current_portfolio=current_portfolio,
            bypass_llm=bypass_llm,
        )

    # ────────────────────────────────────────────────────
    #  HELPERS
    # ────────────────────────────────────────────────────

    def _build_allocation_list(
        self,
        opt_result: dict,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        risk_contributions: dict,
        tickers: list[str],
    ) -> list[dict]:
        """Build per-asset allocation details."""
        from ml.asset_universe import ASSET_UNIVERSE

        weights = opt_result.get("weights", {})
        vols = np.sqrt(np.diag(cov_matrix))

        allocation = []
        for i, ticker in enumerate(tickers):
            asset_info = ASSET_UNIVERSE.get(ticker, {})
            w = weights.get(ticker, 0.0)
            allocation.append({
                "ticker": ticker,
                "asset_class": asset_info.get("asset_class", "unknown"),
                "weight": round(w, 4),
                "expected_return": round(float(expected_returns[i]), 4),
                "expected_volatility": round(float(vols[i]), 4),
                "risk_contribution": risk_contributions.get(ticker, 0.0),
                "rationale": self._asset_rationale(ticker, w, opt_result.get("strategy_type", "")),
            })

        return allocation

    @staticmethod
    def _asset_rationale(ticker: str, weight: float, strategy: str) -> str:
        """Generate per-asset allocation rationale."""
        rationales = {
            "SPY": f"US equity core ({weight:.0%}): primary growth driver via broad market exposure",
            "BND": f"Fixed income ({weight:.0%}): portfolio stabilizer, low correlation to equity",
            "GLD": f"Gold ({weight:.0%}): inflation hedge and crisis alpha during market dislocations",
            "BTC": f"Crypto ({weight:.0%}): high-conviction growth asset with asymmetric return profile",
            "CASH": f"Cash ({weight:.0%}): liquidity buffer and optionality for future deployment",
        }
        return rationales.get(ticker, f"{ticker}: {weight:.0%} allocation")

    def _apply_evolution_engine(
        self,
        current_portfolio: Optional[dict],
        target_allocation: list[dict],
        target_metrics: dict,
        regime: str,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        rf_rate: float
    ) -> tuple[dict, list[dict], dict]:
        """
        Evaluate drift, transaction costs, and threshold rebalancing.
        """
        if not current_portfolio:
            # First deployment: no drift, zero turnover assumed for cash injection
            evo = {
                "requires_rebalance": True,
                "max_drift_detected": 0.0,
                "portfolio_turnover": 0.0,
                "estimated_transaction_costs": 0.0,
            }
            return evo, target_allocation, target_metrics

        # Calculate exact per-asset drift relative to target
        max_drift = 0.0
        turnover = 0.0
        target_weights_dict = {a["ticker"]: a["weight"] for a in target_allocation}
        
        for t, w_curr in current_portfolio.items():
            w_target = target_weights_dict.get(t, 0.0)
            drift = abs(w_target - w_curr)
            if drift > max_drift:
                max_drift = drift
            turnover += drift
            
        turnover = turnover / 2.0  # Turnover is one-way

        # Threshold Rebalancing Rule: 5%
        # Exception: Highly stressed regimes force rebalances regardless of drift
        is_stressed_regime = regime in ["bear_high_vol", "bear_low_vol"]
        requires_rebalance = (max_drift >= 0.05) or is_stressed_regime

        if not requires_rebalance:
            self._execution_log.append(f"EVOLUTION ENGINE: Hold steady. Max drift {max_drift:.1%} < 5%.")
            logger.info(f"Threshold rule triggered: max drift {max_drift:.1%} < 5%. No rebalance needed.")
            
            # Recalculate metrics based on current retained weights
            curr_weights_arr = np.array([current_portfolio.get(t, 0.0) for t in self._universe.tickers])
            port_ret = float(np.dot(curr_weights_arr, expected_returns))
            port_vol = float(np.sqrt(np.dot(curr_weights_arr.T, np.dot(cov_matrix, curr_weights_arr))))
            sharpe = (port_ret - rf_rate) / port_vol if port_vol > 0 else 0
            
            downside_vol = port_vol * 0.7
            sortino = (port_ret - rf_rate) / downside_vol if downside_vol > 0 else 0
            
            held_metrics = target_metrics.copy()
            held_metrics.update({
                "expected_annual_return": port_ret,
                "expected_annual_volatility": port_vol,
                "sharpe_ratio": sharpe,
                "sortino_ratio": round(sortino, 4),
            })
            
            # Build retained allocation list
            held_alloc = []
            for a in target_allocation:
                held = a.copy()
                held["weight"] = current_portfolio.get(a["ticker"], 0.0)
                held["rationale"] = f"Maintained position (drift < 5%). Target was {a['weight']:.0%}."
                held_alloc.append(held)
                
            evo = {
                "requires_rebalance": False,
                "max_drift_detected": float(max_drift),
                "portfolio_turnover": 0.0,
                "estimated_transaction_costs": 0.0,
            }
            return evo, held_alloc, held_metrics

        # If rebalance is required, apply transaction costs
        tx_cost_bps = 10  # 10 bps per unit of turnover
        tx_cost = turnover * (tx_cost_bps / 10000.0)
        
        self._execution_log.append(f"EVOLUTION ENGINE: Rebalance triggered. Drift: {max_drift:.1%}. Turnover: {turnover:.1%}. Cost: {tx_cost:.2%}")
        
        post_tx_metrics = target_metrics.copy()
        post_tx_metrics["expected_annual_return"] -= tx_cost
        
        # Adjust Sharpe for post tx return
        port_vol = post_tx_metrics["expected_annual_volatility"]
        post_tx_metrics["sharpe_ratio"] = (post_tx_metrics["expected_annual_return"] - rf_rate) / port_vol if port_vol > 0 else 0
        
        evo = {
            "requires_rebalance": True,
            "max_drift_detected": float(max_drift),
            "portfolio_turnover": float(turnover),
            "estimated_transaction_costs": float(tx_cost),
        }
        return evo, target_allocation, post_tx_metrics

    @staticmethod
    def _mock_agent1() -> dict:
        """Mock Agent 1 output."""
        return {
            "market_regime": {
                "primary_regime": "bull_low_vol",
                "confidence": 0.85,
                "models_agree": True,
                "regime_duration_days": 45,
                "transition_probability": 0.15,
                "description": "Bullish regime with contained volatility",
            },
            "volatility_state": {
                "current_state": "normal",
                "vix_level": 18.5,
                "vol_trend": "stable",
            },
            "systemic_risk": {
                "overall_risk_level": 0.15,
                "risk_category": "low",
            },
            "macro_environment": {
                "macro_regime": "stable_growth",
                "monetary_policy": "neutral",
                "inflation_state": "target_range",
                "growth_state": "moderate_growth",
                "key_indicators": {"fed_funds_rate": 4.5},
                "yield_curve": {"inverted": False},
            },
            "cross_asset_analysis": {
                "median_correlation": 0.25,
                "key_correlations": {"SPY_GLD": 0.05},
            },
        }

    @staticmethod
    def _mock_agent2() -> dict:
        """Mock Agent 2 output (phase2_profile)."""
        return {
            "session_id": "mock_session",
            "risk_classification": {
                "risk_score": 0.55,
                "behavioral_type": "moderate_balanced",
                "max_acceptable_drawdown": 0.15,
                "liquidity_preference": "medium",
                "time_horizon": "medium",
            },
            "behavioral_profile": {
                "consistency_score": 0.78,
                "emotional_stability": "stable",
                "stress_response_pattern": "adapt",
            },
        }

    # ────────────────────────────────────────────────────
    #  LOGGING
    # ────────────────────────────────────────────────────

    def _log_banner(self, title: str):
        logger.info(f"\n{'+'*60}")
        logger.info(f"|  {title:<56}|")
        logger.info(f"|  Hybrid Intelligence Portfolio System v1.0.0{' '*11}|")
        logger.info(f"{'+'*60}")

    @staticmethod
    def _log_step(step: int, total: int, msg: str):
        logger.info(f"STEP {step}/{total} -- {msg}")

    def _log_execution(self, step_name: str, start_time: float):
        duration = (time.time() - start_time) * 1000
        self._execution_log.append({
            "step": step_name,
            "status": "success",
            "duration_ms": round(duration),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        })
