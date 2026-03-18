"""
Hybrid Intelligence Portfolio System -- Agent 4 Orchestrator
===============================================================
Meta-Risk & Supervision Agent

The final gatekeeper in the 4-agent pipeline.
Independently audits Agent 3's allocation and either
approves, adjusts, or rejects it.

Pipeline:
  1. Load all upstream outputs (Agent 1 + 2 + 3)
  2. Run 5 independent risk audits
  3. Compute adjusted allocation if needed
  4. LLM CRO adjudication → final verdict
  5. Assemble validated output
"""

import json
import logging
import time
from datetime import datetime
from typing import Optional

from ml.risk_auditor import RiskAuditor
from ml.allocation_adjuster import AllocationAdjuster
from llm.risk_adjudicator import RiskAdjudicator

logger = logging.getLogger(__name__)


class Agent4RiskSupervisor:
    """
    Agent 4 -- Meta-Risk & Supervision Agent.

    Acts as Chief Risk Officer: independently evaluates the
    portfolio allocation from Agent 3 against market context
    (Agent 1) and investor profile (Agent 2).
    """

    def __init__(self):
        self._adjudicator = RiskAdjudicator()
        self._execution_log = []

    def run(
        self,
        agent1_output: dict,
        agent2_output: dict,
        agent3_output: dict,
    ) -> dict:
        """
        Run the full Agent 4 pipeline.

        Args:
            agent1_output: Agent 1 market context
            agent2_output: Agent 2 investor profile
            agent3_output: Agent 3 proposed allocation

        Returns:
            Agent4Output-compatible dict
        """
        self._log_banner("AGENT 4 -- META-RISK & SUPERVISION")
        self._execution_log.clear()
        start = time.time()

        # Normalize agent2 output
        agent2_profile = agent2_output
        if "phase2_profile" in agent2_output:
            agent2_profile = agent2_output["phase2_profile"]

        # Extract context for output
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown")
        risk_cls = agent2_profile.get("risk_classification", agent2_profile)
        risk_score = risk_cls.get("risk_score", 0.5)
        strategy = agent3_output.get("optimization", {}).get("strategy_type", "unknown")

        # ── Step 1: Run 5 Independent Risk Audits ─────────
        self._log_step(1, 4, "Running 5 independent risk audits...")
        step_start = time.time()

        audit_results = RiskAuditor.run_all_audits(
            agent1_output=agent1_output,
            agent2_output=agent2_output,
            agent3_output=agent3_output,
        )

        self._log_execution("risk_audits", step_start)

        # Count results
        n_pass = sum(1 for a in audit_results if a["verdict"] == "pass")
        n_warn = sum(1 for a in audit_results if a["verdict"] == "warning")
        n_fail = sum(1 for a in audit_results if a["verdict"] == "fail")
        max_severity = max((a["severity"] for a in audit_results), default=0)

        logger.info(
            f"  Audit Summary: {n_pass} PASS, {n_warn} WARNING, {n_fail} FAIL "
            f"(max severity: {max_severity:.2f})"
        )

        # ── Step 2: Compute Adjusted Allocation ──────────
        self._log_step(2, 4, "Computing risk-adjusted allocation...")
        step_start = time.time()

        needs_adjustment = n_fail > 0 or n_warn > 0
        adjusted_allocation = None
        original_preserved = True

        if needs_adjustment:
            adjusted_allocation = AllocationAdjuster.adjust(
                original_allocation=agent3_output.get("allocation", []),
                audit_results=audit_results,
                agent1_output=agent1_output,
                agent2_output=agent2_profile,
            )
            # Check if any actual changes were made
            any_change = any(
                abs(a.get("change", 0)) > 0.005 for a in adjusted_allocation
            )
            original_preserved = not any_change
        else:
            logger.info("  No adjustments needed — all audits passed")

        self._log_execution("allocation_adjustment", step_start)

        # ── Step 3: LLM CRO Adjudication ─────────────────
        self._log_step(3, 4, "LLM Chief Risk Officer adjudication...")
        step_start = time.time()

        verdict = self._adjudicator.adjudicate(
            audit_results=audit_results,
            agent1_output=agent1_output,
            agent2_output=agent2_profile,
            agent3_output=agent3_output,
            adjusted_allocation=adjusted_allocation,
        )

        self._log_execution("cro_adjudication", step_start)

        # ── Step 4: Assemble Output ───────────────────────
        self._log_step(4, 4, "Assembling final risk oversight output...")

        validation_status = verdict.get("decision", "approved_with_adjustments")

        # Determine overall risk level
        if max_severity >= 0.7:
            overall_risk = "critical"
        elif max_severity >= 0.5:
            overall_risk = "high"
        elif max_severity >= 0.3:
            overall_risk = "elevated"
        elif max_severity >= 0.15:
            overall_risk = "moderate"
        else:
            overall_risk = "low"

        total_ms = (time.time() - start) * 1000

        output = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "session_id": agent3_output.get("session_id", ""),
            "validation_status": validation_status,
            "overall_risk_level": overall_risk,
            "confidence": verdict.get("confidence", 0.5),
            "risk_audits": audit_results,
            "total_audits": 5,
            "audits_passed": n_pass,
            "audits_warned": n_warn,
            "audits_failed": n_fail,
            "adjusted_allocation": adjusted_allocation or [],
            "original_allocation_preserved": original_preserved,
            "risk_verdict": verdict,
            "market_regime": regime,
            "investor_risk_score": risk_score,
            "proposed_strategy": strategy,
            "agent_metadata": {
                "agent_id": "agent4_risk_supervisor",
                "version": "1.0.0",
                "execution_time_ms": round(total_ms),
                "llm_calls": self._adjudicator.llm_calls,
                "llm_total_latency_ms": round(self._adjudicator.llm_latency_ms),
                "models_used": [
                    "regime_consistency_auditor_v1",
                    "profile_alignment_auditor_v1",
                    "drawdown_guardrails_v1",
                    "concentration_auditor_v1",
                    "cross_agent_coherence_v1",
                    "allocation_adjuster_v1",
                    "llm_core_model",
                ],
                "execution_log": self._execution_log.copy(),
            },
        }

        # Log summary
        status_emoji = {"approved": "✓", "approved_with_adjustments": "⚠", "rejected": "✗"}
        logger.info(
            f"\n{'='*60}\n"
            f"AGENT 4 COMPLETE -- RISK OVERSIGHT VERDICT\n"
            f"  Status:       {status_emoji.get(validation_status, '?')} {validation_status.upper()}\n"
            f"  Risk Level:   {overall_risk}\n"
            f"  Confidence:   {verdict.get('confidence', 0):.0%}\n"
            f"  Audits:       {n_pass} pass, {n_warn} warn, {n_fail} fail\n"
            f"  Adjusted:     {'YES' if not original_preserved else 'NO'}\n"
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
        agent3_output: Optional[dict] = None,
    ) -> dict:
        """Run Agent 4 with mock or provided upstream data."""
        if agent1_output is None:
            agent1_output = self._mock_agent1()
        if agent2_output is None:
            agent2_output = self._mock_agent2()
        if agent3_output is None:
            agent3_output = self._mock_agent3()

        return self.run(
            agent1_output=agent1_output,
            agent2_output=agent2_output,
            agent3_output=agent3_output,
        )

    # ────────────────────────────────────────────────────
    #  MOCK DATA
    # ────────────────────────────────────────────────────

    @staticmethod
    def _mock_agent1() -> dict:
        return {
            "market_regime": {
                "primary_regime": "bear_high_vol",
                "confidence": 0.82,
                "models_agree": False,
            },
            "volatility_state": {"current_state": "elevated", "vix_level": 28},
            "systemic_risk": {"overall_risk_level": 0.55, "risk_category": "elevated"},
            "macro_environment": {
                "key_indicators": {"fed_funds_rate": 5.25},
            },
            "cross_asset_analysis": {"median_correlation": 0.55},
        }

    @staticmethod
    def _mock_agent2() -> dict:
        return {
            "risk_classification": {
                "risk_score": 0.40,
                "behavioral_type": "moderate_balanced",
                "max_acceptable_drawdown": 0.12,
                "liquidity_preference": "high",
                "time_horizon": "medium",
            },
            "behavioral_profile": {
                "consistency_score": 0.75,
                "emotional_stability": "stable",
                "contradiction_flags": [],
            },
        }

    @staticmethod
    def _mock_agent3() -> dict:
        """
        Intentionally PROBLEMATIC allocation to test audits.
        Growth allocation in bear market for moderate investor.
        """
        return {
            "allocation": [
                {"ticker": "SPY", "asset_class": "equity", "weight": 0.45,
                 "expected_return": 0.06, "expected_volatility": 0.21,
                 "risk_contribution": 0.50},
                {"ticker": "BND", "asset_class": "bond", "weight": 0.10,
                 "expected_return": 0.04, "expected_volatility": 0.065,
                 "risk_contribution": 0.02},
                {"ticker": "GLD", "asset_class": "commodity", "weight": 0.15,
                 "expected_return": 0.08, "expected_volatility": 0.20,
                 "risk_contribution": 0.10},
                {"ticker": "BTC", "asset_class": "crypto", "weight": 0.25,
                 "expected_return": 0.20, "expected_volatility": 0.85,
                 "risk_contribution": 0.38},
                {"ticker": "CASH", "asset_class": "cash", "weight": 0.05,
                 "expected_return": 0.04, "expected_volatility": 0.001,
                 "risk_contribution": 0.0},
            ],
            "portfolio_metrics": {
                "expected_annual_return": 0.095,
                "expected_annual_volatility": 0.28,
                "sharpe_ratio": 0.20,
                "max_drawdown_estimate": 0.22,
            },
            "monte_carlo": {
                "num_simulations": 10000,
                "median_max_drawdown": 0.22,
                "probability_of_loss": 0.42,
                "probability_of_severe_loss": 0.18,
                "simulation_var_95": 0.25,
                "simulation_cvar_95": 0.32,
                "worst_case_return": -0.45,
            },
            "optimization": {
                "method_used": "mean_variance",
                "strategy_type": "aggressive_growth",
            },
            "session_id": "mock_session",
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
