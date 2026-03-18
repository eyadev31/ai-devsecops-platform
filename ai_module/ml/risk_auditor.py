"""
Hybrid Intelligence Portfolio System -- Risk Auditor
======================================================
5 Independent Risk Audit Submodules for Agent 4.

Each audit independently evaluates a specific risk dimension
and returns a standardized verdict (pass/warning/fail).

Think like a Chief Risk Officer at a hedge fund:
  - Be skeptical
  - Never assume the previous agent is correct
  - Flag everything that could cause catastrophic loss
"""

import logging
import math
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════
#  REGIME RULES: What allocations are acceptable per regime
# ════════════════════════════════════════════════════════

REGIME_ALLOCATION_RULES = {
    "bear": {
        "max_equity": 0.40,
        "max_crypto": 0.08,
        "min_defensive": 0.35,  # bonds + cash + gold
        "description": "Bear regime: defensive posture required",
    },
    "bear_high_vol": {
        "max_equity": 0.35,
        "max_crypto": 0.05,
        "min_defensive": 0.45,
        "description": "Bear + high vol: maximum defensive stance",
    },
    "bull_low_vol": {
        "max_equity": 0.65,
        "max_crypto": 0.20,
        "min_defensive": 0.15,
        "description": "Bull low vol: growth-oriented acceptable",
    },
    "bull_high_vol": {
        "max_equity": 0.50,
        "max_crypto": 0.12,
        "min_defensive": 0.25,
        "description": "Bull high vol: moderate caution warranted",
    },
    "unknown": {
        "max_equity": 0.50,
        "max_crypto": 0.10,
        "min_defensive": 0.25,
        "description": "Uncertain regime: moderate defaults",
    },
}

# Behavioral type → allocation style mapping
PROFILE_ALLOCATION_EXPECTATIONS = {
    "conservative_stable": {"max_equity": 0.35, "max_crypto": 0.03, "min_cash": 0.15},
    "conservative_anxious": {"max_equity": 0.30, "max_crypto": 0.02, "min_cash": 0.20},
    "moderate_balanced": {"max_equity": 0.50, "max_crypto": 0.10, "min_cash": 0.08},
    "moderate_volatile": {"max_equity": 0.45, "max_crypto": 0.08, "min_cash": 0.10},
    "growth_seeker": {"max_equity": 0.55, "max_crypto": 0.15, "min_cash": 0.05},
    "aggressive_speculator": {"max_equity": 0.60, "max_crypto": 0.20, "min_cash": 0.03},
    "aggressive_contrarian": {"max_equity": 0.60, "max_crypto": 0.18, "min_cash": 0.03},
}


class RiskAuditor:
    """
    Independent risk auditor for Agent 4.

    Runs 5 separate audit checks, each returning a standardized
    verdict dict with pass/warning/fail status.
    """

    # ════════════════════════════════════════════════════
    #  AUDIT 1: REGIME CONSISTENCY
    # ════════════════════════════════════════════════════

    @staticmethod
    def audit_regime_consistency(
        agent1_output: dict,
        agent3_output: dict,
    ) -> dict:
        """
        Check if the proposed allocation is consistent with
        the current market regime detected by Agent 1.

        Flags:
          - Growth allocation in bear regime
          - High crypto in elevated volatility
          - Too little defensive in crisis
        """
        regime = agent1_output.get("market_regime", {}).get("primary_regime", "unknown").lower()
        vol_state = agent1_output.get("volatility_state", {}).get("current_state", "normal").lower()
        risk_level = agent1_output.get("systemic_risk", {}).get("overall_risk_level", 0.0)
        risk_cat = agent1_output.get("systemic_risk", {}).get("risk_category", "low").lower()

        # Get allocation weights
        allocation = {
            a["ticker"]: a["weight"]
            for a in agent3_output.get("allocation", [])
        }
        strategy = agent3_output.get("optimization", {}).get("strategy_type", "unknown")

        equity_w = allocation.get("SPY", 0)
        crypto_w = allocation.get("BTC", 0)
        bond_w = allocation.get("BND", 0)
        cash_w = allocation.get("CASH", 0)
        gold_w = allocation.get("GLD", 0)
        defensive_w = bond_w + cash_w + gold_w

        # Find matching regime rules
        regime_key = "unknown"
        for key in REGIME_ALLOCATION_RULES:
            if key in regime:
                regime_key = key
                break

        rules = REGIME_ALLOCATION_RULES[regime_key]
        findings = []
        severity = 0.0

        # Check equity cap
        if equity_w > rules["max_equity"]:
            excess = equity_w - rules["max_equity"]
            findings.append(
                f"Equity ({equity_w:.0%}) exceeds regime limit ({rules['max_equity']:.0%}) "
                f"by {excess:.0%} in {regime} regime"
            )
            severity = max(severity, min(1.0, 0.4 + excess * 3))

        # Check crypto cap
        if crypto_w > rules["max_crypto"]:
            excess = crypto_w - rules["max_crypto"]
            findings.append(
                f"Crypto ({crypto_w:.0%}) exceeds regime limit ({rules['max_crypto']:.0%}) "
                f"by {excess:.0%} in {regime} regime"
            )
            severity = max(severity, min(1.0, 0.5 + excess * 5))

        # Check defensive floor
        if defensive_w < rules["min_defensive"]:
            deficit = rules["min_defensive"] - defensive_w
            findings.append(
                f"Defensive ({defensive_w:.0%}) below regime minimum ({rules['min_defensive']:.0%}) "
                f"— short by {deficit:.0%}"
            )
            severity = max(severity, min(1.0, 0.3 + deficit * 3))

        # Strategy mismatch check
        if "bear" in regime and strategy in ("aggressive_growth", "max_growth"):
            findings.append(
                f"CRITICAL: {strategy} strategy selected during {regime} regime — "
                f"extreme mismatch between market conditions and allocation philosophy"
            )
            severity = max(severity, 0.9)

        # Volatility-specific crypto check
        if vol_state in ("elevated", "extreme") and crypto_w > 0.05:
            findings.append(
                f"Crypto ({crypto_w:.0%}) is high for {vol_state} volatility state — "
                f"crypto correlations increase in stress"
            )
            severity = max(severity, 0.4)

        # Determine verdict
        if severity >= 0.6:
            verdict = "fail"
        elif severity >= 0.3:
            verdict = "warning"
        else:
            verdict = "pass"

        return {
            "audit_name": "regime_consistency",
            "verdict": verdict,
            "severity": round(severity, 3),
            "finding": "; ".join(findings) if findings else "Allocation consistent with regime",
            "recommendation": (
                f"Reduce equity to ≤{rules['max_equity']:.0%}, crypto to ≤{rules['max_crypto']:.0%}, "
                f"increase defensive to ≥{rules['min_defensive']:.0%}"
                if findings else "No adjustments needed"
            ),
            "details": {
                "regime": regime,
                "regime_rules": rules,
                "allocation_weights": allocation,
                "strategy": strategy,
                "vol_state": vol_state,
            },
        }

    # ════════════════════════════════════════════════════
    #  AUDIT 2: PROFILE ALIGNMENT
    # ════════════════════════════════════════════════════

    @staticmethod
    def audit_profile_alignment(
        agent2_output: dict,
        agent3_output: dict,
    ) -> dict:
        """
        Verify allocation matches the investor's behavioral profile.

        Checks:
          - Risk score → appropriate risk level in allocation
          - Behavioral type → allocation style match
          - Max drawdown → portfolio drawdown within tolerance
          - Liquidity preference → adequate cash position
        """
        # Extract profile
        risk_cls = agent2_output.get("risk_classification", agent2_output)
        risk_score = risk_cls.get("risk_score", 0.5)
        beh_type = risk_cls.get("behavioral_type", "moderate_balanced")
        max_dd = risk_cls.get("max_acceptable_drawdown", 0.15)
        liquidity = risk_cls.get("liquidity_preference", "medium")

        # Extract allocation
        allocation = {
            a["ticker"]: a["weight"]
            for a in agent3_output.get("allocation", [])
        }
        portfolio_vol = agent3_output.get("portfolio_metrics", {}).get(
            "expected_annual_volatility", 0
        )
        mc_dd = agent3_output.get("monte_carlo", {}).get("median_max_drawdown", 0)
        strategy = agent3_output.get("optimization", {}).get("strategy_type", "unknown")

        equity_w = allocation.get("SPY", 0)
        crypto_w = allocation.get("BTC", 0)
        cash_w = allocation.get("CASH", 0)

        findings = []
        severity = 0.0

        # Get profile expectations
        profile_rules = PROFILE_ALLOCATION_EXPECTATIONS.get(
            beh_type, {"max_equity": 0.50, "max_crypto": 0.10, "min_cash": 0.08}
        )

        # Risk score vs allocation aggressiveness
        risky_weight = equity_w + crypto_w
        expected_risky_max = 0.30 + risk_score * 0.50  # 30% at score=0, 80% at score=1
        if risky_weight > expected_risky_max + 0.10:
            findings.append(
                f"Risky assets ({risky_weight:.0%} equity+crypto) too high for "
                f"risk score {risk_score:.2f} (expected max ~{expected_risky_max:.0%})"
            )
            severity = max(severity, 0.5)

        # Behavioral type specific checks
        if equity_w > profile_rules["max_equity"]:
            findings.append(
                f"Equity ({equity_w:.0%}) exceeds {beh_type} profile limit ({profile_rules['max_equity']:.0%})"
            )
            severity = max(severity, 0.3)

        if crypto_w > profile_rules["max_crypto"]:
            findings.append(
                f"Crypto ({crypto_w:.0%}) exceeds {beh_type} profile limit ({profile_rules['max_crypto']:.0%})"
            )
            severity = max(severity, 0.4)

        if cash_w < profile_rules["min_cash"]:
            findings.append(
                f"Cash ({cash_w:.0%}) below {beh_type} profile minimum ({profile_rules['min_cash']:.0%})"
            )
            severity = max(severity, 0.2)

        # Liquidity check
        if liquidity == "high" and cash_w < 0.15:
            findings.append(
                f"Investor requires high liquidity but cash is only {cash_w:.0%} (need ≥15%)"
            )
            severity = max(severity, 0.4)

        # Conservative investor with aggressive strategy
        if risk_score < 0.35 and strategy in ("growth", "aggressive_growth", "max_growth"):
            findings.append(
                f"MISMATCH: Conservative investor (score={risk_score:.2f}) "
                f"assigned {strategy} strategy"
            )
            severity = max(severity, 0.8)

        # Determine verdict
        if severity >= 0.6:
            verdict = "fail"
        elif severity >= 0.25:
            verdict = "warning"
        else:
            verdict = "pass"

        return {
            "audit_name": "profile_alignment",
            "verdict": verdict,
            "severity": round(severity, 3),
            "finding": "; ".join(findings) if findings else "Allocation aligns with investor profile",
            "recommendation": (
                f"Adjust allocation to match {beh_type} profile expectations"
                if findings else "No adjustments needed"
            ),
            "details": {
                "risk_score": risk_score,
                "behavioral_type": beh_type,
                "profile_rules": profile_rules,
                "risky_weight": round(risky_weight, 4),
                "expected_risky_max": round(expected_risky_max, 4),
            },
        }

    # ════════════════════════════════════════════════════
    #  AUDIT 3: DRAWDOWN & TAIL RISK GUARDRAILS
    # ════════════════════════════════════════════════════

    @staticmethod
    def audit_drawdown_guardrails(
        agent2_output: dict,
        agent3_output: dict,
    ) -> dict:
        """
        Verify Monte Carlo simulation results against
        the investor's stated maximum acceptable drawdown.

        This is the most critical safety check — it prevents
        catastrophic losses that violate the investor's tolerance.
        """
        risk_cls = agent2_output.get("risk_classification", agent2_output)
        max_dd = risk_cls.get("max_acceptable_drawdown", 0.15)

        mc = agent3_output.get("monte_carlo", {})
        median_dd = mc.get("median_max_drawdown", 0)
        prob_loss = mc.get("probability_of_loss", 0)
        prob_severe = mc.get("probability_of_severe_loss", 0)
        worst_case = mc.get("worst_case_return", 0)
        var_95 = mc.get("simulation_var_95", 0)
        cvar_95 = mc.get("simulation_cvar_95", 0)

        portfolio_dd = agent3_output.get("portfolio_metrics", {}).get(
            "max_drawdown_estimate", 0
        )

        findings = []
        severity = 0.0

        # Check median drawdown vs tolerance
        if median_dd > max_dd:
            excess = median_dd - max_dd
            findings.append(
                f"Median max drawdown ({median_dd:.1%}) EXCEEDS investor tolerance ({max_dd:.0%}) "
                f"by {excess:.1%}"
            )
            severity = max(severity, min(1.0, 0.5 + excess * 5))

        # Check probability of severe loss
        if prob_severe > 0.05:
            findings.append(
                f"Probability of severe loss ({prob_severe:.1%}) exceeds 5% safety threshold — "
                f"{prob_severe*100:.1f}% chance of drawdown exceeding {max_dd:.0%}"
            )
            severity = max(severity, 0.6)

        # Check CVaR vs drawdown tolerance
        if cvar_95 > max_dd * 1.5:
            findings.append(
                f"CVaR95 ({cvar_95:.1%}) is {cvar_95/max_dd:.1f}x the investor's drawdown tolerance — "
                f"tail risk is excessive"
            )
            severity = max(severity, 0.5)

        # Check overall loss probability
        if prob_loss > 0.40:
            findings.append(
                f"Probability of any loss ({prob_loss:.0%}) is above 40% — "
                f"more than 2 in 5 chance of negative return over 1 year"
            )
            severity = max(severity, 0.3)

        # Worst case check
        if abs(worst_case) > 0.30:
            findings.append(
                f"Worst case scenario return is {worst_case:.0%} — catastrophic tail event"
            )
            severity = max(severity, 0.4)

        # VaR sanity check
        if var_95 > max_dd:
            findings.append(
                f"VaR95 ({var_95:.1%}) exceeds investor max drawdown ({max_dd:.0%}) — "
                f"5% chance of loss exceeding stated tolerance"
            )
            severity = max(severity, 0.45)

        if severity >= 0.6:
            verdict = "fail"
        elif severity >= 0.3:
            verdict = "warning"
        else:
            verdict = "pass"

        return {
            "audit_name": "drawdown_guardrails",
            "verdict": verdict,
            "severity": round(severity, 3),
            "finding": "; ".join(findings) if findings else "Drawdown risk within acceptable limits",
            "recommendation": (
                f"Reduce portfolio volatility to keep drawdowns within {max_dd:.0%} tolerance. "
                f"Increase bonds/cash, reduce equity/crypto."
                if findings else "No adjustments needed"
            ),
            "details": {
                "investor_max_dd": max_dd,
                "median_dd": median_dd,
                "prob_loss": prob_loss,
                "prob_severe": prob_severe,
                "var_95": var_95,
                "cvar_95": cvar_95,
                "worst_case": worst_case,
            },
        }

    # ════════════════════════════════════════════════════
    #  AUDIT 4: CONCENTRATION & LEVERAGE
    # ════════════════════════════════════════════════════

    @staticmethod
    def audit_concentration(
        agent1_output: dict,
        agent3_output: dict,
    ) -> dict:
        """
        Check portfolio concentration risk using:
          - Herfindahl-Hirschman Index (HHI)
          - Single-asset concentration limits
          - Risk contribution concentration
          - Crypto limits based on volatility state
        """
        allocation = agent3_output.get("allocation", [])
        vol_state = agent1_output.get("volatility_state", {}).get("current_state", "normal").lower()

        weights = {a["ticker"]: a["weight"] for a in allocation}
        risk_contribs = {a["ticker"]: a.get("risk_contribution", 0) for a in allocation}

        findings = []
        severity = 0.0

        # HHI Index (sum of squared weights)
        # HHI < 0.20 = well diversified, > 0.30 = concentrated
        hhi = sum(w**2 for w in weights.values())

        if hhi > 0.35:
            findings.append(
                f"HHI concentration index ({hhi:.3f}) indicates a highly concentrated portfolio "
                f"(threshold: 0.35)"
            )
            severity = max(severity, 0.5)
        elif hhi > 0.25:
            findings.append(
                f"HHI concentration index ({hhi:.3f}) indicates moderate concentration "
                f"(well-diversified: <0.20)"
            )
            severity = max(severity, 0.2)

        # Single-asset concentration
        for ticker, weight in weights.items():
            if weight > 0.50:
                findings.append(
                    f"CRITICAL: {ticker} weight ({weight:.0%}) exceeds 50% — "
                    f"single-asset dominant portfolio"
                )
                severity = max(severity, 0.7)
            elif weight > 0.40 and ticker != "SPY":
                findings.append(
                    f"{ticker} weight ({weight:.0%}) exceeds 40% — high concentration"
                )
                severity = max(severity, 0.4)

        # Risk contribution concentration
        max_rc_ticker = max(risk_contribs, key=risk_contribs.get) if risk_contribs else None
        max_rc = risk_contribs.get(max_rc_ticker, 0) if max_rc_ticker else 0
        if max_rc > 0.60:
            findings.append(
                f"{max_rc_ticker} contributes {max_rc:.0%} of total portfolio risk — "
                f"single-asset risk dominance"
            )
            severity = max(severity, 0.5)

        # Crypto concentration in volatile markets
        crypto_w = weights.get("BTC", 0)
        if vol_state in ("elevated", "extreme"):
            if crypto_w > 0.05:
                findings.append(
                    f"Crypto at {crypto_w:.0%} during {vol_state} volatility — "
                    f"correlations spike in stress, cap at 5%"
                )
                severity = max(severity, 0.4)

        # Zero diversification check
        non_zero = sum(1 for w in weights.values() if w > 0.01)
        if non_zero < 3:
            findings.append(
                f"Only {non_zero} assets have meaningful weight (>1%) — "
                f"insufficient diversification"
            )
            severity = max(severity, 0.6)

        if severity >= 0.5:
            verdict = "fail"
        elif severity >= 0.2:
            verdict = "warning"
        else:
            verdict = "pass"

        return {
            "audit_name": "concentration_audit",
            "verdict": verdict,
            "severity": round(severity, 3),
            "finding": "; ".join(findings) if findings else "Portfolio adequately diversified",
            "recommendation": (
                "Reduce concentration in dominant assets. Rebalance towards equal risk contribution."
                if findings else "No adjustments needed"
            ),
            "details": {
                "hhi_index": round(hhi, 4),
                "weights": weights,
                "risk_contributions": risk_contribs,
                "non_zero_assets": non_zero,
                "max_risk_contributor": max_rc_ticker,
            },
        }

    # ════════════════════════════════════════════════════
    #  AUDIT 5: CROSS-AGENT COHERENCE
    # ════════════════════════════════════════════════════

    @staticmethod
    def audit_cross_agent_coherence(
        agent1_output: dict,
        agent2_output: dict,
        agent3_output: dict,
    ) -> dict:
        """
        Detect contradictions between what Agent 1/2 say and what Agent 3 actually did.
        -- MATHEMATICAL REGIME RELIABILITY REDESIGN IMPLEMENTATION --
        Uses Geometric Distance between Effective Market Risk and Target Portfolio Risk,
        expanding tolerance dynamically under model uncertainty.
        """
        mr = agent1_output.get("market_regime", {})
        adj_conf = mr.get("adjusted_confidence", 0.0)
        eff_risk = mr.get("effective_risk_state", 0.5)
        
        risk_cls = agent2_output.get("risk_classification", agent2_output)
        inv_risk = risk_cls.get("risk_score", 0.5)
        
        allocation = {a["ticker"]: a["weight"] for a in agent3_output.get("allocation", [])}
        equity_w = sum(w for t, w in allocation.items() if t in ["SPY", "QQQ", "IWM", "DIA"])
        crypto_w = allocation.get("BTC", 0) + allocation.get("ETH", 0)
        
        # Define structural portfolio risk [0,1]
        portfolio_risk = min(1.0, equity_w + (crypto_w * 1.5))
        
        # Target Risk assumes if market is 90% dangerous (0.9 returns), we should target 10% risk.
        target_risk = 1.0 - eff_risk
        
        # Geometric Coherence Distance
        distance = abs(target_risk - portfolio_risk)
        
        # Dynamic Acceptable Leniency Thresholds
        base_warn = 0.15
        base_reject = 0.25
        
        # As Agent 1 regime confidence drops, Supervisor grants leeway
        uncertainty_leniency = (1.0 - adj_conf) * 0.10
        # Aggressive investors get wider bound overrides
        profile_leniency = (inv_risk - 0.5) * 0.08
        
        t_warn = base_warn + uncertainty_leniency + profile_leniency
        t_reject = base_reject + uncertainty_leniency + profile_leniency
        
        findings = []
        severity = 0.0
        
        # Adjudication Matrix
        if distance >= t_reject:
            findings.append(
                f"FAIL: Portfolio Risk ({portfolio_risk:.2f}) diverges from Safe Target ({target_risk:.2f}). "
                f"Distance {distance:.3f} >= Dynamic Reject Threshold {t_reject:.3f}."
            )
            severity = max(severity, 0.75)
        elif distance >= t_warn:
            findings.append(
                f"WARNING: Portfolio Risk ({portfolio_risk:.2f}) stretching Safe Target ({target_risk:.2f}). "
                f"Distance {distance:.3f} >= Dynamic Warn Threshold {t_warn:.3f}."
            )
            severity = max(severity, 0.40)
            
        # Legacy Sub-Checks as minor modifiers
        strategy = agent3_output.get("optimization", {}).get("strategy_type", "unknown")
        models_agree = mr.get("models_agree", True)
        consistency = agent2_output.get("behavioral_profile", {}).get("consistency_score", 1.0)
        
        if not models_agree and strategy in ("aggressive_growth", "max_growth"):
            findings.append("Models disagree on regime, yet optimizer selected aggressive strategy.")
            severity = max(severity, 0.35)
            
        if consistency < 0.50 and max(allocation.values(), default=0) > 0.40:
            findings.append("High single-asset conviction conflicts with extreme investor profiling uncertainty.")
            severity = max(severity, 0.35)

        if severity >= 0.6:
            verdict = "fail"
        elif severity >= 0.3:
            verdict = "warning"
        else:
            verdict = "pass"

        return {
            "audit_name": "cross_agent_coherence",
            "verdict": verdict,
            "severity": round(severity, 3),
            "finding": "; ".join(findings) if findings else "Mathematical Coherence Distance (D_c) Within Safe Thresholds.",
            "recommendation": (
                "Supervisor triggers dynamic weight contraction mechanism."
                if verdict in ("warning", "fail") else "No adjustments needed"
            ),
            "details": {
                "portfolio_risk": round(portfolio_risk, 3),
                "target_risk": round(target_risk, 3),
                "coherence_distance": round(distance, 3),
                "dynamic_warn_threshold": round(t_warn, 3),
                "dynamic_reject_threshold": round(t_reject, 3),
                "adjusted_regime_confidence": adj_conf,
            },
        }

    # ════════════════════════════════════════════════════
    #  RUN ALL AUDITS
    # ════════════════════════════════════════════════════

    @classmethod
    def run_all_audits(
        cls,
        agent1_output: dict,
        agent2_output: dict,
        agent3_output: dict,
    ) -> list[dict]:
        """Run all 5 risk audits and return results."""
        # Normalize agent2 output (might be full or phase2 only)
        agent2_profile = agent2_output
        if "phase2_profile" in agent2_output:
            agent2_profile = agent2_output["phase2_profile"]

        audits = [
            cls.audit_regime_consistency(agent1_output, agent3_output),
            cls.audit_profile_alignment(agent2_profile, agent3_output),
            cls.audit_drawdown_guardrails(agent2_profile, agent3_output),
            cls.audit_concentration(agent1_output, agent3_output),
            cls.audit_cross_agent_coherence(agent1_output, agent2_profile, agent3_output),
        ]

        for audit in audits:
            logger.info(
                f"  AUDIT [{audit['audit_name']}]: {audit['verdict'].upper()} "
                f"(severity: {audit['severity']:.2f})"
            )

        return audits
