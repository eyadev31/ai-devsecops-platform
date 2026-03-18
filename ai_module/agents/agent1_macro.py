"""
Hybrid Intelligence Portfolio System — Agent 1 Pipeline Orchestrator
======================================================================
AGENT 1: MACRO & MARKET INTELLIGENCE AGENT

The central orchestrator that executes the full Agent 1 pipeline:
  1. Fetch market data (Binance + TwelveData)
  2. Fetch macro data (FRED)
  3. Run feature engineering
  4. Run regime detection (HMM + RF ensemble)
  5. Run volatility classification
  6. Run macro environment analysis
  7. Run systemic risk detection
  8. Pass ML outputs through multi-stage LLM reasoning
  9. Build final context JSON
  10. Validate output against Pydantic schema

Supports graceful degradation: if any component fails, the pipeline
continues with available data and flags degraded quality.
"""

import json
import time
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from config.settings import SystemMeta, APIKeys, RiskConfig
from data.market_data import MarketDataAggregator, TwelveDataFetcher
from data.macro_data import MacroDataFetcher
from ml.feature_engine import FeatureEngine
from ml.regime_detector import EnsembleRegimeDetector
from ml.volatility_classifier import VolatilityClassifier
from ml.macro_analyzer import MacroAnalyzer
from ml.risk_detector import SystemicRiskDetector
from llm.context_builder import ContextBuilder
from schemas.agent1_output import validate_output

logger = logging.getLogger(__name__)


def _setup_logging(level: str = "INFO") -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


class Agent1MacroIntelligence:
    """
    Agent 1 — Senior Global Macro Intelligence System

    Analyzes multi-asset financial markets (equities, ETFs, forex,
    commodities, crypto) to produce a structured intelligence context
    consumed by downstream AI agents.

    Pipeline Architecture:
    ┌─────────────┐     ┌────────────┐     ┌──────────────┐
    │ Market Data  │────▶│  Feature   │────▶│   Regime     │
    │ (Binance +   │     │  Engine    │     │  Detector    │
    │  TwelveData) │     └────────────┘     │ (HMM + RF)   │
    └─────────────┘                         └──────┬───────┘
    ┌─────────────┐     ┌────────────┐            │
    │ Macro Data   │────▶│  Macro     │            │
    │ (FRED)       │     │  Analyzer  │            │
    └─────────────┘     └────────────┘            │
                         ┌────────────┐            │
                         │ Volatility │            │
                         │ Classifier │            │
                         └────────────┘            │
                         ┌────────────┐            │
                         │   Risk     │            │
                         │  Detector  │            ▼
                         └────────────┘     ┌──────────────┐
                                            │  LLM Context │
                                            │   Builder    │
                                            │ (3-stage     │
                                            │  reasoning)  │
                                            └──────┬───────┘
                                                   ▼
                                            ┌──────────────┐
                                            │ Agent 1 JSON │
                                            │   Output     │
                                            └──────────────┘
    """

    def __init__(self):
        self.regime_detector = EnsembleRegimeDetector()
        self.context_builder = ContextBuilder()
        self._execution_log: list[dict] = []

    def run(self, mock: bool = False) -> dict:
        """
        Execute the full Agent 1 pipeline.

        Args:
            mock: If True, use mock data (no API calls required)

        Returns:
            Complete Agent 1 output JSON (validated against schema)
        """
        start_time = time.time()

        logger.info("+" + "=" * 60 + "+")
        logger.info("|  AGENT 1 -- MACRO & MARKET INTELLIGENCE SYSTEM           |")
        logger.info("|  Hybrid Intelligence Portfolio System v{:<18s}|".format(SystemMeta.VERSION))
        logger.info("+" + "=" * 60 + "+")
        logger.info("")

        if mock:
            return self._run_mock_pipeline()

        # Validate API keys
        key_status = APIKeys.validate()
        logger.info(f"API Keys: {json.dumps(key_status, indent=2)}")

        degraded_components = []

        # ═══════════════════════════════════════════════
        #  STEP 1: FETCH MARKET DATA
        # ═══════════════════════════════════════════════
        logger.info("STEP 1/9 — Fetching market data...")
        step_start = time.time()

        market_snapshot = {}
        try:
            market_snapshot = MarketDataAggregator.fetch_full_snapshot()
            self._log_step("market_data", "success", time.time() - step_start)
        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            degraded_components.append("market_data")
            self._log_step("market_data", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 2: FETCH MACRO DATA
        # ═══════════════════════════════════════════════
        logger.info("STEP 2/9 — Fetching macroeconomic data from FRED...")
        step_start = time.time()

        macro_snapshot = {}
        try:
            if key_status.get("fred"):
                macro_snapshot = MacroDataFetcher.compute_macro_snapshot()
                self._log_step("macro_data", "success", time.time() - step_start)
            else:
                logger.warning("FRED API key not configured — skipping macro data")
                degraded_components.append("macro_data")
                self._log_step("macro_data", "skipped", 0, "No FRED API key")
        except Exception as e:
            logger.error(f"Macro data fetch failed: {e}")
            degraded_components.append("macro_data")
            self._log_step("macro_data", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 3: FEATURE ENGINEERING
        # ═══════════════════════════════════════════════
        logger.info("STEP 3/9 — Running feature engineering...")
        step_start = time.time()

        features = {}
        regime_features = pd.DataFrame()
        benchmark_prices = None
        vix_data = None

        try:
            # Get benchmark prices (SPY or first available equity)
            benchmark_prices = self._get_benchmark_prices(market_snapshot)
            vix_data = market_snapshot.get("vix")

            # Get VIX close for features
            vix_close = None
            if vix_data is not None and not vix_data.empty and "close" in vix_data.columns:
                vix_close = vix_data["close"]

            if benchmark_prices is not None and not benchmark_prices.empty:
                # Build cross-asset close prices for correlation analysis
                all_closes = self._build_cross_asset_closes(market_snapshot)

                features = FeatureEngine.build_features(
                    benchmark_prices=benchmark_prices,
                    all_close_prices=all_closes,
                    vix_prices=vix_close,
                )

                regime_features = FeatureEngine.build_regime_features(
                    benchmark_prices=benchmark_prices,
                    vix_prices=vix_close,
                )

                self._log_step("feature_engineering", "success", time.time() - step_start)
            else:
                logger.warning("No benchmark prices available — features degraded")
                degraded_components.append("feature_engineering")
                self._log_step("feature_engineering", "degraded", time.time() - step_start)

        except Exception as e:
            logger.error(f"Feature engineering failed: {e}")
            degraded_components.append("feature_engineering")
            self._log_step("feature_engineering", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 4: REGIME DETECTION
        # ═══════════════════════════════════════════════
        logger.info("STEP 4/9 — Running ensemble regime detection...")
        step_start = time.time()

        regime_result = {"primary_regime": "unknown", "confidence": 0.0, "models_agree": False,
                         "hmm_result": {}, "rf_result": {}, "regime_duration_days": 0,
                         "transition_probability": 0.0, "description": "", "ensemble_method": "none"}

        try:
            if not regime_features.empty:
                regime_result = self.regime_detector.detect_regime(regime_features)
                self._log_step("regime_detection", "success", time.time() - step_start)
            else:
                degraded_components.append("regime_detection")
                self._log_step("regime_detection", "skipped", 0, "No features available")
        except Exception as e:
            logger.error(f"Regime detection failed: {e}")
            degraded_components.append("regime_detection")
            self._log_step("regime_detection", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 5: VOLATILITY CLASSIFICATION
        # ═══════════════════════════════════════════════
        logger.info("STEP 5/9 — Classifying volatility state...")
        step_start = time.time()

        volatility_state = {"current_state": "unknown", "vix_level": None,
                            "realized_vol_percentile": 50, "vol_trend": "unknown",
                            "vol_of_vol": "unknown", "term_structure": "unknown", "confidence": 0.0}

        try:
            if features:
                volatility_state = VolatilityClassifier.classify(features, vix_data)
                self._log_step("volatility_classification", "success", time.time() - step_start)
            else:
                degraded_components.append("volatility")
                self._log_step("volatility_classification", "skipped", 0)
        except Exception as e:
            logger.error(f"Volatility classification failed: {e}")
            degraded_components.append("volatility")
            self._log_step("volatility_classification", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 6: MACRO ENVIRONMENT ANALYSIS
        # ═══════════════════════════════════════════════
        logger.info("STEP 6/9 — Analyzing macroeconomic environment...")
        step_start = time.time()

        macro_analysis = {"macro_regime": "unknown", "composite_score": 0.0, "confidence": 0.0,
                          "monetary_policy_state": "unknown", "inflation_state": "unknown",
                          "growth_state": "unknown", "liquidity_state": "unknown",
                          "key_indicators": {}, "yield_curve": {}, "risk_factors": []}

        try:
            if macro_snapshot:
                macro_analysis = MacroAnalyzer.analyze(macro_snapshot)
                self._log_step("macro_analysis", "success", time.time() - step_start)
            else:
                degraded_components.append("macro_analysis")
                self._log_step("macro_analysis", "skipped", 0, "No macro data")
        except Exception as e:
            logger.error(f"Macro analysis failed: {e}")
            degraded_components.append("macro_analysis")
            self._log_step("macro_analysis", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 7: SYSTEMIC RISK DETECTION
        # ═══════════════════════════════════════════════
        logger.info("STEP 7/9 — Detecting systemic risk signals...")
        step_start = time.time()

        risk_assessment = {"overall_risk_level": 0.0, "risk_category": "unknown",
                           "risk_signals": {}, "risk_assessment": "", "recommended_caution": False,
                           "confidence": 0.0}

        try:
            risk_assessment = SystemicRiskDetector.detect(features, macro_analysis, volatility_state)
            self._log_step("risk_detection", "success", time.time() - step_start)
        except Exception as e:
            logger.error(f"Risk detection failed: {e}")
            degraded_components.append("risk_detection")
            self._log_step("risk_detection", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  STEP 7b: COMPUTE EFFECTIVE RISK STATE
        # ═══════════════════════════════════════════════
        # Blend the categorical regime risk with the macro composite score based on confidence.
        regime_risk_map = {
            "bull_low_vol": 0.20,
            "bull_high_vol": 0.40,
            "bear_low_vol": 0.60,
            "bear_high_vol": 0.90
        }
        
        base_regime = regime_result.get("primary_regime", "unknown").split("_0")[0].split("_1")[0].split("_2")[0].split("_3")[0]
        regime_risk = regime_risk_map.get(base_regime, 0.50)
        
        # Macro composite is [-1, 1] where 1 is low risk. Convert to [0, 1] scale.
        comp_score = macro_analysis.get("composite_score", 0.0)
        macro_risk = (1.0 - comp_score) / 2.0
        
        adj_conf = regime_result.get("adjusted_confidence", 0.0)
        
        effective_risk_state = (adj_conf * regime_risk) + ((1.0 - adj_conf) * macro_risk)
        regime_result["effective_risk_state"] = round(effective_risk_state, 4)

        # ═══════════════════════════════════════════════
        #  STEP 8-9: LLM REASONING + CONTEXT BUILD
        # ═══════════════════════════════════════════════
        logger.info("STEP 8-9/9 — Running LLM reasoning pipeline & building context...")
        step_start = time.time()

        try:
            if self.context_builder._llm.is_available():
                output = self.context_builder.build_context(
                    regime_result=regime_result,
                    volatility_state=volatility_state,
                    macro_analysis=macro_analysis,
                    risk_assessment=risk_assessment,
                    features=features,
                    vix_data=vix_data if isinstance(vix_data, dict) else
                             {"close": vix_data["close"]} if vix_data is not None and not vix_data.empty and "close" in vix_data.columns else None,
                    market_metadata=market_snapshot.get("metadata"),
                )
                self._log_step("llm_context_build", "success", time.time() - step_start)
            else:
                logger.warning("LLM provider not available — building context without AI reasoning")
                output = self._build_output_without_llm(
                    regime_result, volatility_state, macro_analysis,
                    risk_assessment, features, market_snapshot,
                    time.time() - start_time,
                )
                degraded_components.append("llm_reasoning")
                self._log_step("llm_context_build", "skipped", time.time() - step_start, "LLM unavailable")

        except Exception as e:
            logger.error(f"LLM context build failed: {e}")
            output = self._build_output_without_llm(
                regime_result, volatility_state, macro_analysis,
                risk_assessment, features, market_snapshot,
                time.time() - start_time,
            )
            degraded_components.append("llm_reasoning")
            self._log_step("llm_context_build", "failed", time.time() - step_start, str(e))

        # ═══════════════════════════════════════════════
        #  FINAL: VALIDATION & METADATA
        # ═══════════════════════════════════════════════
        total_time = time.time() - start_time

        # Add degradation info
        if degraded_components:
            output["degraded_components"] = degraded_components
            output["agent_metadata"]["data_quality"] = "degraded"
            logger.warning(f"Pipeline completed with degradations: {degraded_components}")
        else:
            output["agent_metadata"]["data_quality"] = "full"

        output["agent_metadata"]["execution_time_ms"] = round(total_time * 1000)
        output["agent_metadata"]["execution_log"] = self._execution_log

        # Validate against schema
        is_valid, error = validate_output(output)
        if is_valid:
            logger.info("[OK] Output validates against Agent1Output schema")
        else:
            logger.warning(f"[WARN] Output validation warning: {error}")

        logger.info("")
        logger.info("+" + "=" * 60 + "+")
        logger.info(f"|  AGENT 1 COMPLETE -- {total_time:.1f}s total execution time" + " " * max(0, 21 - len(f"{total_time:.1f}")) + "|")
        logger.info(f"|  Regime: {regime_result.get('primary_regime', 'N/A'):<49s}|")
        logger.info(f"|  Risk: {risk_assessment.get('risk_category', 'N/A'):<51s}|")
        logger.info("+" + "=" * 60 + "+")

        return output

    # ─────────────────────────────────────────────────
    #  HELPER METHODS
    # ─────────────────────────────────────────────────

    def _get_benchmark_prices(self, snapshot: dict) -> Optional[pd.Series]:
        """Extract benchmark close prices (SPY preferred)."""
        equities = snapshot.get("equities", {})
        if "SPY" in equities and not equities["SPY"].empty:
            return equities["SPY"]["close"]

        # Fallback to first available equity
        for symbol, df in equities.items():
            if not df.empty and "close" in df.columns:
                logger.info(f"Using {symbol} as benchmark (SPY unavailable)")
                return df["close"]

        # Fallback to crypto benchmark
        crypto = snapshot.get("crypto", {})
        benchmark = "BTCUSDT"
        if benchmark in crypto and not crypto[benchmark].empty:
            logger.info(f"Using {benchmark} as benchmark (no equity data)")
            return crypto[benchmark]["close"]

        return None

    def _build_cross_asset_closes(self, snapshot: dict) -> pd.DataFrame:
        """Build combined close price DataFrame for correlation analysis."""
        closes = {}

        # Sample key assets from each class
        priority_equities = ["SPY", "QQQ", "IWM", "XLE", "XLF", "XLK", "XLU"]
        for sym in priority_equities:
            eq = snapshot.get("equities", {}).get(sym)
            if eq is not None and not eq.empty and "close" in eq.columns:
                closes[sym] = eq["close"]

        priority_etfs = ["TLT", "GLD", "HYG", "VNQ"]
        for sym in priority_etfs:
            etf = snapshot.get("etfs", {}).get(sym)
            if etf is not None and not etf.empty and "close" in etf.columns:
                closes[sym] = etf["close"]

        priority_crypto = ["BTCUSDT", "ETHUSDT"]
        for sym in priority_crypto:
            cr = snapshot.get("crypto", {}).get(sym)
            if cr is not None and not cr.empty and "close" in cr.columns:
                closes[sym] = cr["close"]

        if not closes:
            return pd.DataFrame()

        df = pd.DataFrame(closes)
        # Forward-fill minor gaps, then drop remaining NaN
        df = df.ffill().dropna()
        return df

    def _build_output_without_llm(
        self,
        regime_result: dict,
        volatility_state: dict,
        macro_analysis: dict,
        risk_assessment: dict,
        features: dict,
        market_snapshot: dict,
        elapsed_time: float,
    ) -> dict:
        """Build output when LLM is unavailable (pure quantitative output)."""
        correlations = features.get("correlations", {})

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "data_freshness": market_snapshot.get("metadata", {}).get("timestamp", "unknown"),
            "market_regime": {
                "primary_regime": regime_result.get("primary_regime", "unknown"),
                "confidence": regime_result.get("confidence", 0.0),
                "hmm_regime": regime_result.get("hmm_result", {}).get("current_regime", "unknown"),
                "rf_regime": regime_result.get("rf_result", {}).get("current_regime", "unknown"),
                "models_agree": regime_result.get("models_agree", False),
                "regime_duration_days": regime_result.get("regime_duration_days", 0),
                "transition_probability": regime_result.get("transition_probability", 0.0),
                "observations_count": regime_result.get("observations_count", 0),
                "convergence_warning": regime_result.get("convergence_warning", False),
                "adjusted_confidence": regime_result.get("adjusted_confidence", 0.0),
                "effective_risk_state": regime_result.get("effective_risk_state", 0.5),
                "description": regime_result.get("description", ""),
            },
            "volatility_state": {
                "current_state": volatility_state.get("current_state", "unknown"),
                "vix_level": volatility_state.get("vix_level"),
                "realized_vol_percentile": volatility_state.get("realized_vol_percentile", 50),
                "vol_trend": volatility_state.get("vol_trend", "unknown"),
                "vol_of_vol": volatility_state.get("vol_of_vol", "unknown"),
                "term_structure": volatility_state.get("term_structure", "unknown"),
            },
            "macro_environment": {
                "macro_regime": macro_analysis.get("macro_regime", "unknown"),
                "monetary_policy": macro_analysis.get("monetary_policy_state", "unknown"),
                "inflation_state": macro_analysis.get("inflation_state", "unknown"),
                "growth_state": macro_analysis.get("growth_state", "unknown"),
                "liquidity": macro_analysis.get("liquidity_state", "unknown"),
                "composite_score": macro_analysis.get("composite_score", 0.0),
                "key_indicators": macro_analysis.get("key_indicators", {}),
                "yield_curve": macro_analysis.get("yield_curve", {}),
            },
            "systemic_risk": {
                "overall_risk_level": risk_assessment.get("overall_risk_level", 0.0),
                "risk_category": risk_assessment.get("risk_category", "unknown"),
                "risk_signals": risk_assessment.get("risk_signals", {}),
                "risk_assessment": risk_assessment.get("risk_assessment", ""),
                "recommended_caution": risk_assessment.get("recommended_caution", False),
            },
            "cross_asset_analysis": {
                "correlation_state": "increasing" if correlations.get("median_correlation", 0) > 0.5 else "normal",
                "median_correlation": correlations.get("median_correlation", 0.0),
                "risk_appetite_index": max(0, min(1, 1 - risk_assessment.get("overall_risk_level", 0.5))),
                "key_correlations": {},
            },
            "llm_reasoning": {
                "market_narrative": "LLM analysis unavailable -- pure quantitative output.",
                "key_risks": macro_analysis.get("risk_factors", []),
                "opportunities": [],
                "asset_class_outlook": {},
                "sector_implications": {},
                "risk_budget_suggestion": {},
                "confidence_level": 0.0,
                "uncertainty_factors": ["LLM reasoning not available"],
            },
            "agent_metadata": {
                "agent_id": SystemMeta.AGENT_ID,
                "version": SystemMeta.VERSION,
                "execution_time_ms": round(elapsed_time * 1000),
                "llm_calls": 0,
                "llm_total_latency_ms": 0,
                "models_used": ["hmm_regime_v1", "rf_regime_v1", "vol_classifier_v1",
                                "macro_analyzer_v1", "risk_detector_v1"],
                "data_sources": market_snapshot.get("metadata", {}).get("data_sources", []),
            },
        }

    def _log_step(self, step: str, status: str, duration: float, error: str = None) -> None:
        """Log pipeline step execution."""
        entry = {
            "step": step,
            "status": status,
            "duration_ms": round(duration * 1000),
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        if error:
            entry["error"] = error
        self._execution_log.append(entry)

    def run_scenario(self, scenario_data: dict) -> dict:
        """
        Run the REAL ML pipeline against pre-built adversarial scenario data.
        
        This is NOT a mock. The HMM, Random Forest, VolatilityClassifier,
        and SystemicRiskDetector all run with genuine computation against
        the injected data. Only the API calls are bypassed.
        
        Args:
            scenario_data: dict from ScenarioDataFactory containing:
                - benchmark_prices: pd.Series
                - vix_series: pd.Series or None
                - cross_asset_closes: pd.DataFrame
                - macro_snapshot: dict
                - scenario_name: str
        
        Returns:
            Complete Agent 1 output JSON (validated against schema)
        """
        start_time = time.time()
        scenario_name = scenario_data.get("scenario_name", "unknown_scenario")
        logger.info(f"Running ADVERSARIAL SCENARIO: {scenario_name}")

        benchmark = scenario_data["benchmark_prices"]
        vix_series = scenario_data.get("vix_series")
        all_closes = scenario_data.get("cross_asset_closes", pd.DataFrame())
        macro_snapshot = scenario_data.get("macro_snapshot", {})

        degraded_components = []

        # ── STEP 3: Feature Engineering (REAL) ──────────────
        logger.info(f"[{scenario_name}] Running REAL feature engineering...")
        features = {}
        regime_features = pd.DataFrame()

        try:
            features = FeatureEngine.build_features(
                benchmark_prices=benchmark,
                all_close_prices=all_closes,
                vix_prices=vix_series,
            )
            regime_features = FeatureEngine.build_regime_features(
                benchmark_prices=benchmark,
                vix_prices=vix_series,
            )
        except Exception as e:
            logger.error(f"[{scenario_name}] Feature engineering failed: {e}")
            degraded_components.append("feature_engineering")

        # ── STEP 4: Regime Detection (REAL HMM + RF) ────────
        logger.info(f"[{scenario_name}] Running REAL regime detection (HMM + RF)...")
        regime_result = {"primary_regime": "unknown", "confidence": 0.0, "models_agree": False,
                         "ensemble_method": "failed", "regime_duration_days": 0, "transition_probability": 0.0,
                         "description": "", "hmm_result": {}, "rf_result": {},
                         "observations_count": 0, "convergence_warning": False, "adjusted_confidence": 0.0}
        try:
            if not regime_features.empty:
                regime_result = self.regime_detector.detect_regime(regime_features)
        except Exception as e:
            logger.error(f"[{scenario_name}] Regime detection failed: {e}")
            degraded_components.append("regime_detection")

        # ── STEP 5: Volatility Classification (REAL) ────────
        logger.info(f"[{scenario_name}] Running REAL volatility classification...")
        vix_df = None
        if vix_series is not None:
            vix_df = pd.DataFrame({"close": vix_series})
        volatility_state = {"current_state": "unknown", "vix_level": None,
                           "realized_vol_percentile": 50, "vol_trend": "unknown",
                           "vol_of_vol": "unknown", "term_structure": "unknown"}
        try:
            volatility_state = VolatilityClassifier.classify(features, vix_df)
        except Exception as e:
            logger.error(f"[{scenario_name}] Volatility classification failed: {e}")
            degraded_components.append("volatility_classification")

        # ── STEP 6: Macro Analysis ──────────────────────────
        logger.info(f"[{scenario_name}] Processing macro data...")
        macro_analysis = {"macro_regime": "unknown", "composite_score": 0.0, "confidence": 0.0,
                          "monetary_policy_state": "unknown", "inflation_state": "unknown",
                          "growth_state": "unknown", "liquidity_state": "unknown",
                          "key_indicators": {}, "yield_curve": {}, "risk_factors": []}
        if macro_snapshot:
            macro_analysis = macro_snapshot
        else:
            degraded_components.append("macro_analysis")

        # ── STEP 7: Systemic Risk Detection (REAL) ──────────
        logger.info(f"[{scenario_name}] Running REAL systemic risk detection...")
        risk_assessment = {"overall_risk_level": 0.0, "risk_category": "unknown",
                          "risk_signals": {}, "risk_assessment": "", "recommended_caution": False,
                          "confidence": 0.0}
        try:
            risk_assessment = SystemicRiskDetector.detect(features, macro_analysis, volatility_state)
        except Exception as e:
            logger.error(f"[{scenario_name}] Risk detection failed: {e}")
            degraded_components.append("risk_detection")

        # ── STEP 7b: Effective Risk State ───────────────────
        regime_risk_map = {
            "bull_low_vol": 0.20, "bull_high_vol": 0.40,
            "bear_low_vol": 0.60, "bear_high_vol": 0.90,
        }
        base_regime = regime_result.get("primary_regime", "unknown").split("_0")[0].split("_1")[0].split("_2")[0].split("_3")[0]
        regime_risk = regime_risk_map.get(base_regime, 0.50)
        comp_score = macro_analysis.get("composite_score", 0.0)
        macro_risk = (1.0 - comp_score) / 2.0
        adj_conf = regime_result.get("adjusted_confidence", 0.0)
        effective_risk_state = (adj_conf * regime_risk) + ((1.0 - adj_conf) * macro_risk)
        regime_result["effective_risk_state"] = round(effective_risk_state, 4)

        # ── Build Output ────────────────────────────────────
        total_time = time.time() - start_time
        output = self._build_output_without_llm(
            regime_result, volatility_state, macro_analysis,
            risk_assessment, features,
            {"metadata": {"timestamp": datetime.utcnow().isoformat() + "Z", "data_sources": [f"scenario_{scenario_name}"]}},
            total_time,
        )

        if degraded_components:
            output["degraded_components"] = degraded_components
            output["agent_metadata"]["data_quality"] = "degraded"
        else:
            output["agent_metadata"]["data_quality"] = "scenario"

        output["agent_metadata"]["data_sources"] = [f"adversarial_scenario_{scenario_name}"]

        # Validate
        is_valid, error = validate_output(output)
        if is_valid:
            logger.info(f"[{scenario_name}] Output validates against schema ✓")
        else:
            logger.warning(f"[{scenario_name}] Validation warning: {error}")

        return output

    def _run_mock_pipeline(self) -> dict:
        """Run pipeline with synthetic mock data for testing."""
        logger.info("Running in MOCK mode -- using synthetic data")
        import numpy as np

        # Generate synthetic benchmark prices
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=300, freq="B")
        prices = 400 * np.exp(np.cumsum(np.random.normal(0.0003, 0.012, len(dates))))
        benchmark = pd.Series(prices, index=dates, name="SPY_mock")

        # Generate VIX mock
        vix_prices = 18 + np.random.normal(0, 3, len(dates))
        vix_prices = np.maximum(10, vix_prices)
        vix_series = pd.Series(vix_prices, index=dates, name="VIX_mock")

        # Generate cross-asset prices
        assets = {}
        for name in ["QQQ", "TLT", "GLD", "BTCUSDT"]:
            noise = np.random.normal(0.0002, 0.015, len(dates))
            asset_prices = 100 * np.exp(np.cumsum(noise))
            assets[name] = pd.Series(asset_prices, index=dates)
        all_closes = pd.DataFrame({"SPY": benchmark, **assets})

        # Feature engineering
        logger.info("Mock: Building features...")
        features = FeatureEngine.build_features(
            benchmark_prices=benchmark,
            all_close_prices=all_closes,
            vix_prices=vix_series,
        )
        regime_features = FeatureEngine.build_regime_features(
            benchmark_prices=benchmark,
            vix_prices=vix_series,
        )

        # Regime detection
        logger.info("Mock: Running regime detection...")
        regime_result = self.regime_detector.detect_regime(regime_features)

        # Volatility classification
        logger.info("Mock: Classifying volatility...")
        vix_df = pd.DataFrame({"close": vix_series})
        volatility_state = VolatilityClassifier.classify(features, vix_df)

        # Mock macro analysis (since FRED requires API key)
        logger.info("Mock: Using synthetic macro data...")
        macro_analysis = {
            "macro_regime": "stable_growth",
            "monetary_policy_state": "tightening",
            "inflation_state": "above_target",
            "growth_state": "moderate_growth",
            "liquidity_state": "neutral",
            "composite_score": -0.1,
            "key_indicators": {
                "fed_funds_rate": 5.25,
                "treasury_10y": 4.35,
                "treasury_2y": 4.60,
                "unemployment": 3.9,
            },
            "yield_curve": {
                "inverted": True,
                "spreads": {"10y_2y": -0.25, "10y_3m": -0.15},
                "signal": "inverted_but_improving",
            },
            "risk_factors": [
                "Monetary policy tightening -- negative for risk assets",
                "Yield curve inverted (10Y-2Y: -0.25%) -- historic recession predictor",
            ],
            "confidence": 0.75,
        }

        # Risk detection
        logger.info("Mock: Detecting systemic risks...")
        risk_assessment = SystemicRiskDetector.detect(features, macro_analysis, volatility_state)

        # Build output without LLM in mock mode
        logger.info("Mock: Building output (without LLM)...")
        output = self._build_output_without_llm(
            regime_result, volatility_state, macro_analysis,
            risk_assessment, features,
            {"metadata": {"timestamp": datetime.utcnow().isoformat() + "Z", "data_sources": ["mock"]}},
            0,
        )

        output["agent_metadata"]["data_sources"] = ["mock_synthetic"]
        output["agent_metadata"]["data_quality"] = "mock"

        # Validate
        is_valid, error = validate_output(output)
        if is_valid:
            logger.info("[OK] Mock output validates against schema")
        else:
            logger.warning(f"[WARN] Mock validation warning: {error}")

        return output
