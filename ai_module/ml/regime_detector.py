"""
Hybrid Intelligence Portfolio System — Market Regime Detection
================================================================
Dual-model ensemble regime detection combining:
  1. Hidden Markov Model (HMM) — unsupervised regime discovery
  2. Random Forest Classifier — supervised regime classification

The ensemble provides robust regime identification with confidence scoring.
When models agree, confidence is high. When they disagree, the system
flags uncertainty and uses a probability-weighted blend.

Detected Regimes:
  - bull_low_vol:  Uptrend with stable/low volatility
  - bull_high_vol: Uptrend with elevated volatility (late cycle)
  - bear_low_vol:  Downtrend with controlled selling
  - bear_high_vol: Crisis / panic selling / capitulation
"""

import logging
import warnings
from typing import Optional

import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

from config.settings import MLConfig

logger = logging.getLogger(__name__)

class RegimeAdjuster:
    """Calculates true statistical reliability of the market regime."""
    
    def __init__(self, lambda_decay: float = 0.012):
        self.lambda_decay = lambda_decay
        self.WARNING_PENALTY = 0.70
        
    def calculate_adjusted_confidence(self, raw_conf: float, n_obs: int, has_warning: bool) -> float:
        # Scale for sample size
        sample_scaling = 1.0 - math.exp(-self.lambda_decay * max(1, n_obs))
        
        # Apply solver penalties
        penalty = self.WARNING_PENALTY if has_warning else 1.0
        
        adjusted_conf = raw_conf * sample_scaling * penalty
        return float(round(max(0.0, min(1.0, adjusted_conf)), 4))

logger = logging.getLogger(__name__)

# Suppress convergence warnings from HMM
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*KMeans.*")

# Regime labels
REGIME_LABELS = {
    0: "bull_low_vol",
    1: "bull_high_vol",
    2: "bear_low_vol",
    3: "bear_high_vol",
}

REGIME_DESCRIPTIONS = {
    "bull_low_vol": "Bullish trend with low volatility — risk-on environment",
    "bull_high_vol": "Bullish with elevated volatility — late cycle or recovery",
    "bear_low_vol": "Bearish with controlled selling — orderly correction",
    "bear_high_vol": "Bearish crisis — panic selling or capitulation regime",
}


class HMMRegimeDetector:
    """
    Hidden Markov Model for unsupervised market regime detection.

    Uses return and volatility features to identify latent market states.
    The HMM discovers regimes without labels, making it adaptable to
    novel market conditions that haven't been seen before.
    """

    def __init__(
        self,
        n_regimes: int = MLConfig.HMM_N_REGIMES,
        n_iter: int = MLConfig.HMM_N_ITER,
        covariance_type: str = MLConfig.HMM_COVARIANCE_TYPE,
        random_state: int = MLConfig.HMM_RANDOM_STATE,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self._regime_mapping: dict[int, str] = {}

    def fit_predict(self, features: pd.DataFrame) -> dict:
        """
        Fit HMM on feature data and predict current regime.

        Args:
            features: DataFrame from FeatureEngine.build_regime_features()

        Returns:
            {
                "current_regime": str,
                "regime_sequence": list,
                "transition_matrix": list[list],
                "regime_probabilities": dict,
                "confidence": float,
                "regime_duration_days": int,
                "stationary_distribution": dict,
            }
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.error("hmmlearn not installed. Run: pip install hmmlearn")
            return self._empty_result()

        if features.empty or len(features) < 50:
            logger.warning(f"Insufficient data for HMM: {len(features)} rows (need >= 50)")
            return self._empty_result()

        # Select key features for HMM
        hmm_cols = self._select_hmm_features(features)
        if not hmm_cols:
            return self._empty_result()

        X = features[hmm_cols].dropna().values
        if len(X) < 50:
            return self._empty_result()

        # Normalize features
        X_scaled = self.scaler.fit_transform(X)

        # Fit HMM
        logger.info(f"Fitting HMM with {self.n_regimes} regimes on {X_scaled.shape[0]} observations...")
        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )

        convergence_warning = False
        try:
            self.model.fit(X_scaled)
            if hasattr(self.model, "monitor_") and hasattr(self.model.monitor_, "converged"):
                if not self.model.monitor_.converged:
                    convergence_warning = True
                    logger.warning("HMM failed to formally converge within n_iter")
        except Exception as e:
            logger.error(f"HMM fitting failed: {e}")
            return self._empty_result()

        # Predict hidden states
        hidden_states = self.model.predict(X_scaled)
        state_probs = self.model.predict_proba(X_scaled)

        # Map HMM states to meaningful regime labels
        self._map_regimes(features, hidden_states, hmm_cols)

        current_state = int(hidden_states[-1])
        current_regime = self._regime_mapping.get(current_state, f"regime_{current_state}")

        # Compute regime duration
        duration = 1
        for i in range(len(hidden_states) - 2, -1, -1):
            if hidden_states[i] == current_state:
                duration += 1
            else:
                break

        # Transition matrix
        trans_matrix = self.model.transmat_.tolist()

        # Current regime probabilities
        current_probs = state_probs[-1]
        regime_probs = {}
        for state_idx, prob in enumerate(current_probs):
            label = self._regime_mapping.get(state_idx, f"regime_{state_idx}")
            regime_probs[label] = round(float(prob), 4)

        # Confidence = probability of predicted state
        confidence = float(current_probs[current_state])

        # Stationary distribution
        stationary = {}
        try:
            eigenvalues, eigenvectors = np.linalg.eig(self.model.transmat_.T)
            stationary_idx = np.argmin(np.abs(eigenvalues - 1))
            stationary_dist = np.real(eigenvectors[:, stationary_idx])
            stationary_dist = stationary_dist / stationary_dist.sum()
            for state_idx, prob in enumerate(stationary_dist):
                label = self._regime_mapping.get(state_idx, f"regime_{state_idx}")
                stationary[label] = round(float(prob), 4)
        except Exception:
            stationary = {}

        result = {
            "current_regime": current_regime,
            "regime_sequence": [self._regime_mapping.get(int(s), f"regime_{s}") for s in hidden_states[-30:]],
            "transition_matrix": [[round(p, 4) for p in row] for row in trans_matrix],
            "regime_probabilities": regime_probs,
            "confidence": round(confidence, 4),
            "regime_duration_days": duration,
            "stationary_distribution": stationary,
            "observations_count": len(X_scaled),
            "convergence_warning": convergence_warning,
        }

        logger.info(f"HMM regime: {current_regime} (confidence: {confidence:.2%}, duration: {duration}d)")
        return result

    def _select_hmm_features(self, features: pd.DataFrame) -> list[str]:
        """Select the most informative features for HMM."""
        priority = [
            "log_return", "vol_21d", "vol_5d", "return_21d",
            "rsi", "drawdown", "vol_ratio_5_63", "vix",
        ]
        available = [col for col in priority if col in features.columns]
        if not available:
            available = list(features.columns[:6])
        return available

    def _map_regimes(self, features: pd.DataFrame, states: np.ndarray, hmm_cols: list[str]) -> None:
        """
        Map HMM state indices to meaningful regime labels based on
        average return and volatility within each state.
        """
        temp_df = features[hmm_cols].dropna().copy()
        temp_df = temp_df.iloc[:len(states)]
        temp_df["state"] = states

        state_stats = {}
        for state in range(self.n_regimes):
            mask = temp_df["state"] == state
            state_data = temp_df[mask]

            avg_return = state_data["log_return"].mean() if "log_return" in state_data.columns else 0
            avg_vol = state_data["vol_21d"].mean() if "vol_21d" in state_data.columns else (
                state_data["vol_5d"].mean() if "vol_5d" in state_data.columns else 0
            )
            state_stats[state] = {"return": avg_return, "vol": avg_vol}

        # Sort by return (bull vs bear) and vol (low vs high)
        median_return = np.median([s["return"] for s in state_stats.values()])
        median_vol = np.median([s["vol"] for s in state_stats.values()])

        self._regime_mapping = {}
        for state, stat in state_stats.items():
            is_bull = stat["return"] >= median_return
            is_high_vol = stat["vol"] >= median_vol

            if is_bull and not is_high_vol:
                label = "bull_low_vol"
            elif is_bull and is_high_vol:
                label = "bull_high_vol"
            elif not is_bull and not is_high_vol:
                label = "bear_low_vol"
            else:
                label = "bear_high_vol"

            # Avoid duplicates
            used = set(self._regime_mapping.values())
            if label in used:
                # Append state index to disambiguate
                label = f"{label}_{state}"
            self._regime_mapping[state] = label

    def _empty_result(self) -> dict:
        return {
            "current_regime": "unknown",
            "regime_sequence": [],
            "transition_matrix": [],
            "regime_probabilities": {},
            "confidence": 0.0,
            "regime_duration_days": 0,
            "stationary_distribution": {},
            "observations_count": 0,
            "convergence_warning": False,
        }


class RFRegimeClassifier:
    """
    Random Forest Classifier for supervised market regime prediction.

    Uses engineered features to classify the current market state.
    Self-labeled: generates training labels from quantile-based rules
    on return and volatility (since we don't have external labels).

    This provides an alternative perspective to the unsupervised HMM,
    and the ensemble of both models yields more robust predictions.
    """

    def __init__(
        self,
        n_estimators: int = MLConfig.RF_N_ESTIMATORS,
        max_depth: int = MLConfig.RF_MAX_DEPTH,
        min_samples_split: int = MLConfig.RF_MIN_SAMPLES_SPLIT,
        random_state: int = MLConfig.RF_RANDOM_STATE,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.scaler = StandardScaler()
        self._is_fitted = False
        self._feature_importances: Optional[pd.Series] = None

    def _generate_labels(self, features: pd.DataFrame) -> pd.Series:
        """
        Generate regime labels from data using quantile-based rules.
        This is the 'self-supervised' component.

        Logic:
          - Bull = return_21d > median
          - Bear = return_21d <= median
          - Low vol = vol_21d < 75th percentile
          - High vol = vol_21d >= 75th percentile
        """
        labels = pd.Series(index=features.index, dtype=str)

        ret_col = "return_21d" if "return_21d" in features.columns else "log_return"
        vol_col = "vol_21d" if "vol_21d" in features.columns else "vol_5d"

        if ret_col not in features.columns or vol_col not in features.columns:
            logger.error(f"Cannot generate labels: missing {ret_col} or {vol_col}")
            return labels

        ret_median = features[ret_col].median()
        vol_75 = features[vol_col].quantile(0.75)

        for idx in features.index:
            is_bull = features.loc[idx, ret_col] > ret_median
            is_high_vol = features.loc[idx, vol_col] >= vol_75

            if is_bull and not is_high_vol:
                labels.loc[idx] = "bull_low_vol"
            elif is_bull and is_high_vol:
                labels.loc[idx] = "bull_high_vol"
            elif not is_bull and not is_high_vol:
                labels.loc[idx] = "bear_low_vol"
            else:
                labels.loc[idx] = "bear_high_vol"

        return labels

    def fit_predict(self, features: pd.DataFrame) -> dict:
        """
        Train RF classifier on historical features and predict current regime.

        Uses time-series aware split to avoid look-ahead bias.

        Returns:
            {
                "current_regime": str,
                "regime_probabilities": dict,
                "confidence": float,
                "feature_importances": dict (top 10),
                "cv_accuracy": float,
            }
        """
        if features.empty or len(features) < 100:
            logger.warning(f"Insufficient data for RF: {len(features)} rows (need >= 100)")
            return self._empty_result()

        # Select feature columns (exclude non-numeric)
        feature_cols = features.select_dtypes(include=[np.number]).columns.tolist()
        X = features[feature_cols].dropna()

        if len(X) < 100:
            return self._empty_result()

        # Generate labels
        labels = self._generate_labels(X)
        valid_mask = labels != ""
        X = X[valid_mask]
        y = labels[valid_mask]

        if len(X) < 100 or y.nunique() < 2:
            return self._empty_result()

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            index=X.index,
            columns=X.columns,
        )

        # Time-series cross-validation
        cv_scores = []
        tscv = TimeSeriesSplit(n_splits=3)
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_scaled.iloc[test_idx]
            y_test = y.iloc[test_idx]

            if y_train.nunique() < 2:
                continue

            self.model.fit(X_train, y_train)
            score = self.model.score(X_test, y_test)
            cv_scores.append(score)

        cv_accuracy = float(np.mean(cv_scores)) if cv_scores else 0.0

        # Final fit on all data
        self.model.fit(X_scaled, y)
        self._is_fitted = True

        # Feature importances
        importances = pd.Series(
            self.model.feature_importances_,
            index=feature_cols,
        ).sort_values(ascending=False)
        self._feature_importances = importances

        # Predict current regime
        current_features = X_scaled.iloc[[-1]]
        predicted_regime = self.model.predict(current_features)[0]
        probabilities = self.model.predict_proba(current_features)[0]
        class_labels = self.model.classes_

        regime_probs = {}
        for label, prob in zip(class_labels, probabilities):
            regime_probs[label] = round(float(prob), 4)

        confidence = float(max(probabilities))

        result = {
            "current_regime": predicted_regime,
            "regime_probabilities": regime_probs,
            "confidence": round(confidence, 4),
            "feature_importances": {k: round(float(v), 4) for k, v in importances.head(10).items()},
            "cv_accuracy": round(cv_accuracy, 4),
        }

        logger.info(f"RF regime: {predicted_regime} (confidence: {confidence:.2%}, CV accuracy: {cv_accuracy:.2%})")
        return result

    def _empty_result(self) -> dict:
        return {
            "current_regime": "unknown",
            "regime_probabilities": {},
            "confidence": 0.0,
            "feature_importances": {},
            "cv_accuracy": 0.0,
        }


class EnsembleRegimeDetector:
    """
    Ensemble regime detector combining HMM and Random Forest.

    Decision Logic:
      - Both agree → high confidence output
      - Disagree → probability-weighted blend + uncertainty flag
      - One model fails → fall back to the working model
      - Both fail → return 'unknown' with confidence = 0

    This ensemble approach provides institutional-grade robustness:
    two independent analytical perspectives that cross-validate each other.
    """

    def __init__(self):
        self.hmm = HMMRegimeDetector()
        self.rf = RFRegimeClassifier()

    def detect_regime(self, features: pd.DataFrame) -> dict:
        """
        Run full ensemble regime detection.

        Args:
            features: DataFrame from FeatureEngine.build_regime_features()

        Returns:
            {
                "primary_regime": str,
                "confidence": float,
                "hmm_result": dict,
                "rf_result": dict,
                "models_agree": bool,
                "ensemble_method": str,
                "regime_duration_days": int,
                "transition_probability": float,
                "description": str,
            }
        """
        logger.info("═" * 60)
        logger.info("ENSEMBLE REGIME DETECTION — Running dual-model analysis")
        logger.info("═" * 60)

        # Run both models
        hmm_result = self.hmm.fit_predict(features)
        rf_result = self.rf.fit_predict(features)

        hmm_regime = hmm_result.get("current_regime", "unknown")
        rf_regime = rf_result.get("current_regime", "unknown")
        hmm_conf = hmm_result.get("confidence", 0.0)
        rf_conf = rf_result.get("confidence", 0.0)

        # ── Ensemble Logic ───────────────────────────────
        if hmm_regime == "unknown" and rf_regime == "unknown":
            # Both failed
            primary_regime = "unknown"
            confidence = 0.0
            method = "both_models_failed"
            models_agree = False

        elif hmm_regime == "unknown":
            # Only RF available
            primary_regime = rf_regime
            confidence = rf_conf * 0.8  # Discount for single model
            method = "rf_only"
            models_agree = False

        elif rf_regime == "unknown":
            # Only HMM available
            primary_regime = hmm_regime
            confidence = hmm_conf * 0.8
            method = "hmm_only"
            models_agree = False

        elif hmm_regime == rf_regime:
            # Both agree — high confidence
            primary_regime = hmm_regime
            confidence = min(0.98, (hmm_conf + rf_conf) / 2 + 0.1)  # Bonus for agreement
            method = "full_agreement"
            models_agree = True

        else:
            # Disagreement — use probability-weighted blend
            models_agree = False

            # Compare base regime type (bull vs bear)
            hmm_is_bull = "bull" in hmm_regime
            rf_is_bull = "bull" in rf_regime

            if hmm_is_bull == rf_is_bull:
                # Same direction, different vol regime
                method = "partial_agreement_direction"
                # Use higher-confidence model
                if hmm_conf >= rf_conf:
                    primary_regime = hmm_regime
                    confidence = (hmm_conf * 0.6 + rf_conf * 0.4)
                else:
                    primary_regime = rf_regime
                    confidence = (rf_conf * 0.6 + hmm_conf * 0.4)
            else:
                # Complete disagreement on direction
                method = "disagreement"
                if hmm_conf >= rf_conf:
                    primary_regime = hmm_regime
                else:
                    primary_regime = rf_regime
                confidence = max(hmm_conf, rf_conf) * 0.65  # Heavy discount

        # ── Transition Probability ───────────────────────
        transition_prob = 0.0
        if hmm_result.get("transition_matrix") and hmm_result["current_regime"] != "unknown":
            trans_matrix = hmm_result["transition_matrix"]
            regime_idx = list(self.hmm._regime_mapping.values()).index(hmm_regime) if hmm_regime in self.hmm._regime_mapping.values() else 0
            if regime_idx < len(trans_matrix):
                # Probability of leaving current regime
                stay_prob = trans_matrix[regime_idx][regime_idx] if regime_idx < len(trans_matrix[regime_idx]) else 0.5
                transition_prob = round(1.0 - stay_prob, 4)

        # Get description
        # Clean regime name for description lookup
        base_regime = primary_regime.split("_0")[0].split("_1")[0].split("_2")[0].split("_3")[0]
        description = REGIME_DESCRIPTIONS.get(base_regime, f"Market in {primary_regime} regime")

        # Compute adjusted confidence based on statistical robustness (Sample size & Convergence)
        n_obs = hmm_result.get("observations_count", 0)
        has_warning = hmm_result.get("convergence_warning", False)
        
        adjuster = RegimeAdjuster()
        adj_confidence = adjuster.calculate_adjusted_confidence(
            raw_conf=confidence, 
            n_obs=n_obs, 
            has_warning=has_warning
        )

        result = {
            "primary_regime": primary_regime,
            "confidence": round(confidence, 4),
            "hmm_result": hmm_result,
            "rf_result": rf_result,
            "models_agree": models_agree,
            "ensemble_method": method,
            "regime_duration_days": hmm_result.get("regime_duration_days", 0),
            "transition_probability": transition_prob,
            "description": description,
            "observations_count": n_obs,
            "convergence_warning": has_warning,
            "adjusted_confidence": adj_confidence,
        }

        logger.info(f"Ensemble regime: {primary_regime}")
        logger.info(f"  HMM: {hmm_regime} ({hmm_conf:.2%}) | RF: {rf_regime} ({rf_conf:.2%})")
        logger.info(f"  Agreement: {models_agree} | Method: {method} | Final confidence: {confidence:.2%}")

        return result
