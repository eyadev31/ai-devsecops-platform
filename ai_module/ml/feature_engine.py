"""
Hybrid Intelligence Portfolio System — Feature Engineering Engine
==================================================================
Institutional-grade feature engineering for multi-asset market analysis.
Computes volatility, momentum, correlation, drawdown, and market breadth
features from raw OHLCV data across all asset classes.

All features are computed as pandas DataFrames/Series for downstream
consumption by ML models (regime detector, volatility classifier, etc.).
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

from config.settings import MLConfig

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Computes institutional-grade features from raw market data.

    Feature Categories:
      1. Volatility Features — realized vol, vol-of-vol, vol percentiles
      2. Momentum Features  — RSI, MACD, rate-of-change, SMA crossovers
      3. Correlation Features — rolling cross-asset correlation matrix
      4. Drawdown Features  — current drawdown, max drawdown, recovery
      5. Return Features    — log returns, cumulative returns, risk-adjusted
      6. Market Breadth     — sector dispersion, advance-decline proxy
    """

    # ═══════════════════════════════════════════════
    #  1. RETURN FEATURES
    # ═══════════════════════════════════════════════

    @staticmethod
    def compute_returns(prices: pd.Series) -> pd.Series:
        """Compute simple daily returns."""
        return prices.pct_change().dropna()

    @staticmethod
    def compute_log_returns(prices: pd.Series) -> pd.Series:
        """Compute log returns (preferred for statistical analysis)."""
        return np.log(prices / prices.shift(1)).dropna()

    @staticmethod
    def compute_cumulative_returns(prices: pd.Series, window: int = 21) -> pd.Series:
        """Compute rolling cumulative returns over window."""
        return prices.pct_change(window).dropna()

    # ═══════════════════════════════════════════════
    #  2. VOLATILITY FEATURES
    # ═══════════════════════════════════════════════

    @classmethod
    def compute_realized_volatility(
        cls,
        prices: pd.Series,
        windows: Optional[list[int]] = None,
        annualize: bool = True,
    ) -> pd.DataFrame:
        """
        Compute realized volatility across multiple windows.

        Args:
            prices: Close price series
            windows: Rolling window sizes (default: config)
            annualize: If True, annualize vol (×√252)

        Returns:
            DataFrame with columns like 'vol_5d', 'vol_21d', etc.
        """
        windows = windows or MLConfig.VOL_WINDOWS
        log_returns = cls.compute_log_returns(prices)
        factor = np.sqrt(252) if annualize else 1.0

        vol_df = pd.DataFrame(index=log_returns.index)
        for w in windows:
            vol_df[f"vol_{w}d"] = log_returns.rolling(window=w).std() * factor

        return vol_df.dropna()

    @classmethod
    def compute_vol_of_vol(cls, prices: pd.Series, vol_window: int = 21, vov_window: int = 63) -> pd.Series:
        """
        Compute volatility-of-volatility: the standard deviation of rolling vol.
        High vol-of-vol indicates regime instability.
        """
        log_returns = cls.compute_log_returns(prices)
        rolling_vol = log_returns.rolling(window=vol_window).std() * np.sqrt(252)
        return rolling_vol.rolling(window=vov_window).std().dropna()

    @classmethod
    def compute_vol_percentile(cls, prices: pd.Series, window: int = 21, lookback: int = 252) -> pd.Series:
        """
        Compute percentile rank of current volatility relative to history.
        Returns value between 0 and 100.
        """
        log_returns = cls.compute_log_returns(prices)
        rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)

        def percentile_rank(series):
            if len(series) < 2:
                return 50.0
            return stats.percentileofscore(series[:-1], series.iloc[-1])

        return rolling_vol.rolling(window=lookback).apply(percentile_rank, raw=False).dropna()

    @classmethod
    def compute_vol_zscore(cls, prices: pd.Series, window: int = 21, lookback: int = 252) -> pd.Series:
        """
        Compute z-score of current volatility relative to its own history.
        Z > 2 = extreme regime, Z < -1 = suppressed vol.
        """
        log_returns = cls.compute_log_returns(prices)
        rolling_vol = log_returns.rolling(window=window).std() * np.sqrt(252)

        vol_mean = rolling_vol.rolling(window=lookback).mean()
        vol_std = rolling_vol.rolling(window=lookback).std()

        zscore = (rolling_vol - vol_mean) / vol_std.replace(0, np.nan)
        return zscore.dropna()

    # ═══════════════════════════════════════════════
    #  3. MOMENTUM FEATURES
    # ═══════════════════════════════════════════════

    @staticmethod
    def compute_rsi(prices: pd.Series, period: int = MLConfig.RSI_PERIOD) -> pd.Series:
        """
        Compute Relative Strength Index (RSI).
        RSI < 30 = oversold, RSI > 70 = overbought.
        """
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period, min_periods=period).mean()
        avg_loss = loss.rolling(window=period, min_periods=period).mean()

        # Use exponential smoothing after initial period
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.dropna()

    @staticmethod
    def compute_macd(
        prices: pd.Series,
        fast: int = MLConfig.MACD_FAST,
        slow: int = MLConfig.MACD_SLOW,
        signal: int = MLConfig.MACD_SIGNAL,
    ) -> pd.DataFrame:
        """
        Compute MACD (Moving Average Convergence Divergence).
        Returns DataFrame with macd_line, signal_line, histogram.
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        result = pd.DataFrame({
            "macd_line": macd_line,
            "signal_line": signal_line,
            "histogram": histogram,
        })
        return result.dropna()

    @staticmethod
    def compute_rate_of_change(prices: pd.Series, windows: Optional[list[int]] = None) -> pd.DataFrame:
        """Compute rate-of-change (ROC) over multiple windows."""
        windows = windows or MLConfig.MOMENTUM_WINDOWS
        roc_df = pd.DataFrame(index=prices.index)
        for w in windows:
            roc_df[f"roc_{w}d"] = prices.pct_change(w) * 100
        return roc_df.dropna()

    @staticmethod
    def compute_sma_crossovers(prices: pd.Series) -> pd.DataFrame:
        """
        Compute SMA crossover signals.
        - price_vs_sma50: price relative to 50-day SMA (>1 = above)
        - price_vs_sma200: price relative to 200-day SMA
        - golden_cross: 50 SMA > 200 SMA (bullish)
        """
        sma50 = prices.rolling(50).mean()
        sma200 = prices.rolling(200).mean()

        result = pd.DataFrame({
            "price_vs_sma50": prices / sma50,
            "price_vs_sma200": prices / sma200,
            "golden_cross": (sma50 > sma200).astype(float),
            "sma50": sma50,
            "sma200": sma200,
        }, index=prices.index)

        return result.dropna()

    # ═══════════════════════════════════════════════
    #  4. CORRELATION FEATURES
    # ═══════════════════════════════════════════════

    @staticmethod
    def compute_correlation_matrix(
        close_prices: pd.DataFrame,
        window: int = MLConfig.CORRELATION_WINDOW,
    ) -> pd.DataFrame:
        """
        Compute current rolling correlation matrix from a DataFrame of close prices.
        Columns = different assets, rows = dates.

        Returns the most recent correlation matrix.
        """
        if close_prices.empty or close_prices.shape[1] < 2:
            return pd.DataFrame()

        returns = close_prices.pct_change().dropna()
        if len(returns) < window:
            return returns.corr()

        # Use last `window` days
        recent_returns = returns.iloc[-window:]
        return recent_returns.corr()

    @staticmethod
    def compute_rolling_correlation(
        series_a: pd.Series,
        series_b: pd.Series,
        window: int = MLConfig.CORRELATION_WINDOW,
    ) -> pd.Series:
        """Compute rolling pairwise correlation between two price series."""
        returns_a = series_a.pct_change().dropna()
        returns_b = series_b.pct_change().dropna()

        # Align indices
        aligned = pd.DataFrame({"a": returns_a, "b": returns_b}).dropna()
        if len(aligned) < window:
            return pd.Series(dtype=float)

        return aligned["a"].rolling(window=window).corr(aligned["b"]).dropna()

    @staticmethod
    def compute_median_correlation(corr_matrix: pd.DataFrame) -> float:
        """
        Compute median pairwise correlation from a correlation matrix.
        High median correlation = convergence = crisis-like conditions.
        """
        if corr_matrix.empty:
            return 0.0

        # Extract upper triangle (exclude diagonal)
        mask = np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        upper_values = corr_matrix.where(mask).stack().values

        if len(upper_values) == 0:
            return 0.0

        return float(np.median(upper_values))

    # ═══════════════════════════════════════════════
    #  5. DRAWDOWN FEATURES
    # ═══════════════════════════════════════════════

    @staticmethod
    def compute_drawdown(prices: pd.Series) -> pd.DataFrame:
        """
        Compute drawdown analysis:
          - current_drawdown: how far below peak (negative %)
          - max_drawdown: worst drawdown in period
          - peak: running peak price
          - recovery: whether price is recovering from drawdown
        """
        peak = prices.cummax()
        drawdown = (prices - peak) / peak

        result = pd.DataFrame({
            "drawdown": drawdown,
            "peak": peak,
            "is_at_peak": (prices == peak).astype(float),
        }, index=prices.index)

        return result

    @staticmethod
    def compute_max_drawdown(prices: pd.Series, window: int = 252) -> pd.Series:
        """Compute rolling maximum drawdown over a window."""
        def max_dd(window_prices):
            peak = window_prices.cummax()
            dd = (window_prices - peak) / peak
            return dd.min()

        return prices.rolling(window=window).apply(max_dd, raw=False).dropna()

    # ═══════════════════════════════════════════════
    #  6. MARKET BREADTH FEATURES
    # ═══════════════════════════════════════════════

    @staticmethod
    def compute_sector_dispersion(sector_returns: pd.DataFrame) -> pd.Series:
        """
        Compute cross-sector return dispersion: std of sector returns.
        High dispersion = rotation opportunity. Low dispersion = crisis convergence.
        """
        return sector_returns.std(axis=1)

    @staticmethod
    def compute_advance_decline_ratio(sector_returns: pd.DataFrame) -> pd.Series:
        """
        Compute advance/decline ratio from sector ETFs.
        > 1 = more sectors advancing. < 1 = more declining.
        """
        advancing = (sector_returns > 0).sum(axis=1)
        declining = (sector_returns < 0).sum(axis=1)
        return (advancing / declining.replace(0, 1)).fillna(1.0)

    # ═══════════════════════════════════════════════
    #  COMPREHENSIVE FEATURE BUILDER
    # ═══════════════════════════════════════════════

    @classmethod
    def build_features(
        cls,
        benchmark_prices: pd.Series,
        all_close_prices: Optional[pd.DataFrame] = None,
        vix_prices: Optional[pd.Series] = None,
    ) -> dict:
        """
        Build comprehensive feature set from benchmark prices.

        Args:
            benchmark_prices: Close prices of the primary benchmark (e.g., SPY)
            all_close_prices: DataFrame of close prices across assets (for correlation)
            vix_prices: VIX close prices (for implied vol features)

        Returns:
            Dict of feature category -> DataFrame/Series/scalar
        """
        logger.info("Building comprehensive feature set...")
        features = {}

        # --- Returns ---
        features["returns"] = {
            "daily": cls.compute_returns(benchmark_prices),
            "log_daily": cls.compute_log_returns(benchmark_prices),
        }

        # --- Volatility ---
        features["volatility"] = {
            "realized": cls.compute_realized_volatility(benchmark_prices),
            "vol_of_vol": cls.compute_vol_of_vol(benchmark_prices),
            "vol_percentile": cls.compute_vol_percentile(benchmark_prices),
            "vol_zscore": cls.compute_vol_zscore(benchmark_prices),
        }

        if vix_prices is not None and not vix_prices.empty:
            features["volatility"]["vix"] = vix_prices
            features["volatility"]["vix_zscore"] = cls.compute_vol_zscore(vix_prices, window=5, lookback=126)

        # --- Momentum ---
        features["momentum"] = {
            "rsi": cls.compute_rsi(benchmark_prices),
            "macd": cls.compute_macd(benchmark_prices),
            "roc": cls.compute_rate_of_change(benchmark_prices),
            "sma_crossovers": cls.compute_sma_crossovers(benchmark_prices),
        }

        # --- Drawdown ---
        features["drawdown"] = {
            "current": cls.compute_drawdown(benchmark_prices),
            "max_rolling": cls.compute_max_drawdown(benchmark_prices),
        }

        # --- Correlations ---
        if all_close_prices is not None and not all_close_prices.empty:
            features["correlations"] = {
                "matrix": cls.compute_correlation_matrix(all_close_prices),
                "median_correlation": cls.compute_median_correlation(
                    cls.compute_correlation_matrix(all_close_prices)
                ),
            }
        else:
            features["correlations"] = {
                "matrix": pd.DataFrame(),
                "median_correlation": 0.0,
            }

        logger.info("Feature computation complete")
        return features

    @classmethod
    def build_regime_features(
        cls,
        benchmark_prices: pd.Series,
        vix_prices: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Build the specific feature DataFrame consumed by the regime detection model.
        Each row = one date. Columns = features used for ML classification.

        Returns:
            DataFrame with columns suitable for HMM / Random Forest input.
        """
        log_returns = cls.compute_log_returns(benchmark_prices)
        vol_df = cls.compute_realized_volatility(benchmark_prices, windows=[5, 21, 63])
        rsi = cls.compute_rsi(benchmark_prices)
        roc = cls.compute_rate_of_change(benchmark_prices, windows=[5, 21])
        sma = cls.compute_sma_crossovers(benchmark_prices)

        regime_features = pd.DataFrame(index=benchmark_prices.index)
        regime_features["log_return"] = log_returns
        regime_features["return_5d"] = benchmark_prices.pct_change(5)
        regime_features["return_21d"] = benchmark_prices.pct_change(21)

        # Volatility features
        for col in vol_df.columns:
            regime_features[col] = vol_df[col]

        # Vol trend: ratio of short-term to long-term vol
        if "vol_5d" in vol_df.columns and "vol_63d" in vol_df.columns:
            regime_features["vol_ratio_5_63"] = vol_df["vol_5d"] / vol_df["vol_63d"].replace(0, np.nan)

        # Momentum features
        regime_features["rsi"] = rsi
        for col in roc.columns:
            regime_features[col] = roc[col]

        # Trend features
        if "price_vs_sma50" in sma.columns:
            regime_features["price_vs_sma50"] = sma["price_vs_sma50"]
        if "price_vs_sma200" in sma.columns:
            regime_features["price_vs_sma200"] = sma["price_vs_sma200"]
        if "golden_cross" in sma.columns:
            regime_features["golden_cross"] = sma["golden_cross"]

        # VIX features
        if vix_prices is not None and not vix_prices.empty:
            regime_features["vix"] = vix_prices
            regime_features["vix_change_5d"] = vix_prices.pct_change(5)

        # Drawdown
        dd = cls.compute_drawdown(benchmark_prices)
        regime_features["drawdown"] = dd["drawdown"]

        # Drop NaN rows
        regime_features = regime_features.dropna()
        logger.info(f"Regime features built: {regime_features.shape[0]} rows × {regime_features.shape[1]} columns")

        return regime_features
