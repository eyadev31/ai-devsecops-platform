"""
Hybrid Intelligence Portfolio System — Configuration
=====================================================
Centralized configuration management with environment variable support,
validation, and sensible defaults for all system components.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────
# Load environment variables from .env file
# ──────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_PATH)


# ══════════════════════════════════════════════════════
#  DATABASE CONFIGURATION
# ══════════════════════════════════════════════════════
class DatabaseConfig:
    """
    Database connection configuration.

    Production (AWS RDS PostgreSQL):
      DATABASE_URL=postgresql://admin:password@your-rds.amazonaws.com:5432/portfolio_db

    Local development (SQLite):
      DATABASE_URL=sqlite:///portfolio_system.db
    """

    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///portfolio_system.db")
    POOL_SIZE: int = int(os.getenv("DB_POOL_SIZE", "10"))
    MAX_OVERFLOW: int = int(os.getenv("DB_MAX_OVERFLOW", "20"))
    POOL_TIMEOUT: int = int(os.getenv("DB_POOL_TIMEOUT", "30"))
    ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"



# ══════════════════════════════════════════════════════
#  API KEYS
# ══════════════════════════════════════════════════════
class APIKeys:
    """Centralized API key management with validation."""

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    TWELVEDATA_API_KEY: str = os.getenv("TWELVEDATA_API_KEY", "")

    @classmethod
    def validate(cls) -> dict[str, bool]:
        """Check which API keys are configured."""
        return {
            "gemini": bool(cls.GEMINI_API_KEY and cls.GEMINI_API_KEY != "your_gemini_api_key_here"),
            "groq": bool(cls.GROQ_API_KEY and cls.GROQ_API_KEY != "your_groq_api_key_here"),
            "fred": bool(cls.FRED_API_KEY and cls.FRED_API_KEY != "your_fred_api_key_here"),
            "twelvedata": bool(cls.TWELVEDATA_API_KEY and cls.TWELVEDATA_API_KEY != "your_twelvedata_api_key_here"),
        }


# ══════════════════════════════════════════════════════
#  LLM CONFIGURATION
# ══════════════════════════════════════════════════════
class LLMConfig:
    """LLM provider configuration — abstracted for future migration."""

    PROVIDER: str = os.getenv("LLM_PROVIDER", "gemini")
    
    # Dynamically select the best default model depending on the provider
    _default_model = "llama-3.3-70b-versatile" if PROVIDER == "groq" else "gemini-1.5-pro"
    MODEL: str = os.getenv("LLM_MODEL", _default_model)
    TEMPERATURE: float = 0.1          # Low temperature for deterministic analytical output
    MAX_OUTPUT_TOKENS: int = 4096
    TOP_P: float = 0.8
    TIMEOUT_SECONDS: int = 60
    MAX_RETRIES: int = 3
    RETRY_BACKOFF_BASE: float = 2.0   # Exponential backoff base


# ══════════════════════════════════════════════════════
#  MARKET DATA CONFIGURATION
# ══════════════════════════════════════════════════════
class MarketDataConfig:
    """Market data source configuration."""

    # --- Binance (Crypto) ---
    BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
    BINANCE_KLINES_LIMIT: int = 500
    CRYPTO_SYMBOLS: list[str] = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
        "ADAUSDT", "AVAXUSDT", "DOTUSDT", "MATICUSDT", "LINKUSDT"
    ]
    CRYPTO_BENCHMARK: str = "BTCUSDT"

    # --- TwelveData (Equities / ETFs / Forex) ---
    TWELVEDATA_BASE_URL: str = "https://api.twelvedata.com"
    EQUITY_SYMBOLS: list[str] = [
        "SPY",    # S&P 500 ETF
        "QQQ",    # NASDAQ 100 ETF
        "IWM",    # Russell 2000 ETF
        "DIA",    # Dow Jones ETF
        "AAPL",   # Apple (tech bellwether)
        "MSFT",   # Microsoft
        "NVDA",   # NVIDIA (AI proxy)
        "JPM",    # JPMorgan (financials)
        "XLE",    # Energy Sector ETF
        "XLF",    # Financial Sector ETF
        "XLK",    # Technology Sector ETF
        "XLV",    # Healthcare Sector ETF
        "XLU",    # Utilities Sector ETF (defensive)
        "XLP",    # Consumer Staples ETF (defensive)
    ]
    ETF_SYMBOLS: list[str] = [
        "VTI",    # Total US Market
        "VXUS",   # International ex-US
        "BND",    # Total Bond Market
        "GLD",    # Gold
        "SLV",    # Silver
        "USO",    # Oil
        "TLT",    # 20+ Year Treasury
        "HYG",    # High Yield Corporate Bond
        "LQD",    # Investment Grade Corporate Bond
        "VNQ",    # Real Estate
    ]
    FOREX_PAIRS: list[str] = [
        "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF",
        "AUD/USD", "USD/CAD", "NZD/USD"
    ]
    COMMODITY_SYMBOLS: list[str] = ["GLD", "SLV", "USO"]  # via ETF proxies

    # --- VIX & Volatility ---
    VIX_SYMBOL: str = "VIX"

    # --- Data Parameters ---
    LOOKBACK_DAYS: int = 252         # 1 year of trading days
    CACHE_TTL_MINUTES: int = int(os.getenv("DATA_CACHE_TTL_MINUTES", "15"))


# ══════════════════════════════════════════════════════
#  MACRO DATA CONFIGURATION (FRED)
# ══════════════════════════════════════════════════════
class MacroConfig:
    """FRED macroeconomic data series configuration."""

    SERIES: dict[str, str] = {
        # Monetary Policy
        "fed_funds_rate": "FEDFUNDS",
        "treasury_10y": "DGS10",
        "treasury_2y": "DGS2",
        "treasury_3m": "DGS3MO",
        # Inflation
        "cpi_yoy": "CPIAUCSL",
        "core_pce": "PCEPILFE",
        "breakeven_5y": "T5YIE",
        # Labor Market
        "unemployment": "UNRATE",
        "nonfarm_payrolls": "PAYEMS",
        "initial_claims": "ICSA",
        # Growth
        "gdp_growth": "A191RL1Q225SBEA",
        "industrial_production": "INDPRO",
        "retail_sales": "RSAFS",
        # Liquidity & Credit
        "m2_money_supply": "M2SL",
        "credit_spread_hy": "BAMLH0A0HYM2",     # ICE BofA High Yield Spread
        "credit_spread_ig": "BAMLC0A0CM",         # ICE BofA IG Spread
        # Sentiment
        "consumer_sentiment": "UMCSENT",
        "leading_index": "USSLIND",
    }
    LOOKBACK_YEARS: int = 3


# ══════════════════════════════════════════════════════
#  ML MODEL CONFIGURATION
# ══════════════════════════════════════════════════════
class MLConfig:
    """Machine learning model hyperparameters."""

    # --- Hidden Markov Model ---
    HMM_N_REGIMES: int = 4                # bull_low_vol, bull_high_vol, bear_low_vol, bear_high_vol
    HMM_N_ITER: int = 100
    HMM_COVARIANCE_TYPE: str = "full"
    HMM_RANDOM_STATE: int = 42

    # --- Random Forest Regime Classifier ---
    RF_N_ESTIMATORS: int = 200
    RF_MAX_DEPTH: int = 10
    RF_MIN_SAMPLES_SPLIT: int = 10
    RF_RANDOM_STATE: int = 42

    # --- Volatility Classifier ---
    VOL_WINDOWS: list[int] = [5, 10, 21, 63]     # Trading days
    VOL_ZSCORE_THRESHOLDS: dict[str, float] = {
        "extremely_low": -2.0,
        "low": -1.0,
        "normal_low": -0.5,
        "normal_high": 0.5,
        "elevated": 1.0,
        "extreme": 2.0,
    }

    # --- Feature Engineering ---
    ROLLING_WINDOWS: list[int] = [5, 10, 21, 63, 126, 252]
    CORRELATION_WINDOW: int = 63     # ~3 months rolling correlation
    MOMENTUM_WINDOWS: list[int] = [5, 10, 21, 63]
    RSI_PERIOD: int = 14
    MACD_FAST: int = 12
    MACD_SLOW: int = 26
    MACD_SIGNAL: int = 9


# ══════════════════════════════════════════════════════
#  RISK DETECTION CONFIGURATION
# ══════════════════════════════════════════════════════
class RiskConfig:
    """Systemic risk detection thresholds."""

    # Correlation convergence — when median cross-asset correlation exceeds this
    CORRELATION_CONVERGENCE_THRESHOLD: float = 0.7

    # Volatility regime break — z-score threshold for sudden vol spikes
    VOL_BREAK_ZSCORE: float = 2.5

    # Yield curve inversion threshold (10Y - 2Y spread, in percentage points)
    YIELD_CURVE_INVERSION_THRESHOLD: float = 0.0

    # Credit stress — high-yield spread threshold (basis points)
    CREDIT_STRESS_THRESHOLD_BPS: float = 500.0

    # Overall risk level thresholds
    RISK_LEVEL_LOW: float = 0.3
    RISK_LEVEL_MODERATE: float = 0.5
    RISK_LEVEL_ELEVATED: float = 0.7
    RISK_LEVEL_CRITICAL: float = 0.85

    # Confidence threshold — below this, agent explicitly flags low confidence
    CONFIDENCE_THRESHOLD: float = 0.6


# ══════════════════════════════════════════════════════
#  NEWS SENTIMENT AGENT CONFIGURATION
# ══════════════════════════════════════════════════════
class NewsConfig:
    """
    Agent 5 — News Sentiment Intelligence configuration.
    Multi-source financial news ingestion, NLP processing, and signal generation.
    """

    # --- API Keys ---
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    CRYPTOPANIC_API_KEY: str = os.getenv("CRYPTOPANIC_API_KEY", "")

    # --- RSS Feed Sources ---
    RSS_FEEDS: dict[str, str] = {
        "reuters_markets": "https://www.reutersagency.com/feed/?best-topics=business-finance",
        "cnbc_finance": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "ft_markets": "https://www.ft.com/markets?format=rss",
        "yahoo_finance": "https://finance.yahoo.com/news/rssindex",
        "marketwatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
        "investing_com": "https://www.investing.com/rss/news.rss",
    }

    # --- API Endpoints ---
    NEWS_API_BASE_URL: str = "https://newsapi.org/v2"
    GDELT_API_URL: str = "https://api.gdeltproject.org/api/v2/doc/doc"
    CRYPTOPANIC_API_URL: str = "https://cryptopanic.com/api/v1/posts/"

    # --- Financial Search Keywords ---
    FINANCIAL_KEYWORDS: list[str] = [
        "federal reserve", "interest rate", "inflation", "GDP", "unemployment",
        "stock market", "S&P 500", "nasdaq", "dow jones", "bitcoin", "ethereum",
        "crypto", "ETF", "bond", "treasury", "yield curve", "recession",
        "earnings", "IPO", "merger", "acquisition", "regulation", "sanctions",
        "oil", "gold", "commodity", "forex", "currency", "bank", "central bank",
        "monetary policy", "fiscal policy", "trade war", "tariff", "geopolitical",
    ]

    # --- Source Reputation Weights ---
    SOURCE_WEIGHTS: dict[str, float] = {
        # Tier 1 — Premier institutional sources
        "reuters": 1.00,
        "bloomberg": 1.00,
        "financial times": 0.95,
        "wall street journal": 0.95,
        # Tier 2 — Major financial media
        "cnbc": 0.85,
        "marketwatch": 0.80,
        "yahoo finance": 0.80,
        "investing.com": 0.75,
        # Tier 3 — Aggregators & specialized
        "coindesk": 0.75,
        "cointelegraph": 0.70,
        "cryptopanic": 0.70,
        "seeking alpha": 0.65,
        # Tier 4 — General media & blogs
        "general": 0.50,
        "blog": 0.30,
        "unknown": 0.40,
    }

    # --- Topic Importance Weights ---
    TOPIC_WEIGHTS: dict[str, float] = {
        "interest_rates": 1.00,
        "monetary_policy": 1.00,
        "regulation": 0.95,
        "geopolitical": 0.90,
        "inflation": 0.90,
        "recession": 0.90,
        "earnings": 0.75,
        "crypto": 0.75,
        "commodities": 0.70,
        "equities": 0.70,
        "forex": 0.65,
        "ipo_merger": 0.60,
        "general_market": 0.50,
    }

    # --- Sentiment Thresholds ---
    SENTIMENT_BULLISH_THRESHOLD: float = 0.3
    SENTIMENT_BEARISH_THRESHOLD: float = -0.3
    SENTIMENT_STRONG_THRESHOLD: float = 0.6
    FINBERT_CONFIDENCE_MINIMUM: float = 0.5

    # --- Embedding Model ---
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384
    SIMILARITY_THRESHOLD: float = 0.78  # Above this = duplicate

    # --- Collection Parameters ---
    MAX_ARTICLES_PER_BATCH: int = 100
    COLLECTION_INTERVAL_MINUTES: int = 30
    ARTICLE_MAX_AGE_HOURS: int = 168  # 7 days
    CACHE_TTL_MINUTES: int = 15

    # --- Temporal Aggregation Windows ---
    TEMPORAL_WINDOWS: dict[str, int] = {
        "1h": 1,
        "6h": 6,
        "24h": 24,
        "3d": 72,
        "7d": 168,
    }

    # --- Impact Score Decay ---
    IMPACT_HALF_LIFE_HOURS: float = 6.0  # Articles lose 50% impact every 6 hours

    # --- Event Detection Keywords ---
    EVENT_KEYWORDS: dict[str, list[str]] = {
        "fed_meeting": ["federal reserve", "fomc", "fed meeting", "rate decision", "fed chair", "powell"],
        "etf_approval": ["etf approval", "etf approved", "etf filing", "etf launch", "spot etf"],
        "regulation": ["sec", "regulation", "regulatory", "compliance", "ban", "crackdown", "lawsuit"],
        "black_swan": ["crash", "collapse", "crisis", "default", "bankruptcy", "bank run", "contagion"],
        "geopolitical": ["war", "conflict", "sanctions", "invasion", "missile", "nuclear", "embargo"],
        "macro_shock": ["recession", "depression", "stagflation", "hyperinflation", "debt ceiling"],
    }

    # --- Asset Entity Mapping ---
    ENTITY_ASSET_MAP: dict[str, list[str]] = {
        "bitcoin": ["crypto"],
        "btc": ["crypto"],
        "ethereum": ["crypto"],
        "eth": ["crypto"],
        "crypto": ["crypto"],
        "s&p 500": ["equities"],
        "spy": ["equities"],
        "nasdaq": ["equities"],
        "qqq": ["equities"],
        "dow jones": ["equities"],
        "apple": ["equities"],
        "microsoft": ["equities"],
        "nvidia": ["equities"],
        "gold": ["commodities"],
        "oil": ["commodities"],
        "crude": ["commodities"],
        "silver": ["commodities"],
        "treasury": ["bonds"],
        "bond": ["bonds"],
        "yield": ["bonds"],
        "federal reserve": ["bonds", "equities"],
        "interest rate": ["bonds", "equities"],
        "dollar": ["forex"],
        "euro": ["forex"],
        "yen": ["forex"],
    }


# ══════════════════════════════════════════════════════
#  LOGGING CONFIGURATION
# ══════════════════════════════════════════════════════
class LogConfig:
    """Structured logging configuration."""

    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FORMAT: str = "json"       # json | console
    LOG_DIR: Path = _PROJECT_ROOT / "logs"


# ══════════════════════════════════════════════════════
#  SYSTEM METADATA
# ══════════════════════════════════════════════════════
class SystemMeta:
    """System-level metadata for agent identification."""

    AGENT_ID: str = "agent1_macro_intelligence"
    VERSION: str = "1.0.0"
    SYSTEM_NAME: str = "Hybrid Intelligence Portfolio System"
    DESCRIPTION: str = "Senior Global Macro Intelligence System — Multi-asset market regime analysis"
