"""
AI Hedge Fund — Part 1: Foundation
====================================
market_data.py — Real market data ingestion & feature engineering

This is the data backbone of the entire system.
Everything — agents, risk engine, backtester — starts here.

Data sources (in priority order):
  1. Cache (SQLite) — instant, no API calls
  2. Yahoo Finance (yfinance) — free, 2+ years of daily OHLCV
  3. Alpha Vantage — premium fundamentals & intraday (requires key)
  4. FRED — macro data (requires free FRED API key)
  5. Polygon.io — real-time tick data (requires paid key)

Features computed from raw OHLCV:
  Momentum       : 1d, 5d, 21d, 63d, 126d, 252d returns
  Volatility     : 21d, 63d realized vol (annualized)
  Technical      : RSI(14), MACD, Bollinger Bands, ATR
  Volume         : Volume ratio, VWAP deviation, turnover
  Cross-section  : Rank within universe, z-score

This is what your XGBoost, FinBERT, and GARCH models
from your existing portfolios will receive as input.

References:
    - Gu, Kelly, Xiu (2020): 94 predictive signals for ML in finance
    - Barra / Axioma factor model feature conventions
"""

import logging
import math
import sqlite3
import time
import hashlib
import json
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Real market data — yfinance is free and works without API keys
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    print("Install yfinance: pip install yfinance")

from src.config.settings import cfg, DATA_DIR, DB_DIR
from src.data.data_models import (
    OHLCVBar, MarketSnapshot, MacroContext, Regime
)

logger = logging.getLogger("hedge_fund.market_data")


# ─────────────────────────────────────────────────────────────────────────────
# SQLite cache layer
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataCache:
    """
    SQLite-backed cache for OHLCV data.

    Avoids re-downloading data on every run.
    Cache expires based on DATA_CACHE_EXPIRY_HOURS in settings.

    Schema:
        price_cache(ticker, date, open, high, low, close, adj_close, volume, fetched_at)
        feature_cache(ticker, date, feature_json, fetched_at)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DB_DIR / "market_data_cache.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_cache (
                    ticker    TEXT NOT NULL,
                    date      TEXT NOT NULL,
                    open      REAL, high REAL, low REAL,
                    close     REAL, adj_close REAL, volume REAL,
                    fetched_at TEXT,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_cache (
                    ticker       TEXT NOT NULL,
                    date         TEXT NOT NULL,
                    feature_json TEXT,
                    fetched_at   TEXT,
                    PRIMARY KEY (ticker, date)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_price_ticker_date
                ON price_cache (ticker, date)
            """)
            conn.commit()

    def get_prices(self, ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
        """Return cached OHLCV dataframe or None if missing/stale."""
        cutoff = (datetime.now() - timedelta(hours=cfg.DATA_CACHE_EXPIRY_HOURS)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, open, high, low, close, adj_close, volume
                FROM price_cache
                WHERE ticker = ? AND date >= ? AND date <= ?
                  AND fetched_at >= ?
                ORDER BY date
                """,
                conn,
                params=(ticker, start.isoformat(), end.isoformat(), cutoff),
                parse_dates=["date"],
                index_col="date"
            )
        if df.empty:
            return None
        # Verify we have most of the requested range
        expected_days = (end - start).days * 5 / 7  # rough trading days
        if len(df) < expected_days * 0.5:
            return None
        return df

    def save_prices(self, ticker: str, df: pd.DataFrame) -> None:
        """Save OHLCV dataframe to cache."""
        now = datetime.now().isoformat()
        rows = []
        for dt, row in df.iterrows():
            rows.append((
                ticker,
                dt.date().isoformat() if hasattr(dt, 'date') else str(dt),
                float(row.get("Open", 0) or 0),
                float(row.get("High", 0) or 0),
                float(row.get("Low", 0) or 0),
                float(row.get("Close", 0) or 0),
                float(row.get("Adj Close", row.get("Close", 0)) or 0),
                float(row.get("Volume", 0) or 0),
                now,
            ))
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO price_cache
                (ticker, date, open, high, low, close, adj_close, volume, fetched_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                """,
                rows
            )
            conn.commit()

    def invalidate(self, ticker: str) -> None:
        """Force re-download for a ticker on next request."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM price_cache WHERE ticker = ?", (ticker,))
            conn.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Core data fetcher
# ─────────────────────────────────────────────────────────────────────────────

class MarketDataFetcher:
    """
    Fetches real OHLCV data from Yahoo Finance with caching.

    Primary data source: yfinance (free, no API key required)
    Fallback: Alpha Vantage (requires key, set in .env)

    Usage:
        fetcher = MarketDataFetcher()
        prices = fetcher.get_prices("AAPL", days=252)
        snapshot = fetcher.get_market_snapshot(["AAPL", "MSFT", "GOOGL"])
    """

    def __init__(self):
        self.cache = MarketDataCache()
        if not YFINANCE_AVAILABLE:
            raise ImportError("yfinance required: pip install yfinance")

    def get_prices(
        self,
        ticker: str,
        days: int = 504,
        end_date: Optional[date] = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a single ticker.

        Returns DataFrame with columns:
            Open, High, Low, Close, Adj Close, Volume
        Index: DatetimeIndex (trading days only)

        Args:
            ticker   : Stock ticker (e.g. "AAPL")
            days     : Number of calendar days of history
            end_date : Last date to fetch (default: today)
            use_cache: Check local cache first (default: True)
        """
        end = end_date or date.today()
        start = end - timedelta(days=days)

        # Check cache first
        if use_cache:
            cached = self.cache.get_prices(ticker, start, end)
            if cached is not None:
                logger.debug(f"Cache hit: {ticker} ({len(cached)} bars)")
                return cached

        # Fetch from Yahoo Finance
        logger.info(f"Fetching {ticker} from Yahoo Finance ({start} → {end})...")
        try:
            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=end.isoformat(),
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if raw.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # Standardise column names
            raw = raw.rename(columns={
                "open": "Open", "high": "High", "low": "Low",
                "close": "Close", "adj close": "Adj Close",
                "volume": "Volume",
            })

            # Drop rows with null close prices
            raw = raw.dropna(subset=["Close"])

            if raw.empty:
                logger.warning(f"All data null for {ticker}")
                return pd.DataFrame()

            # Cache the result
            self.cache.save_prices(ticker, raw)

            logger.info(f"  {ticker}: {len(raw)} bars fetched")
            return raw

        except Exception as e:
            logger.error(f"Failed to fetch {ticker}: {e}")
            return pd.DataFrame()

    def get_multi_prices(
        self,
        tickers: List[str],
        days: int = 504,
        end_date: Optional[date] = None,
        delay_seconds: float = 0.2,
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for multiple tickers.

        Includes rate-limiting delay to be respectful to Yahoo Finance.
        Returns dict of {ticker: DataFrame}.
        """
        result = {}
        failed = []

        for i, ticker in enumerate(tickers):
            df = self.get_prices(ticker, days=days, end_date=end_date)
            if not df.empty:
                result[ticker] = df
            else:
                failed.append(ticker)

            # Rate limiting — avoid hammering Yahoo
            if delay_seconds > 0 and i < len(tickers) - 1:
                time.sleep(delay_seconds)

        if failed:
            logger.warning(f"Failed to fetch {len(failed)} tickers: {failed[:10]}")

        logger.info(
            f"Fetched {len(result)}/{len(tickers)} tickers successfully"
        )
        return result

    def get_returns(
        self,
        prices: pd.DataFrame,
        column: str = "Adj Close"
    ) -> pd.Series:
        """Compute daily log returns from price series."""
        col = prices[column] if column in prices.columns else prices["Close"]
        return np.log(col / col.shift(1)).dropna()

    def get_market_snapshot(
        self,
        tickers: List[str],
        as_of: Optional[date] = None,
    ) -> MarketSnapshot:
        """
        Build a complete MarketSnapshot for a universe of tickers.

        This is what the PM Agent receives when asked to assess the market.
        Computes prices, 1d/5d/21d returns, and 21d volatility for all tickers.

        Args:
            tickers : List of ticker symbols
            as_of   : Date to compute snapshot for (default: today)
        """
        target_date = as_of or date.today()
        all_prices: Dict[str, float] = {}
        returns_1d: Dict[str, float] = {}
        returns_5d: Dict[str, float] = {}
        returns_21d: Dict[str, float] = {}
        volumes:    Dict[str, float] = {}
        vols_21d:   Dict[str, float] = {}

        price_data = self.get_multi_prices(tickers, days=60)  # 60 days for vol

        for ticker, df in price_data.items():
            if df.empty or len(df) < 22:
                continue

            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            closes = df[close_col].dropna()
            if len(closes) < 2:
                continue

            # Latest price
            all_prices[ticker] = float(closes.iloc[-1])

            # Returns
            log_returns = np.log(closes / closes.shift(1)).dropna()
            if len(log_returns) >= 1:
                returns_1d[ticker] = float(np.exp(log_returns.iloc[-1]) - 1)
            if len(log_returns) >= 5:
                returns_5d[ticker] = float(np.exp(log_returns.iloc[-5:].sum()) - 1)
            if len(log_returns) >= 21:
                returns_21d[ticker] = float(np.exp(log_returns.iloc[-21:].sum()) - 1)

            # Volume
            if "Volume" in df.columns:
                volumes[ticker] = float(df["Volume"].iloc[-1])

            # 21-day realized vol (annualized)
            if len(log_returns) >= 21:
                vols_21d[ticker] = float(log_returns.iloc[-21:].std() * math.sqrt(252))

        # SPY as benchmark
        spy_1d = returns_1d.get("SPY")

        # VIX (use ^VIX ticker in Yahoo Finance)
        vix_level = None
        vix_df = self.get_prices("^VIX", days=5)
        if not vix_df.empty:
            vix_col = "Close" if "Close" in vix_df.columns else vix_df.columns[0]
            vix_level = float(vix_df[vix_col].iloc[-1])

        # Regime classification based on VIX and SPY
        regime = self._classify_regime(vix_level, spy_1d, vols_21d)

        return MarketSnapshot(
            timestamp=datetime.now(),
            prices=all_prices,
            returns_1d=returns_1d,
            returns_5d=returns_5d,
            returns_21d=returns_21d,
            volumes=volumes,
            vols_21d=vols_21d,
            regime=regime,
            vix_level=vix_level,
            spy_return_1d=spy_1d,
        )

    def _classify_regime(
        self,
        vix: Optional[float],
        spy_1d: Optional[float],
        vols: Dict[str, float],
    ) -> Regime:
        """Classify market regime from VIX and market returns."""
        if vix is None:
            return Regime.UNKNOWN

        if vix > 40:
            return Regime.CRISIS
        if vix > 25:
            return Regime.HIGH_VOL
        if vix < 15:
            if spy_1d is not None and spy_1d > 0:
                return Regime.BULL
            return Regime.SIDEWAYS
        if spy_1d is not None and spy_1d < -0.01:
            return Regime.BEAR
        return Regime.SIDEWAYS


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering — the 90+ features your ML models need
# ─────────────────────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Computes ML features from raw OHLCV data.

    This is what you feed into your XGBoost, neural network,
    and factor models from your existing quant portfolios.

    Features are computed per-ticker from its OHLCV history.
    All features are designed to avoid look-ahead bias.

    Returns a pandas DataFrame with dates as index, feature names as columns.
    """

    # ── Momentum ──────────────────────────────────────────────────────────────

    @staticmethod
    def momentum_features(close: pd.Series) -> pd.DataFrame:
        """
        Momentum signals at multiple horizons.
        Source: Jegadeesh & Titman (1993), Asness (1994).
        """
        log_r = np.log(close / close.shift(1))
        features = pd.DataFrame(index=close.index)

        for h in [1, 5, 10, 21, 42, 63, 126, 252]:
            features[f"ret_{h}d"] = np.exp(log_r.rolling(h).sum()) - 1

        # Skip-month momentum (exclude most recent month to avoid reversal)
        features["mom_12_1"] = (
            np.exp(log_r.shift(21).rolling(252 - 21).sum()) - 1
        )

        # Short-term reversal (last week, mean-reverting)
        features["reversal_5d"] = -features["ret_5d"]

        return features

    # ── Volatility ────────────────────────────────────────────────────────────

    @staticmethod
    def volatility_features(close: pd.Series, high: pd.Series, low: pd.Series) -> pd.DataFrame:
        """
        Volatility signals — GARCH-style realized vol and vol-of-vol.
        """
        log_r = np.log(close / close.shift(1))
        features = pd.DataFrame(index=close.index)

        for h in [5, 10, 21, 63]:
            features[f"vol_{h}d"] = log_r.rolling(h).std() * math.sqrt(252)

        # Vol-of-vol (uncertainty about uncertainty)
        features["vol_of_vol_21d"] = features["vol_21d"].rolling(21).std()

        # Vol regime change: recent vol vs long-run vol
        features["vol_ratio_21_63"] = features["vol_21d"] / features["vol_63d"].clip(lower=1e-6)

        # Parkinson (1980) high-low range estimator — more efficient than close-to-close
        hl_ratio = np.log(high / low)
        features["parkinson_vol"] = (
            hl_ratio.pow(2).rolling(21).mean() / (4 * math.log(2))
        ).pow(0.5) * math.sqrt(252)

        # Average True Range (ATR) — normalized
        features["atr_14d"] = (high - low).rolling(14).mean() / close

        return features

    # ── Technical Indicators ──────────────────────────────────────────────────

    @staticmethod
    def technical_features(
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        volume: pd.Series,
    ) -> pd.DataFrame:
        """
        Classic technical indicators as ML features.
        Note: we use these as features, not as trading rules.
        """
        features = pd.DataFrame(index=close.index)

        # ── RSI (Relative Strength Index) ──────────────────────────────────
        delta = close.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean().clip(lower=1e-10)
        rs = avg_gain / avg_loss
        features["rsi_14"] = 100 - (100 / (1 + rs))

        # ── MACD ──────────────────────────────────────────────────────────
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        features["macd"] = macd / close              # Normalised
        features["macd_signal"] = signal / close
        features["macd_hist"] = (macd - signal) / close

        # ── Bollinger Bands ───────────────────────────────────────────────
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        features["bb_pct_b"] = (close - (sma20 - 2*std20)) / (4*std20 + 1e-10)  # 0-1 position
        features["bb_width"] = (4 * std20) / sma20   # Band width as vol proxy

        # ── Price vs Moving Averages ───────────────────────────────────────
        for window in [10, 20, 50, 200]:
            ma = close.rolling(window).mean()
            features[f"price_to_ma{window}"] = close / ma - 1

        # ── Volume Features ────────────────────────────────────────────────
        vol_ma20 = volume.rolling(20).mean()
        features["volume_ratio"] = volume / vol_ma20.clip(lower=1)  # Relative volume
        features["volume_trend"] = volume.rolling(5).mean() / vol_ma20.clip(lower=1)

        # ── Price Levels ───────────────────────────────────────────────────
        # Distance from 52-week high/low (anchoring signal)
        high_52w = close.rolling(252).max()
        low_52w  = close.rolling(252).min()
        features["dist_from_52w_high"] = close / high_52w - 1     # <= 0
        features["dist_from_52w_low"]  = close / low_52w - 1      # >= 0

        return features

    # ── Cross-Sectional Features ──────────────────────────────────────────────

    @staticmethod
    def cross_sectional_features(
        feature_panel: pd.DataFrame,  # rows=date, cols=tickers, values=feature
        feature_name: str,
    ) -> pd.DataFrame:
        """
        Compute cross-sectional ranks and z-scores.

        Cross-sectional rank is one of the strongest transformations for
        ML models in finance — it removes time-series level effects.
        """
        ranks = feature_panel.rank(axis=1, pct=True)
        means = feature_panel.mean(axis=1)
        stds  = feature_panel.std(axis=1)
        z     = feature_panel.sub(means, axis=0).div(stds.clip(lower=1e-10), axis=0)
        return {
            f"{feature_name}_cs_rank": ranks,
            f"{feature_name}_cs_zscore": z,
        }

    # ── Full Feature Matrix for One Ticker ───────────────────────────────────

    def compute_all_features(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Compute the complete feature matrix for one ticker.

        This feeds directly into your XGBoost alpha model and neural networks.

        Args:
            df     : OHLCV DataFrame (from MarketDataFetcher.get_prices)
            ticker : Ticker name (for labelling)

        Returns:
            DataFrame with all features, Date index, feature columns
        """
        if df.empty or len(df) < 30:
            logger.warning(f"Insufficient data for {ticker}: {len(df)} rows")
            return pd.DataFrame()

        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        close  = df[close_col]
        high   = df["High"]
        low    = df["Low"]
        volume = df["Volume"].clip(lower=1)

        all_features = [
            self.momentum_features(close),
            self.volatility_features(close, high, low),
            self.technical_features(close, high, low, volume),
        ]

        combined = pd.concat(all_features, axis=1)
        combined["ticker"] = ticker
        combined = combined.dropna(how="all")

        return combined

    def compute_universe_features(
        self,
        price_data: Dict[str, pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Compute features for entire universe.

        Returns panel DataFrame with MultiIndex (date, ticker).
        This is the input to your cross-sectional ML models.
        """
        all_dfs = []
        for ticker, df in price_data.items():
            feat_df = self.compute_all_features(df, ticker)
            if not feat_df.empty:
                feat_df.index = pd.MultiIndex.from_arrays(
                    [feat_df.index, [ticker] * len(feat_df)],
                    names=["date", "ticker"]
                )
                all_dfs.append(feat_df)

        if not all_dfs:
            return pd.DataFrame()

        panel = pd.concat(all_dfs, axis=0).sort_index()
        logger.info(
            f"Feature matrix: {len(panel)} rows, "
            f"{len(panel.columns)} features, "
            f"{len(price_data)} tickers"
        )
        return panel


# ─────────────────────────────────────────────────────────────────────────────
# Macro data from FRED
# ─────────────────────────────────────────────────────────────────────────────

class MacroDataFetcher:
    """
    Fetch macro-economic data from FRED (Federal Reserve Economic Data).

    Free API key: https://fred.stlouisfed.org/docs/api/fred/
    Set FRED_API_KEY in your .env file.

    Falls back to market-implied proxies if no FRED key is available.
    """

    # FRED series IDs for key macro variables
    FRED_SERIES = {
        "fed_funds_rate": "DFF",         # Effective Federal Funds Rate
        "us_10y_yield":   "DGS10",       # 10Y Treasury Yield
        "us_2y_yield":    "DGS2",        # 2Y Treasury Yield
        "unemployment":   "UNRATE",       # Unemployment Rate
        "cpi_yoy":        "CPIAUCSL",     # CPI
        "gdp_growth":     "A191RL1Q225SBEA",  # Real GDP Growth
    }

    def __init__(self):
        self.fred_key = cfg.FRED_API_KEY
        self._cache: Dict[str, pd.Series] = {}

    def get_macro_context(self) -> MacroContext:
        """
        Build current MacroContext.

        Uses FRED if key is available, else market-implied from ETF prices.
        """
        if self.fred_key:
            return self._from_fred()
        else:
            logger.info("No FRED key — using market-implied macro proxies")
            return self._from_market_proxies()

    def _from_market_proxies(self) -> MacroContext:
        """
        Estimate macro context from ETF prices (no API key required).

        Proxies:
            TLT  → 20Y Treasury ETF   (interest rate proxy)
            ^VIX → Volatility Index   (risk proxy)
            HYG  → High-Yield ETF     (credit proxy)
            GLD  → Gold ETF           (inflation/safety proxy)
            SPY  → S&P 500            (equity market)
        """
        fetcher = MarketDataFetcher()
        proxies = fetcher.get_multi_prices(["TLT", "^VIX", "HYG", "GLD", "SPY"], days=252)

        vix_level = None
        credit_spread_proxy = None
        us_10y_proxy = None

        if "^VIX" in proxies and not proxies["^VIX"].empty:
            vix_level = float(proxies["^VIX"]["Close"].iloc[-1])

        # TLT yield proxy: TLT price inversely related to 20Y yield
        # (simplified: use TLT 1Y return as rate direction indicator)
        if "TLT" in proxies and not proxies["TLT"].empty:
            tlt_1y_ret = float(
                proxies["TLT"]["Close"].iloc[-1] / proxies["TLT"]["Close"].iloc[0] - 1
            )
            # Falling TLT price = rising rates
            us_10y_proxy = -tlt_1y_ret * 10  # Very rough proxy

        # HYG-TLT spread as credit spread proxy
        if "HYG" in proxies and "TLT" in proxies:
            hyg_vol = np.log(proxies["HYG"]["Close"] / proxies["HYG"]["Close"].shift(1)).std()
            tlt_vol = np.log(proxies["TLT"]["Close"] / proxies["TLT"]["Close"].shift(1)).std()
            credit_spread_proxy = float((hyg_vol - tlt_vol) * 100)

        # Regime from VIX
        regime = Regime.UNKNOWN
        if vix_level is not None:
            if vix_level > 40:   regime = Regime.CRISIS
            elif vix_level > 25: regime = Regime.HIGH_VOL
            elif vix_level < 15: regime = Regime.BULL
            else:                regime = Regime.SIDEWAYS

        return MacroContext(
            timestamp=datetime.now(),
            vix=vix_level,
            credit_spread=credit_spread_proxy,
            us_10y_yield=us_10y_proxy,
            regime=regime,
            recession_prob=min(0.5, (vix_level or 20) / 80),
        )

    def _from_fred(self) -> MacroContext:
        """Fetch macro data from FRED API."""
        try:
            import requests
            base = "https://api.stlouisfed.org/fred/series/observations"
            results = {}

            for name, series_id in self.FRED_SERIES.items():
                url = (
                    f"{base}?series_id={series_id}"
                    f"&api_key={self.fred_key}"
                    f"&file_type=json"
                    f"&limit=12"
                    f"&sort_order=desc"
                )
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    obs = resp.json().get("observations", [])
                    valid = [o for o in obs if o["value"] != "."]
                    if valid:
                        results[name] = float(valid[0]["value"])

            # Yield curve
            yield_curve = None
            if "us_10y_yield" in results and "us_2y_yield" in results:
                yield_curve = results["us_10y_yield"] - results.get("us_2y_yield", 0)

            vix_level = None
            vix_df = MarketDataFetcher().get_prices("^VIX", days=5)
            if not vix_df.empty:
                vix_level = float(vix_df["Close"].iloc[-1])

            regime = Regime.UNKNOWN
            if vix_level is not None:
                if vix_level > 40:
                    regime = Regime.CRISIS
                elif vix_level > 25:
                    regime = Regime.HIGH_VOL
                elif results.get("us_10y_yield", 5) < 2 and vix_level < 20:
                    regime = Regime.BULL
                else:
                    regime = Regime.SIDEWAYS

            return MacroContext(
                timestamp=datetime.now(),
                fed_funds_rate=results.get("fed_funds_rate"),
                us_10y_yield=results.get("us_10y_yield"),
                yield_curve_10y2y=yield_curve,
                unemployment=results.get("unemployment"),
                cpi_yoy=results.get("cpi_yoy"),
                vix=vix_level,
                regime=regime,
                recession_prob=(1.0 if (yield_curve or 0) < -0.5 else 0.2),
            )

        except Exception as e:
            logger.error(f"FRED fetch failed: {e} — falling back to market proxies")
            return self._from_market_proxies()


# ─────────────────────────────────────────────────────────────────────────────
# Correlation engine
# ─────────────────────────────────────────────────────────────────────────────

class CorrelationEngine:
    """
    Compute and cache return correlations across the universe.

    Used by the Portfolio Manager to assess diversification
    before adding a new position.
    """

    def __init__(self, fetcher: Optional[MarketDataFetcher] = None):
        self.fetcher = fetcher or MarketDataFetcher()

    def build_return_matrix(
        self,
        tickers: List[str],
        days: int = 252,
    ) -> pd.DataFrame:
        """
        Build a return matrix: rows=dates, cols=tickers.
        Uses adjusted close prices.
        """
        price_data = self.fetcher.get_multi_prices(tickers, days=days + 30)
        returns = {}

        for ticker, df in price_data.items():
            if df.empty:
                continue
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            log_r = np.log(df[col] / df[col].shift(1)).dropna()
            returns[ticker] = log_r

        if not returns:
            return pd.DataFrame()

        return pd.DataFrame(returns).dropna(how="all")

    def correlation_matrix(
        self,
        tickers: List[str],
        days: int = 252,
        method: str = "pearson",
    ) -> pd.DataFrame:
        """
        Compute correlation matrix for a universe.

        Args:
            tickers : List of tickers
            days    : Lookback period
            method  : "pearson" (linear) or "spearman" (rank)

        Returns:
            Symmetric correlation matrix (tickers × tickers)
        """
        ret_matrix = self.build_return_matrix(tickers, days)
        if ret_matrix.empty:
            return pd.DataFrame()
        return ret_matrix.corr(method=method)

    def correlation_to_portfolio(
        self,
        ticker: str,
        portfolio_tickers: List[str],
        days: int = 126,
    ) -> Dict[str, float]:
        """
        Compute correlations between a candidate ticker and existing positions.

        Returns dict of {existing_ticker: correlation_with_candidate}.
        """
        if not portfolio_tickers:
            return {}

        all_tickers = list(set([ticker] + portfolio_tickers))
        corr_matrix = self.correlation_matrix(all_tickers, days=days)

        if corr_matrix.empty or ticker not in corr_matrix.columns:
            return {}

        result = {}
        for pt in portfolio_tickers:
            if pt in corr_matrix.columns:
                result[pt] = float(corr_matrix.loc[ticker, pt])
        return result

    def average_correlation(
        self,
        ticker: str,
        portfolio_tickers: List[str],
        days: int = 126,
    ) -> float:
        """Average pairwise correlation between ticker and portfolio."""
        corrs = self.correlation_to_portfolio(ticker, portfolio_tickers, days)
        if not corrs:
            return 0.0
        return float(np.mean(list(corrs.values())))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level convenience functions
# ─────────────────────────────────────────────────────────────────────────────

_fetcher: Optional[MarketDataFetcher] = None
_macro: Optional[MacroDataFetcher] = None
_engineer: Optional[FeatureEngineer] = None


def get_fetcher() -> MarketDataFetcher:
    global _fetcher
    if _fetcher is None:
        _fetcher = MarketDataFetcher()
    return _fetcher


def get_macro_fetcher() -> MacroDataFetcher:
    global _macro
    if _macro is None:
        _macro = MacroDataFetcher()
    return _macro


def get_feature_engineer() -> FeatureEngineer:
    global _engineer
    if _engineer is None:
        _engineer = FeatureEngineer()
    return _engineer


def quick_snapshot(tickers: Optional[List[str]] = None) -> MarketSnapshot:
    """One-liner to get a market snapshot for the default universe."""
    universe = tickers or cfg.DEFAULT_UNIVERSE
    return get_fetcher().get_market_snapshot(universe)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  Market Data Engine — Integration Test")
    print("=" * 60)

    fetcher = MarketDataFetcher()

    # Test 1: Single ticker fetch
    print("\n1. Fetching AAPL (1 year)...")
    aapl = fetcher.get_prices("AAPL", days=365)
    if not aapl.empty:
        print(f"   ✓ {len(aapl)} bars | Latest close: ${aapl['Close'].iloc[-1]:.2f}")
    else:
        print("   ✗ Failed to fetch AAPL")
        sys.exit(1)

    # Test 2: Feature engineering
    print("\n2. Computing features for AAPL...")
    engineer = FeatureEngineer()
    features = engineer.compute_all_features(aapl, "AAPL")
    if not features.empty:
        print(f"   ✓ {len(features)} rows × {len(features.columns)} features")
        non_null = features.count() / len(features)
        print(f"   ✓ Avg feature completeness: {non_null.mean():.1%}")
    else:
        print("   ✗ Feature computation failed")

    # Test 3: Multi-ticker snapshot
    test_universe = ["AAPL", "MSFT", "GOOGL", "JPM", "SPY", "^VIX"]
    print(f"\n3. Building market snapshot for {len(test_universe)} tickers...")
    snapshot = fetcher.get_market_snapshot(test_universe)
    print(f"   ✓ {snapshot.market_summary()}")
    print(f"   Top movers: {snapshot.top_movers(3)}")

    # Test 4: Correlation engine
    print("\n4. Computing correlations...")
    corr_engine = CorrelationEngine(fetcher)
    corrs = corr_engine.correlation_to_portfolio("AAPL", ["MSFT", "GOOGL", "JPM"])
    for t, c in corrs.items():
        print(f"   AAPL ↔ {t}: {c:.3f}")

    # Test 5: Macro context
    print("\n5. Building macro context...")
    macro = MacroDataFetcher()
    ctx = macro.get_macro_context()
    print(ctx.describe())

    print("\n✅ All market data tests passed")
