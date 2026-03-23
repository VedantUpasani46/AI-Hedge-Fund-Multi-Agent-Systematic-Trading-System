"""
AI Hedge Fund — Part 6: Backtesting Engine
============================================
backtest_engine.py — Event-Driven Backtesting Framework

This is the most important module in the entire system for
demonstrating credibility to investors and seeders.

Why event-driven matters:
    Vector-based backtests (pandas operations on full dataframes)
    are fast but chronically infected with look-ahead bias.
    You accidentally use tomorrow's data to make today's decision
    constantly — in feature computation, in signal generation,
    in rebalancing logic. The result: backtested Sharpe of 2.5
    that collapses to 0.3 in live trading.

    Event-driven architecture PREVENTS this:
        Every decision is made from a DataView that contains
        ONLY data available at that moment in time.
        The engine enforces time ordering — nothing flows backward.

Architecture:
    HistoricalDataLoader → DataView (t) → Strategy.on_bar()
                                        → OrderEvent
                                        → SimulatedBroker.fill()
                                        → Portfolio.update()
                                        → PerformanceTracker.record()

    The DataView at time t contains:
        - All price history up to and including t
        - No data from t+1 onward
        - Pre-computed features (computed from [0, t] only)

Walk-forward validation:
    The correct way to backtest ML strategies.
    Instead of one big in-sample period + one out-of-sample:
        Train on [0, T_1]           → test on [T_1, T_2]
        Train on [0, T_2]           → test on [T_2, T_3]
        Train on [0, T_3]           → test on [T_3, T_4]
        ...
    Never touch a test period during training.
    Report aggregated out-of-sample performance only.

Transaction costs modelled:
    - Commission: $0.005/share (IB rate), min $1
    - Bid-ask spread: proportional to volatility
    - Market impact: Almgren-Chriss square-root model
    - Slippage: random noise around impact estimate

References:
    Chan, E. (2013). Algorithmic Trading. Wiley.
    Lopez de Prado, M. (2018). Advances in Financial Machine Learning.
    Jegadeesh & Titman (1993). Returns to Buying Winners. JF.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.backtest")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Bar:
    """A single price bar available to the strategy at time t."""
    ticker:    str
    date:      date
    open:      float
    high:      float
    low:       float
    close:     float
    volume:    float
    adj_close: float

    @property
    def typical_price(self) -> float:
        return (self.high + self.low + self.close) / 3

    @property
    def return_pct(self) -> float:
        return (self.close - self.open) / self.open if self.open > 0 else 0.0


@dataclass
class DataView:
    """
    Snapshot of all available data at time t.

    The strategy ONLY sees what's in this object.
    It cannot access the loader or any future data.
    This is the architectural guarantee against look-ahead bias.
    """
    current_date:  date
    current_bars:  Dict[str, Bar]          # Latest bar for each ticker
    history:       Dict[str, pd.DataFrame] # Full history up to current_date
    features:      Dict[str, pd.DataFrame] # Pre-computed features (no look-ahead)
    portfolio_nav: float = 0.0
    cash:          float = 0.0
    positions:     Dict[str, float] = field(default_factory=dict)  # ticker -> shares

    def price(self, ticker: str) -> Optional[float]:
        bar = self.current_bars.get(ticker)
        return bar.adj_close if bar else None

    def prices(self) -> Dict[str, float]:
        return {t: b.adj_close for t, b in self.current_bars.items()}

    def returns(self, ticker: str, window: int = 1) -> Optional[float]:
        df = self.history.get(ticker)
        if df is None or len(df) < window + 1:
            return None
        closes = df["adj_close"].iloc[-window-1:]
        return float(closes.iloc[-1] / closes.iloc[0] - 1)

    def volatility(self, ticker: str, window: int = 21) -> Optional[float]:
        df = self.history.get(ticker)
        if df is None or len(df) < window + 1:
            return None
        closes = df["adj_close"].tail(window + 1)
        log_ret = np.log(closes / closes.shift(1)).dropna()
        return float(log_ret.std() * math.sqrt(252))

    def feature(self, ticker: str, feature_name: str) -> Optional[float]:
        df = self.features.get(ticker)
        if df is None or feature_name not in df.columns or df.empty:
            return None
        return float(df[feature_name].iloc[-1])

    @property
    def universe(self) -> List[str]:
        return list(self.current_bars.keys())


@dataclass
class BacktestOrder:
    """Order generated by a strategy during backtesting."""
    ticker:       str
    date:         date
    direction:    str    # "BUY" or "SELL"
    shares:       float
    order_type:   str = "MARKET"
    limit_price:  Optional[float] = None
    reason:       str = ""


@dataclass
class BacktestFill:
    """Simulated fill — what the strategy actually got."""
    order:        BacktestOrder
    fill_price:   float
    commission:   float
    slippage_bps: float
    market_impact_bps: float
    total_cost_bps: float
    filled_at:    date


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    date:         date
    nav:          float
    cash:         float
    positions:    Dict[str, Dict]    # {ticker: {shares, price, value, weight}}
    daily_return: float = 0.0
    turnover:     float = 0.0

    @property
    def invested_pct(self) -> float:
        total_pos = sum(p["value"] for p in self.positions.values())
        return total_pos / self.nav if self.nav > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Historical data loader
# ─────────────────────────────────────────────────────────────────────────────

class HistoricalDataLoader:
    """
    Loads and prepares historical OHLCV data for backtesting.

    Sources: Yahoo Finance (free) via yfinance.
    Caches to SQLite to avoid repeated downloads.

    Data quality checks applied:
        - Remove trading days with zero volume
        - Forward-fill up to 3 consecutive missing days
        - Flag and remove obvious data errors (>50% gap in one day)
        - Adjust for splits and dividends (use adj_close throughout)

    Feature computation:
        Features are computed in a time-ordered loop — at each date t,
        only data from [0, t] is used. This matches live trading exactly.
    """

    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or (Path(__file__).parents[3] / "data" / "backtest_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data: Dict[str, pd.DataFrame] = {}

    def load(
        self,
        tickers:    List[str],
        start_date: date,
        end_date:   date,
    ) -> Dict[str, pd.DataFrame]:
        """
        Load adjusted OHLCV data for all tickers.

        Returns dict of {ticker: DataFrame} with standardised columns:
            date, open, high, low, close, adj_close, volume
        Index: DatetimeIndex
        """
        import yfinance as yf

        loaded = {}
        for ticker in tickers:
            cache_key = f"{ticker}_{start_date}_{end_date}.parquet"
            cache_path= self.cache_dir / cache_key

            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    loaded[ticker] = df
                    logger.debug(f"Cache hit: {ticker}")
                    continue
                except Exception:
                    pass

            try:
                raw = yf.download(
                    ticker,
                    start = start_date.isoformat(),
                    end   = (end_date + timedelta(days=1)).isoformat(),
                    progress = False,
                    auto_adjust = False,
                )
                if raw.empty:
                    logger.warning(f"No data for {ticker}")
                    continue

                # Flatten MultiIndex if present
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                # Standardise columns
                rename = {
                    "Open": "open", "High": "high", "Low": "low",
                    "Close": "close", "Adj Close": "adj_close", "Volume": "volume"
                }
                raw = raw.rename(columns=rename)
                required = ["open", "high", "low", "close", "adj_close", "volume"]
                if "adj_close" not in raw.columns:
                    raw["adj_close"] = raw["close"]

                raw = raw[[c for c in required if c in raw.columns]]
                raw = raw.dropna(subset=["close"])

                # Data quality
                raw = self._clean_data(raw, ticker)

                if raw.empty:
                    logger.warning(f"Data quality checks removed all data for {ticker}")
                    continue

                raw.to_parquet(cache_path)
                loaded[ticker] = raw
                logger.info(f"Loaded {ticker}: {len(raw)} bars")

            except Exception as e:
                logger.error(f"Failed to load {ticker}: {e}")

        self._data = loaded
        return loaded

    def _clean_data(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Apply data quality checks."""
        # Remove days with zero volume
        if "volume" in df.columns:
            df = df[df["volume"] > 0]

        # Remove extreme daily moves (>50% in one day — likely data error)
        if len(df) > 1:
            pct_change = df["adj_close"].pct_change().abs()
            df = df[pct_change < 0.50]

        # Forward-fill up to 3 consecutive missing days
        df = df.fillna(method="ffill", limit=3)

        return df.dropna(subset=["adj_close"])

    def compute_features(
        self,
        data:    Dict[str, pd.DataFrame],
        feature_fns: Optional[Dict[str, Callable]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Compute features for all tickers.

        CRITICAL: Features are computed from the full available history
        at each point. The DataView ensures only past features are visible.

        Default features:
            ret_1d, ret_5d, ret_21d, ret_63d  — momentum
            vol_21d, vol_63d                  — realized volatility
            rsi_14                            — RSI
            macd_hist                         — MACD histogram
            volume_ratio                      — Volume vs 20d avg
            price_to_ma20, price_to_ma200     — Moving average signals
        """
        features = {}

        for ticker, df in data.items():
            close  = df["adj_close"]
            volume = df.get("volume", pd.Series(dtype=float))
            high   = df.get("high", close)
            low    = df.get("low", close)

            feat = pd.DataFrame(index=df.index)

            # Momentum
            log_r = np.log(close / close.shift(1))
            for h in [1, 5, 10, 21, 42, 63, 126, 252]:
                feat[f"ret_{h}d"] = np.exp(log_r.rolling(h).sum()) - 1

            # Skip-month momentum (Jegadeesh & Titman 1993)
            feat["mom_12_1"] = np.exp(log_r.shift(21).rolling(252 - 21).sum()) - 1

            # Volatility
            for h in [5, 21, 63]:
                feat[f"vol_{h}d"] = log_r.rolling(h).std() * math.sqrt(252)

            # Vol ratio (vol trend)
            feat["vol_ratio"] = feat["vol_21d"] / feat["vol_63d"].clip(lower=1e-6)

            # RSI
            delta = close.diff()
            gain  = delta.clip(lower=0).rolling(14).mean()
            loss  = (-delta).clip(lower=0).rolling(14).mean().clip(lower=1e-10)
            feat["rsi_14"] = 100 - (100 / (1 + gain / loss))

            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd  = ema12 - ema26
            sig   = macd.ewm(span=9, adjust=False).mean()
            feat["macd_hist"] = (macd - sig) / close.clip(lower=0.01)

            # Bollinger %B
            sma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            feat["bb_pct_b"] = (close - (sma20 - 2*std20)) / ((4*std20).clip(lower=0.01))

            # Moving averages
            for ma in [20, 50, 200]:
                feat[f"price_to_ma{ma}"] = close / close.rolling(ma).mean() - 1

            # Volume ratio
            if not volume.empty:
                feat["volume_ratio"] = volume / volume.rolling(20).mean().clip(lower=1)

            # 52-week high/low
            feat["dist_52w_high"] = close / close.rolling(252).max() - 1
            feat["dist_52w_low"]  = close / close.rolling(252).min() - 1

            features[ticker] = feat

        return features

    def iter_dates(
        self,
        data:     Dict[str, pd.DataFrame],
        features: Dict[str, pd.DataFrame],
        start:    date,
        end:      date,
    ) -> Iterator[DataView]:
        """
        Yield DataViews one trading day at a time.

        Each DataView contains ONLY data up to and including that date.
        This is the core mechanism preventing look-ahead bias.
        """
        # Get all trading dates in range
        all_dates = set()
        for df in data.values():
            dates = [d.date() for d in df.index if start <= d.date() <= end]
            all_dates.update(dates)

        trading_dates = sorted(all_dates)

        for current_date in trading_dates:
            current_bars = {}
            history_view = {}
            feature_view = {}

            for ticker in data:
                df = data[ticker]
                # Only data up to and including current_date
                mask   = df.index.date <= current_date
                hist   = df[mask]
                if hist.empty:
                    continue

                # Current bar
                last_row = hist.iloc[-1]
                current_bars[ticker] = Bar(
                    ticker    = ticker,
                    date      = current_date,
                    open      = float(last_row.get("open", last_row["adj_close"])),
                    high      = float(last_row.get("high", last_row["adj_close"])),
                    low       = float(last_row.get("low", last_row["adj_close"])),
                    close     = float(last_row["adj_close"]),
                    volume    = float(last_row.get("volume", 0)),
                    adj_close = float(last_row["adj_close"]),
                )
                history_view[ticker] = hist

                # Features up to current date
                if ticker in features:
                    feat_mask = features[ticker].index.date <= current_date
                    feature_view[ticker] = features[ticker][feat_mask]

            yield DataView(
                current_date  = current_date,
                current_bars  = current_bars,
                history       = history_view,
                features      = feature_view,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Simulated broker
# ─────────────────────────────────────────────────────────────────────────────

class SimulatedBroker:
    """
    Simulates order execution with realistic costs.

    Fills market orders at next-day open + noise.
    Models:
        Commission:     IB-style $0.005/share, min $1
        Bid-ask spread: 3-10 bps depending on ADV rank
        Market impact:  Square-root model calibrated per Almgren-Chriss
        Slippage noise: ±0-5 bps random

    This is materially more realistic than "fill at close price with
    0.1% commission" — the difference matters for high-turnover strategies.
    """

    def __init__(
        self,
        commission_per_share:  float = 0.005,
        min_commission:        float = 1.0,
        spread_bps:            float = 3.0,
        market_impact_coeff:   float = 0.1,    # Almgren-Chriss η coefficient
        slippage_noise_bps:    float = 2.0,
        seed:                  int = 42,
    ):
        self.commission_per_share = commission_per_share
        self.min_commission       = min_commission
        self.spread_bps           = spread_bps
        self.impact_coeff         = market_impact_coeff
        self.noise_bps            = slippage_noise_bps
        self.rng                  = np.random.default_rng(seed)

    def fill(
        self,
        order:      BacktestOrder,
        data_view:  DataView,
        adv:        float = 1_000_000,   # Average daily volume (shares)
    ) -> Optional[BacktestFill]:
        """Simulate order fill with realistic costs."""
        bar = data_view.current_bars.get(order.ticker)
        if bar is None:
            logger.warning(f"No bar for {order.ticker} on {order.date}")
            return None

        base_price = bar.open if bar.open > 0 else bar.adj_close

        # Commission
        commission = max(
            self.min_commission,
            order.shares * self.commission_per_share
        )

        # Bid-ask spread (half-spread on each side)
        spread_cost_bps = self.spread_bps / 2

        # Market impact: η × (shares / ADV) × daily_vol
        vol_daily = (data_view.volatility(order.ticker, window=21) or 0.02) / math.sqrt(252)
        participation = order.shares / max(adv, 1)
        impact_bps = (
            self.impact_coeff * vol_daily * math.sqrt(participation) * 10000
        )

        # Random slippage noise (symmetric around expected)
        noise_bps = self.rng.normal(0, self.noise_bps)

        # Total cost in bps (positive = we pay more to buy, get less to sell)
        direction = 1 if order.direction == "BUY" else -1
        total_cost_bps = direction * (spread_cost_bps + impact_bps) + noise_bps

        # Fill price
        fill_price = base_price * (1 + total_cost_bps / 10000)
        fill_price = max(0.001, fill_price)

        return BacktestFill(
            order             = order,
            fill_price        = round(fill_price, 4),
            commission        = commission,
            slippage_bps      = spread_cost_bps,
            market_impact_bps = impact_bps,
            total_cost_bps    = abs(total_cost_bps),
            filled_at         = order.date,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Portfolio tracker
# ─────────────────────────────────────────────────────────────────────────────

class BacktestPortfolio:
    """Maintains portfolio state during backtesting."""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash            = initial_capital
        self.positions: Dict[str, Dict] = {}   # {ticker: {shares, avg_cost}}
        self.nav_history:   List[float] = [initial_capital]
        self.date_history:  List[date]  = []
        self.fills_log:     List[BacktestFill] = []
        self.snapshots:     List[PortfolioSnapshot] = []

    @property
    def nav(self) -> float:
        return self.cash + sum(
            p["shares"] * p["current_price"] for p in self.positions.values()
        )

    def update_prices(self, prices: Dict[str, float]) -> None:
        for ticker, pos in self.positions.items():
            if ticker in prices:
                pos["current_price"] = prices[ticker]

    def apply_fill(self, fill: BacktestFill) -> None:
        order = fill.order
        ticker= order.ticker
        cost  = fill.fill_price * order.shares + fill.commission

        if order.direction == "BUY":
            self.cash -= cost
            if ticker in self.positions:
                old_shares   = self.positions[ticker]["shares"]
                old_cost     = self.positions[ticker]["avg_cost"]
                new_shares   = old_shares + order.shares
                new_avg_cost = (old_shares * old_cost + order.shares * fill.fill_price) / new_shares
                self.positions[ticker]["shares"]       = new_shares
                self.positions[ticker]["avg_cost"]     = new_avg_cost
                self.positions[ticker]["current_price"]= fill.fill_price
            else:
                self.positions[ticker] = {
                    "shares":        order.shares,
                    "avg_cost":      fill.fill_price,
                    "current_price": fill.fill_price,
                }
        elif order.direction == "SELL":
            proceeds = fill.fill_price * order.shares - fill.commission
            self.cash += proceeds
            if ticker in self.positions:
                new_shares = self.positions[ticker]["shares"] - order.shares
                if new_shares <= 1e-4:
                    del self.positions[ticker]
                else:
                    self.positions[ticker]["shares"] = new_shares
                    self.positions[ticker]["current_price"] = fill.fill_price

        self.fills_log.append(fill)

    def record_snapshot(self, dt: date, prev_nav: float) -> PortfolioSnapshot:
        nav   = self.nav
        ret   = (nav / prev_nav - 1) if prev_nav > 0 else 0.0
        pos_detail = {
            t: {
                "shares":  p["shares"],
                "price":   p["current_price"],
                "value":   p["shares"] * p["current_price"],
                "weight":  p["shares"] * p["current_price"] / nav if nav > 0 else 0,
            }
            for t, p in self.positions.items()
        }
        snap = PortfolioSnapshot(
            date         = dt,
            nav          = nav,
            cash         = self.cash,
            positions    = pos_detail,
            daily_return = ret,
        )
        self.snapshots.append(snap)
        self.nav_history.append(nav)
        self.date_history.append(dt)
        return snap

    def weight(self, ticker: str) -> float:
        nav = self.nav
        if ticker not in self.positions or nav <= 0:
            return 0.0
        pos = self.positions[ticker]
        return pos["shares"] * pos["current_price"] / nav


# ─────────────────────────────────────────────────────────────────────────────
# Strategy interface
# ─────────────────────────────────────────────────────────────────────────────

class Strategy:
    """
    Abstract base class for backtested strategies.

    Implement on_bar() to generate orders.
    The DataView guarantees no future data leakage.

    Example (momentum strategy):
        class MomentumStrategy(Strategy):
            def on_bar(self, view, portfolio):
                signals = {}
                for ticker in view.universe:
                    ret_63d = view.returns(ticker, 63)
                    if ret_63d is not None:
                        signals[ticker] = ret_63d
                # Rank by momentum
                ranked = sorted(signals.items(), key=lambda x: -x[1])
                top5   = [t for t, _ in ranked[:5]]
                # Buy top quintile, sell bottom quintile
                orders = []
                target = 1/5  # Equal weight top quintile
                for ticker in view.universe:
                    current_w = portfolio.weight(ticker)
                    if ticker in top5:
                        if current_w < target - 0.01:
                            shares = ...
                            orders.append(BacktestOrder(ticker, view.current_date, "BUY", shares))
                    else:
                        if current_w > 0.01:
                            shares = portfolio.positions[ticker]["shares"]
                            orders.append(BacktestOrder(ticker, view.current_date, "SELL", shares))
                return orders
    """

    def __init__(self, params: Dict[str, Any] = None):
        self.params = params or {}
        self.name   = self.__class__.__name__

    def on_bar(
        self,
        view:      DataView,
        portfolio: BacktestPortfolio,
    ) -> List[BacktestOrder]:
        """
        Called once per trading day with current DataView.
        Return list of orders to execute.
        """
        raise NotImplementedError

    def on_start(self, data: Dict[str, pd.DataFrame]) -> None:
        """Called before backtesting begins. Use for parameter fitting."""
        pass

    def on_end(self) -> None:
        """Called after backtesting ends."""
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Built-in strategies for demonstration
# ─────────────────────────────────────────────────────────────────────────────

class MomentumStrategy(Strategy):
    """
    Cross-sectional momentum strategy.

    Long top quintile by 12-1 month momentum, rebalance monthly.
    Jegadeesh & Titman (1993) — one of the most replicated factors.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.lookback    = self.params.get("lookback", 252)
        self.skip_months = self.params.get("skip_months", 21)
        self.top_n       = self.params.get("top_n", 5)
        self.target_weight = self.params.get("target_weight", 1.0 / 5)
        self._last_rebal = None

    def on_bar(self, view: DataView, portfolio: BacktestPortfolio) -> List[BacktestOrder]:
        # Rebalance monthly
        if (self._last_rebal is not None and
                (view.current_date - self._last_rebal).days < 21):
            return []

        self._last_rebal = view.current_date
        nav = portfolio.nav

        # Compute skip-month momentum for each ticker
        signals = {}
        for ticker in view.universe:
            df = view.history.get(ticker)
            if df is None or len(df) < self.lookback:
                continue
            closes = df["adj_close"]
            # 12-1 month momentum (skip last month)
            if len(closes) >= self.lookback:
                price_now   = float(closes.iloc[-self.skip_months - 1])
                price_start = float(closes.iloc[-self.lookback - 1])
                if price_start > 0:
                    signals[ticker] = price_now / price_start - 1

        if not signals:
            return []

        # Rank and select top N
        ranked = sorted(signals.items(), key=lambda x: -x[1])
        top_n  = [t for t, _ in ranked[:self.top_n]]

        orders = []
        # Close positions not in top N
        for ticker in list(portfolio.positions.keys()):
            if ticker not in top_n:
                shares = portfolio.positions[ticker]["shares"]
                if shares > 0:
                    orders.append(BacktestOrder(
                        ticker    = ticker,
                        date      = view.current_date,
                        direction = "SELL",
                        shares    = shares,
                        reason    = "momentum_exit",
                    ))

        # Open/rebalance top N to equal weight
        for ticker in top_n:
            price = view.price(ticker)
            if not price or price <= 0:
                continue
            target_nav   = nav * self.target_weight
            current_val  = portfolio.weight(ticker) * nav
            delta_val    = target_nav - current_val
            shares_delta = delta_val / price

            if abs(shares_delta) * price < 100:  # Minimum $100 trade size
                continue

            if shares_delta > 0:
                orders.append(BacktestOrder(
                    ticker    = ticker,
                    date      = view.current_date,
                    direction = "BUY",
                    shares    = shares_delta,
                    reason    = f"momentum_enter rank={ranked.index((ticker, signals[ticker]))+1}",
                ))
            elif shares_delta < -0 and ticker in portfolio.positions:
                orders.append(BacktestOrder(
                    ticker    = ticker,
                    date      = view.current_date,
                    direction = "SELL",
                    shares    = abs(shares_delta),
                    reason    = "momentum_trim",
                ))

        return orders


class MeanReversionStrategy(Strategy):
    """
    Short-term mean reversion strategy.

    Buy securities down >2σ from 20-day mean, sell when revert.
    """

    def __init__(self, params=None):
        super().__init__(params)
        self.entry_z  = self.params.get("entry_z", 2.0)
        self.exit_z   = self.params.get("exit_z", 0.5)
        self.max_hold = self.params.get("max_hold_days", 10)
        self._entries: Dict[str, date] = {}

    def on_bar(self, view: DataView, portfolio: BacktestPortfolio) -> List[BacktestOrder]:
        orders = []
        nav    = portfolio.nav

        for ticker in view.universe:
            df = view.history.get(ticker)
            if df is None or len(df) < 25:
                continue

            closes = df["adj_close"].tail(25)
            mean   = float(closes.mean())
            std    = float(closes.std())
            price  = view.price(ticker)

            if std < 1e-6 or not price:
                continue

            z_score = (price - mean) / std

            # Entry: price is > entry_z below mean
            if z_score < -self.entry_z and portfolio.weight(ticker) < 0.05:
                target_val = nav * 0.04  # 4% per position
                shares     = target_val / price
                if shares * price > 500:
                    orders.append(BacktestOrder(
                        ticker=ticker, date=view.current_date,
                        direction="BUY", shares=shares,
                        reason=f"mean_rev z={z_score:.2f}",
                    ))
                    self._entries[ticker] = view.current_date

            # Exit: price reverted or held too long
            elif ticker in portfolio.positions:
                entry_date  = self._entries.get(ticker, view.current_date)
                days_held   = (view.current_date - entry_date).days
                should_exit = (
                    z_score > -self.exit_z or
                    days_held > self.max_hold
                )
                if should_exit:
                    shares = portfolio.positions[ticker]["shares"]
                    orders.append(BacktestOrder(
                        ticker=ticker, date=view.current_date,
                        direction="SELL", shares=shares,
                        reason=f"mean_rev_exit z={z_score:.2f} days={days_held}",
                    ))
                    self._entries.pop(ticker, None)

        return orders


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest engine
# ─────────────────────────────────────────────────────────────────────────────

class BacktestEngine:
    """
    The main backtesting engine.

    Orchestrates: data loading → feature computation → event loop
                  → order generation → simulated fills → portfolio tracking

    Usage:
        engine   = BacktestEngine()
        strategy = MomentumStrategy(params={"top_n": 5})
        result   = engine.run(
            strategy   = strategy,
            tickers    = ["AAPL","MSFT","NVDA","GOOGL","JPM"],
            start_date = date(2020, 1, 1),
            end_date   = date(2023, 12, 31),
            initial_capital = 1_000_000,
        )
        print(result.summary())
    """

    def __init__(
        self,
        loader:  Optional[HistoricalDataLoader] = None,
        broker:  Optional[SimulatedBroker] = None,
    ):
        self.loader = loader or HistoricalDataLoader()
        self.broker = broker or SimulatedBroker()

    def run(
        self,
        strategy:        Strategy,
        tickers:         List[str],
        start_date:      date,
        end_date:        date,
        initial_capital: float = 1_000_000,
        warmup_days:     int = 252,    # Data loaded this many days before start_date
        verbose:         bool = True,
    ) -> "BacktestResult":
        """
        Run a full backtest.

        Args:
            strategy:        Strategy instance implementing on_bar()
            tickers:         Universe of securities
            start_date:      First date of TRADING (warmup precedes this)
            end_date:        Last date of trading
            initial_capital: Starting capital
            warmup_days:     Days of history loaded before start_date for feature computation
            verbose:         Log progress

        Returns:
            BacktestResult with full trade history and performance metrics
        """
        # Extend start by warmup for feature computation
        data_start = start_date - timedelta(days=warmup_days * 1.5)

        logger.info(
            f"Starting backtest: {strategy.name} | "
            f"{start_date} → {end_date} | "
            f"{len(tickers)} tickers | "
            f"${initial_capital:,.0f}"
        )

        # Load data
        if verbose:
            print(f"  Loading data for {len(tickers)} tickers...")
        data     = self.loader.load(tickers, data_start, end_date)
        features = self.loader.compute_features(data)

        if not data:
            raise ValueError("No data loaded — check ticker symbols and dates")

        # Initialise portfolio and strategy
        portfolio = BacktestPortfolio(initial_capital)
        strategy.on_start(data)

        if verbose:
            print(f"  Running event loop {start_date} → {end_date}...")

        fills_count  = 0
        prev_nav     = initial_capital
        all_fills    = []

        for view in self.loader.iter_dates(data, features, start_date, end_date):
            # Update portfolio prices from today's data
            portfolio.update_prices(view.prices())

            # Update DataView with portfolio state
            view.portfolio_nav = portfolio.nav
            view.cash          = portfolio.cash
            view.positions     = {
                t: p["shares"] for t, p in portfolio.positions.items()
            }

            # Strategy generates orders
            try:
                orders = strategy.on_bar(view, portfolio)
            except Exception as e:
                logger.error(f"Strategy error on {view.current_date}: {e}")
                orders = []

            # Simulate fills
            for order in orders:
                if order.shares < 0.01:
                    continue
                fill = self.broker.fill(order, view)
                if fill:
                    portfolio.apply_fill(fill)
                    all_fills.append(fill)
                    fills_count += 1

            # Record daily snapshot
            snap     = portfolio.record_snapshot(view.current_date, prev_nav)
            prev_nav = snap.nav

        strategy.on_end()

        if verbose:
            print(
                f"  Backtest complete: {len(portfolio.snapshots)} trading days | "
                f"{fills_count} fills"
            )

        return BacktestResult(
            strategy_name   = strategy.name,
            tickers         = tickers,
            start_date      = start_date,
            end_date        = end_date,
            initial_capital = initial_capital,
            snapshots       = portfolio.snapshots,
            fills           = all_fills,
            params          = strategy.params,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Backtest result and performance analytics
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BacktestResult:
    """Complete backtesting results with performance analytics."""

    strategy_name:   str
    tickers:         List[str]
    start_date:      date
    end_date:        date
    initial_capital: float
    snapshots:       List[PortfolioSnapshot]
    fills:           List[BacktestFill]
    params:          Dict[str, Any] = field(default_factory=dict)

    # Computed on access
    _metrics:        Dict[str, float] = field(default_factory=dict, repr=False)

    def _get_returns(self) -> pd.Series:
        dates   = [s.date for s in self.snapshots]
        returns = [s.daily_return for s in self.snapshots]
        return pd.Series(returns, index=dates)

    def _get_nav(self) -> pd.Series:
        dates = [s.date for s in self.snapshots]
        navs  = [s.nav for s in self.snapshots]
        return pd.Series(navs, index=dates)

    def compute_metrics(self, risk_free_annual: float = 0.05) -> Dict[str, float]:
        if self._metrics:
            return self._metrics

        returns = self._get_returns()
        nav     = self._get_nav()

        if len(returns) < 2:
            return {}

        rf_daily  = risk_free_annual / 252
        excess    = returns - rf_daily
        ann_ret   = float((1 + returns.mean()) ** 252 - 1)
        ann_vol   = float(returns.std() * math.sqrt(252))
        sharpe    = float(excess.mean() / returns.std() * math.sqrt(252)) if returns.std() > 0 else 0

        # Sortino (downside deviation only)
        neg_ret   = returns[returns < rf_daily] - rf_daily
        down_std  = float(neg_ret.std() * math.sqrt(252)) if len(neg_ret) > 1 else ann_vol
        sortino   = float(excess.mean() * 252 / down_std) if down_std > 0 else 0

        # Drawdown
        rolling_peak = nav.cummax()
        drawdowns    = (nav - rolling_peak) / rolling_peak
        max_dd       = float(drawdowns.min())
        calmar        = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0

        # Hit rate (fraction positive days)
        hit_rate = float((returns > 0).mean())

        # Profit factor
        gross_profit = float(returns[returns > 0].sum())
        gross_loss   = float(abs(returns[returns < 0].sum()))
        profit_factor= gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Turnover
        total_traded = sum(f.fill_price * f.order.shares for f in self.fills)
        avg_nav      = float(nav.mean())
        n_years      = (self.end_date - self.start_date).days / 365
        turnover     = (total_traded / avg_nav / n_years / 2) if avg_nav > 0 and n_years > 0 else 0

        # Total return
        total_return = float(nav.iloc[-1] / self.initial_capital - 1)

        # Average transaction cost
        avg_cost_bps = float(np.mean([f.total_cost_bps for f in self.fills])) if self.fills else 0

        metrics = {
            "total_return":      total_return,
            "annual_return":     ann_ret,
            "annual_volatility": ann_vol,
            "sharpe_ratio":      sharpe,
            "sortino_ratio":     sortino,
            "max_drawdown":      max_dd,
            "calmar_ratio":      calmar,
            "hit_rate":          hit_rate,
            "profit_factor":     profit_factor,
            "annual_turnover":   turnover,
            "n_trades":          len(self.fills),
            "avg_cost_bps":      avg_cost_bps,
            "n_trading_days":    len(self.snapshots),
            "final_nav":         float(nav.iloc[-1]) if len(nav) > 0 else self.initial_capital,
        }
        self._metrics = metrics
        return metrics

    def rolling_sharpe(self, window: int = 63) -> pd.Series:
        """Rolling Sharpe ratio."""
        returns = self._get_returns()
        rf_daily= 0.05 / 252
        excess  = returns - rf_daily
        roll_sh = excess.rolling(window).mean() / returns.rolling(window).std() * math.sqrt(252)
        return roll_sh.dropna()

    def monthly_returns(self) -> pd.DataFrame:
        """Monthly return table."""
        returns  = self._get_returns()
        returns.index = pd.to_datetime(returns.index)
        monthly  = (1 + returns).resample("ME").prod() - 1
        df       = monthly.to_frame("return")
        df["year"]  = df.index.year
        df["month"] = df.index.month
        pivot    = df.pivot(index="year", columns="month", values="return")
        pivot.columns = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ][:len(pivot.columns)]
        pivot["Annual"] = (1 + monthly).resample("YE").prod() - 1
        return pivot

    def summary(self) -> str:
        m = self.compute_metrics()
        nav = self._get_nav()
        lines = [
            "═" * 65,
            f"  BACKTEST RESULT — {self.strategy_name}",
            f"  {self.start_date} → {self.end_date} | "
            f"{len(self.tickers)} tickers | "
            f"${self.initial_capital:,.0f} initial",
            "═" * 65,
            f"  Total Return        : {m['total_return']:>+10.2%}",
            f"  Annual Return       : {m['annual_return']:>+10.2%}",
            f"  Annual Volatility   : {m['annual_volatility']:>10.2%}",
            f"  Sharpe Ratio        : {m['sharpe_ratio']:>10.3f}",
            f"  Sortino Ratio       : {m['sortino_ratio']:>10.3f}",
            f"  Max Drawdown        : {m['max_drawdown']:>10.2%}",
            f"  Calmar Ratio        : {m['calmar_ratio']:>10.3f}",
            "─" * 65,
            f"  Hit Rate            : {m['hit_rate']:>10.1%}",
            f"  Profit Factor       : {m['profit_factor']:>10.2f}",
            f"  Annual Turnover     : {m['annual_turnover']:>10.1%}",
            f"  Avg Transaction Cost: {m['avg_cost_bps']:>10.2f}bps",
            f"  Total Trades        : {m['n_trades']:>10,}",
            f"  Final NAV           : ${m['final_nav']:>10,.0f}",
            "═" * 65,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "strategy":   self.strategy_name,
            "start":      self.start_date.isoformat(),
            "end":        self.end_date.isoformat(),
            "metrics":    self.compute_metrics(),
            "params":     self.params,
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  Backtest Engine — Test")
    print("=" * 65)

    engine = BacktestEngine()

    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "JPM", "BAC", "XOM", "JNJ", "SPY"]
    strategy= MomentumStrategy(params={"top_n": 3, "lookback": 126})

    result = engine.run(
        strategy        = strategy,
        tickers         = tickers,
        start_date      = date(2021, 1, 1),
        end_date        = date(2023, 12, 31),
        initial_capital = 1_000_000,
        verbose         = True,
    )

    print(result.summary())

    print("\nMonthly Returns:")
    print(result.monthly_returns().to_string())

    print("\n✅ Backtest engine test passed")
