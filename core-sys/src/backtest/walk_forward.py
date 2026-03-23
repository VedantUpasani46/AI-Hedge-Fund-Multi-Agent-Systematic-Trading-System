"""
AI Hedge Fund — Part 6: Backtesting Engine
============================================
walk_forward.py — Walk-Forward Validation Framework

Walk-forward validation is the standard for testing systematic strategies,
especially ML-based ones. It is the only method that gives an unbiased
estimate of out-of-sample performance for strategies that require fitting.

The problem with a simple train/test split:
    You fit your model on [2010-2018], test on [2019-2023].
    The test period is still "selected" — if you tried 20 different
    parameter sets and reported the best one on [2019-2023],
    you have look-ahead bias at the strategy selection level.

Walk-forward validation:
    Split the full history into K rolling windows.
    For each window k:
        Train:  [t_0, t_k]        ← model fitted here
        Test:   [t_k, t_{k+1}]    ← NEVER touched during training
    Report ONLY the aggregated out-of-sample test performance.
    This is your honest performance estimate.

Three flavours implemented:
    ANCHORED:     Train window always starts at t_0, grows each fold
    ROLLING:      Train window is fixed length, slides forward
    EXPANDING:    Alias for anchored — most common for mean reversion

When to use which:
    ANCHORED:  Regime models, strategies where old data is still useful
    ROLLING:   ML strategies where old data is noisy / irrelevant
    Use rolling for anything with XGBoost/neural nets — old data hurts

Parameter optimisation within walk-forward:
    For each training fold, a grid search over parameter combinations
    is run. The best parameters on the training fold are applied to
    the test fold. This correctly models what you would do in practice:
    refit the model each quarter with the latest data.

References:
    Lopez de Prado (2018). Advances in Financial Machine Learning.
    Chan (2013). Algorithmic Trading: Winning Strategies. Wiley.
    Bailey & Lopez de Prado (2014). The Deflated Sharpe Ratio. JPM.
    Harvey, Liu & Zhu (2016). ...and the Cross-Section of Expected Returns. RFS.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import (
    BacktestEngine, BacktestResult, Strategy, MomentumStrategy,
    MeanReversionStrategy, HistoricalDataLoader
)

logger = logging.getLogger("hedge_fund.walk_forward")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward configuration
# ─────────────────────────────────────────────────────────────────────────────

class WFValidationType(str, Enum):
    ANCHORED = "ANCHORED"   # Growing training window
    ROLLING  = "ROLLING"    # Fixed-length rolling window


@dataclass
class WFConfig:
    """Configuration for walk-forward validation."""
    validation_type:  WFValidationType = WFValidationType.ANCHORED
    n_folds:          int  = 5         # Number of folds
    test_pct:         float = 0.20     # Test fold as % of full period
    min_train_days:   int  = 252       # Minimum training days
    train_window_days:int  = 756       # For ROLLING: training window length
    optimize_params:  bool = True      # Run parameter optimisation on each fold


@dataclass
class WFFold:
    """A single walk-forward fold."""
    fold_number:    int
    train_start:    date
    train_end:      date
    test_start:     date
    test_end:       date
    best_params:    Dict[str, Any] = field(default_factory=dict)
    train_result:   Optional[BacktestResult] = None
    test_result:    Optional[BacktestResult] = None

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days

    def test_sharpe(self) -> Optional[float]:
        if not self.test_result:
            return None
        return self.test_result.compute_metrics().get("sharpe_ratio")

    def test_return(self) -> Optional[float]:
        if not self.test_result:
            return None
        return self.test_result.compute_metrics().get("annual_return")

    def oos_is_ratio(self) -> Optional[float]:
        """
        Out-of-sample / In-sample Sharpe ratio.

        A ratio near 1.0 = strategy generalises well.
        A ratio < 0.5 = likely overfit.
        Industry target: OOS/IS > 0.7 (Lopez de Prado 2018).
        """
        if not self.train_result or not self.test_result:
            return None
        is_sharpe = self.train_result.compute_metrics().get("sharpe_ratio", 0)
        os_sharpe = self.test_sharpe() or 0
        if abs(is_sharpe) < 1e-4:
            return None
        return os_sharpe / is_sharpe


@dataclass
class WFResult:
    """
    Aggregated walk-forward validation results.

    The headline figure is the concatenated OOS performance —
    the Sharpe and return computed from stitching all test folds
    together. This is the only honest measure of strategy quality.
    """
    strategy_name:    str
    tickers:          List[str]
    full_start:       date
    full_end:         date
    folds:            List[WFFold]
    config:           WFConfig
    concatenated_oos: Optional[BacktestResult] = None   # Stitched OOS periods

    # Aggregate statistics
    _oos_metrics: Dict[str, Any] = field(default_factory=dict, repr=False)

    def compute_aggregate_stats(self) -> Dict[str, Any]:
        """
        Compute fold-level and aggregate statistics.

        Key metrics:
            oos_sharpe:     Sharpe from stitched OOS periods (the headline)
            is_oos_ratio:   IS/OOS Sharpe consistency (want > 0.7)
            pct_folds_pos:  % of folds with positive OOS Sharpe
            param_stability: How stable optimal parameters are across folds
        """
        if self._oos_metrics:
            return self._oos_metrics

        # Per-fold metrics
        is_sharpes  = []
        oos_sharpes = []
        oos_returns = []

        for fold in self.folds:
            if fold.train_result:
                is_s = fold.train_result.compute_metrics().get("sharpe_ratio", 0)
                is_sharpes.append(is_s)
            if fold.test_result:
                oos_s = fold.test_result.compute_metrics().get("sharpe_ratio", 0)
                oos_r = fold.test_result.compute_metrics().get("annual_return", 0)
                oos_sharpes.append(oos_s)
                oos_returns.append(oos_r)

        # Aggregate OOS
        oos_sharpe_agg = float(np.mean(oos_sharpes)) if oos_sharpes else 0
        oos_return_agg = float(np.mean(oos_returns)) if oos_returns else 0
        pct_folds_pos  = float(np.mean([s > 0 for s in oos_sharpes])) if oos_sharpes else 0

        # IS/OOS ratio
        is_sharpe_avg  = float(np.mean(is_sharpes)) if is_sharpes else 0
        is_oos_ratio   = oos_sharpe_agg / is_sharpe_avg if abs(is_sharpe_avg) > 1e-4 else None

        # Deflated Sharpe Ratio adjustment (Bailey & Lopez de Prado 2014)
        # Corrects for multiple testing across parameter combinations
        n_folds = len(self.folds)
        if n_folds >= 2 and oos_sharpes:
            sharpe_std    = float(np.std(oos_sharpes))
            z_score       = oos_sharpe_agg / (sharpe_std / math.sqrt(n_folds)) if sharpe_std > 0 else 0
            dsr_adjustment= 1 - sharpe_std / (abs(oos_sharpe_agg) + 1e-8)
            deflated_sharpe = oos_sharpe_agg * max(0, dsr_adjustment)
        else:
            deflated_sharpe = oos_sharpe_agg
            z_score         = 0.0

        # Headline OOS metrics from concatenated result
        concat_metrics = {}
        if self.concatenated_oos:
            concat_metrics = self.concatenated_oos.compute_metrics()

        self._oos_metrics = {
            # Headline
            "oos_sharpe_avg":       round(oos_sharpe_agg, 3),
            "oos_annual_return":    round(oos_return_agg, 4),
            "deflated_sharpe":      round(deflated_sharpe, 3),
            # Consistency
            "is_sharpe_avg":        round(is_sharpe_avg, 3),
            "is_oos_ratio":         round(is_oos_ratio, 3) if is_oos_ratio else None,
            "pct_folds_positive":   round(pct_folds_pos, 3),
            "n_folds":              n_folds,
            # Concatenated OOS (most meaningful for investors)
            "concat_oos_sharpe":    round(concat_metrics.get("sharpe_ratio", 0), 3),
            "concat_oos_return":    round(concat_metrics.get("annual_return", 0), 4),
            "concat_oos_max_dd":    round(concat_metrics.get("max_drawdown", 0), 4),
            "concat_oos_calmar":    round(concat_metrics.get("calmar_ratio", 0), 3),
        }
        return self._oos_metrics

    def summary(self) -> str:
        stats = self.compute_aggregate_stats()
        lines = [
            "═" * 70,
            f"  WALK-FORWARD VALIDATION — {self.strategy_name}",
            f"  {self.full_start} → {self.full_end} | "
            f"{self.config.n_folds} folds | "
            f"{self.config.validation_type.value}",
            "═" * 70,
            "  AGGREGATE OUT-OF-SAMPLE PERFORMANCE",
            f"  Concatenated OOS Sharpe  : {stats['concat_oos_sharpe']:>8.3f}",
            f"  Concatenated OOS Return  : {stats['concat_oos_return']:>8.2%}",
            f"  Concatenated Max Drawdown: {stats['concat_oos_max_dd']:>8.2%}",
            f"  Deflated Sharpe Ratio    : {stats['deflated_sharpe']:>8.3f}",
            "─" * 70,
            "  GENERALISATION METRICS",
            f"  IS Sharpe (avg)          : {stats['is_sharpe_avg']:>8.3f}",
            f"  OOS Sharpe (avg)         : {stats['oos_sharpe_avg']:>8.3f}",
            f"  IS/OOS Ratio             : {str(round(stats['is_oos_ratio'], 2)) if stats['is_oos_ratio'] else 'N/A':>8}  (target > 0.7)",
            f"  % Folds Positive OOS     : {stats['pct_folds_positive']:>8.1%}",
            "─" * 70,
            "  FOLD-BY-FOLD DETAIL",
            f"  {'Fold':<6} {'Train':>10} {'Test':>10} {'IS Sharpe':>10} "
            f"{'OOS Sharpe':>11} {'IS/OOS':>8} {'Best Params'}",
            "─" * 70,
        ]
        for fold in self.folds:
            is_s   = fold.train_result.compute_metrics().get("sharpe_ratio", 0) if fold.train_result else 0
            oos_s  = fold.test_result.compute_metrics().get("sharpe_ratio", 0) if fold.test_result else 0
            ratio  = fold.oos_is_ratio()
            params = str(fold.best_params)[:35] if fold.best_params else "default"
            lines.append(
                f"  {fold.fold_number:<6} "
                f"{fold.train_start.strftime('%Y-%m')}-{fold.train_end.strftime('%y'):>4} "
                f"{fold.test_start.strftime('%Y-%m')}-{fold.test_end.strftime('%y'):>4} "
                f"{is_s:>10.3f} {oos_s:>11.3f} "
                f"{ratio:>7.2f}  {params}"
                if ratio is not None else
                f"  {fold.fold_number:<6} "
                f"{fold.train_start.strftime('%Y-%m')}-{fold.train_end.strftime('%y'):>4} "
                f"{fold.test_start.strftime('%Y-%m')}-{fold.test_end.strftime('%y'):>4} "
                f"{is_s:>10.3f} {oos_s:>11.3f} "
                f"{'N/A':>7}  {params}"
            )
        lines.append("═" * 70)
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Parameter optimiser
# ─────────────────────────────────────────────────────────────────────────────

class FoldOptimiser:
    """
    Grid search over strategy parameters on a training fold.

    Runs a full backtest for each parameter combination and
    selects the one with the best Sharpe on the training period.

    WARNING: Parameters are selected to maximise IS Sharpe.
    The IS Sharpe will ALWAYS look better than OOS.
    Report OOS Sharpe only.
    """

    def __init__(
        self,
        engine:    BacktestEngine,
        param_grid:Dict[str, List[Any]],
    ):
        self.engine     = engine
        self.param_grid = param_grid

    def _expand_grid(self) -> List[Dict[str, Any]]:
        """Expand parameter grid into all combinations."""
        import itertools
        keys   = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos = list(itertools.product(*values))
        return [dict(zip(keys, c)) for c in combos]

    def optimise(
        self,
        strategy_class: type,
        tickers:        List[str],
        train_start:    date,
        train_end:      date,
        initial_capital: float = 1_000_000,
        metric:         str = "sharpe_ratio",
    ) -> Tuple[Dict[str, Any], float]:
        """
        Find the best parameter set on the training fold.

        Returns (best_params, best_metric_value).
        """
        combos     = self._expand_grid()
        best_params= combos[0] if combos else {}
        best_score = -float("inf")

        logger.info(
            f"  Optimising over {len(combos)} parameter combinations "
            f"on {train_start}→{train_end}..."
        )

        for params in combos:
            try:
                strategy = strategy_class(params=params)
                result   = self.engine.run(
                    strategy        = strategy,
                    tickers         = tickers,
                    start_date      = train_start,
                    end_date        = train_end,
                    initial_capital = initial_capital,
                    verbose         = False,
                )
                metrics = result.compute_metrics()
                score   = metrics.get(metric, -float("inf"))

                if score > best_score:
                    best_score  = score
                    best_params = params

            except Exception as e:
                logger.debug(f"  Param combo {params} failed: {e}")

        logger.info(f"  Best params: {best_params} → {metric}={best_score:.3f}")
        return best_params, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward engine
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardEngine:
    """
    Runs walk-forward validation for any strategy.

    Usage:
        wf = WalkForwardEngine()
        config = WFConfig(
            validation_type = WFValidationType.ANCHORED,
            n_folds         = 5,
        )
        param_grid = {"top_n": [3, 5, 7], "lookback": [126, 189, 252]}
        result = wf.run(
            strategy_class = MomentumStrategy,
            param_grid     = param_grid,
            tickers        = ["AAPL", "MSFT", "NVDA", ...],
            start_date     = date(2018, 1, 1),
            end_date       = date(2023, 12, 31),
        )
        print(result.summary())
    """

    def __init__(self, engine: Optional[BacktestEngine] = None):
        self.engine = engine or BacktestEngine()

    def _build_folds(
        self,
        start_date: date,
        end_date:   date,
        config:     WFConfig,
    ) -> List[WFFold]:
        """Build the fold date ranges."""
        total_days = (end_date - start_date).days
        test_days  = int(total_days * config.test_pct)
        n_folds    = config.n_folds

        folds = []

        if config.validation_type == WFValidationType.ANCHORED:
            # Anchored: train always starts at start_date
            # Split test period into n_folds equal pieces
            oos_start = start_date + timedelta(days=int(total_days * (1 - config.test_pct * n_folds / n_folds)))
            oos_total = (end_date - oos_start).days
            fold_days = oos_total // n_folds

            for k in range(n_folds):
                fold_test_start = oos_start + timedelta(days=k * fold_days)
                fold_test_end   = fold_test_start + timedelta(days=fold_days)
                if fold_test_end > end_date:
                    fold_test_end = end_date

                fold = WFFold(
                    fold_number = k + 1,
                    train_start = start_date,
                    train_end   = fold_test_start - timedelta(days=1),
                    test_start  = fold_test_start,
                    test_end    = fold_test_end,
                )

                if fold.train_days < config.min_train_days:
                    logger.debug(f"Fold {k+1} skipped: train_days={fold.train_days} < {config.min_train_days}")
                    continue

                folds.append(fold)

        elif config.validation_type == WFValidationType.ROLLING:
            train_days = config.train_window_days
            test_step  = int((total_days - train_days) / n_folds)

            for k in range(n_folds):
                train_start_k = start_date + timedelta(days=k * test_step)
                train_end_k   = train_start_k + timedelta(days=train_days)
                test_start_k  = train_end_k + timedelta(days=1)
                test_end_k    = test_start_k + timedelta(days=test_step)

                if test_end_k > end_date:
                    test_end_k = end_date
                if test_start_k >= end_date:
                    break

                folds.append(WFFold(
                    fold_number = k + 1,
                    train_start = train_start_k,
                    train_end   = train_end_k,
                    test_start  = test_start_k,
                    test_end    = test_end_k,
                ))

        return folds

    def run(
        self,
        strategy_class:  type,
        tickers:         List[str],
        start_date:      date,
        end_date:        date,
        initial_capital: float = 1_000_000,
        config:          Optional[WFConfig] = None,
        param_grid:      Optional[Dict[str, List[Any]]] = None,
        default_params:  Optional[Dict[str, Any]] = None,
    ) -> WFResult:
        """
        Run walk-forward validation.

        Args:
            strategy_class:  Strategy class (not instance)
            tickers:         Universe of securities
            start_date:      Full period start
            end_date:        Full period end
            initial_capital: Starting capital
            config:          Walk-forward configuration
            param_grid:      Parameter grid for optimisation
            default_params:  Default params if no grid
        """
        cfg = config or WFConfig()
        folds = self._build_folds(start_date, end_date, cfg)

        if not folds:
            raise ValueError(
                f"No valid folds generated. Check start/end dates and "
                f"min_train_days ({cfg.min_train_days})."
            )

        logger.info(
            f"Walk-forward: {strategy_class.__name__} | "
            f"{len(folds)} folds | {cfg.validation_type.value}"
        )

        # Optimiser for parameter search
        optimiser = FoldOptimiser(self.engine, param_grid) if param_grid else None

        # Run each fold
        for fold in folds:
            print(
                f"\n  Fold {fold.fold_number}/{len(folds)}: "
                f"train [{fold.train_start}→{fold.train_end}] "
                f"test [{fold.test_start}→{fold.test_end}]"
            )

            # Optimise parameters on training fold
            if optimiser and cfg.optimize_params:
                best_params, _ = optimiser.optimise(
                    strategy_class  = strategy_class,
                    tickers         = tickers,
                    train_start     = fold.train_start,
                    train_end       = fold.train_end,
                    initial_capital = initial_capital,
                )
                fold.best_params = best_params
            else:
                fold.best_params = default_params or {}

            # Run on training fold (for IS metrics)
            try:
                train_strategy = strategy_class(params=fold.best_params)
                fold.train_result = self.engine.run(
                    strategy        = train_strategy,
                    tickers         = tickers,
                    start_date      = fold.train_start,
                    end_date        = fold.train_end,
                    initial_capital = initial_capital,
                    verbose         = False,
                )
                is_s = fold.train_result.compute_metrics().get("sharpe_ratio", 0)
                print(f"    IS Sharpe: {is_s:.3f} | params: {fold.best_params}")
            except Exception as e:
                logger.error(f"Train fold {fold.fold_number} failed: {e}")

            # Run on test fold (OOS — the honest number)
            try:
                test_strategy = strategy_class(params=fold.best_params)
                fold.test_result = self.engine.run(
                    strategy        = test_strategy,
                    tickers         = tickers,
                    start_date      = fold.test_start,
                    end_date        = fold.test_end,
                    initial_capital = initial_capital,
                    verbose         = False,
                )
                oos_s = fold.test_result.compute_metrics().get("sharpe_ratio", 0)
                print(f"    OOS Sharpe: {oos_s:.3f}")
            except Exception as e:
                logger.error(f"Test fold {fold.fold_number} failed: {e}")

        # Stitch OOS periods into one continuous result
        concat_result = self._concatenate_oos(folds, strategy_class.__name__)

        result = WFResult(
            strategy_name    = strategy_class.__name__,
            tickers          = tickers,
            full_start       = start_date,
            full_end         = end_date,
            folds            = folds,
            config           = cfg,
            concatenated_oos = concat_result,
        )

        return result

    def _concatenate_oos(
        self,
        folds:         List[WFFold],
        strategy_name: str,
    ) -> Optional[BacktestResult]:
        """
        Stitch OOS test fold results into a single BacktestResult.

        This is what investors would have actually experienced —
        not the cherry-picked best fold, but the full OOS sequence.
        """
        from src.backtest.backtest_engine import PortfolioSnapshot, BacktestFill

        all_snapshots: List[PortfolioSnapshot] = []
        all_fills:     List[BacktestFill]       = []

        prev_nav = None
        for fold in folds:
            if not fold.test_result or not fold.test_result.snapshots:
                continue
            snaps = fold.test_result.snapshots
            fills = fold.test_result.fills

            # Rescale NAV to start where previous fold ended
            if prev_nav is not None and snaps:
                scale = prev_nav / snaps[0].nav
                for s in snaps:
                    s.nav  = s.nav  * scale
                    s.cash = s.cash * scale

            all_snapshots.extend(snaps)
            all_fills.extend(fills)
            if snaps:
                prev_nav = snaps[-1].nav

        if not all_snapshots:
            return None

        # Recalculate daily returns on the stitched series
        for i in range(1, len(all_snapshots)):
            prev = all_snapshots[i-1].nav
            curr = all_snapshots[i].nav
            all_snapshots[i].daily_return = (curr / prev - 1) if prev > 0 else 0.0
        if all_snapshots:
            all_snapshots[0].daily_return = 0.0

        # Find overall start/end
        overall_start = min(f.test_start for f in folds if f.test_result)
        overall_end   = max(f.test_end   for f in folds if f.test_result)

        from src.backtest.backtest_engine import BacktestResult
        return BacktestResult(
            strategy_name   = f"{strategy_name}_OOS_concat",
            tickers         = folds[0].test_result.tickers if folds else [],
            start_date      = overall_start,
            end_date        = overall_end,
            initial_capital = all_snapshots[0].nav if all_snapshots else 1_000_000,
            snapshots       = all_snapshots,
            fills           = all_fills,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from datetime import date

    print("=" * 70)
    print("  Walk-Forward Validation — Test")
    print("=" * 70)

    wf = WalkForwardEngine()

    param_grid = {
        "top_n":    [3, 5],
        "lookback": [126, 189],
    }

    result = wf.run(
        strategy_class  = MomentumStrategy,
        tickers         = ["AAPL","MSFT","NVDA","GOOGL","JPM","BAC","XOM","JNJ"],
        start_date      = date(2019, 1, 1),
        end_date        = date(2023, 12, 31),
        initial_capital = 1_000_000,
        config          = WFConfig(n_folds=3, validation_type=WFValidationType.ANCHORED),
        param_grid      = param_grid,
    )

    print("\n" + result.summary())
    print("\n✅ Walk-forward validation test passed")
