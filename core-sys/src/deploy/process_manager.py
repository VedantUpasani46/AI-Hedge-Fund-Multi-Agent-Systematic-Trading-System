"""
AI Hedge Fund — Part 9: Cloud Production
==========================================
process_manager.py — Background Process Orchestration

In production, the hedge fund system runs as multiple processes:

Process 1: API Server (FastAPI/uvicorn)
    Handles HTTP and WebSocket connections.
    Stateless — can be scaled horizontally.
    Started by: uvicorn directly via CMD in Dockerfile.

Process 2: Risk Monitor
    LiveRiskEngine from Part 7 running continuously.
    Polls prices every 30 seconds.
    Fires circuit breaker alerts via MessageBus.
    This file starts it.

Process 3: Strategy Engine
    Runs the AgentCoordinator from Part 2.
    Scans universe every N hours.
    Generates allocation decisions.
    This file starts it.

Process isolation design:
    Each process has its own SQLite connections (thread-safe via WAL mode).
    Shared state lives in Redis (MessageBus upgrade from Part 2's SQLite).
    Processes communicate via Redis pub/sub — no shared memory.

Why separate processes (not threads)?
    Python GIL: CPU-bound tasks (VaR computation, Monte Carlo) don't
    parallelise within a process due to the GIL.
    Process isolation: a crash in the strategy engine doesn't kill the API.
    Resource limits: Docker can limit each process independently.

Restart policy:
    Each process runs in its own container with `restart: unless-stopped`.
    If it crashes, Docker restarts it within 5 seconds.
    Persistent state (orders, decisions) is in volume-mounted SQLite files
    so nothing is lost on restart.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger("hedge_fund.process_manager")


# ─────────────────────────────────────────────────────────────────────────────
# Risk monitor process
# ─────────────────────────────────────────────────────────────────────────────

class RiskMonitorProcess:
    """
    Runs the LiveRiskEngine continuously.

    Loads the portfolio from the shared database,
    starts monitoring, and keeps running until signalled.
    """

    def __init__(self):
        self._running = True
        self._engine  = None
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Signal {signum} received — shutting down risk monitor")
        self._running = False
        if self._engine:
            self._engine.stop()

    def _load_portfolio(self):
        """Load or create a portfolio for monitoring."""
        from src.data.data_models import Portfolio, Position, Direction

        # In production: load from database
        # Here: use the last known portfolio state from db
        db_path = Path(os.getenv("DB_DIR", "/app/db")) / "portfolio.db"

        if db_path.exists():
            try:
                import sqlite3
                with sqlite3.connect(db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        "SELECT * FROM positions ORDER BY created_at DESC"
                    ).fetchall()
                    if rows:
                        # Reconstruct portfolio
                        portfolio = Portfolio("LIVE_FUND", cash=0, initial_capital=1_000_000)
                        for row in rows:
                            portfolio.positions[row["ticker"]] = Position(
                                ticker        = row["ticker"],
                                direction     = Direction.LONG,
                                shares        = row["shares"],
                                entry_price   = row["avg_cost"],
                                current_price = row["current_price"],
                                sector        = row.get("sector", ""),
                            )
                        nav = sum(
                            p.shares * p.current_price
                            for p in portfolio.positions.values()
                        )
                        portfolio.initial_capital = nav
                        logger.info(
                            f"Loaded portfolio: {len(portfolio.positions)} positions, "
                            f"NAV≈${nav:,.0f}"
                        )
                        return portfolio
            except Exception as e:
                logger.warning(f"Could not load portfolio from DB: {e}")

        # Fallback: empty portfolio (will be updated as trades come in)
        logger.info("Starting with empty portfolio — will populate from trade feed")
        return Portfolio("LIVE_FUND", cash=1_000_000, initial_capital=1_000_000)

    def run(self):
        """Start and run the risk monitor until stopped."""
        logger.info("=== Risk Monitor Process starting ===")

        portfolio = self._load_portfolio()

        from src.risk.live_risk_engine import LiveRiskEngine, build_circuit_breakers
        from src.monitoring.structured_logger import get_monitor_logger

        monitor_log = get_monitor_logger()

        # Alert callback: log and send to Redis
        def on_circuit_breaker(name: str, info: dict):
            logger.critical(f"CIRCUIT BREAKER: {name} — {info}")
            monitor_log.alert(
                event   = "CIRCUIT_BREAKER_TRIGGERED",
                level   = "CRITICAL",
                details = {"breaker": name, **info},
            )
            # Publish to Redis for API server and other consumers
            self._publish_alert(name, info)

        poll_interval = float(os.getenv("MONITOR_INTERVAL", "30"))

        self._engine = LiveRiskEngine(
            portfolio             = portfolio,
            poll_interval_seconds = poll_interval,
            circuit_breakers      = build_circuit_breakers(
                max_daily_loss_pct    = float(os.getenv("MAX_DAILY_LOSS_PCT",   "0.02")),
                max_var_pct           = float(os.getenv("MAX_VAR_PCT",          "0.02")),
                max_intraday_drawdown = float(os.getenv("MAX_INTRADAY_DD",      "0.015")),
            ),
        )
        self._engine.register_alert_callback(on_circuit_breaker)
        self._engine.start()

        logger.info(
            f"Risk monitor running | "
            f"poll={poll_interval}s | "
            f"positions={len(portfolio.positions)}"
        )

        # Keep alive loop
        while self._running:
            snap = self._engine.current_snapshot
            if snap:
                monitor_log.metric(
                    name  = "risk_snapshot",
                    value = snap.nav,
                    tags  = {
                        "daily_pnl_pct":  snap.daily_pnl_pct,
                        "var_95_pct":     snap.var_95_pct,
                        "beta":           snap.portfolio_beta,
                        "risk_level":     snap.risk_level(),
                    }
                )
            time.sleep(60)   # Log every minute

        self._engine.stop()
        logger.info("Risk monitor stopped cleanly")

    def _publish_alert(self, name: str, info: dict):
        """Publish circuit breaker alert to Redis."""
        try:
            import redis, json
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            r.publish("hedgefund:alerts", json.dumps({
                "type":      "circuit_breaker",
                "name":      name,
                "info":      info,
                "timestamp": datetime.now().isoformat(),
            }))
        except Exception as e:
            logger.debug(f"Redis publish failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy engine process
# ─────────────────────────────────────────────────────────────────────────────

class StrategyEngineProcess:
    """
    Runs the AgentCoordinator on a schedule.

    Scans the universe every N hours, generates consensus decisions,
    and queues approved trades for execution.
    """

    def __init__(self):
        self._running = True
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT,  self._handle_signal)

    def _handle_signal(self, signum, frame):
        logger.info(f"Signal {signum} — shutting down strategy engine")
        self._running = False

    def run(self):
        """Run the strategy engine on a schedule."""
        logger.info("=== Strategy Engine Process starting ===")

        from src.monitoring.structured_logger import get_monitor_logger
        from src.data.data_models import Portfolio

        monitor_log = get_monitor_logger()
        scan_interval_hours = float(os.getenv("SCAN_INTERVAL_HOURS", "4"))

        # Load universe from config
        universe = os.getenv(
            "UNIVERSE",
            "AAPL,MSFT,NVDA,GOOGL,META,AMZN,JPM,BAC,GS,XOM,CVX,JNJ,UNH"
        ).split(",")

        logger.info(
            f"Strategy engine running | "
            f"universe={len(universe)} tickers | "
            f"scan_interval={scan_interval_hours}h"
        )

        while self._running:
            scan_start = datetime.now()
            logger.info(f"Starting universe scan: {len(universe)} tickers")

            try:
                portfolio = Portfolio(
                    "LIVE_FUND",
                    cash            = float(os.getenv("INITIAL_CAPITAL", "1000000")),
                    initial_capital = float(os.getenv("INITIAL_CAPITAL", "1000000")),
                )

                from src.agents.agent_coordinator import AgentCoordinator
                coordinator = AgentCoordinator(portfolio)

                results = coordinator.scan_universe(
                    tickers    = universe,
                    fast_mode  = True,
                    top_n      = int(os.getenv("TOP_N_OPPORTUNITIES", "5")),
                )

                buy_decisions = [r for r in results if r.final_decision == "BUY"]

                monitor_log.metric(
                    name  = "strategy_scan",
                    value = len(buy_decisions),
                    tags  = {
                        "universe_size":   len(universe),
                        "buy_decisions":   len(buy_decisions),
                        "total_llm_cost":  coordinator.total_spend(),
                        "duration_seconds":(datetime.now() - scan_start).total_seconds(),
                    }
                )

                logger.info(
                    f"Scan complete: {len(buy_decisions)} BUY decisions | "
                    f"LLM cost=${coordinator.total_spend():.4f} | "
                    f"duration={(datetime.now()-scan_start).total_seconds():.1f}s"
                )

                # Publish decisions for execution
                for decision in buy_decisions:
                    self._queue_decision(decision)

            except Exception as e:
                logger.error(f"Strategy scan failed: {e}")
                monitor_log.alert(
                    event   = "STRATEGY_SCAN_FAILED",
                    level   = "ERROR",
                    details = {"error": str(e)},
                )

            # Wait until next scan
            next_scan = scan_start + timedelta(hours=scan_interval_hours)
            logger.info(f"Next scan at {next_scan:%H:%M:%S}")

            while self._running and datetime.now() < next_scan:
                time.sleep(30)

        logger.info("Strategy engine stopped cleanly")

    def _queue_decision(self, decision) -> None:
        """Queue a BUY decision for execution."""
        try:
            import redis, json
            r = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))
            r.lpush("hedgefund:decisions", json.dumps({
                "ticker":         decision.ticker,
                "final_decision": decision.final_decision,
                "final_weight":   decision.final_weight,
                "avg_confidence": decision.avg_confidence,
                "buy_votes":      decision.buy_votes,
                "timestamp":      datetime.now().isoformat(),
            }))
        except Exception as e:
            logger.debug(f"Redis queue failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level   = os.getenv("LOG_LEVEL", "INFO"),
        format  = "%(asctime)s [%(name)s] %(levelname)s %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="AI Hedge Fund Process Manager")
    parser.add_argument(
        "--process",
        choices = ["risk_monitor", "strategy_engine"],
        required= True,
        help    = "Which process to run",
    )
    args = parser.parse_args()

    if args.process == "risk_monitor":
        RiskMonitorProcess().run()
    elif args.process == "strategy_engine":
        StrategyEngineProcess().run()
