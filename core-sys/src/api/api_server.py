"""
AI Hedge Fund — Part 8: Investor Dashboard
============================================
api_server.py — FastAPI REST + WebSocket Server

This is the fund's operational interface. It exposes every
meaningful piece of the system as a clean HTTP API that:

    1. The React frontend consumes to build the investor portal
    2. Investors access via their browser (with auth)
    3. Counterparties / prime brokers query programmatically
    4. The fund administrator uses for NAV reconciliation
    5. You use during due diligence demos

Endpoints:
    /                       Health check + system status
    /portfolio              Current NAV, positions, weights
    /portfolio/history      NAV time series for charting
    /risk                   Live risk snapshot (VaR, beta, drawdown)
    /risk/factors           Current factor exposures
    /risk/history           Intraday risk time series
    /risk/circuit-breakers  Circuit breaker status
    /performance            Full performance metrics (Sharpe, etc.)
    /performance/monthly    Monthly return table
    /trades                 Recent execution history
    /trades/{order_id}      Single trade detail
    /reports/generate       Trigger PDF report generation
    /reports/list           List available reports
    /reports/{report_id}    Download PDF report
    /ws/risk                WebSocket: live risk stream

WebSocket stream (wss://host/ws/risk):
    Pushes RiskSnapshot JSON every 30 seconds during market hours.
    React frontend subscribes and updates the dashboard in real-time
    without polling.

Authentication:
    API key in header: X-API-Key: <key>
    Keys stored in .env: API_KEYS=key1,key2,key3
    In production: replace with JWT + OAuth2

CORS:
    Configured for localhost:3000 (React dev server)
    In production: restrict to your domain
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sqlite3
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.api")

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models (request/response schemas)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from pydantic import BaseModel, Field

    class PositionOut(BaseModel):
        ticker:         str
        shares:         float
        avg_cost:       float
        current_price:  float
        market_value:   float
        unrealised_pnl: float
        weight_pct:     float
        sector:         str = ""

    class PortfolioOut(BaseModel):
        portfolio_id:    str
        nav:             float
        cash:            float
        invested_value:  float
        invested_pct:    float
        unrealised_pnl:  float
        unrealised_pnl_pct: float
        n_positions:     int
        positions:       List[PositionOut]
        as_of:           str

    class RiskOut(BaseModel):
        timestamp:       str
        nav:             float
        daily_pnl_pct:   float
        var_95_pct:      float
        var_99_pct:      float
        portfolio_beta:  float
        intraday_drawdown_pct: float
        trailing_drawdown_pct: float
        top_position_weight:   float
        risk_level:      str
        breaches:        List[str]
        warnings:        List[str]

    class PerformanceOut(BaseModel):
        start_date:      str
        end_date:        str
        total_return:    float
        annual_return:   float
        annual_vol:      float
        sharpe_ratio:    float
        sortino_ratio:   float
        max_drawdown:    float
        calmar_ratio:    float
        hit_rate:        float
        n_trading_days:  int
        final_nav:       float

    class TradeOut(BaseModel):
        order_id:    str
        ticker:      str
        side:        str
        quantity:    float
        avg_fill:    Optional[float]
        commission:  float
        status:      str
        is_bps:      Optional[float]
        created_at:  str
        filled_at:   Optional[str]

    PYDANTIC_AVAILABLE = True

except ImportError:
    PYDANTIC_AVAILABLE = False
    logger.warning("pydantic not installed — using dict responses")


# ─────────────────────────────────────────────────────────────────────────────
# API state — shared data store
# ─────────────────────────────────────────────────────────────────────────────

class APIState:
    """
    Shared state for the API server.

    In production this would be backed by Redis for
    multi-process support. For single-process operation
    (which covers up to ~$50M AUM comfortably), this works fine.
    """

    def __init__(self):
        self.portfolio        = None
        self.risk_engine      = None
        self.factor_monitor   = None
        self.db_path          = Path(__file__).parents[3] / "db"
        self._nav_history:    List[Dict] = []
        self._start_date:     Optional[date] = None
        self._initial_capital: float = 1_000_000

    def set_portfolio(self, portfolio) -> None:
        self.portfolio         = portfolio
        self._initial_capital  = getattr(portfolio, "initial_capital", 1_000_000)
        self._start_date       = date.today()

    def set_risk_engine(self, engine) -> None:
        self.risk_engine = engine

    def set_factor_monitor(self, monitor) -> None:
        self.factor_monitor = monitor

    def get_portfolio_dict(self) -> Dict:
        if not self.portfolio:
            return _demo_portfolio()

        positions = []
        total_invested = 0.0
        total_upnl     = 0.0

        for ticker, pos in self.portfolio.positions.items():
            shares        = getattr(pos, "shares", 0)
            avg_cost      = getattr(pos, "avg_cost", 0) or getattr(pos, "entry_price", 0)
            current_price = getattr(pos, "current_price", avg_cost)
            market_value  = shares * current_price
            upnl          = (current_price - avg_cost) * shares
            total_invested += market_value
            total_upnl     += upnl

        nav = self.portfolio.net_asset_value

        for ticker, pos in self.portfolio.positions.items():
            shares        = getattr(pos, "shares", 0)
            avg_cost      = getattr(pos, "avg_cost", 0) or getattr(pos, "entry_price", 0)
            current_price = getattr(pos, "current_price", avg_cost)
            market_value  = shares * current_price
            upnl          = (current_price - avg_cost) * shares
            positions.append({
                "ticker":         ticker,
                "shares":         shares,
                "avg_cost":       round(avg_cost, 4),
                "current_price":  round(current_price, 4),
                "market_value":   round(market_value, 2),
                "unrealised_pnl": round(upnl, 2),
                "weight_pct":     round(market_value / nav * 100, 2) if nav > 0 else 0,
                "sector":         getattr(pos, "sector", ""),
            })

        positions.sort(key=lambda x: -abs(x["market_value"]))

        return {
            "portfolio_id":       getattr(self.portfolio, "portfolio_id", "FUND_001"),
            "nav":                round(nav, 2),
            "cash":               round(self.portfolio.cash, 2),
            "invested_value":     round(total_invested, 2),
            "invested_pct":       round(total_invested / nav * 100, 2) if nav > 0 else 0,
            "unrealised_pnl":     round(total_upnl, 2),
            "unrealised_pnl_pct": round(total_upnl / self._initial_capital * 100, 2),
            "n_positions":        len(positions),
            "positions":          positions,
            "as_of":              datetime.now().isoformat(),
        }

    def get_risk_dict(self) -> Dict:
        if self.risk_engine and self.risk_engine.current_snapshot:
            snap = self.risk_engine.current_snapshot
            return snap.to_dict()
        if self.risk_engine:
            snap = self.risk_engine._compute_snapshot()
            if snap:
                return snap.to_dict()
        return _demo_risk()

    def get_nav_history(self, days: int = 30) -> List[Dict]:
        """Return NAV time series for charting."""
        history = list(self._nav_history)

        # Also pull from risk monitor DB if available
        db = self.db_path / "risk_monitor.db"
        if db.exists():
            try:
                cutoff = (datetime.now() - timedelta(days=days)).isoformat()
                with sqlite3.connect(db) as conn:
                    rows = conn.execute("""
                        SELECT timestamp, nav, daily_pnl_pct
                        FROM risk_snapshots
                        WHERE timestamp >= ?
                        ORDER BY timestamp ASC
                    """, (cutoff,)).fetchall()
                for r in rows:
                    history.append({
                        "timestamp": r[0],
                        "nav":       r[1],
                        "daily_pnl_pct": r[2],
                    })
            except Exception as e:
                logger.debug(f"Nav history DB read failed: {e}")

        if not history:
            # Generate synthetic history for demo
            history = _generate_demo_nav_history(self._initial_capital, days)

        return sorted(history, key=lambda x: x["timestamp"])

    def get_performance_dict(self) -> Dict:
        history = self.get_nav_history(days=365)
        if len(history) < 2:
            return _demo_performance()

        navs = [h["nav"] for h in history]
        returns = [
            (navs[i] / navs[i-1] - 1) if navs[i-1] > 0 else 0
            for i in range(1, len(navs))
        ]
        if not returns:
            return _demo_performance()

        r = np.array(returns)
        ann_ret  = float((1 + r.mean()) ** 252 - 1)
        ann_vol  = float(r.std() * np.sqrt(252))
        sharpe   = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0
        neg      = r[r < 0.05/252]
        sortino  = (ann_ret - 0.05) / (neg.std() * np.sqrt(252)) if len(neg) > 0 else 0

        nav_s    = np.array(navs)
        peak     = np.maximum.accumulate(nav_s)
        dd       = (nav_s - peak) / peak
        max_dd   = float(dd.min())
        calmar   = ann_ret / abs(max_dd) if max_dd != 0 else 0

        total_ret = float(navs[-1] / navs[0] - 1)
        hit_rate  = float(np.mean(r > 0))

        start = history[0]["timestamp"][:10]
        end   = history[-1]["timestamp"][:10]

        return {
            "start_date":    start,
            "end_date":      end,
            "total_return":  round(total_ret, 4),
            "annual_return": round(ann_ret, 4),
            "annual_vol":    round(ann_vol, 4),
            "sharpe_ratio":  round(sharpe, 3),
            "sortino_ratio": round(sortino, 3),
            "max_drawdown":  round(max_dd, 4),
            "calmar_ratio":  round(calmar, 3),
            "hit_rate":      round(hit_rate, 3),
            "n_trading_days":len(returns),
            "final_nav":     round(navs[-1], 2),
            "initial_nav":   round(navs[0], 2),
        }

    def get_monthly_returns(self) -> Dict:
        history = self.get_nav_history(days=365 * 3)
        if len(history) < 5:
            return {"note": "Insufficient history for monthly returns"}

        df = pd.DataFrame(history)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp").sort_index()

        monthly = (1 + df["daily_pnl_pct"].fillna(0)).resample("ME").prod() - 1
        result   = {}
        for dt, ret in monthly.items():
            year  = str(dt.year)
            month = dt.strftime("%b")
            if year not in result:
                result[year] = {}
            result[year][month] = round(float(ret) * 100, 2)

        return result

    def get_trades(self, limit: int = 50) -> List[Dict]:
        db = self.db_path / "execution.db"
        if not db.exists():
            return _demo_trades()
        try:
            with sqlite3.connect(db) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT order_id, ticker, side, quantity, avg_fill,
                           commission, status, is_bps, created_at, filled_at
                    FROM orders
                    ORDER BY created_at DESC
                    LIMIT ?
                """, (limit,)).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.debug(f"Trade history read failed: {e}")
            return _demo_trades()


# Singleton
_state = APIState()


def get_state() -> APIState:
    return _state


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI application
# ─────────────────────────────────────────────────────────────────────────────

def create_app(portfolio=None, risk_engine=None, factor_monitor=None):
    """
    Create and configure the FastAPI application.

    Args:
        portfolio:      Portfolio object (from Part 1)
        risk_engine:    LiveRiskEngine (from Part 7)
        factor_monitor: FactorMonitor (from Part 7)

    Returns configured FastAPI app.
    """
    try:
        from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, FileResponse
        from fastapi.security.api_key import APIKeyHeader
    except ImportError:
        raise ImportError(
            "FastAPI not installed. Run: pip install fastapi uvicorn"
        )

    state = get_state()
    if portfolio:
        state.set_portfolio(portfolio)
    if risk_engine:
        state.set_risk_engine(risk_engine)
    if factor_monitor:
        state.set_factor_monitor(factor_monitor)

    app = FastAPI(
        title       = "AI Hedge Fund — Investor Dashboard API",
        description = "Real-time portfolio monitoring and investor reporting",
        version     = "1.0.0",
    )

    # CORS for React frontend
    app.add_middleware(
        CORSMiddleware,
        allow_origins     = ["http://localhost:3000", "http://localhost:5173",
                              "http://127.0.0.1:3000", "http://127.0.0.1:5173"],
        allow_credentials = True,
        allow_methods     = ["*"],
        allow_headers     = ["*"],
    )

    # API key auth
    api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

    def verify_api_key(api_key: str = Depends(api_key_header)):
        valid_keys = os.getenv("API_KEYS", "dev-key-1,dev-key-2").split(",")
        if api_key not in valid_keys:
            # In development, skip auth if no key provided
            if os.getenv("ENV", "development") == "development":
                return "dev"
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key

    # ── Health & system ────────────────────────────────────────────────────────

    @app.get("/")
    async def health():
        return {
            "status":   "running",
            "version":  "1.0.0",
            "timestamp":datetime.now().isoformat(),
            "components": {
                "portfolio":     state.portfolio is not None,
                "risk_engine":   state.risk_engine is not None,
                "factor_monitor":state.factor_monitor is not None,
            }
        }

    @app.get("/health")
    async def health_detail():
        return {
            "api":           "healthy",
            "nav":           state.get_portfolio_dict().get("nav", 0),
            "risk_level":    state.get_risk_dict().get("risk_level", "UNKNOWN"),
            "trading_halted":getattr(state.risk_engine, "is_halted", False) if state.risk_engine else False,
            "timestamp":     datetime.now().isoformat(),
        }

    # ── Portfolio endpoints ────────────────────────────────────────────────────

    @app.get("/portfolio")
    async def get_portfolio(_: str = Depends(verify_api_key)):
        return state.get_portfolio_dict()

    @app.get("/portfolio/history")
    async def get_portfolio_history(
        days: int = Query(default=30, ge=1, le=365),
        _: str = Depends(verify_api_key),
    ):
        return {"history": state.get_nav_history(days=days), "days": days}

    @app.get("/portfolio/positions")
    async def get_positions(_: str = Depends(verify_api_key)):
        p = state.get_portfolio_dict()
        return {
            "n_positions": p["n_positions"],
            "positions":   p["positions"],
            "as_of":       p["as_of"],
        }

    @app.get("/portfolio/sector-allocation")
    async def get_sector_allocation(_: str = Depends(verify_api_key)):
        p = state.get_portfolio_dict()
        sectors: Dict[str, float] = {}
        for pos in p.get("positions", []):
            sector = pos.get("sector") or "Unknown"
            sectors[sector] = sectors.get(sector, 0) + pos.get("weight_pct", 0)
        return {"sectors": sectors, "as_of": p.get("as_of")}

    # ── Risk endpoints ────────────────────────────────────────────────────────

    @app.get("/risk")
    async def get_risk(_: str = Depends(verify_api_key)):
        return state.get_risk_dict()

    @app.get("/risk/factors")
    async def get_factor_exposures(_: str = Depends(verify_api_key)):
        if state.factor_monitor:
            p    = state.get_portfolio_dict()
            positions = {
                pos["ticker"]: pos["market_value"]
                for pos in p.get("positions", [])
            }
            nav  = p.get("nav", 1_000_000)
            snap = state.factor_monitor.compute(positions, nav)
            return {
                "timestamp":  snap.timestamp.isoformat(),
                "r_squared":  snap.r_squared,
                "alpha_daily":snap.alpha_estimate,
                "factors": {
                    name: {
                        "beta":   exp.portfolio_beta,
                        "status": exp.status(),
                        "target": exp.target_beta,
                    }
                    for name, exp in snap.exposures.items()
                },
            }
        return {"note": "Factor monitor not configured"}

    @app.get("/risk/history")
    async def get_risk_history(
        hours: int = Query(default=8, ge=1, le=48),
        _: str = Depends(verify_api_key),
    ):
        if state.risk_engine:
            history = state.risk_engine.get_snapshot_history(hours=hours)
            return {
                "n_snapshots": len(history),
                "snapshots": [s.to_dict() for s in history[-200:]],  # Last 200
            }
        return {"snapshots": [], "note": "Risk engine not running"}

    @app.get("/risk/circuit-breakers")
    async def get_circuit_breakers(_: str = Depends(verify_api_key)):
        if not state.risk_engine:
            return {"circuit_breakers": [], "trading_halted": False}
        return {
            "trading_halted": state.risk_engine.is_halted,
            "any_breached":   state.risk_engine.any_breached(),
            "circuit_breakers": [
                {
                    "name":        cb.name,
                    "description": cb.description,
                    "severity":    cb.severity,
                    "triggered":   cb.triggered,
                    "triggered_at":cb.triggered_at.isoformat() if cb.triggered_at else None,
                }
                for cb in state.risk_engine.circuit_breakers
            ],
        }

    @app.post("/risk/reset-circuit-breakers")
    async def reset_circuit_breakers(_: str = Depends(verify_api_key)):
        if state.risk_engine:
            n = state.risk_engine.reset_circuit_breakers()
            return {"reset": n, "timestamp": datetime.now().isoformat()}
        return {"reset": 0, "note": "Risk engine not running"}

    # ── Performance endpoints ─────────────────────────────────────────────────

    @app.get("/performance")
    async def get_performance(_: str = Depends(verify_api_key)):
        return state.get_performance_dict()

    @app.get("/performance/monthly")
    async def get_monthly_returns(_: str = Depends(verify_api_key)):
        return state.get_monthly_returns()

    @app.get("/performance/benchmark")
    async def get_vs_benchmark(
        benchmark: str = Query(default="SPY"),
        _: str = Depends(verify_api_key),
    ):
        """Compare portfolio performance vs benchmark."""
        try:
            import yfinance as yf
            history = state.get_nav_history(days=365)
            if len(history) < 10:
                return {"note": "Insufficient history"}

            start = history[0]["timestamp"][:10]
            end   = history[-1]["timestamp"][:10]

            bench = yf.download(benchmark, start=start, end=end,
                               progress=False, auto_adjust=True)
            if bench.empty:
                return {"note": f"Could not fetch {benchmark} data"}

            col = "Close" if "Close" in bench.columns else bench.columns[0]
            bench_total = float(bench[col].iloc[-1] / bench[col].iloc[0] - 1)
            fund_total  = state.get_performance_dict().get("total_return", 0)

            return {
                "benchmark":        benchmark,
                "fund_return":      round(fund_total, 4),
                "benchmark_return": round(bench_total, 4),
                "excess_return":    round(fund_total - bench_total, 4),
                "start_date":       start,
                "end_date":         end,
            }
        except Exception as e:
            return {"note": f"Benchmark comparison failed: {e}"}

    # ── Trade endpoints ───────────────────────────────────────────────────────

    @app.get("/trades")
    async def get_trades(
        limit: int = Query(default=50, ge=1, le=500),
        _: str = Depends(verify_api_key),
    ):
        return {"trades": state.get_trades(limit=limit), "count": limit}

    @app.get("/trades/{order_id}")
    async def get_trade(order_id: str, _: str = Depends(verify_api_key)):
        trades = state.get_trades(limit=500)
        for t in trades:
            if t.get("order_id") == order_id:
                return t
        raise HTTPException(status_code=404, detail=f"Trade {order_id} not found")

    # ── Report endpoints ──────────────────────────────────────────────────────

    @app.post("/reports/generate")
    async def generate_report(
        report_type: str = Query(default="monthly"),
        _: str = Depends(verify_api_key),
    ):
        """Trigger PDF report generation."""
        try:
            from src.reports.pdf_generator import generate_fund_report
            report_path = generate_fund_report(
                portfolio_data  = state.get_portfolio_dict(),
                performance_data= state.get_performance_dict(),
                risk_data       = state.get_risk_dict(),
                nav_history     = state.get_nav_history(days=30),
                report_type     = report_type,
            )
            report_id = Path(report_path).name
            return {
                "status":     "generated",
                "report_id":  report_id,
                "report_type":report_type,
                "download_url":f"/reports/{report_id}",
                "generated_at":datetime.now().isoformat(),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")

    @app.get("/reports/list")
    async def list_reports(_: str = Depends(verify_api_key)):
        reports_dir = Path(__file__).parents[3] / "reports"
        if not reports_dir.exists():
            return {"reports": []}
        pdfs = sorted(reports_dir.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
        return {
            "reports": [
                {
                    "report_id":  p.name,
                    "size_kb":    round(p.stat().st_size / 1024, 1),
                    "created_at": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
                    "download_url": f"/reports/{p.name}",
                }
                for p in pdfs[:20]
            ]
        }

    @app.get("/reports/{report_id}")
    async def download_report(report_id: str, _: str = Depends(verify_api_key)):
        report_path = Path(__file__).parents[3] / "reports" / report_id
        if not report_path.exists():
            raise HTTPException(status_code=404, detail="Report not found")
        return FileResponse(
            path         = str(report_path),
            media_type   = "application/pdf",
            filename     = report_id,
        )

    # ── WebSocket — live risk stream ──────────────────────────────────────────

    class ConnectionManager:
        def __init__(self):
            self.active: List[WebSocket] = []

        async def connect(self, ws: WebSocket):
            await ws.accept()
            self.active.append(ws)
            logger.info(f"WS connected: {len(self.active)} clients")

        def disconnect(self, ws: WebSocket):
            if ws in self.active:
                self.active.remove(ws)

        async def broadcast(self, data: str):
            dead = []
            for ws in self.active:
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws)

    ws_manager = ConnectionManager()

    @app.websocket("/ws/risk")
    async def ws_risk_stream(websocket: WebSocket):
        """
        Live risk stream via WebSocket.

        The React dashboard subscribes here for real-time updates.
        Sends a RiskSnapshot JSON every 30 seconds.
        """
        await ws_manager.connect(websocket)
        try:
            # Send initial snapshot immediately
            risk_data = state.get_risk_dict()
            await websocket.send_text(json.dumps({
                "type": "risk_snapshot",
                "data": risk_data,
            }))

            # Then stream every 30s
            while True:
                await asyncio.sleep(30)
                try:
                    risk_data = state.get_risk_dict()
                    await websocket.send_text(json.dumps({
                        "type": "risk_snapshot",
                        "data": risk_data,
                    }))
                except Exception as e:
                    logger.debug(f"WS send failed: {e}")
                    break
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    @app.websocket("/ws/portfolio")
    async def ws_portfolio_stream(websocket: WebSocket):
        """Live portfolio updates via WebSocket."""
        await ws_manager.connect(websocket)
        try:
            while True:
                port_data = state.get_portfolio_dict()
                await websocket.send_text(json.dumps({
                    "type": "portfolio_update",
                    "data": port_data,
                }))
                await asyncio.sleep(60)   # Portfolio updates every minute
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Demo data generators
# ─────────────────────────────────────────────────────────────────────────────

def _demo_portfolio() -> Dict:
    return {
        "portfolio_id": "FUND_001",
        "nav": 1_042_500,
        "cash": 210_000,
        "invested_value": 832_500,
        "invested_pct": 79.9,
        "unrealised_pnl": 42_500,
        "unrealised_pnl_pct": 4.25,
        "n_positions": 5,
        "positions": [
            {"ticker":"AAPL","shares":500,"avg_cost":180.0,"current_price":195.0,"market_value":97500,"unrealised_pnl":7500,"weight_pct":9.35,"sector":"Technology"},
            {"ticker":"MSFT","shares":300,"avg_cost":380.0,"current_price":415.0,"market_value":124500,"unrealised_pnl":10500,"weight_pct":11.94,"sector":"Technology"},
            {"ticker":"NVDA","shares":150,"avg_cost":450.0,"current_price":485.0,"market_value":72750,"unrealised_pnl":5250,"weight_pct":6.98,"sector":"Technology"},
            {"ticker":"JPM", "shares":800,"avg_cost":185.0,"current_price":198.0,"market_value":158400,"unrealised_pnl":10400,"weight_pct":15.19,"sector":"Financials"},
            {"ticker":"XOM", "shares":600,"avg_cost":105.0,"current_price":112.0,"market_value":67200,"unrealised_pnl":4200,"weight_pct":6.45,"sector":"Energy"},
        ],
        "as_of": datetime.now().isoformat(),
    }

def _demo_risk() -> Dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "nav": 1_042_500,
        "daily_pnl_pct": 0.0047,
        "var_95_pct": 0.0142,
        "var_99_pct": 0.0201,
        "portfolio_beta": 0.92,
        "intraday_drawdown_pct": -0.0012,
        "trailing_drawdown_pct": -0.0215,
        "top_position_weight": 0.1519,
        "risk_level": "GREEN",
        "breaches": [],
        "warnings": [],
    }

def _demo_performance() -> Dict:
    return {
        "start_date":    "2023-01-01",
        "end_date":      date.today().isoformat(),
        "total_return":  0.1425,
        "annual_return": 0.1318,
        "annual_vol":    0.1241,
        "sharpe_ratio":  1.28,
        "sortino_ratio": 1.74,
        "max_drawdown":  -0.0873,
        "calmar_ratio":  1.51,
        "hit_rate":      0.535,
        "n_trading_days":250,
        "final_nav":     1_142_500,
        "initial_nav":   1_000_000,
    }

def _demo_trades() -> List[Dict]:
    return [
        {"order_id":"ORD_ABC123","ticker":"AAPL","side":"BUY","quantity":100,"avg_fill":194.85,"commission":0.50,"status":"FILLED","is_bps":2.1,"created_at":datetime.now().isoformat(),"filled_at":datetime.now().isoformat()},
        {"order_id":"ORD_DEF456","ticker":"MSFT","side":"BUY","quantity":50, "avg_fill":414.20,"commission":0.25,"status":"FILLED","is_bps":3.4,"created_at":datetime.now().isoformat(),"filled_at":datetime.now().isoformat()},
    ]

def _generate_demo_nav_history(initial_capital: float, days: int) -> List[Dict]:
    """Generate synthetic NAV history for demo mode."""
    np.random.seed(42)
    nav    = initial_capital
    result = []
    dt     = date.today() - timedelta(days=days)

    for i in range(days):
        daily_ret = np.random.normal(0.0004, 0.012)  # ~10% annual, 12% vol
        nav      *= (1 + daily_ret)
        result.append({
            "timestamp":     datetime.combine(dt + timedelta(days=i), datetime.min.time()).isoformat(),
            "nav":           round(nav, 2),
            "daily_pnl_pct": round(daily_ret, 5),
        })

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Server launcher
# ─────────────────────────────────────────────────────────────────────────────

def start_server(
    portfolio=None,
    risk_engine=None,
    factor_monitor=None,
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
) -> None:
    """Launch the API server."""
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn not installed. Run: pip install uvicorn"
        )

    app = create_app(portfolio, risk_engine, factor_monitor)

    logger.info(f"Starting API server on {host}:{port}")
    logger.info(f"Docs: http://{host}:{port}/docs")
    logger.info(f"API:  http://{host}:{port}/portfolio")

    uvicorn.run(app, host=host, port=port, reload=reload, log_level="warning")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("API server module loaded. Run via run_part8.py")
