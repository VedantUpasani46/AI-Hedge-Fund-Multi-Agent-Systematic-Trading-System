"""
AI Hedge Fund — Part 9: Cloud Production
==========================================
monitoring.py — Production Monitoring, Structured Logging & Alerting

Production monitoring has three layers:

1. STRUCTURED LOGGING
   Every log line is JSON. Not human-readable strings.
   Why: CloudWatch Insights, Datadog, Loki all parse JSON natively.
   You can query: "all log lines where daily_pnl_pct < -0.01"
   You cannot do that with: "2024-01-15 14:23:11 Portfolio dropped 1.2%"

2. METRICS (push to CloudWatch)
   Time-series numeric data:
     - Portfolio NAV every 5 minutes
     - API latency (p50, p95, p99) per endpoint
     - LLM cost per hour (don't let it runaway)
     - Circuit breaker trigger count
     - VaR vs limit utilisation
   Alarms fire when metrics cross thresholds.

3. HEALTH CHECKS
   The system checks itself every N seconds and reports to an
   external health endpoint. If the health check fails:
     - Load balancer removes the instance from rotation
     - ECS restarts the task
     - Pager fires

Production alerting channels:
    AWS SNS → Email (fund team)
    AWS SNS → PagerDuty (24/7 on-call for critical breaches)
    Slack webhook (for warnings and daily summaries)

The health check hierarchy:
    /health (basic):  Is the process running?
    /health/deep:     Can we reach the database? Redis? Market data?
    /health/risk:     Is the risk monitor running? Last snapshot recent?
"""

from __future__ import annotations

import json
import logging
import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hedge_fund.monitoring")


# ─────────────────────────────────────────────────────────────────────────────
# Structured JSON logger
# ─────────────────────────────────────────────────────────────────────────────

class StructuredLogger:
    """
    JSON-structured logger for production observability.

    Every log line is a JSON object with:
        timestamp:  ISO 8601
        level:      DEBUG/INFO/WARNING/ERROR/CRITICAL
        service:    which component logged this
        event:      machine-readable event type
        message:    human-readable description
        + any additional context fields

    Usage:
        log = StructuredLogger("risk_monitor")
        log.info("circuit_breaker_triggered", breaker="VAR_LIMIT", nav=1_042_000)
        log.metric("var_95_pct", 0.0187, portfolio_id="FUND_001")
        log.alert("TRADING_HALTED", severity="CRITICAL", reason="daily_loss")
    """

    def __init__(self, service: str):
        self.service   = service
        self._log_file = self._init_log_file()

    def _init_log_file(self) -> Optional[Path]:
        log_dir = Path(os.getenv("LOG_DIR", "/app/logs"))
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            return log_dir / f"{self.service}.jsonl"
        except PermissionError:
            return None

    def _emit(self, level: str, event: str, message: str, **kwargs) -> None:
        record = {
            "timestamp": datetime.now().isoformat(),
            "level":     level,
            "service":   self.service,
            "event":     event,
            "message":   message,
            **kwargs,
        }
        line = json.dumps(record)

        # Write to file
        if self._log_file:
            try:
                with open(self._log_file, "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass

        # Also emit via standard Python logger
        log_fn = {
            "DEBUG":    logger.debug,
            "INFO":     logger.info,
            "WARNING":  logger.warning,
            "ERROR":    logger.error,
            "CRITICAL": logger.critical,
        }.get(level, logger.info)
        log_fn(f"[{event}] {message}")

    def debug(self, event: str, message: str = "", **kwargs):
        self._emit("DEBUG", event, message, **kwargs)

    def info(self, event: str, message: str = "", **kwargs):
        self._emit("INFO", event, message, **kwargs)

    def warning(self, event: str, message: str = "", **kwargs):
        self._emit("WARNING", event, message, **kwargs)

    def error(self, event: str, message: str = "", **kwargs):
        self._emit("ERROR", event, message, **kwargs)

    def critical(self, event: str, message: str = "", **kwargs):
        self._emit("CRITICAL", event, message, **kwargs)

    def metric(self, name: str, value: float, **tags):
        """Log a numeric metric."""
        self._emit("INFO", "metric", f"{name}={value}", metric=name, value=value, tags=tags)

    def alert(self, event: str, level: str = "WARNING", **details):
        """Log an alert — high-priority event requiring human attention."""
        self._emit(level, f"ALERT_{event}", f"Alert: {event}", **details)
        # Also send to external channels
        self._send_external_alert(event, level, details)

    def _send_external_alert(self, event: str, level: str, details: dict):
        """Send alert to Slack/SNS (if configured)."""
        slack_url = os.getenv("SLACK_WEBHOOK_URL")
        if slack_url:
            try:
                import urllib.request, json as _json
                color = {"CRITICAL": "#EF4444", "ERROR": "#F59E0B", "WARNING": "#F59E0B"}.get(level, "#6B7280")
                payload = {
                    "attachments": [{
                        "color": color,
                        "title": f"AI Hedge Fund — {level}: {event}",
                        "text":  _json.dumps(details, indent=2)[:1000],
                        "footer": f"Service: {self.service} | {datetime.now():%Y-%m-%d %H:%M:%S}",
                    }]
                }
                req = urllib.request.Request(
                    slack_url,
                    data    = _json.dumps(payload).encode(),
                    headers = {"Content-Type": "application/json"},
                )
                urllib.request.urlopen(req, timeout=5)
            except Exception as e:
                logger.debug(f"Slack alert failed: {e}")


_loggers: Dict[str, StructuredLogger] = {}

def get_monitor_logger(service: str = "monitor") -> StructuredLogger:
    if service not in _loggers:
        _loggers[service] = StructuredLogger(service)
    return _loggers[service]


# ─────────────────────────────────────────────────────────────────────────────
# API request timing middleware
# ─────────────────────────────────────────────────────────────────────────────

def create_monitoring_middleware():
    """
    FastAPI middleware that:
      1. Times every request
      2. Logs structured access logs
      3. Publishes latency metrics to CloudWatch
      4. Tracks error rates
    """
    try:
        from fastapi import Request
        from starlette.middleware.base import BaseHTTPMiddleware
        import time as _time

        class MonitoringMiddleware(BaseHTTPMiddleware):
            def __init__(self, app):
                super().__init__(app)
                self.log     = get_monitor_logger("api")
                self._request_count:     int   = 0
                self._error_count:       int   = 0
                self._total_latency_ms:  float = 0.0

            async def dispatch(self, request: Request, call_next):
                start     = _time.time()
                response  = None
                exc_info  = None

                try:
                    response = await call_next(request)
                except Exception as e:
                    exc_info = e
                    raise
                finally:
                    latency_ms = (_time.time() - start) * 1000
                    status     = response.status_code if response else 500

                    self._request_count    += 1
                    self._total_latency_ms += latency_ms
                    if status >= 400:
                        self._error_count += 1

                    self.log.info(
                        event       = "http_request",
                        message     = f"{request.method} {request.url.path} {status}",
                        method      = request.method,
                        path        = request.url.path,
                        status_code = status,
                        latency_ms  = round(latency_ms, 1),
                        client_ip   = request.client.host if request.client else "",
                    )

                    # Publish to CloudWatch every 100 requests
                    if self._request_count % 100 == 0:
                        self._publish_api_metrics()

                return response

            def _publish_api_metrics(self):
                try:
                    from src.deploy.aws_deploy import AWSConfig, CloudWatchManager
                    cw  = CloudWatchManager(AWSConfig.from_env())
                    avg = self._total_latency_ms / max(self._request_count, 1)
                    err = self._error_count / max(self._request_count, 1)
                    cw.put_metric("api_latency_avg", avg, unit="Milliseconds")
                    cw.put_metric("api_error_rate",  err)
                    cw.put_metric("api_request_count", self._request_count)
                except Exception:
                    pass

        return MonitoringMiddleware

    except ImportError:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Request timer context manager
# ─────────────────────────────────────────────────────────────────────────────

@contextmanager
def timer(operation: str, log: Optional[StructuredLogger] = None):
    """
    Context manager that times a block and logs the duration.

    Usage:
        with timer("llm_inference", log=monitor_log):
            response = llm.complete(prompt)
    """
    start = time.time()
    try:
        yield
    finally:
        elapsed_ms = (time.time() - start) * 1000
        if log:
            log.metric(f"{operation}_ms", elapsed_ms)
        else:
            logger.debug(f"{operation}: {elapsed_ms:.1f}ms")


# ─────────────────────────────────────────────────────────────────────────────
# Deep health checks
# ─────────────────────────────────────────────────────────────────────────────

class HealthChecker:
    """
    Comprehensive system health checks.

    Called by:
        /health       — Basic (is process alive?)
        /health/deep  — All dependencies reachable?
        /health/risk  — Risk monitor running and recent?
    """

    def check_database(self) -> Dict[str, Any]:
        """Check SQLite databases are accessible and not corrupted."""
        db_dir = Path(os.getenv("DB_DIR", "/app/db"))
        results = {}

        for db_name in ["risk_monitor.db", "execution.db", "decisions.db"]:
            db_path = db_dir / db_name
            if not db_path.exists():
                results[db_name] = {"status": "missing", "ok": False}
                continue
            try:
                import sqlite3
                with sqlite3.connect(str(db_path), timeout=5) as conn:
                    conn.execute("SELECT 1")
                    size_mb = db_path.stat().st_size / (1024 * 1024)
                results[db_name] = {
                    "status":  "ok",
                    "ok":      True,
                    "size_mb": round(size_mb, 2),
                }
            except Exception as e:
                results[db_name] = {"status": str(e), "ok": False}

        return results

    def check_redis(self) -> Dict[str, Any]:
        """Check Redis connectivity."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        try:
            import redis as _redis
            r = _redis.from_url(redis_url, socket_timeout=2)
            ping = r.ping()
            info = r.info("memory")
            return {
                "status":      "ok" if ping else "failed",
                "ok":          bool(ping),
                "used_memory": info.get("used_memory_human", "unknown"),
            }
        except ImportError:
            return {"status": "redis not installed", "ok": True}  # Not required
        except Exception as e:
            return {"status": str(e), "ok": False}

    def check_market_data(self) -> Dict[str, Any]:
        """Check we can fetch market data."""
        try:
            import yfinance as yf
            start = time.time()
            df    = yf.download("SPY", period="1d", progress=False, auto_adjust=True)
            latency = (time.time() - start) * 1000
            if df.empty:
                return {"status": "no data returned", "ok": False}
            return {
                "status":     "ok",
                "ok":         True,
                "latency_ms": round(latency, 0),
                "last_price": round(float(df["Close"].iloc[-1]), 2),
            }
        except Exception as e:
            return {"status": str(e), "ok": False}

    def check_risk_monitor(self, max_age_seconds: int = 120) -> Dict[str, Any]:
        """Check that the risk monitor is producing recent snapshots."""
        db_path = Path(os.getenv("DB_DIR", "/app/db")) / "risk_monitor.db"
        if not db_path.exists():
            return {"status": "db not found", "ok": False}

        try:
            import sqlite3
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT timestamp, nav, risk_level FROM risk_snapshots ORDER BY id DESC LIMIT 1"
                ).fetchone()

            if not row:
                return {"status": "no snapshots yet", "ok": False}

            last_ts  = datetime.fromisoformat(row[0])
            age_sec  = (datetime.now() - last_ts).total_seconds()
            is_fresh = age_sec < max_age_seconds

            return {
                "status":      "ok" if is_fresh else "stale",
                "ok":          is_fresh,
                "last_snapshot": row[0],
                "age_seconds": round(age_sec, 0),
                "nav":         row[1],
                "risk_level":  row[2],
            }
        except Exception as e:
            return {"status": str(e), "ok": False}

    def check_disk_space(self) -> Dict[str, Any]:
        """Check available disk space."""
        try:
            import shutil
            usage = shutil.disk_usage(os.getenv("DB_DIR", "/app"))
            free_gb  = usage.free / (1024**3)
            total_gb = usage.total / (1024**3)
            pct_used = (usage.used / usage.total) * 100
            return {
                "status":   "ok" if pct_used < 85 else "warning",
                "ok":       pct_used < 90,
                "free_gb":  round(free_gb, 2),
                "total_gb": round(total_gb, 2),
                "pct_used": round(pct_used, 1),
            }
        except Exception as e:
            return {"status": str(e), "ok": True}

    def full_check(self) -> Dict[str, Any]:
        """Run all health checks and return aggregate status."""
        checks = {
            "database":      self.check_database(),
            "redis":         self.check_redis(),
            "market_data":   self.check_market_data(),
            "risk_monitor":  self.check_risk_monitor(),
            "disk_space":    self.check_disk_space(),
        }

        all_ok   = all(
            (v.get("ok", False) if isinstance(v, dict) else
             all(vv.get("ok", False) for vv in v.values()))
            for v in checks.values()
        )

        return {
            "status":    "healthy" if all_ok else "degraded",
            "ok":        all_ok,
            "timestamp": datetime.now().isoformat(),
            "checks":    checks,
        }


# ─────────────────────────────────────────────────────────────────────────────
# LLM cost tracker
# ─────────────────────────────────────────────────────────────────────────────

class LLMCostTracker:
    """
    Tracks LLM API costs in real-time and alerts on runaway spend.

    At $0.003 per input token / $0.015 per output token (Claude Sonnet):
        A typical allocation decision:  ~2,000 tokens in / ~500 tokens out = ~$0.014
        Universe scan (20 tickers):     ~40,000 tokens in / ~10,000 out   = ~$0.27
        Daily run:                      ~$0.50-2.00 depending on frequency

    Alert thresholds (configurable):
        Hourly:  >$5 (runaway loop?)
        Daily:   >$50 (unexpected behaviour)
        Weekly:  >$200 (budget exceeded)
    """

    def __init__(self):
        self._db_path = Path(os.getenv("DB_DIR", "/app/db")) / "llm_costs.db"
        self._init_db()

    def _init_db(self):
        import sqlite3
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_costs (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp  TEXT NOT NULL,
                    agent      TEXT,
                    model      TEXT,
                    purpose    TEXT,
                    in_tokens  INTEGER,
                    out_tokens INTEGER,
                    cost_usd   REAL,
                    duration_ms REAL
                )
            """)
            conn.commit()

    def record(
        self,
        agent:       str,
        model:       str,
        purpose:     str,
        in_tokens:   int,
        out_tokens:  int,
        cost_usd:    float,
        duration_ms: float = 0.0,
    ) -> None:
        """Record an LLM API call cost."""
        import sqlite3
        with sqlite3.connect(str(self._db_path)) as conn:
            conn.execute("""
                INSERT INTO llm_costs
                (timestamp, agent, model, purpose, in_tokens, out_tokens, cost_usd, duration_ms)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                datetime.now().isoformat(), agent, model, purpose,
                in_tokens, out_tokens, cost_usd, duration_ms,
            ))
            conn.commit()

        # Check alert thresholds
        hourly = self.cost_in_period(hours=1)
        if hourly > float(os.getenv("LLM_HOURLY_ALERT_USD", "5.0")):
            log = get_monitor_logger("llm_cost")
            log.alert(
                "LLM_COST_SPIKE",
                level   = "WARNING",
                hourly_cost_usd = round(hourly, 4),
                latest_call     = cost_usd,
                agent           = agent,
                purpose         = purpose,
            )

    def cost_in_period(self, hours: float = 24) -> float:
        """Total LLM cost in the last N hours."""
        import sqlite3
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute(
                    "SELECT SUM(cost_usd) FROM llm_costs WHERE timestamp >= ?",
                    (cutoff,)
                ).fetchone()
            return float(row[0] or 0)
        except Exception:
            return 0.0

    def cost_summary(self) -> Dict[str, float]:
        """Summary of LLM costs by time period."""
        return {
            "last_hour":  round(self.cost_in_period(1), 5),
            "last_day":   round(self.cost_in_period(24), 4),
            "last_week":  round(self.cost_in_period(168), 3),
            "last_month": round(self.cost_in_period(720), 2),
        }

    def cost_by_agent(self, days: int = 7) -> Dict[str, float]:
        """Cost breakdown by agent."""
        import sqlite3
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                rows = conn.execute("""
                    SELECT agent, SUM(cost_usd) as total
                    FROM llm_costs WHERE timestamp >= ?
                    GROUP BY agent ORDER BY total DESC
                """, (cutoff,)).fetchall()
            return {r[0]: round(r[1], 4) for r in rows}
        except Exception:
            return {}


# ─────────────────────────────────────────────────────────────────────────────
# GitHub Actions health check endpoint additions
# ─────────────────────────────────────────────────────────────────────────────

def add_health_endpoints(app) -> None:
    """
    Add deep health check endpoints to the FastAPI app.
    Called from api_server.create_app() in production.
    """
    checker = HealthChecker()
    cost_tracker = LLMCostTracker()

    @app.get("/health/deep")
    async def deep_health():
        return checker.full_check()

    @app.get("/health/risk")
    async def risk_health():
        return checker.check_risk_monitor()

    @app.get("/metrics/llm-costs")
    async def llm_costs():
        return {
            "summary":  cost_tracker.cost_summary(),
            "by_agent": cost_tracker.cost_by_agent(),
        }

    @app.get("/metrics/system")
    async def system_metrics():
        return {
            "disk":        checker.check_disk_space(),
            "database":    checker.check_database(),
            "market_data": checker.check_market_data(),
            "timestamp":   datetime.now().isoformat(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  Production Monitoring — Health Check Test")
    print("=" * 60)

    checker = HealthChecker()

    print("\n1. Database check:")
    db_result = checker.check_database()
    for name, status in db_result.items():
        icon = "✓" if status.get("ok") else "✗"
        print(f"   {icon} {name}: {status['status']}")

    print("\n2. Market data check:")
    md = checker.check_market_data()
    print(f"   {'✓' if md['ok'] else '✗'} {md['status']}" +
          (f" (SPY ${md.get('last_price')}, {md.get('latency_ms')}ms)" if md["ok"] else ""))

    print("\n3. Disk space:")
    disk = checker.check_disk_space()
    print(f"   {'✓' if disk['ok'] else '⚠'} {disk.get('pct_used', 0)}% used, "
          f"{disk.get('free_gb', 0)}GB free")

    print("\n4. LLM cost tracking:")
    tracker = LLMCostTracker()
    print(f"   Costs: {tracker.cost_summary()}")

    print("\n5. Structured logging test:")
    log = get_monitor_logger("test")
    log.info("health_check_complete", message="All checks passed", test=True)
    print(f"   Log file: {log._log_file}")

    print("\n✅ Monitoring system ready")
