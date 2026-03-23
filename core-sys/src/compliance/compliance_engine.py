"""
AI Hedge Fund — Part 10: Fund Operations & Compliance
=======================================================
compliance_engine.py — Regulatory Compliance & Trade Surveillance

Compliance is what separates a real fund from a hobby portfolio.
Every institutional investor (pension fund, endowment, family office)
will ask about compliance before writing a check. This module provides
the technical backbone for a compliance program.

What compliance means at the $1M-$50M AUM stage:

1. TRADE SURVEILLANCE
   Pre-trade: Is this trade compliant with our mandate?
   Post-trade: Did anything suspicious happen?
   Alerts: Front-running, wash sales, excessive concentration

2. POSITION LIMITS
   Regulatory limits (e.g. 13D/13G filing if >5% of a company)
   Fund mandate limits (sector caps, single-name caps)
   Leverage limits (gross and net exposure)

3. REGULATORY REPORTING
   Form PF:     Annual filing for SEC-registered investment advisors
   Schedule 13F: Quarterly position disclosure (if >$100M equity holdings)
   Form 13D/13G: Disclosure when acquiring >5% of a public company
   AIFMD:        For EU-domiciled funds or EU investors (Annex IV)
   These are TEMPLATES — actual filing requires legal counsel.

4. BEST EXECUTION
   Was the strategy executed at a fair price?
   TCA (Transaction Cost Analysis) feed from Part 6 execution engine
   Quarterly best execution report for institutional LPs

5. PERSONAL ACCOUNT TRADING
   Supervised persons (PM, analysts) must pre-clear personal trades
   Prevent front-running of fund trades
   Track and report employee trading activity

Disclaimer:
   This module provides infrastructure for compliance tracking.
   It is NOT legal advice. A registered fund requires:
   - A Chief Compliance Officer (CCO) or retained compliance consultant
   - Policies and Procedures manual reviewed by counsel
   - Annual compliance review
   - SEC registration (RIA) if warranted by AUM and investor count
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("hedge_fund.compliance")


# ─────────────────────────────────────────────────────────────────────────────
# Surveillance alert types
# ─────────────────────────────────────────────────────────────────────────────

class AlertSeverity(str, Enum):
    INFO     = "INFO"
    WARNING  = "WARNING"
    BREACH   = "BREACH"
    CRITICAL = "CRITICAL"


class AlertType(str, Enum):
    CONCENTRATION_LIMIT   = "CONCENTRATION_LIMIT"
    LEVERAGE_LIMIT        = "LEVERAGE_LIMIT"
    SECTOR_LIMIT          = "SECTOR_LIMIT"
    OWNERSHIP_THRESHOLD   = "OWNERSHIP_THRESHOLD"    # 5% 13D/13G trigger
    WASH_SALE             = "WASH_SALE"
    BEST_EXECUTION        = "BEST_EXECUTION"
    PERSONAL_TRADE        = "PERSONAL_TRADE"
    UNUSUAL_VOLUME        = "UNUSUAL_VOLUME"
    MANDATE_BREACH        = "MANDATE_BREACH"


@dataclass
class ComplianceAlert:
    alert_id:   str
    alert_type: AlertType
    severity:   AlertSeverity
    ticker:     Optional[str]
    message:    str
    details:    Dict[str, Any]
    timestamp:  datetime = field(default_factory=datetime.now)
    resolved:   bool = False
    resolved_at: Optional[datetime] = None
    resolution_note: str = ""

    def to_dict(self) -> dict:
        return {
            "alert_id":   self.alert_id,
            "type":       self.alert_type.value,
            "severity":   self.severity.value,
            "ticker":     self.ticker,
            "message":    self.message,
            "details":    self.details,
            "timestamp":  self.timestamp.isoformat(),
            "resolved":   self.resolved,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Investment mandate limits
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class InvestmentMandate:
    """
    Fund investment mandate — the rules governing what the fund can hold.

    These are described in the fund's offering documents (PPM/LPA)
    and enforced by this compliance engine.
    """
    fund_name:              str
    strategy:               str = "Systematic Long/Short Equity"

    # Position limits
    max_single_position_pct: float = 0.15    # 15% of NAV in one name
    max_sector_pct:          float = 0.35    # 35% in any GICS sector
    max_country_pct:         float = 0.80    # 80% in any single country

    # Leverage limits
    max_gross_exposure:      float = 1.50    # 150% gross (long + short)
    max_net_exposure:        float = 1.20    # 120% net (long - short)
    max_leverage_ratio:      float = 2.00    # 2x leverage

    # Liquidity
    max_illiquid_pct:        float = 0.20    # Max 20% in illiquid positions
    min_days_to_liquidate:   float = 5.0     # Portfolio should be liquidatable in 5 days

    # Prohibited instruments
    prohibited_tickers:      List[str] = field(default_factory=list)
    prohibited_sectors:      List[str] = field(default_factory=list)

    # Regulatory thresholds
    ownership_13d_threshold: float = 0.05    # 5% of outstanding shares → 13D
    ownership_13g_threshold: float = 0.05    # Same threshold, passive vs active

    # ESG constraints (optional)
    esg_exclusions:          List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Compliance engine
# ─────────────────────────────────────────────────────────────────────────────

class ComplianceEngine:
    """
    Real-time and post-trade compliance monitoring.

    Pre-trade checks: called before any order is submitted
    Post-trade surveillance: runs daily after market close
    Regulatory reporting: generates required filings

    Usage:
        mandate = InvestmentMandate(fund_name="AI Systematic Fund LP")
        engine  = ComplianceEngine(mandate)

        # Pre-trade check:
        ok, alerts = engine.pre_trade_check(
            ticker="AAPL", proposed_weight=0.12,
            positions=portfolio.positions, nav=1_000_000
        )

        # Daily surveillance:
        alerts = engine.daily_surveillance(positions, trades, nav)
    """

    def __init__(
        self,
        mandate:  InvestmentMandate,
        db_path:  Optional[Path] = None,
    ):
        self.mandate = mandate
        self._alerts: List[ComplianceAlert] = []
        db = db_path or (Path(__file__).parents[3] / "db" / "compliance.db")
        db.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS compliance_alerts (
                    alert_id     TEXT PRIMARY KEY,
                    alert_type   TEXT,
                    severity     TEXT,
                    ticker       TEXT,
                    message      TEXT,
                    details      TEXT,
                    timestamp    TEXT,
                    resolved     INTEGER DEFAULT 0,
                    resolved_at  TEXT,
                    resolution   TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trade_log (
                    trade_id      TEXT PRIMARY KEY,
                    ticker        TEXT,
                    side          TEXT,
                    quantity      REAL,
                    price         REAL,
                    trader        TEXT,
                    compliance_ok INTEGER,
                    alert_ids     TEXT,
                    timestamp     TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS personal_trades (
                    trade_id    TEXT PRIMARY KEY,
                    person_name TEXT,
                    ticker      TEXT,
                    side        TEXT,
                    quantity    REAL,
                    preclear_status TEXT,
                    preclear_at TEXT,
                    executed_at TEXT
                )
            """)
            conn.commit()

    # ── Pre-trade compliance check ────────────────────────────────────────────

    def pre_trade_check(
        self,
        ticker:           str,
        proposed_weight:  float,       # As fraction of NAV
        current_positions: Dict[str, float],   # {ticker: current_weight}
        nav:              float,
        proposed_side:    str = "BUY",
    ) -> Tuple[bool, List[ComplianceAlert]]:
        """
        Pre-trade compliance check.

        Returns (is_compliant, list_of_alerts).
        Blocking alerts prevent the trade.
        Warning alerts allow the trade but are logged.
        """
        alerts = []

        # 1. Check prohibited list
        if ticker in self.mandate.prohibited_tickers:
            alerts.append(self._make_alert(
                AlertType.MANDATE_BREACH,
                AlertSeverity.CRITICAL,
                ticker,
                f"{ticker} is on the prohibited instruments list",
                {"ticker": ticker, "list": "prohibited_tickers"},
            ))

        # 2. Position concentration limit
        if proposed_weight > self.mandate.max_single_position_pct:
            alerts.append(self._make_alert(
                AlertType.CONCENTRATION_LIMIT,
                AlertSeverity.BREACH,
                ticker,
                f"{ticker}: proposed weight {proposed_weight:.1%} exceeds limit "
                f"{self.mandate.max_single_position_pct:.1%}",
                {"proposed": proposed_weight, "limit": self.mandate.max_single_position_pct},
            ))
        elif proposed_weight > self.mandate.max_single_position_pct * 0.85:
            alerts.append(self._make_alert(
                AlertType.CONCENTRATION_LIMIT,
                AlertSeverity.WARNING,
                ticker,
                f"{ticker}: proposed weight {proposed_weight:.1%} approaching "
                f"limit {self.mandate.max_single_position_pct:.1%}",
                {"proposed": proposed_weight, "limit": self.mandate.max_single_position_pct},
            ))

        # 3. 13D/13G ownership check (simplified — needs shares outstanding data)
        if proposed_weight > self.mandate.ownership_13d_threshold:
            alerts.append(self._make_alert(
                AlertType.OWNERSHIP_THRESHOLD,
                AlertSeverity.WARNING,
                ticker,
                f"{ticker}: weight {proposed_weight:.1%} may approach 5% ownership "
                f"threshold. Verify shares outstanding before executing.",
                {"proposed_weight": proposed_weight, "threshold": "5% of outstanding shares"},
            ))

        # 4. Gross exposure check
        current_gross = sum(abs(w) for w in current_positions.values())
        new_gross     = current_gross + abs(proposed_weight)
        if new_gross > self.mandate.max_gross_exposure:
            alerts.append(self._make_alert(
                AlertType.LEVERAGE_LIMIT,
                AlertSeverity.BREACH,
                ticker,
                f"Adding {ticker} would bring gross exposure to {new_gross:.1%} "
                f"(limit {self.mandate.max_gross_exposure:.1%})",
                {"current_gross": current_gross, "new_gross": new_gross,
                 "limit": self.mandate.max_gross_exposure},
            ))

        # Persist and return
        for alert in alerts:
            self._persist_alert(alert)
            self._alerts.append(alert)

        is_compliant = not any(
            a.severity in (AlertSeverity.BREACH, AlertSeverity.CRITICAL)
            for a in alerts
        )
        return is_compliant, alerts

    # ── Daily post-trade surveillance ─────────────────────────────────────────

    def daily_surveillance(
        self,
        positions:     Dict[str, Dict],   # {ticker: {weight, sector, shares}}
        trades_today:  List[Dict],
        nav:           float,
        surveillance_date: Optional[date] = None,
    ) -> List[ComplianceAlert]:
        """
        Run daily post-trade compliance surveillance.

        Checks:
            - Concentration limits
            - Sector limits
            - Leverage/exposure limits
            - Wash sale detection
            - Unusual trade activity
            - Best execution review
        """
        alerts = []
        dt = surveillance_date or date.today()

        # 1. Concentration check on all current positions
        for ticker, pos in positions.items():
            weight = pos.get("weight", 0) if isinstance(pos, dict) else 0
            if weight > self.mandate.max_single_position_pct:
                alerts.append(self._make_alert(
                    AlertType.CONCENTRATION_LIMIT,
                    AlertSeverity.BREACH,
                    ticker,
                    f"Position concentration breach: {ticker} = {weight:.1%} "
                    f"(limit {self.mandate.max_single_position_pct:.1%})",
                    {"current_weight": weight, "limit": self.mandate.max_single_position_pct},
                ))

        # 2. Sector concentration
        sector_weights: Dict[str, float] = {}
        for ticker, pos in positions.items():
            sector = (pos.get("sector", "Unknown") if isinstance(pos, dict) else "Unknown")
            w      = pos.get("weight", 0) if isinstance(pos, dict) else 0
            sector_weights[sector] = sector_weights.get(sector, 0) + abs(w)

        for sector, total_weight in sector_weights.items():
            if total_weight > self.mandate.max_sector_pct:
                alerts.append(self._make_alert(
                    AlertType.SECTOR_LIMIT,
                    AlertSeverity.BREACH,
                    None,
                    f"Sector concentration breach: {sector} = {total_weight:.1%} "
                    f"(limit {self.mandate.max_sector_pct:.1%})",
                    {"sector": sector, "weight": total_weight,
                     "limit": self.mandate.max_sector_pct},
                ))

        # 3. Leverage / exposure check
        gross_exposure = sum(abs(pos.get("weight", 0) if isinstance(pos, dict) else 0)
                             for pos in positions.values())
        if gross_exposure > self.mandate.max_gross_exposure:
            alerts.append(self._make_alert(
                AlertType.LEVERAGE_LIMIT,
                AlertSeverity.BREACH,
                None,
                f"Gross exposure {gross_exposure:.1%} exceeds limit "
                f"{self.mandate.max_gross_exposure:.1%}",
                {"gross_exposure": gross_exposure, "limit": self.mandate.max_gross_exposure},
            ))

        # 4. Wash sale detection (buy and sell same security within 30 days)
        sell_dates: Dict[str, date] = {}
        for trade in sorted(trades_today, key=lambda t: t.get("created_at", "")):
            ticker = trade.get("ticker", "")
            side   = trade.get("side", "")
            if side == "SELL":
                sell_dates[ticker] = dt
            elif side == "BUY" and ticker in sell_dates:
                days_since_sell = (dt - sell_dates[ticker]).days
                if days_since_sell < 30:
                    alerts.append(self._make_alert(
                        AlertType.WASH_SALE,
                        AlertSeverity.WARNING,
                        ticker,
                        f"Potential wash sale: {ticker} sold and repurchased "
                        f"within {days_since_sell} days",
                        {"ticker": ticker, "days_between": days_since_sell},
                    ))

        # 5. Best execution check (flag trades with high implementation shortfall)
        for trade in trades_today:
            is_bps = trade.get("is_bps")
            ticker = trade.get("ticker", "")
            if is_bps is not None and abs(is_bps) > 20:   # >20bps IS is high
                alerts.append(self._make_alert(
                    AlertType.BEST_EXECUTION,
                    AlertSeverity.WARNING,
                    ticker,
                    f"High implementation shortfall: {ticker} IS = {is_bps:+.1f}bps",
                    {"ticker": ticker, "is_bps": is_bps, "threshold": 20},
                ))

        # Persist all alerts
        for alert in alerts:
            self._persist_alert(alert)
            self._alerts.append(alert)

        logger.info(
            f"Daily surveillance [{dt}]: "
            f"{sum(1 for a in alerts if a.severity == AlertSeverity.BREACH)} breaches, "
            f"{sum(1 for a in alerts if a.severity == AlertSeverity.WARNING)} warnings"
        )

        return alerts

    # ── Personal account trading ───────────────────────────────────────────────

    def preclear_personal_trade(
        self,
        person_name:    str,
        ticker:         str,
        side:           str,
        quantity:       float,
        fund_positions: Dict[str, float],   # Current fund positions
    ) -> Tuple[bool, str]:
        """
        Pre-clearance for personal account trading by supervised persons.

        Checks:
            - Fund is not currently trading this security
            - Security is not on the restricted list
            - 5-day blackout after fund trade in this security
        """
        # Check restricted list
        if ticker in self.mandate.prohibited_tickers:
            return False, f"{ticker} is on the fund's restricted list"

        # Check fund's current activity (simplified)
        if ticker in fund_positions and abs(fund_positions[ticker]) > 0.02:
            return False, (
                f"{ticker} is a material position (>{fund_positions[ticker]:.1%}) in the fund. "
                f"Personal trading requires CCO approval."
            )

        # Log the pre-clearance
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO personal_trades
                    (trade_id, person_name, ticker, side, quantity,
                     preclear_status, preclear_at)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    f"PT_{uuid.uuid4().hex[:8].upper()}",
                    person_name, ticker, side, quantity,
                    "APPROVED", datetime.now().isoformat(),
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Personal trade log failed: {e}")

        return True, f"Pre-clearance approved for {person_name}: {side} {quantity:.0f} {ticker}"

    # ── Regulatory report generators ──────────────────────────────────────────

    def generate_form_pf_summary(
        self,
        nav:              float,
        gross_exposure:   float,
        net_exposure:     float,
        fund_type:        str = "3c1_hedge_fund",
        as_of_date:       Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Generate Form PF data summary (Section 1 — Large Traders).

        Form PF is required for SEC-registered investment advisors
        with >$150M AUM in private funds.
        At <$150M, only an annual abbreviated filing is required.

        IMPORTANT: This generates the data summary only.
        Actual Form PF filing must be done through the SEC's PFRD system.
        Consult legal counsel before filing.
        """
        as_of = as_of_date or date.today()
        quarter_end = self._get_quarter_end(as_of)

        return {
            "_disclaimer": (
                "TEMPLATE ONLY — Not a legal filing. "
                "Form PF must be filed through SEC PFRD. "
                "Consult legal counsel before submitting."
            ),
            "form_type":          "Form PF",
            "reporting_period":   quarter_end.isoformat(),
            "filing_type":        "Annual" if nav < 150_000_000 else "Quarterly",
            "section_1a": {
                "fund_name":          self.mandate.fund_name,
                "fund_type":          fund_type,
                "nav_usd":            round(nav, 0),
                "regulatory_assets":  round(nav * gross_exposure, 0),
                "gross_exposure_pct": round(gross_exposure * 100, 1),
                "net_exposure_pct":   round(net_exposure * 100, 1),
                "leverage_ratio":     round(gross_exposure, 2),
                "liquidity_days":     5,   # Estimated days to liquidate
            },
            "generated_at": datetime.now().isoformat(),
        }

    def generate_13f_holdings(
        self,
        positions: Dict[str, float],    # {ticker: market_value}
        nav:       float,
        as_of_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Generate Schedule 13F data.

        Required quarterly if US equity holdings > $100M.
        Lists all equity positions with CUSIP.

        IMPORTANT: Template only — requires actual CUSIP mapping and
        legal review before submission to SEC EDGAR.
        """
        as_of = as_of_date or date.today()

        # 13F threshold check
        total_equity = sum(v for v in positions.values() if v > 0)
        requires_13f = total_equity > 100_000_000

        # Build holdings table
        holdings = []
        for ticker, market_value in positions.items():
            if market_value < 10_000:  # 13F excludes positions <$10K
                continue
            holdings.append({
                "name_of_issuer": ticker,
                "title_of_class": "COM",   # Common Stock
                "cusip":          "UNKNOWN",   # Requires CUSIP database
                "value_1000s":    round(market_value / 1000, 0),
                "shares":         "UNKNOWN",   # Requires shares data
                "discretion":     "SOLE",
                "put_call":       None,
            })

        return {
            "_disclaimer": (
                "TEMPLATE ONLY — CUSIP data not included. "
                "Schedule 13F must be filed through SEC EDGAR. "
                "Requires actual CUSIP for each security."
            ),
            "form_type":       "Schedule 13F-HR",
            "period_of_report":as_of.isoformat(),
            "requires_13f":    requires_13f,
            "total_equity_usd":round(total_equity, 0),
            "threshold":       100_000_000,
            "holdings":        holdings,
            "n_holdings":      len(holdings),
            "generated_at":    datetime.now().isoformat(),
        }

    def generate_13d_check(
        self,
        ticker:              str,
        fund_shares:         float,
        company_shares_out:  float,
        as_of_date:          Optional[date] = None,
    ) -> Dict[str, Any]:
        """
        Check if a position triggers 13D/13G filing requirements.

        >5% of outstanding shares → file Schedule 13D (active) or 13G (passive)
        within 10 days of crossing the threshold.

        IMMEDIATELY notify legal counsel if this triggers.
        """
        ownership_pct = fund_shares / company_shares_out if company_shares_out > 0 else 0
        requires_filing = ownership_pct >= 0.05

        return {
            "_disclaimer": (
                "If filing required: notify legal counsel IMMEDIATELY. "
                "13D/13G must be filed within 10 calendar days of crossing 5%. "
                "Late filing is an SEC violation."
            ),
            "ticker":            ticker,
            "fund_shares":       fund_shares,
            "company_shares_out":company_shares_out,
            "ownership_pct":     round(ownership_pct * 100, 4),
            "threshold_pct":     5.0,
            "requires_filing":   requires_filing,
            "filing_type":       "13G (passive)" if requires_filing else None,
            "deadline":          (date.today() + timedelta(days=10)).isoformat() if requires_filing else None,
            "urgency":           "IMMEDIATE — NOTIFY LEGAL COUNSEL" if requires_filing else "OK",
        }

    def generate_best_execution_report(
        self,
        trades: List[Dict],
        period_start: date,
        period_end:   date,
    ) -> Dict[str, Any]:
        """
        Quarterly Best Execution Report.

        Required for RIAs under SEC Advisers Act Section 28(e).
        Shows average implementation shortfall, by broker, by ticker.
        """
        if not trades:
            return {"note": "No trades in period"}

        is_values = [t.get("is_bps", 0) or 0 for t in trades if t.get("status") == "FILLED"]
        comm_vals = [t.get("commission", 0) or 0 for t in trades if t.get("status") == "FILLED"]

        by_ticker: Dict[str, List[float]] = {}
        for t in trades:
            if t.get("status") == "FILLED" and t.get("is_bps") is not None:
                by_ticker.setdefault(t["ticker"], []).append(t["is_bps"])

        return {
            "period":          f"{period_start} to {period_end}",
            "n_trades":        len(trades),
            "n_filled":        len(is_values),
            "avg_is_bps":      round(float(sum(is_values) / len(is_values)), 2) if is_values else 0,
            "p90_is_bps":      round(float(sorted(is_values)[int(0.9 * len(is_values))]), 2) if len(is_values) > 2 else 0,
            "total_commission":round(sum(comm_vals), 2),
            "by_ticker":       {t: round(sum(v)/len(v), 2) for t, v in by_ticker.items()},
            "assessment":      (
                "GOOD — avg IS < 10bps" if is_values and sum(is_values)/len(is_values) < 10 else
                "ACCEPTABLE — avg IS 10-20bps" if is_values and sum(is_values)/len(is_values) < 20 else
                "REVIEW REQUIRED — avg IS > 20bps"
            ),
            "generated_at":    datetime.now().isoformat(),
        }

    # ── Alert management ──────────────────────────────────────────────────────

    def _make_alert(
        self,
        alert_type: AlertType,
        severity:   AlertSeverity,
        ticker:     Optional[str],
        message:    str,
        details:    Dict,
    ) -> ComplianceAlert:
        return ComplianceAlert(
            alert_id   = f"CA_{uuid.uuid4().hex[:8].upper()}",
            alert_type = alert_type,
            severity   = severity,
            ticker     = ticker,
            message    = message,
            details    = details,
        )

    def _persist_alert(self, alert: ComplianceAlert) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO compliance_alerts
                    (alert_id, alert_type, severity, ticker, message, details, timestamp)
                    VALUES (?,?,?,?,?,?,?)
                """, (
                    alert.alert_id, alert.alert_type.value, alert.severity.value,
                    alert.ticker, alert.message, json.dumps(alert.details),
                    alert.timestamp.isoformat(),
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Alert persist failed: {e}")

    def resolve_alert(self, alert_id: str, note: str = "") -> bool:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    UPDATE compliance_alerts
                    SET resolved=1, resolved_at=?, resolution=?
                    WHERE alert_id=?
                """, (datetime.now().isoformat(), note, alert_id))
                conn.commit()
            return True
        except Exception:
            return False

    def get_open_alerts(
        self,
        min_severity: AlertSeverity = AlertSeverity.WARNING,
    ) -> List[Dict]:
        severity_rank = {
            AlertSeverity.INFO: 0,
            AlertSeverity.WARNING: 1,
            AlertSeverity.BREACH: 2,
            AlertSeverity.CRITICAL: 3,
        }
        threshold = severity_rank[min_severity]

        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT * FROM compliance_alerts
                    WHERE resolved=0
                    ORDER BY timestamp DESC
                    LIMIT 100
                """).fetchall()
            return [
                dict(r) for r in rows
                if severity_rank.get(r["severity"], 0) >= threshold
            ]
        except Exception:
            return []

    def compliance_summary(self) -> Dict[str, Any]:
        """Summary for the compliance dashboard."""
        open_alerts = self.get_open_alerts(AlertSeverity.INFO)
        breaches    = [a for a in open_alerts if a.get("severity") in ("BREACH", "CRITICAL")]
        warnings    = [a for a in open_alerts if a.get("severity") == "WARNING"]

        return {
            "status":          "BREACH" if breaches else ("WARNING" if warnings else "CLEAN"),
            "open_alerts":     len(open_alerts),
            "breaches":        len(breaches),
            "warnings":        len(warnings),
            "latest_breach":   breaches[0]["message"] if breaches else None,
            "mandate":         self.mandate.fund_name,
            "as_of":           datetime.now().isoformat(),
        }

    @staticmethod
    def _get_quarter_end(dt: date) -> date:
        quarter_end_months = {1: 3, 2: 3, 3: 3, 4: 6, 5: 6, 6: 6,
                              7: 9, 8: 9, 9: 9, 10: 12, 11: 12, 12: 12}
        end_month = quarter_end_months[dt.month]
        end_day   = 31 if end_month in (3, 12) else 30
        return date(dt.year, end_month, end_day)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  Compliance Engine — Test")
    print("=" * 65)

    mandate = InvestmentMandate(
        fund_name             = "AI Systematic Fund LP",
        max_single_position_pct = 0.15,
        max_sector_pct          = 0.35,
        max_gross_exposure      = 1.50,
        prohibited_tickers      = ["XYZ"],
    )
    engine = ComplianceEngine(mandate)

    positions = {
        "AAPL": {"weight": 0.12, "sector": "Technology", "shares": 500},
        "MSFT": {"weight": 0.11, "sector": "Technology", "shares": 300},
        "NVDA": {"weight": 0.08, "sector": "Technology", "shares": 150},
        "JPM":  {"weight": 0.10, "sector": "Financials", "shares": 800},
        "XOM":  {"weight": 0.07, "sector": "Energy",     "shares": 600},
    }

    print("\n1. Pre-trade check (AAPL at 14% — should warn):")
    ok, alerts = engine.pre_trade_check(
        ticker            = "AAPL",
        proposed_weight   = 0.14,
        current_positions = {k: v["weight"] for k, v in positions.items()},
        nav               = 1_000_000,
    )
    print(f"   Compliant: {ok}")
    for a in alerts:
        print(f"   [{a.severity.value}] {a.message}")

    print("\n2. Pre-trade check (MSFT at 18% — should breach):")
    ok2, alerts2 = engine.pre_trade_check(
        ticker            = "MSFT",
        proposed_weight   = 0.18,
        current_positions = {k: v["weight"] for k, v in positions.items()},
        nav               = 1_000_000,
    )
    print(f"   Compliant: {ok2}")
    for a in alerts2:
        print(f"   [{a.severity.value}] {a.message}")

    print("\n3. Daily surveillance:")
    surv_alerts = engine.daily_surveillance(positions, [], 1_000_000)
    print(f"   {len(surv_alerts)} alerts generated")
    for a in surv_alerts[:3]:
        print(f"   [{a.severity.value}] {a.message}")

    print("\n4. 13D ownership check:")
    check = engine.generate_13d_check("AAPL", 5_000, 15_700_000_000)
    print(f"   Ownership: {check['ownership_pct']:.4f}%")
    print(f"   Requires filing: {check['requires_filing']}")

    print("\n5. Compliance summary:")
    for k, v in engine.compliance_summary().items():
        print(f"   {k:<25}: {v}")

    print("\n✅ Compliance Engine tests passed")
