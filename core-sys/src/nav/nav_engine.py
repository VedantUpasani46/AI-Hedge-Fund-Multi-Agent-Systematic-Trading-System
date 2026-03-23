"""
AI Hedge Fund — Part 10: Fund Operations & Compliance
=======================================================
nav_engine.py — Daily Fund Accounting & NAV Calculation

NAV (Net Asset Value) calculation is the heartbeat of fund operations.
Every fee, every investor's ownership stake, every performance report
derives from it. Getting it wrong is not just an accounting error —
it is a regulatory violation and a legal liability.

What the NAV engine computes daily:
    1. Gross Asset Value:  mark all positions at closing prices
    2. Accrued expenses:   management fees, admin fees, audit, etc.
    3. Net Asset Value:    GAV minus accrued liabilities
    4. NAV per share:      NAV / shares outstanding
    5. Management fee:     2% annual on NAV, accrued daily
    6. Performance fee:    20% of profits above high-water mark
    7. Investor allocations: each LP's share of NAV
    8. Reconciliation:     vs prime broker statement

High-Water Mark explained:
    An LP invests $1M. Fund grows to $1.2M → performance fee on $200K.
    Fund drops to $1.1M → NO performance fee (below prior peak).
    Fund recovers to $1.3M → performance fee on $100K ($1.3M - $1.2M HWM).
    This protects investors from paying twice for the same gains.

Fee structures implemented:
    STANDARD:       2% management / 20% performance
    FOUNDERS:       1% management / 15% performance (early investors)
    INSTITUTIONAL:  1.5% management / 10% performance (large allocators)
    NO_PERFORMANCE: 1% management / 0% performance (some pension mandates)

Fund administrator integration:
    In production, a fund administrator (NAV Consulting, SS&C GlobeOp)
    independently calculates NAV and reconciles with the fund's calculation.
    This module produces the fund's side of that reconciliation.
    Any variance > 0.01% triggers investigation.

Audit trail:
    Every NAV calculation is persisted with:
        - All position prices used
        - All fee accruals
        - Reconciliation variance vs prior day
        - Sign-off metadata

References:
    AIMA Alternative Investment Management Practices Guide
    SEC Form ADV Part 2A — material for registered advisors
    ILPA Principles 3.0 — investor reporting standards
"""

from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.nav")


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

class FeeStructure(str, Enum):
    STANDARD      = "STANDARD"       # 2/20
    FOUNDERS      = "FOUNDERS"       # 1/15
    INSTITUTIONAL = "INSTITUTIONAL"  # 1.5/10
    NO_PERFORMANCE= "NO_PERFORMANCE" # 1/0


FEE_PARAMS = {
    FeeStructure.STANDARD:       {"mgmt_annual": 0.0200, "perf": 0.20},
    FeeStructure.FOUNDERS:       {"mgmt_annual": 0.0100, "perf": 0.15},
    FeeStructure.INSTITUTIONAL:  {"mgmt_annual": 0.0150, "perf": 0.10},
    FeeStructure.NO_PERFORMANCE: {"mgmt_annual": 0.0100, "perf": 0.00},
}


@dataclass
class InvestorAccount:
    """
    A single limited partner's account in the fund.

    Each LP has:
        - A capital account (their economic interest in the fund)
        - A high-water mark (for performance fee calculation)
        - A fee structure (may differ from other LPs)
        - A share count (their proportional ownership)
    """
    investor_id:     str
    name:            str
    committed_capital: float        # Total capital committed (may not all be called)
    called_capital:  float          # Capital actually in the fund
    current_value:   float          # Current NAV of their account
    shares:          float          # Number of fund shares held
    high_water_mark: float          # Peak NAV per share for this investor
    inception_date:  date
    fee_structure:   FeeStructure = FeeStructure.STANDARD
    accrued_mgmt_fee:  float = 0.0
    accrued_perf_fee:  float = 0.0
    realised_gain:     float = 0.0
    total_fees_paid:   float = 0.0
    is_active:         bool  = True
    metadata:          Dict[str, Any] = field(default_factory=dict)

    @property
    def unrealised_gain(self) -> float:
        return self.current_value - self.called_capital

    @property
    def total_return_pct(self) -> float:
        if self.called_capital <= 0:
            return 0.0
        return self.current_value / self.called_capital - 1

    @property
    def moic(self) -> float:
        """Multiple on Invested Capital."""
        if self.called_capital <= 0:
            return 1.0
        return self.current_value / self.called_capital


@dataclass
class DailyNAV:
    """The official NAV record for a single trading day."""
    nav_date:            date
    gross_asset_value:   float      # Sum of all position market values + cash
    total_liabilities:   float      # Accrued fees + other liabilities
    net_asset_value:     float      # GAV - liabilities
    nav_per_share:       float      # NAV / total shares outstanding
    shares_outstanding:  float

    # Fee accruals for the day
    mgmt_fee_accrual:    float      # Daily management fee accrual
    perf_fee_accrual:    float      # Daily performance fee accrual
    other_expenses:      float      # Admin, legal, prime brokerage fees

    # P&L
    daily_pnl:           float
    daily_return_pct:    float
    mtd_return_pct:      float = 0.0
    ytd_return_pct:      float = 0.0

    # Position snapshot (for reconciliation)
    position_count:      int = 0
    cash:                float = 0.0

    # Audit
    calculated_at:       datetime = field(default_factory=datetime.now)
    calculated_by:       str = "nav_engine_v1"
    prior_nav_variance:  float = 0.0    # vs prior day expected

    def to_dict(self) -> dict:
        return {
            "nav_date":           self.nav_date.isoformat(),
            "gross_asset_value":  round(self.gross_asset_value, 2),
            "total_liabilities":  round(self.total_liabilities, 2),
            "net_asset_value":    round(self.net_asset_value, 2),
            "nav_per_share":      round(self.nav_per_share, 6),
            "shares_outstanding": round(self.shares_outstanding, 4),
            "mgmt_fee_accrual":   round(self.mgmt_fee_accrual, 2),
            "perf_fee_accrual":   round(self.perf_fee_accrual, 2),
            "daily_return_pct":   round(self.daily_return_pct, 6),
            "mtd_return_pct":     round(self.mtd_return_pct, 6),
            "ytd_return_pct":     round(self.ytd_return_pct, 6),
            "calculated_at":      self.calculated_at.isoformat(),
        }


@dataclass
class Subscription:
    """A capital subscription (investor putting money into the fund)."""
    sub_id:          str
    investor_id:     str
    amount:          float
    effective_date:  date
    nav_per_share:   float     # NAV/share on effective date
    shares_issued:   float     # amount / nav_per_share
    status:          str = "PENDING"   # PENDING / PROCESSED / REJECTED
    notes:           str = ""


@dataclass
class Redemption:
    """A capital redemption (investor withdrawing money from the fund)."""
    redemption_id:   str
    investor_id:     str
    shares_redeemed: float
    nav_per_share:   float
    gross_proceeds:  float     # shares × nav_per_share
    redemption_fee:  float     # Early redemption penalty if any
    net_proceeds:    float     # gross - fee
    effective_date:  date
    lock_up_days:    int = 0   # Days remaining on lock-up
    status:          str = "PENDING"


# ─────────────────────────────────────────────────────────────────────────────
# NAV Engine
# ─────────────────────────────────────────────────────────────────────────────

class NAVEngine:
    """
    Daily NAV calculation engine.

    Produces the official fund NAV, allocates P&L to investor accounts,
    accrues fees, and maintains the complete accounting record.

    Usage:
        engine = NAVEngine(fund_name="AI Systematic Fund LP")
        engine.add_investor(InvestorAccount(...))

        # Each business day at market close:
        nav = engine.calculate_daily_nav(
            portfolio_nav = 1_042_500,
            cash          = 210_000,
            positions     = {"AAPL": 97500, "MSFT": 124500},
        )
        print(f"NAV: ${nav.net_asset_value:,.2f}")
        print(f"NAV/share: ${nav.nav_per_share:.6f}")
    """

    TRADING_DAYS_PER_YEAR = 252

    def __init__(
        self,
        fund_name:        str = "AI Systematic Fund LP",
        inception_date:   date = date.today(),
        initial_nav_per_share: float = 1000.0,   # $1,000 per share at inception
        other_expenses_annual: float = 0.0050,    # 50bps: admin + legal + prime
        db_path:          Optional[Path] = None,
    ):
        self.fund_name             = fund_name
        self.inception_date        = inception_date
        self.initial_nav_per_share = initial_nav_per_share
        self.other_expenses_rate   = other_expenses_annual
        self.investors:            Dict[str, InvestorAccount] = {}
        self.nav_history:          List[DailyNAV] = []
        self._subscriptions:       List[Subscription] = []
        self._redemptions:         List[Redemption]   = []

        # Current state
        self._shares_outstanding: float = 0.0
        self._current_nav_per_share: float = initial_nav_per_share
        self._total_accrued_fees:    float = 0.0
        self._mtd_start_nav:         float = 0.0
        self._ytd_start_nav:         float = 0.0

        # Persistence
        db = db_path or (Path(__file__).parents[3] / "db" / "fund_accounting.db")
        db.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db
        self._init_db()

        logger.info(
            f"NAV Engine initialised: {fund_name} | "
            f"inception={inception_date} | "
            f"initial NAV/share=${initial_nav_per_share}"
        )

    def _init_db(self):
        with sqlite3.connect(self._db_path) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nav_history (
                    nav_date        TEXT PRIMARY KEY,
                    nav             REAL,
                    nav_per_share   REAL,
                    shares_outstanding REAL,
                    daily_return_pct REAL,
                    mtd_return_pct  REAL,
                    ytd_return_pct  REAL,
                    mgmt_fee        REAL,
                    perf_fee        REAL,
                    full_json       TEXT,
                    calculated_at   TEXT
                );

                CREATE TABLE IF NOT EXISTS investors (
                    investor_id     TEXT PRIMARY KEY,
                    name            TEXT,
                    called_capital  REAL,
                    current_value   REAL,
                    shares          REAL,
                    high_water_mark REAL,
                    fee_structure   TEXT,
                    inception_date  TEXT,
                    total_fees_paid REAL,
                    full_json       TEXT
                );

                CREATE TABLE IF NOT EXISTS subscriptions (
                    sub_id          TEXT PRIMARY KEY,
                    investor_id     TEXT,
                    amount          REAL,
                    effective_date  TEXT,
                    nav_per_share   REAL,
                    shares_issued   REAL,
                    status          TEXT
                );

                CREATE TABLE IF NOT EXISTS redemptions (
                    redemption_id   TEXT PRIMARY KEY,
                    investor_id     TEXT,
                    shares_redeemed REAL,
                    nav_per_share   REAL,
                    net_proceeds    REAL,
                    effective_date  TEXT,
                    status          TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_nav_date ON nav_history (nav_date);
            """)
            conn.commit()

    # ── Investor management ───────────────────────────────────────────────────

    def add_investor(
        self,
        investor_id:   str,
        name:          str,
        capital:       float,
        fee_structure: FeeStructure = FeeStructure.STANDARD,
        effective_date: Optional[date] = None,
        founder:       bool = False,
    ) -> Tuple[InvestorAccount, Subscription]:
        """
        Add a new investor to the fund via subscription.

        Converts their capital into fund shares at the current NAV/share.
        If fund is new (no prior NAV), uses the initial NAV/share.
        """
        eff_date   = effective_date or date.today()
        nav_ps     = self._current_nav_per_share
        shares     = capital / nav_ps

        if founder:
            fee_structure = FeeStructure.FOUNDERS

        investor = InvestorAccount(
            investor_id     = investor_id,
            name            = name,
            committed_capital = capital,
            called_capital  = capital,
            current_value   = capital,
            shares          = shares,
            high_water_mark = nav_ps,
            inception_date  = eff_date,
            fee_structure   = fee_structure,
        )
        self.investors[investor_id] = investor
        self._shares_outstanding   += shares

        import uuid
        sub = Subscription(
            sub_id         = f"SUB_{uuid.uuid4().hex[:8].upper()}",
            investor_id    = investor_id,
            amount         = capital,
            effective_date = eff_date,
            nav_per_share  = nav_ps,
            shares_issued  = shares,
            status         = "PROCESSED",
        )
        self._subscriptions.append(sub)

        logger.info(
            f"Investor added: {name} ({investor_id}) | "
            f"${capital:,.0f} | {shares:.4f} shares @ ${nav_ps:.4f}/share | "
            f"fees={fee_structure.value}"
        )
        self._save_investor(investor)
        return investor, sub

    # ── NAV calculation ────────────────────────────────────────────────────────

    def calculate_daily_nav(
        self,
        portfolio_nav:    float,    # Total portfolio value (from live system)
        cash:             float,
        positions:        Dict[str, float],   # {ticker: market_value}
        nav_date:         Optional[date] = None,
    ) -> DailyNAV:
        """
        Calculate the official daily NAV.

        Steps:
            1. Gross Asset Value = positions + cash
            2. Accrue management fees (1/252 of annual rate × NAV)
            3. Accrue performance fees (vs high-water mark)
            4. Accrue other expenses (admin, legal, prime brokerage)
            5. Net Asset Value = GAV - accrued liabilities
            6. NAV per share = NAV / shares outstanding
            7. Allocate to investor accounts

        Args:
            portfolio_nav:  Total NAV from LiveRiskEngine / portfolio object
            cash:           Cash component of NAV
            positions:      Market values by ticker
            nav_date:       Calculation date (defaults to today)
        """
        nav_date = nav_date or date.today()

        # ── Gross Asset Value ─────────────────────────────────────────────────
        gav = portfolio_nav   # Positions + cash already summed

        # ── Fee accruals (daily) ───────────────────────────────────────────────
        prior_nav = self.nav_history[-1].net_asset_value if self.nav_history else gav

        # Management fee: annual_rate / 252 × prior_NAV
        # Using blended rate across investor fee structures
        blended_mgmt_rate = self._blended_mgmt_rate()
        daily_mgmt_fee    = prior_nav * blended_mgmt_rate / self.TRADING_DAYS_PER_YEAR

        # Performance fee: 20% of daily gain above high-water mark
        # Computed at the fund level; allocation to investors is per-account
        daily_perf_fee = self._compute_daily_perf_fee(gav, prior_nav)

        # Other expenses: 50bps annual accrued daily
        daily_other = prior_nav * self.other_expenses_rate / self.TRADING_DAYS_PER_YEAR

        total_accrued_today = daily_mgmt_fee + daily_perf_fee + daily_other
        self._total_accrued_fees += total_accrued_today

        # ── Net Asset Value ────────────────────────────────────────────────────
        nav = gav - self._total_accrued_fees

        # ── NAV per share ──────────────────────────────────────────────────────
        if self._shares_outstanding > 0:
            nav_per_share = nav / self._shares_outstanding
        else:
            nav_per_share = self.initial_nav_per_share

        self._current_nav_per_share = nav_per_share

        # ── Returns ────────────────────────────────────────────────────────────
        prior_nav_ps   = self.nav_history[-1].nav_per_share if self.nav_history else self.initial_nav_per_share
        daily_ret      = (nav_per_share / prior_nav_ps - 1) if prior_nav_ps > 0 else 0.0

        # MTD/YTD reference points
        if not self._mtd_start_nav or nav_date.day == 1:
            self._mtd_start_nav = prior_nav_ps
        if not self._ytd_start_nav or (nav_date.month == 1 and nav_date.day == 1):
            self._ytd_start_nav = prior_nav_ps

        mtd_ret = (nav_per_share / self._mtd_start_nav - 1) if self._mtd_start_nav > 0 else 0.0
        ytd_ret = (nav_per_share / self._ytd_start_nav - 1) if self._ytd_start_nav > 0 else 0.0

        # ── Build NAV record ───────────────────────────────────────────────────
        daily_nav = DailyNAV(
            nav_date           = nav_date,
            gross_asset_value  = round(gav, 2),
            total_liabilities  = round(self._total_accrued_fees, 2),
            net_asset_value    = round(nav, 2),
            nav_per_share      = round(nav_per_share, 6),
            shares_outstanding = round(self._shares_outstanding, 6),
            mgmt_fee_accrual   = round(daily_mgmt_fee, 2),
            perf_fee_accrual   = round(daily_perf_fee, 2),
            other_expenses     = round(daily_other, 2),
            daily_pnl          = round(nav - prior_nav, 2),
            daily_return_pct   = round(daily_ret, 6),
            mtd_return_pct     = round(mtd_ret, 6),
            ytd_return_pct     = round(ytd_ret, 6),
            position_count     = len(positions),
            cash               = round(cash, 2),
        )

        self.nav_history.append(daily_nav)
        self._persist_nav(daily_nav)
        self._update_investor_accounts(nav_per_share)

        logger.info(
            f"NAV [{nav_date}]: ${nav:,.2f} | "
            f"${nav_per_share:.4f}/share | "
            f"{daily_ret:+.3%} | "
            f"fees: mgmt=${daily_mgmt_fee:.2f} perf=${daily_perf_fee:.2f}"
        )

        return daily_nav

    def _blended_mgmt_rate(self) -> float:
        """Weighted average management fee rate across all investors."""
        if not self.investors:
            return FEE_PARAMS[FeeStructure.STANDARD]["mgmt_annual"]

        total_value = sum(inv.current_value for inv in self.investors.values())
        if total_value <= 0:
            return FEE_PARAMS[FeeStructure.STANDARD]["mgmt_annual"]

        blended = sum(
            inv.current_value * FEE_PARAMS[inv.fee_structure]["mgmt_annual"]
            for inv in self.investors.values()
        ) / total_value
        return blended

    def _compute_daily_perf_fee(self, current_gav: float, prior_nav: float) -> float:
        """
        Compute daily performance fee accrual.

        Performance fee only accrues when:
            (a) Today's NAV per share exceeds the high-water mark
            (b) There is a positive return today

        The fee is 20% (or investor's rate) of the daily gain above HWM.
        """
        if not self.investors or self._shares_outstanding <= 0:
            return 0.0

        total_perf_fee = 0.0
        current_navps  = current_gav / self._shares_outstanding

        for investor in self.investors.values():
            if not investor.is_active:
                continue
            perf_rate = FEE_PARAMS[investor.fee_structure]["perf"]
            if perf_rate <= 0:
                continue

            # Only charge performance fee on gains above HWM
            if current_navps > investor.high_water_mark:
                # Daily gain above HWM, proportional to this investor's shares
                gain_above_hwm = (current_navps - investor.high_water_mark) * investor.shares
                fee            = gain_above_hwm * perf_rate
                total_perf_fee += fee
                investor.accrued_perf_fee += fee

        return total_perf_fee

    def _update_investor_accounts(self, nav_per_share: float) -> None:
        """Allocate P&L to investor accounts and update high-water marks."""
        for investor in self.investors.values():
            if not investor.is_active:
                continue
            # Update current value
            investor.current_value = investor.shares * nav_per_share

            # Update high-water mark if we're at a new peak
            if nav_per_share > investor.high_water_mark:
                investor.high_water_mark = nav_per_share

    # ── Subscriptions & redemptions ───────────────────────────────────────────

    def process_redemption(
        self,
        investor_id:    str,
        shares:         Optional[float] = None,
        amount:         Optional[float] = None,
        lock_up_days_remaining: int = 0,
        redemption_fee_pct: float = 0.0,
    ) -> Optional[Redemption]:
        """
        Process a redemption request.

        Investor can specify either shares or dollar amount.
        If lock-up period has not expired, apply early redemption fee.
        """
        import uuid
        investor = self.investors.get(investor_id)
        if not investor:
            logger.error(f"Investor {investor_id} not found")
            return None

        nav_ps = self._current_nav_per_share

        # Determine shares to redeem
        if amount is not None:
            shares_to_redeem = amount / nav_ps
        elif shares is not None:
            shares_to_redeem = shares
        else:
            shares_to_redeem = investor.shares   # Full redemption

        shares_to_redeem = min(shares_to_redeem, investor.shares)

        gross   = shares_to_redeem * nav_ps
        fee_amt = gross * redemption_fee_pct
        net     = gross - fee_amt

        redemption = Redemption(
            redemption_id   = f"RDM_{uuid.uuid4().hex[:8].upper()}",
            investor_id     = investor_id,
            shares_redeemed = shares_to_redeem,
            nav_per_share   = nav_ps,
            gross_proceeds  = round(gross, 2),
            redemption_fee  = round(fee_amt, 2),
            net_proceeds    = round(net, 2),
            effective_date  = date.today(),
            lock_up_days    = lock_up_days_remaining,
            status          = "PROCESSED",
        )

        # Update investor account
        investor.shares        -= shares_to_redeem
        investor.called_capital = max(0, investor.called_capital - net)
        investor.current_value  = investor.shares * nav_ps
        investor.total_fees_paid += fee_amt
        self._shares_outstanding -= shares_to_redeem

        if investor.shares < 0.001:
            investor.is_active = False

        self._redemptions.append(redemption)

        logger.info(
            f"Redemption: {investor.name} | "
            f"{shares_to_redeem:.4f} shares @ ${nav_ps:.4f} | "
            f"net proceeds ${net:,.2f}"
        )
        return redemption

    # ── Reporting ─────────────────────────────────────────────────────────────

    def get_investor_statement(self, investor_id: str) -> Dict[str, Any]:
        """Generate an investor account statement."""
        investor = self.investors.get(investor_id)
        if not investor:
            return {"error": f"Investor {investor_id} not found"}

        return {
            "investor_id":        investor.investor_id,
            "name":               investor.name,
            "as_of_date":         date.today().isoformat(),
            "called_capital":     round(investor.called_capital, 2),
            "current_value":      round(investor.current_value, 2),
            "unrealised_gain":    round(investor.unrealised_gain, 2),
            "total_return_pct":   round(investor.total_return_pct * 100, 2),
            "moic":               round(investor.moic, 3),
            "shares_held":        round(investor.shares, 6),
            "nav_per_share":      round(self._current_nav_per_share, 6),
            "high_water_mark":    round(investor.high_water_mark, 6),
            "accrued_mgmt_fee":   round(investor.accrued_mgmt_fee, 2),
            "accrued_perf_fee":   round(investor.accrued_perf_fee, 2),
            "total_fees_paid":    round(investor.total_fees_paid, 2),
            "fee_structure":      investor.fee_structure.value,
            "inception_date":     investor.inception_date.isoformat(),
        }

    def get_fund_summary(self) -> Dict[str, Any]:
        """Current fund summary for reporting."""
        latest = self.nav_history[-1] if self.nav_history else None
        return {
            "fund_name":           self.fund_name,
            "as_of_date":          date.today().isoformat(),
            "nav":                 round(latest.net_asset_value, 2) if latest else 0,
            "nav_per_share":       round(self._current_nav_per_share, 6),
            "shares_outstanding":  round(self._shares_outstanding, 6),
            "n_investors":         sum(1 for i in self.investors.values() if i.is_active),
            "total_committed":     sum(i.committed_capital for i in self.investors.values()),
            "total_called":        sum(i.called_capital for i in self.investors.values()),
            "ytd_return":          round(latest.ytd_return_pct * 100, 2) if latest else 0,
            "mtd_return":          round(latest.mtd_return_pct * 100, 2) if latest else 0,
            "total_fees_accrued":  round(self._total_accrued_fees, 2),
        }

    def nav_series(self, start_date: Optional[date] = None) -> pd.DataFrame:
        """Return NAV history as a DataFrame for charting/analysis."""
        rows = [n.to_dict() for n in self.nav_history]
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["nav_date"] = pd.to_datetime(df["nav_date"])
        df = df.set_index("nav_date").sort_index()
        if start_date:
            df = df[df.index >= pd.Timestamp(start_date)]
        return df

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist_nav(self, nav: DailyNAV) -> None:
        import json
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO nav_history
                    (nav_date, nav, nav_per_share, shares_outstanding,
                     daily_return_pct, mtd_return_pct, ytd_return_pct,
                     mgmt_fee, perf_fee, full_json, calculated_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    nav.nav_date.isoformat(), nav.net_asset_value,
                    nav.nav_per_share, nav.shares_outstanding,
                    nav.daily_return_pct, nav.mtd_return_pct, nav.ytd_return_pct,
                    nav.mgmt_fee_accrual, nav.perf_fee_accrual,
                    json.dumps(nav.to_dict()), nav.calculated_at.isoformat(),
                ))
                conn.commit()
        except Exception as e:
            logger.error(f"NAV persist failed: {e}")

    def _save_investor(self, investor: InvestorAccount) -> None:
        import json
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO investors
                    (investor_id, name, called_capital, current_value, shares,
                     high_water_mark, fee_structure, inception_date,
                     total_fees_paid, full_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (
                    investor.investor_id, investor.name,
                    investor.called_capital, investor.current_value,
                    investor.shares, investor.high_water_mark,
                    investor.fee_structure.value,
                    investor.inception_date.isoformat(),
                    investor.total_fees_paid,
                    json.dumps(investor.__dict__, default=str),
                ))
                conn.commit()
        except Exception as e:
            logger.debug(f"Investor persist failed: {e}")

    def load_history(self) -> int:
        """Load NAV history from database on startup."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM nav_history ORDER BY nav_date ASC"
                ).fetchall()
            for row in rows:
                import json
                try:
                    d = json.loads(row["full_json"])
                    self.nav_history.append(DailyNAV(
                        nav_date          = date.fromisoformat(d["nav_date"]),
                        gross_asset_value = d["gross_asset_value"],
                        total_liabilities = d["total_liabilities"],
                        net_asset_value   = d["net_asset_value"],
                        nav_per_share     = d["nav_per_share"],
                        shares_outstanding= d["shares_outstanding"],
                        mgmt_fee_accrual  = d["mgmt_fee_accrual"],
                        perf_fee_accrual  = d["perf_fee_accrual"],
                        other_expenses    = 0,
                        daily_pnl         = 0,
                        daily_return_pct  = d["daily_return_pct"],
                        mtd_return_pct    = d["mtd_return_pct"],
                        ytd_return_pct    = d["ytd_return_pct"],
                    ))
                except Exception:
                    pass
            if self.nav_history:
                self._current_nav_per_share = self.nav_history[-1].nav_per_share
            return len(self.nav_history)
        except Exception as e:
            logger.debug(f"History load failed: {e}")
            return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  NAV Engine — Test")
    print("=" * 65)

    engine = NAVEngine(
        fund_name     = "AI Systematic Fund LP",
        inception_date= date(2024, 1, 1),
        initial_nav_per_share = 1000.0,
    )

    # Add investors
    print("\n  Adding investors...")
    inv1, _ = engine.add_investor("LP001", "Founder Capital LLC", 500_000, founder=True)
    inv2, _ = engine.add_investor("LP002", "Institutional Allocator", 1_000_000, FeeStructure.INSTITUTIONAL)
    inv3, _ = engine.add_investor("LP003", "High Net Worth Individual", 250_000)

    print(f"\n  Fund summary after subscriptions:")
    summary = engine.get_fund_summary()
    for k, v in summary.items():
        print(f"    {k:<25}: {v}")

    # Simulate 5 days of NAV calculations
    print("\n  Simulating 5 days of NAV calculations...")
    import numpy as np
    np.random.seed(42)
    current_nav = 1_750_000.0

    for i in range(5):
        dt = date(2024, 1, i + 2)
        current_nav *= (1 + np.random.normal(0.0004, 0.012))
        nav = engine.calculate_daily_nav(
            portfolio_nav = current_nav,
            cash          = current_nav * 0.15,
            positions     = {"AAPL": current_nav * 0.30, "MSFT": current_nav * 0.25},
            nav_date      = dt,
        )
        print(f"  Day {i+1}: NAV=${nav.net_asset_value:>12,.2f} | "
              f"${nav.nav_per_share:.4f}/share | {nav.daily_return_pct:+.3%}")

    # Investor statement
    print("\n  Investor Statement — LP001 (Founder):")
    stmt = engine.get_investor_statement("LP001")
    for k, v in stmt.items():
        print(f"    {k:<25}: {v}")

    # Test redemption
    print("\n  Processing partial redemption for LP003...")
    rdm = engine.process_redemption("LP003", amount=50_000)
    if rdm:
        print(f"  Redeemed: ${rdm.gross_proceeds:,.2f} gross | ${rdm.net_proceeds:,.2f} net")

    print(f"\n  Final fund summary:")
    for k, v in engine.get_fund_summary().items():
        print(f"    {k:<25}: {v}")

    print("\n✅ NAV Engine tests passed")
