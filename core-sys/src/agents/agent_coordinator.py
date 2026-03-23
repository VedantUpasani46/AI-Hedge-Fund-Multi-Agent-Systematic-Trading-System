"""
AI Hedge Fund — Part 2: Multi-Agent System
============================================
agent_coordinator.py — Multi-Agent Orchestration & Consensus

The Coordinator is the conductor of the agent orchestra.
It receives investment ideas from external triggers (daily scan,
manual request, signal alert) and orchestrates the full
multi-agent workflow to reach a final decision.

Full workflow for an allocation decision:
  1. Research Analyst runs deep analysis (3-5 min)
  2. Risk Manager runs pre-trade checks (30 sec)
  3. Portfolio Manager synthesises signals + research (1-2 min)
  4. CONSENSUS: All three agents vote (10 sec)
  5. Coordinator reviews consensus, makes final call
  6. Decision logged, execution triggered (if approved)

The consensus protocol is what makes this system institutional-grade.
No single agent can push through a bad trade — it needs agreement
from the risk function (or a documented override).

In a real fund:
  Single PM making all decisions = key-man risk, conflict of interest
  Multi-agent with independent risk = institutional-grade governance

Decision rules:
  UNANIMOUS BUY     → Execute at full Kelly size
  PM+ANALYST BUY, RISK APPROVED   → Execute at 75% Kelly
  PM BUY, ANALYST HOLD, RISK OK   → Execute at 50% Kelly
  RISK REJECTION (any)            → PASS, log override attempt
  PM+ANALYST disagree             → HOLD, re-analyse in 24h
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("hedge_fund.coordinator")


# ─────────────────────────────────────────────────────────────────────────────
# Consensus structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentVote:
    """A single agent's vote in the consensus process."""
    agent_name:      str
    vote:            str            # BUY / SELL / HOLD / PASS
    confidence:      float          # 0-1
    reasoning:       str
    key_factors:     List[str]
    timestamp:       datetime = field(default_factory=datetime.now)
    llm_cost:        float = 0.0

    def to_dict(self) -> dict:
        return {
            "agent":       self.agent_name,
            "vote":        self.vote,
            "confidence":  round(self.confidence, 3),
            "reasoning":   self.reasoning[:300],
            "key_factors": self.key_factors,
            "timestamp":   self.timestamp.isoformat(),
        }


@dataclass
class ConsensusResult:
    """
    Outcome of the multi-agent consensus process.

    Contains all votes, the final decision, position sizing,
    and full audit trail.
    """
    consensus_id:    str
    ticker:          str
    votes:           List[AgentVote]
    final_decision:  str            # BUY / SELL / HOLD / PASS
    final_weight:    float          # Portfolio weight (0-1)
    size_rationale:  str            # Why this size?
    risk_approved:   bool
    risk_check_detail: Optional[Dict]
    total_llm_cost:  float
    latency_seconds: float
    timestamp:       datetime = field(default_factory=datetime.now)

    @property
    def buy_votes(self) -> int:
        return sum(1 for v in self.votes if v.vote in ("BUY", "STRONG BUY"))

    @property
    def sell_votes(self) -> int:
        return sum(1 for v in self.votes if v.vote in ("SELL", "STRONG SELL"))

    @property
    def total_votes(self) -> int:
        return len(self.votes)

    @property
    def avg_confidence(self) -> float:
        if not self.votes:
            return 0.0
        return float(np.mean([v.confidence for v in self.votes]))

    def summary(self) -> str:
        lines = [
            "═" * 65,
            f"  CONSENSUS DECISION — {self.ticker}",
            f"  ID: {self.consensus_id}",
            f"  {self.timestamp:%Y-%m-%d %H:%M:%S}",
            "═" * 65,
            f"  FINAL DECISION : {self.final_decision}",
            f"  FINAL WEIGHT   : {self.final_weight:.1%} of NAV",
            f"  RISK APPROVED  : {'YES ✅' if self.risk_approved else 'NO ❌'}",
            f"  AVG CONFIDENCE : {self.avg_confidence:.0%}",
            f"  TOTAL COST     : ${self.total_llm_cost:.4f}",
            f"  LATENCY        : {self.latency_seconds:.1f}s",
            "─" * 65,
            "  VOTES:",
        ]
        for v in self.votes:
            lines.append(
                f"    {v.agent_name:<20} {v.vote:<10} conf={v.confidence:.0%}"
            )
            if v.key_factors:
                lines.append(f"      → {v.key_factors[0]}")
        lines += [
            "─" * 65,
            f"  SIZING RATIONALE: {self.size_rationale}",
            "═" * 65,
        ]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "consensus_id":   self.consensus_id,
            "ticker":         self.ticker,
            "final_decision": self.final_decision,
            "final_weight":   self.final_weight,
            "size_rationale": self.size_rationale,
            "risk_approved":  self.risk_approved,
            "buy_votes":      self.buy_votes,
            "sell_votes":     self.sell_votes,
            "total_votes":    self.total_votes,
            "avg_confidence": self.avg_confidence,
            "total_llm_cost": self.total_llm_cost,
            "latency_seconds": self.latency_seconds,
            "votes":          [v.to_dict() for v in self.votes],
            "timestamp":      self.timestamp.isoformat(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Decision database
# ─────────────────────────────────────────────────────────────────────────────

class DecisionDatabase:
    """
    Persistent store for all consensus decisions.

    Every decision is logged permanently for:
      - Performance attribution (did consensus decisions outperform solo?)
      - Audit trail (regulators, investors, due diligence)
      - Learning (future models can train on which signals worked)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (Path(__file__).parents[3] / "db" / "decisions.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS consensus_decisions (
                    consensus_id    TEXT PRIMARY KEY,
                    ticker          TEXT NOT NULL,
                    timestamp       TEXT NOT NULL,
                    final_decision  TEXT NOT NULL,
                    final_weight    REAL,
                    buy_votes       INTEGER,
                    sell_votes      INTEGER,
                    avg_confidence  REAL,
                    risk_approved   INTEGER,
                    total_llm_cost  REAL,
                    latency_seconds REAL,
                    full_json       TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_votes (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    consensus_id    TEXT NOT NULL,
                    agent_name      TEXT,
                    vote            TEXT,
                    confidence      REAL,
                    reasoning       TEXT,
                    timestamp       TEXT,
                    FOREIGN KEY (consensus_id) REFERENCES consensus_decisions(consensus_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_decisions_ticker
                ON consensus_decisions (ticker, timestamp)
            """)
            conn.commit()

    def save(self, result: ConsensusResult) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO consensus_decisions
                (consensus_id, ticker, timestamp, final_decision, final_weight,
                 buy_votes, sell_votes, avg_confidence, risk_approved,
                 total_llm_cost, latency_seconds, full_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                result.consensus_id, result.ticker,
                result.timestamp.isoformat(), result.final_decision,
                result.final_weight, result.buy_votes, result.sell_votes,
                result.avg_confidence, int(result.risk_approved),
                result.total_llm_cost, result.latency_seconds,
                json.dumps(result.to_dict()),
            ))
            for vote in result.votes:
                conn.execute("""
                    INSERT INTO agent_votes
                    (consensus_id, agent_name, vote, confidence, reasoning, timestamp)
                    VALUES (?,?,?,?,?,?)
                """, (
                    result.consensus_id, vote.agent_name, vote.vote,
                    vote.confidence, vote.reasoning[:500],
                    vote.timestamp.isoformat(),
                ))
            conn.commit()

    def get_recent(self, ticker: str, days: int = 30) -> List[Dict]:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("""
                SELECT consensus_id, ticker, timestamp, final_decision,
                       final_weight, avg_confidence, risk_approved
                FROM consensus_decisions
                WHERE ticker = ? AND timestamp >= ?
                ORDER BY timestamp DESC
            """, (ticker, cutoff)).fetchall()
        return [dict(r) for r in rows]

    def decision_stats(self, days: int = 30) -> Dict:
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT final_decision, COUNT(*) as cnt,
                       AVG(avg_confidence) as avg_conf,
                       SUM(total_llm_cost) as total_cost
                FROM consensus_decisions
                WHERE timestamp >= ?
                GROUP BY final_decision
            """, (cutoff,)).fetchall()
        return {r[0]: {"count": r[1], "avg_confidence": r[2], "total_cost": r[3]}
                for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Agent Coordinator
# ─────────────────────────────────────────────────────────────────────────────

class AgentCoordinator:
    """
    Orchestrates the multi-agent decision process.

    This is the top-level entry point for investment decisions.
    External code calls coordinator.decide(ticker) and gets back
    a fully reasoned, risk-checked, consensus decision.

    Coordination flow:
        1. research_analyst.analyse(ticker)    → ResearchReport
        2. portfolio_manager.make_decision()   → AgentDecision
        3. risk_manager.pre_trade_check()      → PreTradeCheck
        4. coordinator.run_consensus()         → ConsensusResult
        5. Apply consensus rules → final decision + size
        6. Persist to database
        7. Return ConsensusResult
    """

    # Consensus rules: (buy_votes_needed, min_confidence, kelly_fraction)
    CONSENSUS_RULES = {
        "UNANIMOUS_BUY":      (3, 0.70, 1.00),   # All 3 BUY → full Kelly
        "STRONG_MAJORITY":    (2, 0.65, 0.75),   # 2/3 BUY, high conf
        "WEAK_MAJORITY":      (2, 0.50, 0.50),   # 2/3 BUY, low conf
        "PASS":               (0, 0.00, 0.00),   # Anything else → PASS
    }

    def __init__(
        self,
        portfolio,
        pm_agent    = None,
        risk_agent  = None,
        analyst     = None,
    ):
        self.portfolio = portfolio

        # Agents (lazy-loaded if not provided)
        self._pm_agent    = pm_agent
        self._risk_agent  = risk_agent
        self._analyst     = analyst

        # State tracking
        self.db      = DecisionDatabase()
        self._decisions: List[ConsensusResult] = []
        self._lock   = threading.Lock()

        logger.info("AgentCoordinator initialised")

    # ── Agent accessors (lazy init) ───────────────────────────────────────────

    @property
    def pm_agent(self):
        if self._pm_agent is None:
            from src.agents.portfolio_manager_agent import PortfolioManagerAgent
            self._pm_agent = PortfolioManagerAgent(self.portfolio)
        return self._pm_agent

    @property
    def risk_agent(self):
        if self._risk_agent is None:
            from src.agents.risk_manager_agent import RiskManagerAgent
            self._risk_agent = RiskManagerAgent(self.portfolio)
        return self._risk_agent

    @property
    def analyst(self):
        if self._analyst is None:
            from src.agents.research_analyst_agent import ResearchAnalystAgent
            self._analyst = ResearchAnalystAgent()
        return self._analyst

    # ── Main decision entry point ─────────────────────────────────────────────

    def decide(
        self,
        ticker:      str,
        fast_mode:   bool = False,    # Skip research for speed
        context:     str  = "",
    ) -> ConsensusResult:
        """
        Full multi-agent decision for a ticker.

        Args:
            ticker    : Security to evaluate
            fast_mode : Skip Research Analyst (faster, less thorough)
            context   : Additional context string

        Returns:
            ConsensusResult with final decision, size, and all votes
        """
        consensus_id = f"CON_{ticker}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        start_time   = datetime.now()
        total_cost   = 0.0

        logger.info(f"Coordinator starting decision: {ticker} | id={consensus_id}")

        votes        = []
        risk_check   = None
        risk_approved = True

        # ── Step 1: Research Analyst ──────────────────────────────────────────
        research_summary = ""
        if not fast_mode:
            try:
                logger.info(f"  Step 1/3: Research Analyst analysing {ticker}...")
                report = self.analyst.analyse(ticker, context)

                # Build vote from research report
                votes.append(AgentVote(
                    agent_name  = "ResearchAnalyst",
                    vote        = self._rec_to_vote(report.recommendation),
                    confidence  = report.conviction_score,
                    reasoning   = report.investment_thesis[:300],
                    key_factors = report.key_catalysts[:3],
                ))
                research_summary = (
                    f"Research: {report.recommendation} | "
                    f"Thesis: {report.investment_thesis[:200]}"
                )
                logger.info(
                    f"  Research complete: {report.recommendation} "
                    f"conf={report.conviction_score:.0%}"
                )
            except Exception as e:
                logger.error(f"  Research Analyst failed: {e}")
                votes.append(AgentVote(
                    agent_name  = "ResearchAnalyst",
                    vote        = "HOLD",
                    confidence  = 0.3,
                    reasoning   = f"Analysis failed: {e}",
                    key_factors = [],
                ))

        # ── Step 2: Portfolio Manager ─────────────────────────────────────────
        logger.info(f"  Step 2/3: Portfolio Manager deciding on {ticker}...")
        try:
            from src.data.market_data import MarketDataFetcher
            fetcher   = MarketDataFetcher()
            snapshot  = fetcher.get_market_snapshot(
                [ticker, "SPY", "^VIX"], as_of=None
            )
            market_ctx = (
                f"Market: {snapshot.regime.value} | "
                f"VIX: {snapshot.vix_level or 'N/A'} | "
                f"{research_summary}"
            )

            pm_decision = self.pm_agent.make_decision(
                ticker     = ticker,
                market_ctx = market_ctx,
            )
            total_cost += pm_decision.llm_cost_usd

            votes.append(AgentVote(
                agent_name  = "PortfolioManager",
                vote        = pm_decision.recommendation,
                confidence  = (
                    0.85 if pm_decision.conviction.value == "HIGH" else
                    0.60 if pm_decision.conviction.value == "MEDIUM" else 0.35
                ),
                reasoning   = pm_decision.reasoning[:300],
                key_factors = pm_decision.key_factors[:3],
                llm_cost    = pm_decision.llm_cost_usd,
            ))
            logger.info(
                f"  PM decision: {pm_decision.recommendation} "
                f"weight={pm_decision.target_weight:.1%}"
            )

        except Exception as e:
            logger.error(f"  Portfolio Manager failed: {e}")
            votes.append(AgentVote(
                agent_name  = "PortfolioManager",
                vote        = "PASS",
                confidence  = 0.1,
                reasoning   = f"PM failed: {e}",
                key_factors = [],
            ))

        # ── Step 3: Risk Manager pre-trade check ──────────────────────────────
        logger.info(f"  Step 3/3: Risk Manager checking {ticker}...")
        proposed_weight = 0.0
        for v in votes:
            if v.agent_name == "PortfolioManager":
                # Extract target weight from PM decision
                try:
                    proposed_weight = pm_decision.target_weight
                except Exception:
                    proposed_weight = 0.05  # Default 5%
                break

        try:
            risk_check   = self.risk_agent.pre_trade_check(
                ticker          = ticker,
                proposed_weight = proposed_weight,
                current_weight  = self.portfolio.position_weight(ticker),
            )
            risk_approved = risk_check.approved

            if risk_check.all_passed:
                risk_vote = "BUY" if any(v.vote in ("BUY","STRONG BUY") for v in votes) else "HOLD"
                risk_conf = 0.80
                risk_msg  = "All risk checks passed"
            elif risk_check.warnings and not risk_check.breaches:
                risk_vote = "HOLD"   # Has warnings but no breaches — cautious
                risk_conf = 0.55
                risk_msg  = f"Warnings: {[w.message for w in risk_check.warnings]}"
            else:
                risk_vote = "PASS"   # Breach → block the trade
                risk_conf = 0.95     # Very confident in the rejection
                risk_msg  = f"BREACHES: {[b.message for b in risk_check.breaches]}"
                risk_approved = False

            votes.append(AgentVote(
                agent_name  = "RiskManager",
                vote        = risk_vote,
                confidence  = risk_conf,
                reasoning   = risk_msg,
                key_factors = [c.message for c in (risk_check.breaches or risk_check.warnings)][:3],
            ))
            logger.info(
                f"  Risk check: {'APPROVED' if risk_approved else 'REJECTED'} | "
                f"vote={risk_vote}"
            )

        except Exception as e:
            logger.error(f"  Risk Manager failed: {e}")
            votes.append(AgentVote(
                agent_name  = "RiskManager",
                vote        = "HOLD",
                confidence  = 0.3,
                reasoning   = f"Risk check failed: {e} — proceeding with caution",
                key_factors = [],
            ))

        # ── Step 4: Apply consensus rules ─────────────────────────────────────
        final_decision, final_weight, size_rationale = self._apply_consensus(
            votes            = votes,
            proposed_weight  = proposed_weight,
            risk_approved    = risk_approved,
        )

        latency = (datetime.now() - start_time).total_seconds()

        result = ConsensusResult(
            consensus_id      = consensus_id,
            ticker            = ticker,
            votes             = votes,
            final_decision    = final_decision,
            final_weight      = final_weight,
            size_rationale    = size_rationale,
            risk_approved     = risk_approved,
            risk_check_detail = risk_check.to_dict() if risk_check else None,
            total_llm_cost    = total_cost,
            latency_seconds   = latency,
        )

        # Persist
        with self._lock:
            self._decisions.append(result)
            try:
                self.db.save(result)
            except Exception as e:
                logger.error(f"Failed to save decision to DB: {e}")

        logger.info(
            f"Consensus complete: {ticker} → {final_decision} "
            f"{final_weight:.1%} | "
            f"cost=${total_cost:.4f} | "
            f"{latency:.1f}s"
        )

        return result

    def _rec_to_vote(self, recommendation: str) -> str:
        """Normalise recommendation string to vote."""
        rec = recommendation.upper().strip()
        if rec in ("STRONG BUY", "BUY"):
            return "BUY"
        elif rec in ("STRONG SELL", "SELL"):
            return "SELL"
        elif rec == "HOLD":
            return "HOLD"
        return "PASS"

    def _apply_consensus(
        self,
        votes:           List[AgentVote],
        proposed_weight: float,
        risk_approved:   bool,
    ) -> Tuple[str, float, str]:
        """
        Apply consensus rules to derive final decision and position size.

        Returns (final_decision, final_weight, size_rationale)
        """
        # Risk veto: if Risk Manager says PASS, always PASS
        risk_votes = [v for v in votes if v.agent_name == "RiskManager"]
        if risk_votes and risk_votes[0].vote == "PASS":
            return (
                "PASS",
                0.0,
                f"Risk Manager rejected trade: {risk_votes[0].reasoning}"
            )

        if not risk_approved:
            return (
                "PASS",
                0.0,
                "Risk limits breached — trade blocked by risk management"
            )

        # Count votes
        buy_votes  = sum(1 for v in votes if v.vote in ("BUY", "STRONG BUY"))
        sell_votes = sum(1 for v in votes if v.vote in ("SELL", "STRONG SELL"))
        total      = len(votes)
        avg_conf   = float(np.mean([v.confidence for v in votes])) if votes else 0

        # SELL consensus
        if sell_votes >= 2:
            return (
                "SELL",
                0.0,
                f"{sell_votes}/{total} agents voted SELL"
            )

        # BUY consensus — apply sizing rules
        if buy_votes == total and total >= 2 and avg_conf >= 0.70:
            # Unanimous BUY with high confidence
            kelly = proposed_weight * 1.00
            return (
                "BUY",
                min(kelly, 0.15),  # Hard cap
                f"Unanimous BUY ({total}/{total} votes, {avg_conf:.0%} avg confidence) → full Kelly"
            )

        if buy_votes >= 2 and avg_conf >= 0.65:
            # Strong majority BUY
            kelly = proposed_weight * 0.75
            return (
                "BUY",
                min(kelly, 0.15),
                f"Strong majority BUY ({buy_votes}/{total} votes, {avg_conf:.0%} conf) → 75% Kelly"
            )

        if buy_votes >= 2 and avg_conf >= 0.50:
            # Weak majority BUY
            kelly = proposed_weight * 0.50
            return (
                "BUY",
                min(kelly, 0.15),
                f"Weak majority BUY ({buy_votes}/{total} votes, {avg_conf:.0%} conf) → 50% Kelly"
            )

        if buy_votes == 1 and avg_conf >= 0.70:
            # Only one BUY vote but high confidence — small starter
            kelly = proposed_weight * 0.30
            return (
                "BUY",
                min(kelly, 0.05),  # Max 5% for lone BUY vote
                f"Single BUY vote with high confidence ({avg_conf:.0%}) → 30% Kelly, capped at 5%"
            )

        # Default: insufficient consensus
        return (
            "PASS",
            0.0,
            f"Insufficient consensus: {buy_votes} BUY, {sell_votes} SELL "
            f"out of {total} votes ({avg_conf:.0%} avg confidence)"
        )

    # ── Universe scan ─────────────────────────────────────────────────────────

    def scan_universe(
        self,
        tickers:    Optional[List[str]] = None,
        fast_mode:  bool = True,
        top_n:      int = 5,
    ) -> List[ConsensusResult]:
        """
        Scan universe and return top N BUY opportunities.

        Uses fast_mode=True by default to skip full research
        (faster and cheaper for universe scanning).
        Use fast_mode=False for deep analysis of fewer names.

        Args:
            tickers   : Universe to scan (default: settings.DEFAULT_UNIVERSE)
            fast_mode : Skip Research Analyst (2x faster, 2x cheaper)
            top_n     : Return top N results

        Returns:
            List of ConsensusResults sorted by confidence descending
        """
        try:
            from src.config.settings import cfg
            universe = tickers or cfg.DEFAULT_UNIVERSE
        except ImportError:
            universe = tickers or ["AAPL","MSFT","NVDA","GOOGL","JPM","BAC","XOM"]

        # Filter out ETFs and macro instruments
        exclude = {"SPY","QQQ","IWM","TLT","GLD","^VIX","VIX","XLK","XLF"}
        universe = [t for t in universe if t not in exclude]

        logger.info(
            f"Scanning {len(universe)} securities | "
            f"fast_mode={fast_mode}"
        )

        results = []
        for ticker in universe:
            try:
                result = self.decide(ticker, fast_mode=fast_mode)
                results.append(result)
            except Exception as e:
                logger.error(f"Scan failed for {ticker}: {e}")

        # Sort: BUY first, then by buy_votes × avg_confidence
        def score(r: ConsensusResult) -> float:
            if r.final_decision == "BUY":
                return r.buy_votes * r.avg_confidence * r.final_weight
            return -1.0

        results.sort(key=score, reverse=True)

        buy_results = [r for r in results if r.final_decision == "BUY"]
        logger.info(
            f"Scan complete: {len(buy_results)} BUY out of "
            f"{len(results)} analysed"
        )

        return results[:top_n]

    # ── Status and reporting ──────────────────────────────────────────────────

    def get_agent_status(self) -> Dict[str, str]:
        """Return status of each agent."""
        status = {}
        for name, agent in [
            ("PortfolioManager", self._pm_agent),
            ("RiskManager",      self._risk_agent),
            ("ResearchAnalyst",  self._analyst),
        ]:
            if agent is not None:
                status[name] = "RUNNING"
            else:
                status[name] = "NOT_STARTED"
        return status

    def get_decision_history(self, ticker: Optional[str] = None) -> List[Dict]:
        """Get recent decision history, optionally filtered by ticker."""
        if ticker:
            return self.db.get_recent(ticker, days=30)
        return self.db.decision_stats(days=30)

    def print_daily_report(self) -> str:
        """Generate a daily decision summary."""
        stats = self.db.decision_stats(days=1)
        recent = self._decisions[-10:] if self._decisions else []

        lines = [
            "=" * 65,
            f"  DAILY DECISION REPORT — {datetime.now():%Y-%m-%d}",
            "=" * 65,
            f"  Decisions today: {sum(v['count'] for v in stats.values())}",
        ]
        for decision_type, data in stats.items():
            lines.append(
                f"    {decision_type:<10} {data['count']} decisions | "
                f"avg confidence {data['avg_confidence']:.0%} | "
                f"cost ${data['total_cost']:.4f}"
            )

        if recent:
            lines += ["─" * 65, "  RECENT DECISIONS:"]
            for r in recent[-5:]:
                lines.append(
                    f"  {r.ticker:<6} {r.final_decision:<5} "
                    f"{r.final_weight:.1%} | conf={r.avg_confidence:.0%} | "
                    f"{r.timestamp:%H:%M}"
                )

        lines.append("=" * 65)
        return "\n".join(lines)

    def total_spend(self) -> float:
        """Total LLM cost across all decisions."""
        return sum(r.total_llm_cost for r in self._decisions)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, str(Path(__file__).parents[3]))
    logging.basicConfig(level=logging.INFO)

    print("=" * 65)
    print("  Agent Coordinator — Integration Test")
    print("=" * 65)

    from src.data.data_models import Portfolio, Position, Direction

    portfolio = Portfolio("COORD_TEST_001", cash=1_000_000, initial_capital=1_000_000)
    portfolio.positions["MSFT"] = Position(
        "MSFT", Direction.LONG, 200, 380.0, 415.0, sector="Technology"
    )

    coordinator = AgentCoordinator(portfolio)

    if not os.getenv("ANTHROPIC_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        print("\n⚠️  No LLM API key — testing risk and data components only\n")

        # Test risk agent directly
        from src.agents.risk_manager_agent import RiskManagerAgent
        rm = RiskManagerAgent(portfolio)
        check = rm.pre_trade_check("AAPL", 0.10, 0.0)
        print(check.summary())

        print("\n✅ Component tests passed. Add API key for full coordinator test.")
    else:
        print("\nRunning full consensus decision for AAPL...")
        result = coordinator.decide("AAPL", fast_mode=False)
        print(result.summary())

        print(f"\nTotal spend: ${coordinator.total_spend():.4f}")
        print(coordinator.print_daily_report())
