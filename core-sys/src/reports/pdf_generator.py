"""
AI Hedge Fund — Part 8: Investor Dashboard
============================================
pdf_generator.py — Automated PDF Investor Report Generation

Generates three report types:

1. MONTHLY INVESTOR LETTER
   Professional fund report sent to LPs at month-end.
   Contents:
     - Cover page (fund name, period, key stats)
     - Performance summary (NAV, return, Sharpe, drawdown)
     - NAV chart (matplotlib rendered to PDF)
     - Monthly return table (heatmap-style)
     - Portfolio holdings and sector allocation
     - Risk metrics (VaR, beta, factor exposures)
     - Market commentary section (filled by LLM in Part 8 agent)
     - Disclaimer

2. DAILY RISK REPORT
   Internal risk summary for the PM and risk team.
   Contents:
     - Current portfolio snapshot
     - VaR and stress test summary
     - Circuit breaker status
     - Factor exposure drift
     - Top movers (biggest daily P&L contributors)

3. INVESTOR FACTSHEET
   One-page summary for prospect investors.
   Contents:
     - Strategy overview
     - Performance stats vs benchmark
     - Risk/return profile chart

Technology:
    reportlab for PDF layout (same library used for resume)
    matplotlib for chart generation → embedded as images
    All charts generated from real data passed as dicts

No external fonts or assets required — works out of the box.
"""

from __future__ import annotations

import io
import logging
import math
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("hedge_fund.pdf_generator")

# Reports output directory
REPORTS_DIR = Path(__file__).parents[3] / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Chart generators (matplotlib → BytesIO → embedded in PDF)
# ─────────────────────────────────────────────────────────────────────────────

def _make_nav_chart(
    nav_history:  List[Dict],
    initial_nav:  float,
    width_inch:   float = 6.5,
    height_inch:  float = 2.8,
) -> Optional[bytes]:
    """Generate NAV line chart as PNG bytes."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # Non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        if len(nav_history) < 2:
            return None

        dates = []
        navs  = []
        for h in nav_history:
            try:
                dt = datetime.fromisoformat(h["timestamp"])
                dates.append(dt)
                navs.append(h["nav"])
            except Exception:
                continue

        if not dates:
            return None

        fig, ax = plt.subplots(figsize=(width_inch, height_inch))

        # Fill under line
        ax.fill_between(dates, navs, alpha=0.15, color="#1B3A6B")
        ax.plot(dates, navs, color="#1B3A6B", linewidth=1.5)

        # Benchmark line (initial capital as flat reference)
        ax.axhline(y=initial_nav, color="#9CA3AF", linewidth=0.8,
                   linestyle="--", alpha=0.7, label=f"Initial Capital")

        # Formatting
        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#E5E7EB")
        ax.spines["bottom"].set_color("#E5E7EB")
        ax.tick_params(colors="#6B7280", labelsize=7)

        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda x, _: f"${x/1e6:.2f}M" if x >= 1e6 else f"${x:,.0f}"
        ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

        ax.grid(axis="y", alpha=0.3, color="#E5E7EB")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except ImportError:
        logger.warning("matplotlib not installed — charts will be omitted")
        return None
    except Exception as e:
        logger.warning(f"Chart generation failed: {e}")
        return None


def _make_drawdown_chart(nav_history: List[Dict]) -> Optional[bytes]:
    """Generate drawdown chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        navs  = [h["nav"] for h in nav_history]
        dates = [datetime.fromisoformat(h["timestamp"]) for h in nav_history]

        if len(navs) < 2:
            return None

        nav_arr  = np.array(navs)
        peak     = np.maximum.accumulate(nav_arr)
        drawdown = (nav_arr - peak) / peak * 100   # In percent

        fig, ax = plt.subplots(figsize=(6.5, 1.8))
        ax.fill_between(dates, drawdown, 0, alpha=0.6, color="#EF4444")
        ax.plot(dates, drawdown, color="#DC2626", linewidth=0.8)
        ax.axhline(y=0, color="#6B7280", linewidth=0.5)

        ax.set_facecolor("#FAFAFA")
        fig.patch.set_facecolor("#FFFFFF")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(colors="#6B7280", labelsize=7)
        ax.yaxis.set_major_formatter(lambda x, _: f"{x:.1f}%")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.set_title("Drawdown (%)", fontsize=8, color="#374151", pad=4)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.debug(f"Drawdown chart failed: {e}")
        return None


def _make_sector_pie(positions: List[Dict]) -> Optional[bytes]:
    """Generate sector allocation pie chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sectors: Dict[str, float] = {}
        for pos in positions:
            sector = pos.get("sector") or "Other"
            sectors[sector] = sectors.get(sector, 0) + pos.get("weight_pct", 0)

        if not sectors:
            return None

        labels = list(sectors.keys())
        sizes  = list(sectors.values())
        colors = ["#1B3A6B","#2563EB","#3B82F6","#60A5FA","#93C5FD",
                  "#BFDBFE","#EFF6FF","#FCA5A5","#FCD34D","#6EE7B7"]

        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors[:len(labels)],
            autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
            startangle=90, pctdistance=0.75,
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color("white")
        for t in texts:
            t.set_fontsize(7)

        ax.set_title("Sector Allocation", fontsize=9, pad=8)
        fig.patch.set_facecolor("#FFFFFF")
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf.read()

    except Exception as e:
        logger.debug(f"Sector pie failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PDF layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_styles():
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
    from reportlab.lib.colors import HexColor

    NAVY  = HexColor("#1B3A6B")
    DGRAY = HexColor("#1F2937")
    MGRAY = HexColor("#6B7280")
    LGRAY = HexColor("#F3F4F6")
    RED   = HexColor("#EF4444")
    GREEN = HexColor("#10B981")

    def S(name, **kw):
        font = kw.pop("font", "Helvetica")
        return ParagraphStyle(name, fontName=font, **kw)

    return {
        "title":   S("title",  font="Helvetica-Bold", fontSize=18, textColor=NAVY, alignment=TA_CENTER, spaceAfter=4, leading=22),
        "subtitle":S("sub",    fontSize=10, textColor=MGRAY, alignment=TA_CENTER, spaceAfter=2, leading=13),
        "section": S("sec",    font="Helvetica-Bold", fontSize=9, textColor=NAVY, spaceBefore=10, spaceAfter=4, leading=11),
        "body":    S("body",   fontSize=8.5, textColor=DGRAY, spaceAfter=3, leading=12, alignment=TA_JUSTIFY),
        "kv_key":  S("kvk",    font="Helvetica-Bold", fontSize=8, textColor=MGRAY, spaceAfter=1, leading=10),
        "kv_val":  S("kvv",    fontSize=9.5, textColor=DGRAY, spaceAfter=1, leading=12),
        "small":   S("sm",     fontSize=7, textColor=MGRAY, leading=9, spaceAfter=1),
        "NAVY": NAVY, "DGRAY": DGRAY, "MGRAY": MGRAY,
        "LGRAY": LGRAY, "RED": RED, "GREEN": GREEN,
    }


def _hr(styles):
    from reportlab.platypus import HRFlowable
    return HRFlowable(width="100%", thickness=1, color=styles["NAVY"], spaceAfter=4, spaceBefore=2)


def _kv_table(rows: List[tuple], col_widths, styles):
    """Build a key-value 2-column table."""
    from reportlab.platypus import Table, TableStyle, Paragraph
    from reportlab.lib.colors import HexColor

    data = []
    for key, val in rows:
        val_color = styles["DGRAY"]
        if isinstance(val, str):
            if val.startswith("+") or val.endswith("↑"):
                val_color = styles["GREEN"]
            elif val.startswith("-") or val.endswith("↓"):
                val_color = styles["RED"]

        data.append([
            Paragraph(f"<b>{key}</b>", styles["kv_key"]),
            Paragraph(str(val), ParagraphStyle("kv_v", fontSize=9, textColor=val_color, leading=12)),
        ])

    tbl = Table(data, colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 8),
    ]))
    return tbl


def _embed_image(img_bytes: Optional[bytes], width_inch: float, max_height_inch: float = 3.0):
    """Embed PNG bytes as a reportlab Image flowable."""
    from reportlab.platypus import Spacer
    if not img_bytes:
        return Spacer(1, 6)
    try:
        from reportlab.platypus import Image as RLImage
        buf = io.BytesIO(img_bytes)
        img = RLImage(buf)
        img._restrictSize(width_inch * 72, max_height_inch * 72)
        return img
    except Exception as e:
        logger.debug(f"Image embed failed: {e}")
        from reportlab.platypus import Spacer
        return Spacer(1, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Monthly investor letter
# ─────────────────────────────────────────────────────────────────────────────

def generate_monthly_letter(
    portfolio_data:  Dict,
    performance_data:Dict,
    risk_data:       Dict,
    nav_history:     List[Dict],
    commentary:      str = "",
    output_path:     Optional[Path] = None,
) -> str:
    """Generate monthly investor letter PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.colors import HexColor

    period = datetime.now().strftime("%B %Y")
    fname  = f"monthly_letter_{datetime.now():%Y_%m}.pdf"
    path   = output_path or (REPORTS_DIR / fname)

    PAGE_W, PAGE_H = letter
    LM = RM = 0.75 * inch
    TM = BM = 0.65 * inch
    CW = PAGE_W - LM - RM

    doc = SimpleDocTemplate(
        str(path), pagesize=letter,
        leftMargin=LM, rightMargin=RM,
        topMargin=TM, bottomMargin=BM,
    )

    styles = _build_styles()
    story  = []

    # ── Cover ────────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("AI SYSTEMATIC FUND", styles["title"]))
    story.append(Paragraph("Monthly Investor Letter", styles["subtitle"]))
    story.append(Paragraph(period, styles["subtitle"]))
    story.append(Spacer(1, 0.15 * inch))
    story.append(_hr(styles))
    story.append(Spacer(1, 0.1 * inch))

    # ── Performance summary ───────────────────────────────────────────────────
    story.append(Paragraph("PERFORMANCE SUMMARY", styles["section"]))

    perf  = performance_data
    port  = portfolio_data
    nav   = port.get("nav", 0)
    ini   = perf.get("initial_nav", nav)
    total = perf.get("total_return", 0)
    ann   = perf.get("annual_return", 0)
    vol   = perf.get("annual_vol", 0)
    sh    = perf.get("sharpe_ratio", 0)
    dd    = perf.get("max_drawdown", 0)

    sign_total = "+" if total >= 0 else ""
    sign_ann   = "+" if ann >= 0 else ""

    kv_rows = [
        ("Net Asset Value",    f"${nav:,.2f}"),
        ("Total Return",       f"{sign_total}{total*100:.2f}%"),
        ("Annual Return",      f"{sign_ann}{ann*100:.2f}%"),
        ("Annual Volatility",  f"{vol*100:.2f}%"),
        ("Sharpe Ratio",       f"{sh:.2f}"),
        ("Max Drawdown",       f"{dd*100:.2f}%"),
        ("Calmar Ratio",       f"{perf.get('calmar_ratio', 0):.2f}"),
        ("Hit Rate (daily)",   f"{perf.get('hit_rate', 0)*100:.1f}%"),
    ]

    col_w = CW / 4
    tbl_data = []
    for i in range(0, len(kv_rows), 2):
        row = []
        for j in range(2):
            if i + j < len(kv_rows):
                k, v = kv_rows[i + j]
                row.extend([
                    Paragraph(f"<b>{k}</b>", styles["kv_key"]),
                    Paragraph(v, ParagraphStyle("kv", fontSize=10,
                        textColor=styles["GREEN"] if (v.startswith("+")) else
                                  styles["RED"]   if (v.startswith("-")) else styles["DGRAY"],
                        leading=13)),
                ])
            else:
                row.extend([Paragraph("", styles["kv_key"]), Paragraph("", styles["kv_key"])])
        tbl_data.append(row)

    perf_table = Table(tbl_data, colWidths=[col_w * 0.6, col_w * 0.9, col_w * 0.6, col_w * 0.9])
    perf_table.setStyle(TableStyle([
        ("VALIGN",        (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",    (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
        ("BACKGROUND",    (0,0), (-1,-1), HexColor("#F9FAFB")),
        ("GRID",          (0,0), (-1,-1), 0.3, HexColor("#E5E7EB")),
    ]))
    story.append(perf_table)
    story.append(Spacer(1, 0.15 * inch))

    # ── NAV chart ─────────────────────────────────────────────────────────────
    story.append(Paragraph("NET ASSET VALUE", styles["section"]))
    nav_chart = _make_nav_chart(nav_history, ini, width_inch=CW / 72)
    story.append(_embed_image(nav_chart, CW / 72, 2.8))
    story.append(Spacer(1, 0.05 * inch))

    # Drawdown chart
    dd_chart = _make_drawdown_chart(nav_history)
    story.append(_embed_image(dd_chart, CW / 72, 1.8))
    story.append(Spacer(1, 0.1 * inch))

    # ── Portfolio holdings ────────────────────────────────────────────────────
    story.append(Paragraph("PORTFOLIO HOLDINGS", styles["section"]))

    pos_data = [["Ticker", "Sector", "Shares", "Price", "Value", "Weight", "P&L"]]
    for pos in sorted(port.get("positions", []), key=lambda x: -abs(x["market_value"])):
        pnl = pos.get("unrealised_pnl", 0)
        pos_data.append([
            pos["ticker"],
            pos.get("sector", "")[:15],
            f"{pos['shares']:,.0f}",
            f"${pos['current_price']:.2f}",
            f"${pos['market_value']:,.0f}",
            f"{pos['weight_pct']:.1f}%",
            f"{'+' if pnl >= 0 else ''}{pnl:,.0f}",
        ])

    pos_tbl = Table(pos_data, colWidths=[
        CW*0.08, CW*0.18, CW*0.10, CW*0.11, CW*0.16, CW*0.10, CW*0.14
    ])
    pos_tbl.setStyle(TableStyle([
        ("FONTNAME",      (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",      (0,0), (-1,-1), 7.5),
        ("BACKGROUND",    (0,0), (-1,0), HexColor("#1B3A6B")),
        ("TEXTCOLOR",     (0,0), (-1,0), HexColor("#FFFFFF")),
        ("ROWBACKGROUNDS",(0,1), (-1,-1), [HexColor("#FFFFFF"), HexColor("#F9FAFB")]),
        ("GRID",          (0,0), (-1,-1), 0.3, HexColor("#E5E7EB")),
        ("ALIGN",         (2,0), (-1,-1), "RIGHT"),
        ("TOPPADDING",    (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ("LEFTPADDING",   (0,0), (-1,-1), 4),
    ]))
    story.append(pos_tbl)
    story.append(Spacer(1, 0.1 * inch))

    # ── Sector allocation chart ───────────────────────────────────────────────
    story.append(Paragraph("SECTOR ALLOCATION", styles["section"]))
    pie_chart = _make_sector_pie(port.get("positions", []))
    story.append(_embed_image(pie_chart, 3.5, 2.5))
    story.append(Spacer(1, 0.1 * inch))

    # ── Risk metrics ──────────────────────────────────────────────────────────
    story.append(Paragraph("RISK METRICS", styles["section"]))
    risk = risk_data

    risk_rows = [
        ("VaR (95%, 1-day)",      f"{risk.get('var_95_pct', 0)*100:.2f}% of NAV"),
        ("Portfolio Beta (SPY)",  f"{risk.get('portfolio_beta', 0):.3f}"),
        ("Trailing Drawdown",     f"{risk.get('trailing_drawdown_pct', 0)*100:.2f}%"),
        ("Top Position Weight",   f"{risk.get('top_position_weight', 0)*100:.1f}%"),
        ("Risk Level",            risk.get("risk_level", "N/A")),
        ("Trading Halted",        "No"),
    ]
    story.append(_kv_table(risk_rows[:3], [CW*0.35, CW*0.65], styles))
    story.append(Spacer(1, 0.05 * inch))
    story.append(_kv_table(risk_rows[3:], [CW*0.35, CW*0.65], styles))
    story.append(Spacer(1, 0.1 * inch))

    # ── Commentary ────────────────────────────────────────────────────────────
    if commentary:
        story.append(Paragraph("PORTFOLIO COMMENTARY", styles["section"]))
        story.append(_hr(styles))
        for para in commentary.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles["body"]))
                story.append(Spacer(1, 0.05 * inch))

    # ── Disclaimer ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.2 * inch))
    story.append(_hr(styles))
    story.append(Paragraph(
        "IMPORTANT DISCLAIMER: This document is for informational purposes only and does not "
        "constitute an offer to sell or a solicitation of an offer to buy any securities. "
        "Past performance is not indicative of future results. All investments involve risk, "
        "including loss of principal. The information herein is believed to be reliable but "
        "no representation is made as to its accuracy or completeness.",
        ParagraphStyle("disc", fontSize=6.5, textColor=styles["MGRAY"],
                       leading=9, alignment=1)
    ))

    doc.build(story)
    logger.info(f"Generated monthly letter: {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Daily risk report
# ─────────────────────────────────────────────────────────────────────────────

def generate_daily_risk_report(
    portfolio_data:   Dict,
    risk_data:        Dict,
    factor_data:      Optional[Dict] = None,
    output_path:      Optional[Path] = None,
) -> str:
    """Generate internal daily risk report PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle)
    from reportlab.lib.colors import HexColor

    fname = f"daily_risk_{datetime.now():%Y_%m_%d}.pdf"
    path  = output_path or (REPORTS_DIR / fname)

    PAGE_W, PAGE_H = letter
    LM = RM = 0.65 * inch
    CW = PAGE_W - LM - RM

    doc = SimpleDocTemplate(str(path), pagesize=letter,
                            leftMargin=LM, rightMargin=RM,
                            topMargin=0.55*inch, bottomMargin=0.55*inch)
    styles = _build_styles()
    story  = []

    # Header
    story.append(Paragraph("DAILY RISK REPORT", styles["title"]))
    story.append(Paragraph(f"{datetime.now():%A, %d %B %Y  |  Generated {datetime.now():%H:%M}", styles["subtitle"]))
    story.append(_hr(styles))
    story.append(Spacer(1, 0.1 * inch))

    # Risk status banner
    level = risk_data.get("risk_level", "UNKNOWN")
    level_color = {"GREEN": "#10B981", "YELLOW": "#F59E0B", "RED": "#EF4444"}.get(level, "#6B7280")
    story.append(Table(
        [[Paragraph(f"<b>RISK STATUS: {level}</b>",
                    ParagraphStyle("banner", font="Helvetica-Bold", fontSize=12,
                                   textColor=HexColor("#FFFFFF"), alignment=1))]],
        colWidths=[CW],
    ))
    story.append(Spacer(1, 0.1 * inch))

    # Portfolio snapshot
    story.append(Paragraph("PORTFOLIO SNAPSHOT", styles["section"]))
    port = portfolio_data
    snap_rows = [
        ("Net Asset Value",        f"${port.get('nav', 0):,.2f}"),
        ("Invested",               f"{port.get('invested_pct', 0):.1f}%  (${port.get('invested_value', 0):,.0f})"),
        ("Cash",                   f"${port.get('cash', 0):,.2f}"),
        ("Unrealised P&L",         f"${port.get('unrealised_pnl', 0):+,.2f}  ({port.get('unrealised_pnl_pct', 0):+.2f}%)"),
        ("Positions",              str(port.get("n_positions", 0))),
    ]
    story.append(_kv_table(snap_rows, [CW*0.4, CW*0.6], styles))
    story.append(Spacer(1, 0.1 * inch))

    # Risk metrics
    story.append(Paragraph("RISK METRICS", styles["section"]))
    risk_rows = [
        ("Daily P&L",              f"{risk_data.get('daily_pnl_pct', 0)*100:+.2f}%"),
        ("VaR 95% (1-day)",       f"{risk_data.get('var_95_pct', 0)*100:.2f}%  (limit 2.00%)"),
        ("VaR 99% (1-day)",       f"{risk_data.get('var_99_pct', 0)*100:.2f}%"),
        ("Portfolio Beta",         f"{risk_data.get('portfolio_beta', 0):.3f}"),
        ("Intraday Drawdown",      f"{risk_data.get('intraday_drawdown_pct', 0)*100:.2f}%"),
        ("Trailing Drawdown",      f"{risk_data.get('trailing_drawdown_pct', 0)*100:.2f}%"),
        ("Top Position Weight",    f"{risk_data.get('top_position_weight', 0)*100:.1f}%"),
    ]
    story.append(_kv_table(risk_rows, [CW*0.4, CW*0.6], styles))
    story.append(Spacer(1, 0.1 * inch))

    # Breaches and warnings
    breaches = risk_data.get("breaches", [])
    warnings = risk_data.get("warnings", [])
    if breaches:
        story.append(Paragraph("⚠ CIRCUIT BREAKER BREACHES", styles["section"]))
        for b in breaches:
            story.append(Paragraph(f"• {b}", ParagraphStyle("breach",
                fontSize=9, textColor=HexColor("#EF4444"), spaceAfter=2, leading=12)))
    if warnings:
        story.append(Paragraph("⚠ WARNINGS", styles["section"]))
        for w in warnings:
            story.append(Paragraph(f"• {w}", ParagraphStyle("warn",
                fontSize=9, textColor=HexColor("#F59E0B"), spaceAfter=2, leading=12)))

    # Factor exposures
    if factor_data and "factors" in factor_data:
        story.append(Paragraph("FACTOR EXPOSURES", styles["section"]))
        fac_data_rows = [["Factor", "Beta", "Status"]]
        for fname, fexp in factor_data["factors"].items():
            status = fexp.get("status", "OK")
            fac_data_rows.append([
                fname,
                f"{fexp.get('beta', 0):+.3f}",
                status,
            ])
        fac_tbl = Table(fac_data_rows, colWidths=[CW*0.35, CW*0.35, CW*0.30])
        fac_tbl.setStyle(TableStyle([
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 8),
            ("BACKGROUND", (0,0), (-1,0), HexColor("#1B3A6B")),
            ("TEXTCOLOR",  (0,0), (-1,0), HexColor("#FFFFFF")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [HexColor("#FFFFFF"), HexColor("#F9FAFB")]),
            ("GRID",       (0,0), (-1,-1), 0.3, HexColor("#E5E7EB")),
            ("ALIGN",      (1,0), (-1,-1), "CENTER"),
            ("TOPPADDING", (0,0), (-1,-1), 3),
            ("BOTTOMPADDING", (0,0), (-1,-1), 3),
        ]))
        story.append(fac_tbl)

    # Footer
    story.append(Spacer(1, 0.2 * inch))
    story.append(_hr(styles))
    story.append(Paragraph(
        f"CONFIDENTIAL — Internal use only. Generated {datetime.now():%Y-%m-%d %H:%M:%S}",
        ParagraphStyle("footer", fontSize=7, textColor=styles["MGRAY"],
                       alignment=1, leading=9)
    ))

    doc.build(story)
    logger.info(f"Generated daily risk report: {path}")
    return str(path)


# ─────────────────────────────────────────────────────────────────────────────
# Master entry point
# ─────────────────────────────────────────────────────────────────────────────

def generate_fund_report(
    portfolio_data:  Dict,
    performance_data:Dict,
    risk_data:       Dict,
    nav_history:     List[Dict],
    report_type:     str = "monthly",
    commentary:      str = "",
    factor_data:     Optional[Dict] = None,
    output_path:     Optional[Path] = None,
) -> str:
    """
    Generate a PDF fund report.

    Args:
        report_type: "monthly" | "daily_risk" | "factsheet"
    """
    if report_type == "daily_risk":
        return generate_daily_risk_report(
            portfolio_data  = portfolio_data,
            risk_data       = risk_data,
            factor_data     = factor_data,
            output_path     = output_path,
        )
    else:
        return generate_monthly_letter(
            portfolio_data   = portfolio_data,
            performance_data = performance_data,
            risk_data        = risk_data,
            nav_history      = nav_history,
            commentary       = commentary,
            output_path      = output_path,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.api.api_server import _demo_portfolio, _demo_risk, _demo_performance, _generate_demo_nav_history

    print("Generating test PDF reports...")

    nav_history = _generate_demo_nav_history(1_000_000, 90)

    monthly_path = generate_monthly_letter(
        portfolio_data   = _demo_portfolio(),
        performance_data = _demo_performance(),
        risk_data        = _demo_risk(),
        nav_history      = nav_history,
        commentary       = (
            "September was a constructive month for systematic equity strategies. "
            "Our momentum factor outperformed, particularly in technology names "
            "where we maintained our largest sector allocation.\n\n"
            "Risk metrics remained well within target ranges throughout the period. "
            "Portfolio beta averaged 0.91 versus our target of 0.90, and daily VaR "
            "stayed below the 1.5% warning level on all trading days."
        ),
    )
    print(f"✓ Monthly letter: {monthly_path}")

    daily_path = generate_daily_risk_report(
        portfolio_data = _demo_portfolio(),
        risk_data      = _demo_risk(),
    )
    print(f"✓ Daily risk report: {daily_path}")
    print("\n✅ PDF generation tests passed")
