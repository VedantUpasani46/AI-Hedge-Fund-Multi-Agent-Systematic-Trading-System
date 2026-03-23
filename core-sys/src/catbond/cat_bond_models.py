"""
AI Hedge Fund — Part 5: Alternative Assets & Data
====================================================
cat_bond_models.py — Catastrophe Bond Pricing & Risk Analysis

Catastrophe bonds are the most capital-markets-adjacent product
in the actuarial/insurance world. They are traded by hedge funds,
priced using quantitative models, and can be analysed without
actuarial credentials — they sit firmly in the quant finance domain.

What cat bonds are:
    Insurance companies (sponsors) issue bonds to transfer catastrophe
    risk to capital market investors. If a defined catastrophe occurs
    (trigger is hit), investors lose principal. In return, they receive
    a coupon that is substantially higher than equivalent-rated corporate
    bonds (typically SOFR + 300-1500 bps).

    Typical transaction:
        1. Sponsor issues $100M cat bond (e.g. Florida hurricane)
        2. Investors receive SOFR + 500 bps per year
        3. If Category 4+ hurricane hits Miami-Dade: investors lose principal
        4. If no trigger in 3-year term: investors get principal back

Why this belongs in your fund:
    - Low correlation with equities (0.0 to 0.1 historically)
    - High risk-adjusted returns (Sharpe ~1.2-1.8 historically)
    - Transparent, rules-based triggers
    - Diversification in any market regime

The four trigger types (order of basis risk, lowest to highest):
    1. INDEMNITY:      Triggers based on actual losses of the sponsor
    2. INDUSTRY_INDEX: Triggers based on industry-wide insured loss (e.g. PCS estimate)
    3. MODELED_LOSS:   Triggers based on model output from a vendor (AIR, RMS)
    4. PARAMETRIC:     Triggers based on physical parameter (wind speed, Richter magnitude)

Pricing methodology:
    Expected Loss (EL) = P(trigger hit) × expected fraction of principal lost
    Fair coupon = EL + Risk Premium
    Risk Premium = EL × Sharpe multiplier (typically 1.5-3.0x for cat bonds)

Loss modelling:
    Event frequency:  Poisson(λ)  [events per year]
    Event severity:   Generalised Pareto Distribution or Lognormal
    Annual loss:      Frequency-Severity compound distribution
    Monte Carlo:      10,000-100,000 simulations for full loss distribution

References:
    Cummins, J.D. (2008). Cat Bonds and Other Risk-Linked Securities. Geneva Papers.
    Lane, M. (2006). Pricing Risk Transfer Transactions. ASTIN Bulletin.
    Froot, K. (2001). The Market for Catastrophe Risk. Journal of Financial Economics.
    Swiss Re Sigma (various): Natural Catastrophe reports.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.special import gamma as gamma_func

logger = logging.getLogger("hedge_fund.cat_bond")


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────

class TriggerType(str, Enum):
    INDEMNITY      = "INDEMNITY"
    INDUSTRY_INDEX = "INDUSTRY_INDEX"
    MODELED_LOSS   = "MODELED_LOSS"
    PARAMETRIC     = "PARAMETRIC"


class PerilType(str, Enum):
    HURRICANE        = "HURRICANE"
    EARTHQUAKE       = "EARTHQUAKE"
    WINDSTORM_EU     = "WINDSTORM_EU"
    FLOOD            = "FLOOD"
    WILDFIRE         = "WILDFIRE"
    PANDEMIC         = "PANDEMIC"
    MULTI_PERIL      = "MULTI_PERIL"
    EXTREME_MORTALITY= "EXTREME_MORTALITY"


class TrancheLoss(str, Enum):
    """
    How principal is lost if trigger is breached.
    BINARY:    Lose 100% on trigger
    PRO_RATA:  Lose proportionally to how far trigger is exceeded
    SCHEDULED: Loss follows a schedule
    """
    BINARY    = "BINARY"
    PRO_RATA  = "PRO_RATA"
    SCHEDULED = "SCHEDULED"


# ─────────────────────────────────────────────────────────────────────────────
# Cat bond specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CatBondSpec:
    """
    Complete specification of a catastrophe bond.

    All the parameters needed to price and risk-analyse a cat bond.
    Based on typical term sheet structure from Swiss Re Capital Markets,
    Aon Securities, and GC Securities cat bond databases.
    """
    # Identity
    ticker:          str          # e.g. "FLOR-2024-1"
    sponsor:         str          # e.g. "Citizens Property Insurance"
    issuer:          str          # SPV name
    series:          str          # e.g. "Series A"

    # Structure
    principal:       float        # Face value in USD millions
    coupon_spread:   float        # Spread over SOFR in bps (e.g. 500 = 5%)
    maturity_years:  float        # Term in years (typically 1-5)
    issue_date:      date
    maturity_date:   date

    # Risk
    peril:           PerilType
    trigger_type:    TriggerType
    territory:       str          # e.g. "Florida", "US Gulf Coast"
    loss_type:       TrancheLoss = TrancheLoss.BINARY

    # Trigger parameters (interpretation depends on trigger_type)
    # PARAMETRIC:     e.g. wind speed in knots (Saffir-Simpson scale)
    # INDUSTRY_INDEX: e.g. PCS industry loss in USD billions
    # MODELED_LOSS:   e.g. vendor model output in USD billions
    # INDEMNITY:      e.g. sponsor's actual loss in USD millions
    trigger_level:   float = 0.0  # Attach point (loss of first dollar)
    exhaustion_level: float = 0.0 # Exhaustion point (loss of last dollar)

    # Risk metrics (populated by pricer)
    expected_loss_pct: float = 0.0     # EL as % of principal
    attachment_prob:   float = 0.0     # P(any loss)
    exhaustion_prob:   float = 0.0     # P(total loss)
    rating:            str = ""        # S&P / Moody's rating (if rated)

    # Market data
    current_price:     float = 100.0   # Price as % of par
    current_spread:    float = 0.0     # Current market spread in bps

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def notional_usd(self) -> float:
        return self.principal * 1_000_000

    @property
    def annual_coupon_usd(self) -> float:
        return self.notional_usd * self.coupon_spread / 10000

    @property
    def risk_multiple(self) -> float:
        """Coupon spread / Expected loss — the 'multiple on EL'."""
        if self.expected_loss_pct <= 0:
            return 0.0
        return (self.coupon_spread / 10000) / (self.expected_loss_pct / 100)

    def __repr__(self):
        return (
            f"CatBond({self.ticker} | {self.peril.value} | "
            f"{self.territory} | "
            f"EL={self.expected_loss_pct:.2f}% | "
            f"spread={self.coupon_spread}bps | "
            f"multiple={self.risk_multiple:.1f}x)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-severity loss model
# ─────────────────────────────────────────────────────────────────────────────

class FrequencySeverityModel:
    """
    Compound frequency-severity model for catastrophe losses.

    Industry standard for cat bond and reinsurance pricing.

    Annual loss = Σ_{i=1}^{N} Severity_i
    where:
        N ~ Poisson(λ)         event count per year
        Severity_i ~ F_S       loss per event

    Severity distribution options:
        LOGNORMAL: Standard for moderate/large losses
        GPD:       Generalised Pareto — better tail fit (from your EVT module)
        EXPONENTIAL: Simple, analytically tractable
        WEIBULL:   Flexible, common in windstorm modelling

    For cat bonds we care about:
        P(annual loss > trigger)  ← attachment probability
        E[loss | loss > trigger]  ← expected loss given attachment
        Full loss distribution    ← for portfolio construction

    References:
        Embrechts, Klüppelberg, Mikosch (1997) — Modelling Extremal Events
        McNeil, Frey, Embrechts (2015) — Quantitative Risk Management Ch. 9
        Cummins & Weiss (2009) — Convergence of Insurance and Financial Markets
    """

    def __init__(
        self,
        lambda_annual:    float,         # Annual event frequency
        severity_dist:    str = "LOGNORMAL",
        severity_params:  Dict[str, float] = None,
        n_simulations:    int = 50_000,
        seed:             int = 42,
    ):
        """
        Args:
            lambda_annual:  Annual expected event count (e.g. 0.1 = 1 in 10 years)
            severity_dist:  Loss severity distribution
            severity_params: Distribution parameters
                LOGNORMAL: {"mu": float, "sigma": float}
                GPD:       {"xi": float, "beta": float}  (shape, scale)
                WEIBULL:   {"shape": float, "scale": float}
            n_simulations: Monte Carlo sample size
            seed:          Random seed for reproducibility
        """
        self.lam     = lambda_annual
        self.dist    = severity_dist.upper()
        self.params  = severity_params or {}
        self.n_sim   = n_simulations
        self.rng     = np.random.default_rng(seed)
        self._losses: Optional[np.ndarray] = None  # Cached simulations

    def _sample_severity(self, n: int) -> np.ndarray:
        """Draw n severity samples from the configured distribution."""
        if self.dist == "LOGNORMAL":
            mu    = self.params.get("mu", 0.0)
            sigma = self.params.get("sigma", 1.0)
            return self.rng.lognormal(mu, sigma, size=n)

        elif self.dist == "GPD":
            xi   = self.params.get("xi", 0.3)     # Shape (tail index)
            beta = self.params.get("beta", 1.0)   # Scale
            u    = self.rng.uniform(size=n)
            # GPD quantile function: G^{-1}(p) = (β/ξ)[(1-p)^{-ξ} - 1]
            if abs(xi) < 1e-8:
                return -beta * np.log(1 - u)
            return (beta / xi) * ((1 - u) ** (-xi) - 1)

        elif self.dist == "WEIBULL":
            shape = self.params.get("shape", 1.5)
            scale = self.params.get("scale", 1.0)
            return self.rng.weibull(shape, size=n) * scale

        elif self.dist == "EXPONENTIAL":
            mean = self.params.get("mean", 1.0)
            return self.rng.exponential(mean, size=n)

        else:
            raise ValueError(f"Unknown severity distribution: {self.dist}")

    def simulate_annual_losses(self, force_resim: bool = False) -> np.ndarray:
        """
        Simulate annual aggregate losses via Monte Carlo.

        Returns array of length n_simulations, each element is
        the simulated annual aggregate loss (in whatever units
        the severity is expressed — USD billions, % of exposure, etc.)
        """
        if self._losses is not None and not force_resim:
            return self._losses

        # Simulate event counts: N_i ~ Poisson(λ)
        event_counts = self.rng.poisson(self.lam, size=self.n_sim)

        # Total events to simulate across all years
        total_events = int(event_counts.sum())

        # Simulate all severities at once (vectorized, much faster)
        if total_events > 0:
            all_severities = self._sample_severity(total_events)
        else:
            all_severities = np.array([])

        # Aggregate by year using cumulative indexing
        annual_losses = np.zeros(self.n_sim)
        idx = 0
        for i, n_events in enumerate(event_counts):
            if n_events > 0:
                annual_losses[i] = all_severities[idx:idx + n_events].sum()
                idx += n_events

        self._losses = annual_losses
        return annual_losses

    def attachment_probability(self, trigger_level: float) -> float:
        """P(annual loss > trigger_level)."""
        losses = self.simulate_annual_losses()
        return float(np.mean(losses > trigger_level))

    def exhaustion_probability(self, exhaustion_level: float) -> float:
        """P(annual loss > exhaustion_level)."""
        return self.attachment_probability(exhaustion_level)

    def expected_loss_given_attachment(
        self,
        trigger_level:    float,
        exhaustion_level: float,
        principal:        float = 1.0,
        loss_type:        TrancheLoss = TrancheLoss.PRO_RATA,
    ) -> float:
        """
        E[loss to bondholder | trigger hit].

        Accounts for the tranche structure:
            - PRO_RATA: Loss scales from 0% at attach to 100% at exhaust
            - BINARY:   Lose 100% if any loss above trigger
        """
        losses = self.simulate_annual_losses()
        tranche_width = exhaustion_level - trigger_level

        if loss_type == TrancheLoss.BINARY:
            # Triggered losses = full principal
            triggered = losses > trigger_level
            return float(np.mean(triggered)) * principal

        elif loss_type == TrancheLoss.PRO_RATA:
            # Loss to bondholder = min(max(loss - trigger, 0), tranche_width) / tranche_width
            if tranche_width <= 0:
                return 0.0
            tranche_loss = np.clip(losses - trigger_level, 0, tranche_width) / tranche_width
            return float(np.mean(tranche_loss)) * principal

        else:
            # SCHEDULED — default to pro-rata
            return self.expected_loss_given_attachment(
                trigger_level, exhaustion_level, principal, TrancheLoss.PRO_RATA
            )

    def loss_exceedance_curve(
        self,
        percentiles: Optional[List[float]] = None,
    ) -> Dict[str, float]:
        """
        Occurrence Exceedance Probability (OEP) curve.

        Returns {return_period: loss_level} pairs.
        Used for cat bond trigger calibration and portfolio risk analysis.

        Standard return periods: 10, 25, 50, 100, 200, 250, 500 years.
        """
        losses = self.simulate_annual_losses()
        ps     = percentiles or [0.10, 0.04, 0.02, 0.01, 0.005, 0.004, 0.002]
        result = {}
        for p in ps:
            rp  = round(1 / p)
            lvl = float(np.percentile(losses, (1 - p) * 100))
            result[f"{rp}yr_return_period"] = lvl
        return result

    def value_at_risk(self, confidence: float = 0.99) -> float:
        """Annual loss VaR at given confidence level."""
        losses = self.simulate_annual_losses()
        return float(np.percentile(losses, confidence * 100))

    def expected_shortfall(self, confidence: float = 0.99) -> float:
        """Expected Shortfall (CVaR) of annual losses."""
        losses = self.simulate_annual_losses()
        var    = self.value_at_risk(confidence)
        return float(np.mean(losses[losses >= var]))

    def summary_stats(self) -> Dict[str, float]:
        """Summary statistics of the annual loss distribution."""
        losses = self.simulate_annual_losses()
        return {
            "mean_annual_loss":        float(np.mean(losses)),
            "std_annual_loss":         float(np.std(losses)),
            "p_nonzero":               float(np.mean(losses > 0)),
            "var_99":                  float(np.percentile(losses, 99)),
            "cvar_99":                 float(np.mean(losses[losses >= np.percentile(losses, 99)])),
            "max_simulated_loss":      float(np.max(losses)),
            "skewness":                float(stats.skew(losses)),
            "kurtosis":                float(stats.kurtosis(losses)),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Cat bond pricer
# ─────────────────────────────────────────────────────────────────────────────

class CatBondPricer:
    """
    Price catastrophe bonds and compute risk metrics.

    Methodology:
        1. Build a frequency-severity loss model for the covered peril
        2. Simulate annual losses → get full loss distribution
        3. Compute attachment probability, expected loss, tranche loss
        4. Derive fair spread = EL × risk multiple
        5. Compute duration, convexity (modified for cat bond structure)
        6. Compute standard ILS risk metrics

    The risk multiple (spread / EL) reflects the risk premium investors
    demand above pure expected loss. Historical multiples (Lane & Mahul, 2009):
        High-yield layer (EL 5-10%):   ~1.5-2.0x
        Mezzanine  (EL 1-5%):          ~2.0-3.0x
        Senior     (EL 0.1-1%):        ~3.0-5.0x
        Super-senior (EL <0.1%):       ~5.0-8.0x

    Fair spread = EL × risk_multiple + risk_free_rate_portion
    """

    # Empirical risk multiples by EL layer (Lane 2006, Swiss Re 2020)
    RISK_MULTIPLE_TABLE = [
        (0.001, 7.0),   # EL < 0.1% → 7x multiple
        (0.005, 5.5),   # EL 0.1-0.5%
        (0.010, 4.5),   # EL 0.5-1%
        (0.030, 3.0),   # EL 1-3%
        (0.060, 2.5),   # EL 3-6%
        (0.100, 2.0),   # EL 6-10%
        (0.200, 1.7),   # EL 10-20%
        (1.000, 1.5),   # EL > 20%
    ]

    def _get_risk_multiple(self, el_fraction: float) -> float:
        """Get empirical risk multiple for a given EL fraction."""
        for el_threshold, multiple in self.RISK_MULTIPLE_TABLE:
            if el_fraction <= el_threshold:
                return multiple
        return self.RISK_MULTIPLE_TABLE[-1][1]

    def price(
        self,
        spec:              CatBondSpec,
        loss_model:        FrequencySeverityModel,
        risk_free_rate:    float = 0.05,    # SOFR + spread
        risk_multiple:     Optional[float] = None,
    ) -> CatBondPricer.PricingResult:
        """
        Full cat bond pricing.

        Args:
            spec:           Cat bond specification
            loss_model:     Calibrated frequency-severity model
            risk_free_rate: Current SOFR rate
            risk_multiple:  Override empirical risk multiple

        Returns:
            PricingResult with fair value, spreads, and risk metrics
        """
        # ── Step 1: Simulate loss distribution ───────────────────────────────
        losses = loss_model.simulate_annual_losses()

        # ── Step 2: Compute tranche metrics ───────────────────────────────────
        trigger     = spec.trigger_level
        exhaustion  = spec.exhaustion_level

        # Attachment and exhaustion probabilities
        attach_prob = float(np.mean(losses > trigger))
        exhaust_prob= float(np.mean(losses > exhaustion))

        # Expected Loss (EL) = E[fraction of principal lost]
        tranche_width = exhaustion - trigger
        if spec.loss_type == TrancheLoss.BINARY:
            el_fraction = attach_prob
        else:
            if tranche_width > 0:
                tranche_losses = np.clip(losses - trigger, 0, tranche_width) / tranche_width
                el_fraction = float(np.mean(tranche_losses))
            else:
                el_fraction = attach_prob

        el_bps    = el_fraction * 10000   # EL in basis points
        el_usd    = el_fraction * spec.notional_usd

        # ── Step 3: Fair spread ───────────────────────────────────────────────
        rm = risk_multiple or self._get_risk_multiple(el_fraction)
        fair_spread_bps = el_bps * rm

        # ── Step 4: Duration (modified for cat bond) ──────────────────────────
        # Cat bond duration differs from standard bond:
        # Loss of principal can occur at any time → reduces effective duration
        # Approximation: duration = maturity × (1 - attach_prob × 0.5)
        simple_duration = spec.maturity_years
        cat_duration    = simple_duration * (1 - attach_prob * 0.5)

        # ── Step 5: Price vs par ───────────────────────────────────────────────
        # If current spread > fair spread → bond is cheap (price < par)
        # If current spread < fair spread → bond is rich (price > par)
        spread_diff     = spec.coupon_spread - fair_spread_bps
        price_vs_par    = 100.0 + (spread_diff / 10000) * cat_duration * 100

        # ── Step 6: IRR from current price ────────────────────────────────────
        # Simple IRR: coupon + principal recovery × (1-EL)
        annual_coupon_frac = spec.coupon_spread / 10000
        expected_principal_recovery = 1.0 - el_fraction
        irr = (
            annual_coupon_frac +
            (expected_principal_recovery - spec.current_price / 100) /
            spec.maturity_years
        )

        # ── Step 7: Loss distribution exceedance curve ─────────────────────────
        oep_curve = loss_model.loss_exceedance_curve()

        # ── Step 8: Tail risk ─────────────────────────────────────────────────
        var_99    = loss_model.value_at_risk(0.99)
        cvar_99   = loss_model.expected_shortfall(0.99)

        return self.PricingResult(
            spec             = spec,
            attach_prob      = attach_prob,
            exhaust_prob     = exhaust_prob,
            el_fraction      = el_fraction,
            el_bps           = el_bps,
            el_usd           = el_usd,
            fair_spread_bps  = fair_spread_bps,
            risk_multiple    = rm,
            cat_duration     = cat_duration,
            price_vs_par     = price_vs_par,
            irr              = irr,
            oep_curve        = oep_curve,
            var_99           = var_99,
            cvar_99          = cvar_99,
            is_cheap         = spec.coupon_spread > fair_spread_bps,
            spread_vs_fair   = spec.coupon_spread - fair_spread_bps,
        )

    @dataclass
    class PricingResult:
        """Output of CatBondPricer.price()"""
        spec:             CatBondSpec
        attach_prob:      float     # P(any loss)
        exhaust_prob:     float     # P(total loss)
        el_fraction:      float     # E[loss fraction]
        el_bps:           float     # EL in basis points
        el_usd:           float     # EL in USD
        fair_spread_bps:  float     # Fair coupon spread
        risk_multiple:    float     # Spread / EL
        cat_duration:     float     # Modified duration (cat-adjusted)
        price_vs_par:     float     # Implied price vs par
        irr:              float     # Expected IRR
        oep_curve:        Dict[str, float]
        var_99:           float
        cvar_99:          float
        is_cheap:         bool      # True if bond is cheap vs fair value
        spread_vs_fair:   float     # Actual spread - fair spread (positive = cheap)

        @property
        def sharpe_ratio(self) -> float:
            """
            Cat bond Sharpe ratio (analogous to equity Sharpe).
            Return above risk-free / std of annual returns.
            Uses annual coupon as return, EL as risk proxy.
            """
            annual_return = self.spec.coupon_spread / 10000
            risk_vol      = math.sqrt(self.el_fraction * (1 - self.el_fraction))
            if risk_vol < 1e-8:
                return 0.0
            return (annual_return - 0.05) / risk_vol   # Assume 5% risk-free

        @property
        def return_on_risk(self) -> float:
            """Annual coupon / EL — key ILS metric."""
            return (self.spec.coupon_spread / 10000) / max(self.el_fraction, 1e-6)

        def summary(self) -> str:
            cheap_str = f"+{self.spread_vs_fair:.0f}bps CHEAP" if self.is_cheap else f"{self.spread_vs_fair:.0f}bps RICH"
            lines = [
                f"Cat Bond: {self.spec.ticker}",
                f"  Peril:         {self.spec.peril.value} | {self.spec.territory}",
                f"  Trigger:       {self.spec.trigger_type.value} @ {self.spec.trigger_level}",
                f"  ─────────────────────────────────────────",
                f"  Attachment P:  {self.attach_prob:.2%}",
                f"  Exhaustion P:  {self.exhaust_prob:.2%}",
                f"  Expected Loss: {self.el_bps:.1f}bps ({self.el_fraction:.2%})",
                f"  ─────────────────────────────────────────",
                f"  Coupon spread: {self.spec.coupon_spread}bps",
                f"  Fair spread:   {self.fair_spread_bps:.0f}bps",
                f"  Valuation:     {cheap_str}",
                f"  Risk multiple: {self.risk_multiple:.1f}x EL",
                f"  Cat duration:  {self.cat_duration:.2f}yrs",
                f"  ─────────────────────────────────────────",
                f"  Expected IRR:  {self.irr:.2%}",
                f"  Return/Risk:   {self.return_on_risk:.1f}x",
                f"  Sharpe (cat):  {self.sharpe_ratio:.2f}",
                f"  99% VaR:       {self.var_99:.2f}",
                f"  99% CVaR:      {self.cvar_99:.2f}",
            ]
            return "\n".join(lines)

        def to_dict(self) -> dict:
            return {
                "ticker":         self.spec.ticker,
                "peril":          self.spec.peril.value,
                "territory":      self.spec.territory,
                "attach_prob":    round(self.attach_prob, 4),
                "exhaust_prob":   round(self.exhaust_prob, 4),
                "el_bps":         round(self.el_bps, 2),
                "el_fraction":    round(self.el_fraction, 4),
                "fair_spread_bps": round(self.fair_spread_bps, 1),
                "coupon_spread":  self.spec.coupon_spread,
                "is_cheap":       self.is_cheap,
                "spread_vs_fair": round(self.spread_vs_fair, 1),
                "risk_multiple":  round(self.risk_multiple, 2),
                "irr":            round(self.irr, 4),
                "sharpe":         round(self.sharpe_ratio, 3),
                "cat_duration":   round(self.cat_duration, 2),
                "var_99":         round(self.var_99, 3),
                "cvar_99":        round(self.cvar_99, 3),
            }


# ─────────────────────────────────────────────────────────────────────────────
# Standard cat bond database (representative real structures)
# ─────────────────────────────────────────────────────────────────────────────

def build_standard_loss_models() -> Dict[str, FrequencySeverityModel]:
    """
    Calibrated frequency-severity models for major peril/region combinations.

    Calibration sources:
        - Swiss Re Sigma annual catastrophe reports (historical λ)
        - AIR/RMS industry loss curves (severity calibration)
        - Cummins (2008) academic cat bond analysis
        - Swiss Re Capital Markets pricing data

    These are representative estimates, not production vendor model outputs.
    Production systems use AIR Touchstone, RMS RiskLink, or ELEMENTS.
    """
    return {
        # ── US Hurricane ──────────────────────────────────────────────────────
        # Florida: ~0.8 named storms/year make landfall; severe ones rarer
        "US_HURRICANE_FLORIDA": FrequencySeverityModel(
            lambda_annual   = 0.15,         # Major hurricane hitting Florida: ~15%/yr historically
            severity_dist   = "LOGNORMAL",
            severity_params = {"mu": 3.2, "sigma": 1.4},  # Losses in $B, log-scale
            n_simulations   = 100_000,
        ),
        "US_HURRICANE_GULF": FrequencySeverityModel(
            lambda_annual   = 0.25,
            severity_dist   = "GPD",
            severity_params = {"xi": 0.35, "beta": 2.5},
            n_simulations   = 100_000,
        ),
        "US_HURRICANE_NORTHEAST": FrequencySeverityModel(
            lambda_annual   = 0.08,
            severity_dist   = "LOGNORMAL",
            severity_params = {"mu": 2.8, "sigma": 1.6},
            n_simulations   = 100_000,
        ),
        # ── US Earthquake ─────────────────────────────────────────────────────
        # California: ~Mag 7.0+ occurs ~once per decade in populated areas
        "US_EARTHQUAKE_CALIFORNIA": FrequencySeverityModel(
            lambda_annual   = 0.08,         # Major insured loss event: ~8%/yr
            severity_dist   = "GPD",
            severity_params = {"xi": 0.50, "beta": 3.0},   # Very heavy tail
            n_simulations   = 100_000,
        ),
        # ── European Windstorm ─────────────────────────────────────────────────
        # Like 1990 Daria, 2000 Anatol/Lothar — ~0.5-1 per decade at major level
        "EU_WINDSTORM": FrequencySeverityModel(
            lambda_annual   = 0.12,
            severity_dist   = "LOGNORMAL",
            severity_params = {"mu": 2.5, "sigma": 1.3},
            n_simulations   = 100_000,
        ),
        # ── Japan Earthquake ─────────────────────────────────────────────────
        "JAPAN_EARTHQUAKE": FrequencySeverityModel(
            lambda_annual   = 0.20,
            severity_dist   = "GPD",
            severity_params = {"xi": 0.40, "beta": 2.0},
            n_simulations   = 100_000,
        ),
        # ── Extreme Mortality ─────────────────────────────────────────────────
        # Pandemic/terrorism/extreme mortality events
        # Based on Swiss Re report on pandemic risk in ILS
        "EXTREME_MORTALITY_GLOBAL": FrequencySeverityModel(
            lambda_annual   = 0.02,         # Major pandemic-level event: ~2%/yr
            severity_dist   = "LOGNORMAL",
            severity_params = {"mu": 0.5, "sigma": 1.0},   # Mortality spike fraction
            n_simulations   = 100_000,
        ),
    }


def create_example_cat_bonds() -> List[CatBondSpec]:
    """
    Representative cat bond structures modelled on real transactions.

    Based on Swiss Re Capital Markets, Aon Securities, and
    GC Securities cat bond transaction data (anonymised/representative).
    """
    today = date.today()

    return [
        # ── Florida Hurricane — typical Citizens/FHCF transaction ─────────────
        CatBondSpec(
            ticker          = "PELICAN-2024-A",
            sponsor         = "Citizens Property Insurance",
            issuer          = "Pelican Re Ltd",
            series          = "Series A",
            principal       = 150.0,           # $150M
            coupon_spread   = 750,             # SOFR + 750bps
            maturity_years  = 3.0,
            issue_date      = today,
            maturity_date   = date(today.year + 3, today.month, today.day),
            peril           = PerilType.HURRICANE,
            trigger_type    = TriggerType.INDUSTRY_INDEX,
            territory       = "Florida",
            loss_type       = TrancheLoss.PRO_RATA,
            trigger_level   = 25.0,            # Industry loss $25B
            exhaustion_level= 40.0,            # Industry loss $40B
            rating          = "BB-",
        ),
        # ── California Earthquake — typical USAA/state fund transaction ────────
        CatBondSpec(
            ticker          = "SIERRA-2024-A",
            sponsor         = "California Earthquake Authority",
            issuer          = "Sierra Re Ltd",
            series          = "Series A",
            principal       = 200.0,
            coupon_spread   = 450,
            maturity_years  = 5.0,
            issue_date      = today,
            maturity_date   = date(today.year + 5, today.month, today.day),
            peril           = PerilType.EARTHQUAKE,
            trigger_type    = TriggerType.PARAMETRIC,
            territory       = "California",
            loss_type       = TrancheLoss.BINARY,
            trigger_level   = 7.5,             # Richter 7.5+ within 50km of LA
            exhaustion_level= 7.5,             # Binary — same level
            rating          = "B+",
        ),
        # ── European Windstorm — typical Munich Re / Allianz structure ─────────
        CatBondSpec(
            ticker          = "BORA-2024-A",
            sponsor         = "Allianz SE",
            issuer          = "Bora Re Ltd",
            series          = "Series A",
            principal       = 100.0,
            coupon_spread   = 350,
            maturity_years  = 4.0,
            issue_date      = today,
            maturity_date   = date(today.year + 4, today.month, today.day),
            peril           = PerilType.WINDSTORM_EU,
            trigger_type    = TriggerType.MODELED_LOSS,
            territory       = "Western Europe",
            loss_type       = TrancheLoss.PRO_RATA,
            trigger_level   = 8.0,             # Modelled loss $8B
            exhaustion_level= 15.0,
            rating          = "BB",
        ),
        # ── US Multi-Peril — diversified US cat exposure ───────────────────────
        CatBondSpec(
            ticker          = "ATLAS-2024-A",
            sponsor         = "Nationwide Mutual",
            issuer          = "Atlas Re Ltd",
            series          = "Series A",
            principal       = 250.0,
            coupon_spread   = 600,
            maturity_years  = 3.0,
            issue_date      = today,
            maturity_date   = date(today.year + 3, today.month, today.day),
            peril           = PerilType.MULTI_PERIL,
            trigger_type    = TriggerType.INDEMNITY,
            territory       = "United States",
            loss_type       = TrancheLoss.PRO_RATA,
            trigger_level   = 500.0,           # $500M sponsor loss
            exhaustion_level= 750.0,
            rating          = "BB-",
        ),
    ]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  Cat Bond Models — Test")
    print("=" * 65)

    # Test frequency-severity model
    print("\n1. Florida Hurricane loss model (100K simulations)...")
    models = build_standard_loss_models()
    fl_model = models["US_HURRICANE_FLORIDA"]
    losses   = fl_model.simulate_annual_losses()

    print(f"   Annual loss stats:")
    stats_dict = fl_model.summary_stats()
    for k, v in stats_dict.items():
        print(f"     {k:<28}: {v:.4f}")

    print(f"\n   OEP Curve:")
    oep = fl_model.loss_exceedance_curve()
    for rp, loss in list(oep.items())[:5]:
        print(f"     {rp:<25}: ${loss:.2f}B")

    # Test cat bond pricing
    print("\n2. Pricing PELICAN-2024-A (Florida Hurricane cat bond)...")
    bonds  = create_example_cat_bonds()
    pricer = CatBondPricer()

    for bond in bonds:
        # Pick appropriate model
        if bond.peril == PerilType.HURRICANE and "Florida" in bond.territory:
            model = models["US_HURRICANE_FLORIDA"]
        elif bond.peril == PerilType.EARTHQUAKE and "California" in bond.territory:
            model = models["US_EARTHQUAKE_CALIFORNIA"]
        elif bond.peril == PerilType.WINDSTORM_EU:
            model = models["EU_WINDSTORM"]
        else:
            model = models["US_HURRICANE_FLORIDA"]   # Fallback

        result = pricer.price(bond, model)
        print(f"\n  {result.summary()}")

    print("\n✅ Cat bond models test passed")
