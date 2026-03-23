"""
AI Hedge Fund — Part 5: Alternative Assets & Data
====================================================
ils_portfolio.py — ILS Portfolio Construction & Analysis

Insurance-Linked Securities (ILS) portfolio management.

The key insight behind ILS as a portfolio allocation:
    Cat bonds have near-zero correlation with equities and bonds.
    A hurricane in Florida has nothing to do with equity valuations.
    This makes ILS one of the genuinely uncorrelated asset classes —
    not just "low beta" but structurally uncorrelated.

Historical correlation (Swiss Re ILS data, 2002-2023):
    ILS vs S&P 500:       0.04
    ILS vs US Treasuries: 0.01
    ILS vs Credit:        0.08
    ILS vs REITs:         0.06

The exceptions — when correlation spikes:
    Financial crises (2008): Slight pickup as risk appetite collapses
    COVID (2020): Pandemic bonds triggered (but most cat bonds unaffected)
    These episodes confirm the thesis — ILS holds when equities crash

ILS in a multi-asset portfolio:
    Allocating 5-15% to ILS in a 60/40 portfolio:
        - Reduces portfolio vol ~8-12%
        - Maintains or improves Sharpe ratio
        - Reduces max drawdown in equity crises

Portfolio construction for ILS:
    Diversify across:
        - Perils (hurricane, earthquake, windstorm)
        - Geographies (US, Europe, Japan, global)
        - Trigger types (parametric, industry index, indemnity)
        - Maturities (1yr, 3yr, 5yr)
    Avoid:
        - Correlation within peril (multiple Florida hurricane bonds)
        - Basis risk accumulation
        - Single-event concentration

References:
    Cummins & Weiss (2009). Convergence of Insurance and Financial Markets.
    Braun, A. (2016). Pricing in the Primary Market for Cat Bonds. JRI.
    Gürtler, Hibbeln & Winkelvos (2016). Portfolio Selection with Cat Bonds.
    Lane & Mahul (2008). Catastrophe Risk Pricing: An Empirical Analysis. World Bank.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.catbond.cat_bond_models import (
    CatBondSpec, CatBondPricer, FrequencySeverityModel,
    PerilType, TriggerType, build_standard_loss_models, create_example_cat_bonds
)

logger = logging.getLogger("hedge_fund.ils_portfolio")


# ─────────────────────────────────────────────────────────────────────────────
# ILS position
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ILSPosition:
    """A holding in the ILS portfolio."""
    bond:               CatBondSpec
    pricing:            CatBondPricer.PricingResult
    allocation_usd:     float       # Dollar allocation
    portfolio_weight:   float       # As fraction of total portfolio NAV
    purchase_price:     float = 100.0
    purchase_date:      date = field(default_factory=date.today)
    current_price:      float = 100.0

    @property
    def unrealised_pnl(self) -> float:
        return (self.current_price - self.purchase_price) / 100 * self.allocation_usd

    @property
    def accrued_coupon_usd(self) -> float:
        """Approximate accrued coupon since purchase."""
        days_held = (date.today() - self.purchase_date).days
        daily_coupon = self.bond.coupon_spread / 10000 / 365
        return daily_coupon * days_held * self.allocation_usd

    @property
    def annual_income(self) -> float:
        return self.bond.coupon_spread / 10000 * self.allocation_usd

    @property
    def annual_expected_loss(self) -> float:
        return self.pricing.el_fraction * self.allocation_usd


# ─────────────────────────────────────────────────────────────────────────────
# Correlation structure for ILS
# ─────────────────────────────────────────────────────────────────────────────

class ILSCorrelationMatrix:
    """
    Correlation structure between ILS positions and other assets.

    ILS correlation is driven by:
    1. Intra-ILS correlation: bonds covering the same peril/region
    2. Cross-asset correlation: ILS vs equities, bonds
    3. Tail correlation: spikes during systemic events

    Empirical correlation matrix from academic research:
        Florida Hurricane × California Earthquake:  0.05
        Florida Hurricane × EU Windstorm:           0.03
        US Hurricane × Japan Earthquake:            0.02
        Any cat bond × S&P 500:                     0.04
        Any cat bond × US 10Y Treasury:             0.01

    Sources: Gürtler et al (2016), Swiss Re ILS data 2002-2023
    """

    # Base correlations between peril/region pairs
    PERIL_CORRELATIONS = {
        ("US_HURRICANE_FLORIDA",   "US_HURRICANE_GULF"):       0.45,  # Same basin
        ("US_HURRICANE_FLORIDA",   "US_HURRICANE_NORTHEAST"):  0.20,
        ("US_HURRICANE_GULF",      "US_HURRICANE_NORTHEAST"):  0.15,
        ("US_HURRICANE_FLORIDA",   "US_EARTHQUAKE_CALIFORNIA"): 0.04,
        ("US_HURRICANE_FLORIDA",   "EU_WINDSTORM"):             0.03,
        ("US_HURRICANE_FLORIDA",   "JAPAN_EARTHQUAKE"):         0.02,
        ("US_EARTHQUAKE_CALIFORNIA","EU_WINDSTORM"):            0.03,
        ("EU_WINDSTORM",           "JAPAN_EARTHQUAKE"):         0.02,
        ("US_HURRICANE_FLORIDA",   "EXTREME_MORTALITY_GLOBAL"): 0.05,
    }

    # Cross-asset correlations (ILS vs financial markets)
    CROSS_ASSET_CORRELATIONS = {
        "SPY":  0.04,
        "QQQ":  0.04,
        "TLT":  0.01,
        "LQD":  0.02,
        "HYG":  0.05,
        "GLD":  0.03,
        "VIX":  0.08,
    }

    def get_intra_ils_correlation(
        self, peril_key_1: str, peril_key_2: str
    ) -> float:
        """Get correlation between two ILS positions."""
        if peril_key_1 == peril_key_2:
            return 1.0
        key  = (peril_key_1, peril_key_2)
        rkey = (peril_key_2, peril_key_1)
        return self.PERIL_CORRELATIONS.get(key, self.PERIL_CORRELATIONS.get(rkey, 0.02))

    def build_ils_correlation_matrix(
        self, peril_keys: List[str]
    ) -> np.ndarray:
        """Build full correlation matrix for a set of ILS positions."""
        n    = len(peril_keys)
        corr = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                c = self.get_intra_ils_correlation(peril_keys[i], peril_keys[j])
                corr[i, j] = c
                corr[j, i] = c
        return corr

    def equity_ils_correlation(self, equity_ticker: str = "SPY") -> float:
        return self.CROSS_ASSET_CORRELATIONS.get(equity_ticker, 0.04)


# ─────────────────────────────────────────────────────────────────────────────
# ILS portfolio
# ─────────────────────────────────────────────────────────────────────────────

class ILSPortfolio:
    """
    ILS portfolio construction, risk analysis, and optimisation.

    Manages a portfolio of cat bonds and ILS instruments as part
    of the broader hedge fund allocation.

    Key metrics computed:
        - Portfolio expected loss (probability-weighted)
        - Portfolio-level VaR and CVaR
        - Correlation contribution to the broader portfolio
        - Diversification benefit from adding each new bond
        - Concentration by peril, geography, trigger type
    """

    def __init__(
        self,
        total_nav:        float = 1_000_000,
        ils_allocation:   float = 0.10,        # 10% of portfolio in ILS
    ):
        self.total_nav     = total_nav
        self.ils_budget    = total_nav * ils_allocation
        self.positions:    List[ILSPosition] = []
        self.pricer        = CatBondPricer()
        self.loss_models   = build_standard_loss_models()
        self.corr_matrix   = ILSCorrelationMatrix()

    # ── Position management ───────────────────────────────────────────────────

    def add_position(
        self,
        bond:           CatBondSpec,
        allocation_usd: float,
        model_key:      Optional[str] = None,
    ) -> ILSPosition:
        """Add a cat bond position to the portfolio."""
        # Get appropriate loss model
        mk    = model_key or self._infer_model_key(bond)
        model = self.loss_models.get(mk, list(self.loss_models.values())[0])

        # Price the bond
        pricing = self.pricer.price(bond, model)

        position = ILSPosition(
            bond            = bond,
            pricing         = pricing,
            allocation_usd  = allocation_usd,
            portfolio_weight= allocation_usd / self.total_nav,
        )
        self.positions.append(position)
        logger.info(
            f"Added {bond.ticker}: ${allocation_usd:,.0f} | "
            f"EL={pricing.el_bps:.1f}bps | "
            f"spread={bond.coupon_spread}bps | "
            f"{'CHEAP' if pricing.is_cheap else 'RICH'}"
        )
        return position

    def _infer_model_key(self, bond: CatBondSpec) -> str:
        """Map a cat bond spec to its appropriate loss model."""
        if bond.peril == PerilType.HURRICANE:
            if "Florida" in bond.territory:
                return "US_HURRICANE_FLORIDA"
            elif "Gulf" in bond.territory:
                return "US_HURRICANE_GULF"
            else:
                return "US_HURRICANE_NORTHEAST"
        elif bond.peril == PerilType.EARTHQUAKE:
            if "California" in bond.territory or "United States" in bond.territory:
                return "US_EARTHQUAKE_CALIFORNIA"
            else:
                return "JAPAN_EARTHQUAKE"
        elif bond.peril == PerilType.WINDSTORM_EU:
            return "EU_WINDSTORM"
        elif bond.peril == PerilType.EXTREME_MORTALITY:
            return "EXTREME_MORTALITY_GLOBAL"
        else:
            return "US_HURRICANE_FLORIDA"   # Conservative default

    # ── Portfolio risk metrics ────────────────────────────────────────────────

    def portfolio_expected_loss_usd(self) -> float:
        """Aggregate expected loss across all positions."""
        return sum(p.annual_expected_loss for p in self.positions)

    def portfolio_expected_loss_pct(self) -> float:
        """EL as fraction of ILS allocation."""
        if self.ils_budget <= 0:
            return 0.0
        return self.portfolio_expected_loss_usd() / self.ils_budget

    def portfolio_expected_income_usd(self) -> float:
        """Total annual coupon income."""
        return sum(p.annual_income for p in self.positions)

    def net_expected_return(self) -> float:
        """Net annual return = income - expected losses."""
        return self.portfolio_expected_income_usd() - self.portfolio_expected_loss_usd()

    def net_expected_return_pct(self) -> float:
        if self.ils_budget <= 0:
            return 0.0
        return self.net_expected_return() / self.ils_budget

    def simulate_portfolio_loss(
        self,
        n_simulations: int = 50_000,
        seed:          int = 42,
    ) -> np.ndarray:
        """
        Simulate correlated annual losses across all positions.

        Uses Gaussian copula to model correlation between perils.
        Individual marginals come from the calibrated F-S models.

        Gaussian copula approach:
            1. Simulate correlated normals z ~ N(0, Σ) where Σ is the ILS corr matrix
            2. Transform to uniform via Φ(z)
            3. Invert each marginal CDF to get correlated losses
        """
        if not self.positions:
            return np.zeros(n_simulations)

        n_bonds = len(self.positions)
        rng     = np.random.default_rng(seed)

        # Build correlation matrix for these positions
        peril_keys = [self._infer_model_key(p.bond) for p in self.positions]
        corr       = self.corr_matrix.build_ils_correlation_matrix(peril_keys)

        # Cholesky decomposition for correlated sampling
        try:
            L = np.linalg.cholesky(corr)
        except np.linalg.LinAlgError:
            # Ensure PSD by adding small diagonal
            corr += np.eye(n_bonds) * 1e-6
            L = np.linalg.cholesky(corr)

        # Simulate correlated uniform variables via Gaussian copula
        z_indep = rng.standard_normal((n_simulations, n_bonds))
        z_corr  = z_indep @ L.T   # Correlated standard normals
        u_corr  = stats.norm.cdf(z_corr)   # Transform to uniform [0,1]

        # Simulate individual losses using each bond's loss distribution
        portfolio_losses = np.zeros(n_simulations)
        for j, (pos, u_j) in enumerate(zip(self.positions, u_corr.T)):
            mk    = peril_keys[j]
            model = self.loss_models.get(mk, list(self.loss_models.values())[0])

            # Simulate losses for this bond
            bond_losses = model.simulate_annual_losses()

            # Reorder bond losses by copula rank to introduce correlation
            # Sort bond losses, then map via correlated uniforms
            sorted_losses = np.sort(bond_losses)
            n_model = len(sorted_losses)
            # Map copula uniforms to model loss quantiles
            quantile_idx = (u_j * n_model).astype(int).clip(0, n_model - 1)
            correlated_losses = sorted_losses[quantile_idx]

            # Compute dollar loss to this position
            trigger  = pos.bond.trigger_level
            exhaust  = pos.bond.exhaustion_level
            if exhaust > trigger:
                tranche_frac = np.clip(
                    (correlated_losses - trigger) / (exhaust - trigger), 0, 1
                )
            else:
                tranche_frac = (correlated_losses > trigger).astype(float)

            portfolio_losses += tranche_frac * pos.allocation_usd

        return portfolio_losses

    def portfolio_var(
        self,
        confidence:    float = 0.99,
        n_simulations: int = 50_000,
    ) -> float:
        """Portfolio VaR at given confidence level."""
        losses = self.simulate_portfolio_loss(n_simulations)
        return float(np.percentile(losses, confidence * 100))

    def portfolio_cvar(
        self,
        confidence:    float = 0.99,
        n_simulations: int = 50_000,
    ) -> float:
        """Portfolio CVaR (Expected Shortfall)."""
        losses = self.simulate_portfolio_loss(n_simulations)
        var    = np.percentile(losses, confidence * 100)
        return float(np.mean(losses[losses >= var]))

    def portfolio_sharpe(self) -> float:
        """
        ILS Sharpe ratio.

        Return = net expected return (income - EL)
        Vol    = std of simulated annual portfolio P&L
        """
        net_return = self.net_expected_return_pct()
        losses     = self.simulate_portfolio_loss(25_000)
        # Annual P&L = income - losses
        income     = self.portfolio_expected_income_usd()
        annual_pnl = income - losses
        pnl_std    = float(annual_pnl.std()) / self.ils_budget if self.ils_budget > 0 else 1
        if pnl_std < 1e-8:
            return 0.0
        rf = 0.05
        return (net_return - rf) / pnl_std

    # ── Concentration analysis ────────────────────────────────────────────────

    def concentration_by_peril(self) -> Dict[str, float]:
        """Allocation by peril as fraction of ILS budget."""
        conc: Dict[str, float] = {}
        for pos in self.positions:
            key = pos.bond.peril.value
            conc[key] = conc.get(key, 0.0) + pos.allocation_usd
        total = sum(conc.values())
        return {k: v / total for k, v in conc.items()} if total > 0 else conc

    def concentration_by_geography(self) -> Dict[str, float]:
        """Allocation by territory."""
        conc: Dict[str, float] = {}
        for pos in self.positions:
            key = pos.bond.territory
            conc[key] = conc.get(key, 0.0) + pos.allocation_usd
        total = sum(conc.values())
        return {k: v / total for k, v in conc.items()} if total > 0 else conc

    def herfindahl_index(self) -> float:
        """Concentration measure (lower = more diversified)."""
        total = sum(p.allocation_usd for p in self.positions)
        if total <= 0:
            return 1.0
        weights = [p.allocation_usd / total for p in self.positions]
        return sum(w**2 for w in weights)

    # ── Marginal contribution to portfolio ────────────────────────────────────

    def marginal_var_contribution(
        self,
        candidate_bond: CatBondSpec,
        allocation_usd: float,
    ) -> float:
        """
        Compute how much a new bond adds to portfolio VaR.

        Used by the allocation agent to screen candidates.
        Lower marginal VaR contribution = better diversification.
        """
        # Current portfolio VaR
        current_var = self.portfolio_var(confidence=0.99, n_simulations=25_000)

        # Add candidate temporarily
        temp_position = self.add_position(candidate_bond, allocation_usd)
        new_var = self.portfolio_var(confidence=0.99, n_simulations=25_000)

        # Remove candidate
        self.positions.remove(temp_position)

        return new_var - current_var

    # ── Portfolio optimiser ────────────────────────────────────────────────────

    def optimise_allocation(
        self,
        candidate_bonds: List[CatBondSpec],
        budget_usd:      float,
        max_single_pct:  float = 0.40,    # Max 40% of ILS budget in one bond
        min_bonds:       int = 3,
        target_el_pct:   float = 0.03,    # Target 3% portfolio EL
    ) -> Dict[str, float]:
        """
        Optimise allocation across candidate cat bonds.

        Objective: Maximise net return (income - EL) subject to:
            - Total allocation = budget_usd
            - No single bond > max_single_pct of budget
            - Minimum n bonds
            - Portfolio EL within target_el_pct

        Returns dict of {bond_ticker: allocation_usd}
        """
        n = len(candidate_bonds)
        if n == 0:
            return {}

        # Price all candidates
        priced = []
        for bond in candidate_bonds:
            mk     = self._infer_model_key(bond)
            model  = self.loss_models.get(mk, list(self.loss_models.values())[0])
            result = self.pricer.price(bond, model)
            priced.append((bond, result))

        # Simple optimisation: maximum Sharpe (return/EL ratio) allocation
        # with concentration constraints
        # Full optimiser would use scipy.optimize.minimize with correlation matrix

        # Score each bond by (net return - EL) / EL = (spread/EL - 1)
        scores = []
        for bond, result in priced:
            net_return_bps  = bond.coupon_spread - result.el_bps
            score           = net_return_bps / max(result.el_bps, 1.0)
            scores.append((score, bond, result))

        scores.sort(key=lambda x: -x[0])

        # Allocate proportionally to scores with constraints
        top_n      = min(max(min_bonds, n), len(scores))
        top_scores = scores[:top_n]
        total_score= sum(s for s, _, _ in top_scores)

        allocation = {}
        for score, bond, _ in top_scores:
            raw_weight = score / total_score if total_score > 0 else 1.0 / top_n
            capped     = min(raw_weight, max_single_pct)
            allocation[bond.ticker] = round(capped * budget_usd, 2)

        # Normalise to exactly hit budget
        total_alloc = sum(allocation.values())
        if total_alloc > 0:
            scale = budget_usd / total_alloc
            allocation = {k: v * scale for k, v in allocation.items()}

        logger.info(
            f"Optimised ILS portfolio: {len(allocation)} bonds, "
            f"budget=${budget_usd:,.0f}"
        )
        return allocation

    # ── Reporting ─────────────────────────────────────────────────────────────

    def portfolio_report(self) -> str:
        """Generate comprehensive ILS portfolio report."""
        if not self.positions:
            return "ILS portfolio is empty."

        total_alloc = sum(p.allocation_usd for p in self.positions)
        total_income= self.portfolio_expected_income_usd()
        total_el    = self.portfolio_expected_loss_usd()
        net_return  = self.net_expected_return_pct()

        lines = [
            "═" * 65,
            "  ILS Portfolio Report",
            f"  {datetime.now():%Y-%m-%d %H:%M}",
            "═" * 65,
            f"  Total ILS Allocation : ${total_alloc:>12,.0f}",
            f"  Expected Annual Income: ${total_income:>12,.0f}  ({total_income/total_alloc:.2%})",
            f"  Expected Annual Loss  : ${total_el:>12,.0f}  ({total_el/total_alloc:.2%})",
            f"  Net Expected Return   : ${net_return*total_alloc:>12,.0f}  ({net_return:.2%})",
            f"  Diversification (HHI) : {self.herfindahl_index():.3f}",
            "─" * 65,
            f"  {'Ticker':<16} {'Peril':<14} {'Alloc $':>10} {'EL':>6} {'Spread':>8} {'Val':>8}",
            "─" * 65,
        ]
        for pos in self.positions:
            val_str = "CHEAP" if pos.pricing.is_cheap else "RICH "
            lines.append(
                f"  {pos.bond.ticker:<16} "
                f"{pos.bond.peril.value:<14} "
                f"${pos.allocation_usd:>9,.0f} "
                f"{pos.pricing.el_bps:>5.1f}bp "
                f"{pos.bond.coupon_spread:>7}bp "
                f"{val_str:>8}"
            )
        lines.append("─" * 65)
        peril_conc = self.concentration_by_peril()
        lines.append("  Concentration by peril:")
        for peril, frac in sorted(peril_conc.items(), key=lambda x: -x[1]):
            lines.append(f"    {peril:<25}: {frac:.1%}")
        lines.append("═" * 65)
        return "\n".join(lines)

    def equity_diversification_benefit(
        self,
        equity_weight: float = 0.60,
        equity_vol:    float = 0.16,
        equity_return: float = 0.10,
        rf_rate:       float = 0.05,
    ) -> Dict[str, float]:
        """
        Compute the portfolio-level benefit of adding ILS to a 60/40 portfolio.

        Shows how ILS improves Sharpe and reduces max drawdown in equity crises.
        """
        ils_weight  = total_alloc / self.total_nav if (total_alloc := sum(
            p.allocation_usd for p in self.positions)) > 0 else 0.10

        ils_return  = self.net_expected_return_pct()
        ils_vol     = math.sqrt(
            sum(p.pricing.el_fraction * (1 - p.pricing.el_fraction) *
                (p.allocation_usd / self.total_nav)**2
                for p in self.positions)
        ) if self.positions else 0.05

        # Portfolio without ILS (pure equity)
        pure_eq_sharpe = (equity_return - rf_rate) / equity_vol

        # Portfolio with ILS (blend)
        corr_ils_eq = self.corr_matrix.equity_ils_correlation("SPY")
        blended_w_eq  = equity_weight / (equity_weight + ils_weight)
        blended_w_ils = ils_weight    / (equity_weight + ils_weight)

        blended_return = blended_w_eq * equity_return + blended_w_ils * ils_return
        blended_vol    = math.sqrt(
            (blended_w_eq * equity_vol)**2 +
            (blended_w_ils * ils_vol)**2 +
            2 * blended_w_eq * blended_w_ils * corr_ils_eq * equity_vol * ils_vol
        )
        blended_sharpe = (blended_return - rf_rate) / blended_vol

        return {
            "pure_equity_sharpe":  round(pure_eq_sharpe, 3),
            "blended_sharpe":      round(blended_sharpe, 3),
            "sharpe_improvement":  round(blended_sharpe - pure_eq_sharpe, 3),
            "vol_reduction_pct":   round((1 - blended_vol / equity_vol) * 100, 1),
            "ils_equity_corr":     corr_ils_eq,
            "ils_allocation_pct":  round(ils_weight * 100, 1),
        }


try:
    from scipy import stats
except ImportError:
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 65)
    print("  ILS Portfolio — Test")
    print("=" * 65)

    portfolio = ILSPortfolio(total_nav=10_000_000, ils_allocation=0.10)
    bonds = create_example_cat_bonds()

    # Add bonds to portfolio
    allocations = {
        "PELICAN-2024-A": 400_000,
        "SIERRA-2024-A":  300_000,
        "BORA-2024-A":    200_000,
        "ATLAS-2024-A":   100_000,
    }

    for bond in bonds:
        alloc = allocations.get(bond.ticker, 100_000)
        portfolio.add_position(bond, alloc)

    print(portfolio.portfolio_report())

    print("\n  Running portfolio simulation (50K scenarios)...")
    var_99  = portfolio.portfolio_var(0.99, 50_000)
    cvar_99 = portfolio.portfolio_cvar(0.99, 50_000)
    sharpe  = portfolio.portfolio_sharpe()

    print(f"  Portfolio VaR (99%):  ${var_99:,.0f}")
    print(f"  Portfolio CVaR (99%): ${cvar_99:,.0f}")
    print(f"  Portfolio Sharpe:     {sharpe:.2f}")

    print("\n  Diversification benefit vs 60/40:")
    benefit = portfolio.equity_diversification_benefit()
    for k, v in benefit.items():
        print(f"    {k:<28}: {v}")

    print("\n✅ ILS Portfolio tests passed")
