import os
import sys
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime

# Try pydantic-settings for production-grade config
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Path constants
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).resolve().parents[3]   # AI_HEDGE_FUND/core-sys/
SRC_DIR  = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
DB_DIR   = ROOT_DIR / "db"

# Create directories if they don't exist
for d in [DATA_DIR, LOGS_DIR, DB_DIR, DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "cache"]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Load .env file manually (no dotenv dependency required)
# ─────────────────────────────────────────────────────────────────────────────

def _load_env_file(env_path: Optional[Path] = None) -> None:
    """Load .env file into os.environ without requiring python-dotenv."""
    candidates = [
        env_path,
        ROOT_DIR / ".env",
        Path.home() / ".env",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            with open(candidate) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, value = line.partition("=")
                        key = key.strip()
                        value = value.strip().strip('"').strip("'")
                        if key and key not in os.environ:
                            os.environ[key] = value
            break


_load_env_file()


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HedgeFundConfig:
    """
    Central configuration for the AI Hedge Fund system.

    All values can be overridden via environment variables.
    Required keys (must be set in .env or environment):
        ANTHROPIC_API_KEY or OPENAI_API_KEY (at least one)

    Optional data provider keys:
        ALPHA_VANTAGE_API_KEY  — premium market data (fallback: Yahoo Finance)
        POLYGON_API_KEY        — real-time tick data
        FRED_API_KEY           — macro data from St. Louis Fed
    """

    # ── LLM API Keys ──────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    OPENAI_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )

    # ── Data Provider Keys ────────────────────────────────────────────────────
    ALPHA_VANTAGE_API_KEY: str = field(
        default_factory=lambda: os.getenv("ALPHA_VANTAGE_API_KEY", "")
    )
    POLYGON_API_KEY: str = field(
        default_factory=lambda: os.getenv("POLYGON_API_KEY", "")
    )
    FRED_API_KEY: str = field(
        default_factory=lambda: os.getenv("FRED_API_KEY", "")
    )

    # ── Default LLM Model ─────────────────────────────────────────────────────
    # FIX: Changed from "claude-sonnet-4-6" (invalid) to the real Anthropic
    # model identifier "claude-sonnet-4-20250514".
    DEFAULT_LLM_MODEL: str = field(
        default_factory=lambda: os.getenv("DEFAULT_LLM_MODEL", "claude-sonnet-4-20250514")
    )
    LLM_TEMPERATURE: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1"))
    )
    LLM_MAX_TOKENS: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "4096"))
    )
    LLM_TIMEOUT_SECONDS: int = field(
        default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "120"))
    )

    # ── Portfolio Parameters ───────────────────────────────────────────────────
    INITIAL_CAPITAL: float = field(
        default_factory=lambda: float(os.getenv("INITIAL_CAPITAL", "1000000"))
    )
    MAX_POSITION_SIZE: float = field(
        default_factory=lambda: float(os.getenv("MAX_POSITION_SIZE", "0.15"))  # 15%
    )
    MAX_SECTOR_CONCENTRATION: float = field(
        default_factory=lambda: float(os.getenv("MAX_SECTOR_CONCENTRATION", "0.30"))  # 30%
    )
    MIN_POSITION_SIZE: float = field(
        default_factory=lambda: float(os.getenv("MIN_POSITION_SIZE", "0.01"))  # 1%
    )
    TARGET_NUM_POSITIONS: int = field(
        default_factory=lambda: int(os.getenv("TARGET_NUM_POSITIONS", "20"))
    )
    REBALANCE_THRESHOLD: float = field(
        default_factory=lambda: float(os.getenv("REBALANCE_THRESHOLD", "0.05"))  # 5% drift
    )

    # ── Risk Limits ────────────────────────────────────────────────────────────
    MAX_PORTFOLIO_VAR_PCT: float = field(
        default_factory=lambda: float(os.getenv("MAX_PORTFOLIO_VAR_PCT", "0.02"))  # 2% NAV/day
    )
    VAR_CONFIDENCE_LEVEL: float = field(
        default_factory=lambda: float(os.getenv("VAR_CONFIDENCE_LEVEL", "0.95"))
    )
    MAX_DRAWDOWN_LIMIT: float = field(
        default_factory=lambda: float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.15"))  # 15%
    )
    TARGET_SHARPE_RATIO: float = field(
        default_factory=lambda: float(os.getenv("TARGET_SHARPE_RATIO", "1.5"))
    )
    MAX_CORRELATION_ADDITION: float = field(
        default_factory=lambda: float(os.getenv("MAX_CORRELATION_ADDITION", "0.70"))
    )

    # ── Universe & Data ────────────────────────────────────────────────────────
    LOOKBACK_DAYS: int = field(
        default_factory=lambda: int(os.getenv("LOOKBACK_DAYS", "504"))  # 2 years
    )
    FEATURE_LOOKBACK_DAYS: int = field(
        default_factory=lambda: int(os.getenv("FEATURE_LOOKBACK_DAYS", "252"))  # 1 year
    )
    DATA_CACHE_EXPIRY_HOURS: int = field(
        default_factory=lambda: int(os.getenv("DATA_CACHE_EXPIRY_HOURS", "4"))
    )
    TRADING_DAYS_PER_YEAR: int = 252
    MARKET_OPEN_HOUR: int = 9
    MARKET_OPEN_MINUTE: int = 30
    MARKET_CLOSE_HOUR: int = 16

    # ── Database ───────────────────────────────────────────────────────────────
    DATABASE_URL: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL",
            f"sqlite:///{DB_DIR}/hedge_fund.db"
        )
    )

    # ── Logging ────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    LOG_FILE: str = field(
        default_factory=lambda: str(LOGS_DIR / f"hedge_fund_{datetime.now():%Y%m%d}.log")
    )

    # ── Default Stock Universe ─────────────────────────────────────────────────
    DEFAULT_UNIVERSE: List[str] = field(default_factory=lambda: [
        # US Large Cap — S&P 500 representative sample across sectors
        # Technology
        "AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "AMD",
        # Financials
        "JPM", "BAC", "GS", "MS", "BLK", "V", "MA",
        # Healthcare
        "JNJ", "UNH", "LLY", "PFE", "ABBV", "MRK",
        # Energy
        "XOM", "CVX", "COP",
        # Industrials
        "CAT", "DE", "BA", "HON", "GE",
        # Consumer
        "PG", "KO", "WMT", "COST", "HD",
        # ETFs for macro hedging
        "SPY", "QQQ", "IWM", "TLT", "GLD", "VIX",
    ])

    # ── Agent Settings ─────────────────────────────────────────────────────────
    AGENT_MAX_ITERATIONS: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_ITERATIONS", "15"))
    )
    AGENT_VERBOSE: bool = field(
        default_factory=lambda: os.getenv("AGENT_VERBOSE", "true").lower() == "true"
    )

    def validate(self) -> "HedgeFundConfig":
        """Validate configuration — call at startup."""
        errors = []

        # Must have at least one LLM key
        if not self.ANTHROPIC_API_KEY and not self.OPENAI_API_KEY:
            errors.append(
                "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env"
            )

        if self.MAX_POSITION_SIZE > 0.50:
            errors.append("MAX_POSITION_SIZE > 50% is dangerously concentrated")

        if self.MAX_PORTFOLIO_VAR_PCT > 0.05:
            errors.append("MAX_PORTFOLIO_VAR_PCT > 5% per day is extremely risky")

        if errors:
            for err in errors:
                print(f"[CONFIG ERROR] {err}", file=sys.stderr)
            if not self.ANTHROPIC_API_KEY and not self.OPENAI_API_KEY:
                raise ValueError("Cannot start without LLM API key")

        return self

    @property
    def has_anthropic(self) -> bool:
        return bool(self.ANTHROPIC_API_KEY)

    @property
    def has_openai(self) -> bool:
        return bool(self.OPENAI_API_KEY)

    @property
    def has_real_time_data(self) -> bool:
        return bool(self.POLYGON_API_KEY)

    @property
    def has_premium_data(self) -> bool:
        return bool(self.ALPHA_VANTAGE_API_KEY)

    def llm_summary(self) -> str:
        lines = ["=== LLM Configuration ==="]
        lines.append(f"  Default model : {self.DEFAULT_LLM_MODEL}")
        lines.append(f"  Temperature   : {self.LLM_TEMPERATURE}")
        lines.append(f"  Anthropic key : {'SET' if self.has_anthropic else 'NOT SET'}")
        lines.append(f"  OpenAI key    : {'SET' if self.has_openai else 'NOT SET'}")
        return "\n".join(lines)

    def portfolio_summary(self) -> str:
        lines = ["=== Portfolio Configuration ==="]
        lines.append(f"  Initial capital  : ${self.INITIAL_CAPITAL:,.0f}")
        lines.append(f"  Max position     : {self.MAX_POSITION_SIZE:.0%}")
        lines.append(f"  Max sector       : {self.MAX_SECTOR_CONCENTRATION:.0%}")
        lines.append(f"  Max port VaR/day : {self.MAX_PORTFOLIO_VAR_PCT:.1%}")
        lines.append(f"  Max drawdown     : {self.MAX_DRAWDOWN_LIMIT:.0%}")
        lines.append(f"  Target Sharpe    : {self.TARGET_SHARPE_RATIO:.1f}")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton config instance — import this everywhere
# ─────────────────────────────────────────────────────────────────────────────

cfg = HedgeFundConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(level: str = None) -> logging.Logger:
    """Configure logging for the hedge fund system."""
    log_level = level or cfg.LOG_LEVEL

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d — %(message)s"
    date_fmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format=fmt,
        datefmt=date_fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(cfg.LOG_FILE, mode="a", encoding="utf-8"),
        ]
    )

    # Quiet noisy third-party libraries
    for noisy in ["urllib3", "httpx", "httpcore", "anthropic", "openai"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logging.getLogger("hedge_fund")


logger = setup_logging()


# ─────────────────────────────────────────────────────────────────────────────
# .env template generator
# ─────────────────────────────────────────────────────────────────────────────

ENV_TEMPLATE = """# AI Hedge Fund — Environment Configuration
# Copy this to .env and fill in your keys
# NEVER commit .env to git

# ── REQUIRED: At least one LLM key ──────────────────────────────────────────
ANTHROPIC_API_KEY=your_anthropic_key_here
OPENAI_API_KEY=your_openai_key_here

# ── OPTIONAL: Market Data Providers ─────────────────────────────────────────
# Yahoo Finance works without a key (free, rate-limited)
# For production, add at least one premium provider:
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here   # Free tier: 25 req/day
POLYGON_API_KEY=your_polygon_key_here               # Real-time data
FRED_API_KEY=your_fred_key_here                     # Macro data (free)

# ── Portfolio Settings ───────────────────────────────────────────────────────
INITIAL_CAPITAL=1000000
MAX_POSITION_SIZE=0.15
MAX_PORTFOLIO_VAR_PCT=0.02

# ── LLM Settings ─────────────────────────────────────────────────────────────
# FIX: Use the real Anthropic model identifier, not "claude-sonnet-4-6"
DEFAULT_LLM_MODEL=claude-sonnet-4-20250514
LLM_TEMPERATURE=0.1
AGENT_VERBOSE=true

# ── System ────────────────────────────────────────────────────────────────────
LOG_LEVEL=INFO
"""


def write_env_template():
    env_example = ROOT_DIR / ".env.example"
    if not env_example.exists():
        env_example.write_text(ENV_TEMPLATE)
        print(f"Created .env.example at {env_example}")


if __name__ == "__main__":
    write_env_template()
    print(cfg.llm_summary())
    print(cfg.portfolio_summary())
    print(f"\nData directory : {DATA_DIR}")
    print(f"Database       : {cfg.DATABASE_URL}")
    print(f"Log file       : {cfg.LOG_FILE}")
    print(f"Universe size  : {len(cfg.DEFAULT_UNIVERSE)} securities")
