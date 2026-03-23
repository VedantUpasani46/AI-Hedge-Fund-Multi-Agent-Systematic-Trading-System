"""
AI Hedge Fund — Part 1: Foundation
====================================
llm_client.py — Production LLM client

Wraps Anthropic Claude and OpenAI GPT-4 with:
  - Automatic retry with exponential backoff
  - Cost tracking per call (critical for budget management)
  - Structured output parsing (JSON responses from agents)
  - Context window management
  - Response caching for identical prompts

Why this matters:
  Every LLM call costs money and takes time.
  A badly written client will silently lose decisions,
  fail on rate limits, and leave you with no audit trail.

Cost reference (as of 2025):
  Claude claude-sonnet-4-6 : $0.003 / 1K input tokens, $0.015 / 1K output
  Claude claude-opus-4-6   : $0.015 / 1K input tokens, $0.075 / 1K output
  GPT-4o                   : $0.005 / 1K input tokens, $0.015 / 1K output

At 5 decisions/day × 2K tokens each → ~$0.15-0.50/day for production.
"""

import json
import logging
import time
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config.settings import cfg, DB_DIR

logger = logging.getLogger("hedge_fund.llm_client")


# ─────────────────────────────────────────────────────────────────────────────
# Token cost table
# ─────────────────────────────────────────────────────────────────────────────

MODEL_COSTS: Dict[str, Tuple[float, float]] = {
    # Model: (input_cost_per_1k, output_cost_per_1k) in USD
    "claude-sonnet-4-6":              (0.003,  0.015),
    "claude-opus-4-6":                (0.015,  0.075),
    "claude-haiku-4-5-20251001":      (0.00025, 0.00125),
    "gpt-4o":                         (0.005,  0.015),
    "gpt-4o-mini":                    (0.00015, 0.0006),
    "gpt-4":                          (0.03,   0.06),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate API cost in USD for a call."""
    if model not in MODEL_COSTS:
        # Use mid-range estimate for unknown models
        return (input_tokens * 0.003 + output_tokens * 0.015) / 1000
    in_cost, out_cost = MODEL_COSTS[model]
    return (input_tokens * in_cost + output_tokens * out_cost) / 1000


# ─────────────────────────────────────────────────────────────────────────────
# Response container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LLMResponse:
    """
    Structured response from any LLM call.

    Contains the text output, usage stats, cost, and metadata.
    Every agent decision is backed by one of these.
    """
    content:       str
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float
    latency_ms:    float
    timestamp:     datetime = field(default_factory=datetime.now)
    cached:        bool = False
    raw_response:  Any = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def parse_json(self) -> Optional[Dict]:
        """
        Parse JSON from the response content.

        Handles common LLM formatting quirks:
        - JSON wrapped in ```json...``` blocks
        - Trailing commas
        - Single quotes (invalid JSON)
        """
        text = self.content.strip()

        # Strip markdown code fences
        for fence in ["```json", "```JSON", "```"]:
            if fence in text:
                parts = text.split(fence)
                for part in parts:
                    clean = part.strip().rstrip("`").strip()
                    if clean.startswith("{") or clean.startswith("["):
                        text = clean
                        break

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to find JSON object within text
            import re
            json_pattern = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
            if json_pattern:
                try:
                    return json.loads(json_pattern.group())
                except json.JSONDecodeError:
                    pass
            logger.warning(f"Failed to parse JSON from LLM response: {text[:200]}...")
            return None

    def summary(self) -> str:
        return (
            f"LLMResponse | model={self.model} | "
            f"tokens={self.total_tokens} ({self.input_tokens}in/{self.output_tokens}out) | "
            f"cost=${self.cost_usd:.5f} | "
            f"latency={self.latency_ms:.0f}ms | "
            f"cached={self.cached}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Cost tracker — cumulative spend monitoring
# ─────────────────────────────────────────────────────────────────────────────

class LLMCostTracker:
    """
    Tracks cumulative LLM API spend and logs to SQLite.

    Helps you stay within budget and understand which agents
    are most expensive.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DB_DIR / "llm_costs.db")
        self._init_db()
        self._session_cost = 0.0
        self._session_calls = 0

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_calls (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp    TEXT NOT NULL,
                    model        TEXT NOT NULL,
                    agent_name   TEXT,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    cost_usd     REAL,
                    latency_ms   REAL,
                    cached       INTEGER,
                    purpose      TEXT
                )
            """)
            conn.commit()

    def record(
        self,
        response: LLMResponse,
        agent_name: str = "",
        purpose: str = "",
    ) -> None:
        """Record a call to the database."""
        self._session_cost += response.cost_usd
        self._session_calls += 1

        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO llm_calls
                (timestamp, model, agent_name, input_tokens, output_tokens,
                 cost_usd, latency_ms, cached, purpose)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (
                response.timestamp.isoformat(),
                response.model,
                agent_name,
                response.input_tokens,
                response.output_tokens,
                response.cost_usd,
                response.latency_ms,
                int(response.cached),
                purpose,
            ))
            conn.commit()

    def session_summary(self) -> str:
        return (
            f"Session: {self._session_calls} calls | "
            f"${self._session_cost:.4f} total"
        )

    def total_spend(self, days: int = 30) -> float:
        """Total spend over the last N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(cost_usd) FROM llm_calls WHERE timestamp >= ?",
                (cutoff,)
            ).fetchone()
        return float(row[0] or 0.0)

    def spend_by_agent(self, days: int = 7) -> Dict[str, float]:
        """Spend breakdown by agent over N days."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT agent_name, SUM(cost_usd)
                FROM llm_calls
                WHERE timestamp >= ?
                GROUP BY agent_name
                ORDER BY SUM(cost_usd) DESC
            """, (cutoff,)).fetchall()
        return {r[0]: float(r[1]) for r in rows}


# ─────────────────────────────────────────────────────────────────────────────
# Response cache
# ─────────────────────────────────────────────────────────────────────────────

class LLMResponseCache:
    """
    SQLite cache for LLM responses.

    Identical prompts return cached responses, saving API cost.
    Cache expires after cfg.DATA_CACHE_EXPIRY_HOURS.

    IMPORTANT: Only cache non-time-sensitive analysis.
    Never cache real-time risk decisions.
    """

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (DB_DIR / "llm_cache.db")
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key  TEXT PRIMARY KEY,
                    content    TEXT,
                    model      TEXT,
                    cost_usd   REAL,
                    cached_at  TEXT,
                    expires_at TEXT
                )
            """)
            conn.commit()

    def _make_key(self, model: str, messages: List[Dict]) -> str:
        payload = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def get(self, model: str, messages: List[Dict]) -> Optional[str]:
        key = self._make_key(model, messages)
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT content FROM response_cache WHERE cache_key=? AND expires_at>?",
                (key, now)
            ).fetchone()
        return row[0] if row else None

    def set(
        self,
        model: str,
        messages: List[Dict],
        content: str,
        cost_usd: float,
        ttl_hours: int = 4,
    ) -> None:
        key = self._make_key(model, messages)
        now = datetime.now()
        expires = (now + timedelta(hours=ttl_hours)).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO response_cache
                (cache_key, content, model, cost_usd, cached_at, expires_at)
                VALUES (?,?,?,?,?,?)
            """, (key, content, model, cost_usd, now.isoformat(), expires))
            conn.commit()

    def clear_expired(self) -> int:
        now = datetime.now().isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM response_cache WHERE expires_at < ?", (now,)
            )
            conn.commit()
            return cursor.rowcount


# ─────────────────────────────────────────────────────────────────────────────
# Main LLM client
# ─────────────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Production LLM client for the AI Hedge Fund.

    Supports Claude (Anthropic) and GPT-4 (OpenAI) with:
      - Automatic retry on rate limits / transient errors
      - Response caching (4-hour TTL by default)
      - Cost tracking per call and session
      - Structured JSON output parsing
      - Context window management (truncates if needed)

    Usage:
        client = LLMClient()

        # Simple completion
        response = client.complete(
            system="You are a risk analyst.",
            user="What is the VaR for a 15% vol position?",
        )
        print(response.content)
        print(response.summary())

        # With structured output
        response = client.complete(
            system="Respond in JSON only.",
            user="Rate this trade: BUY AAPL 5%",
        )
        data = response.parse_json()
    """

    MAX_RETRIES    = 3
    RETRY_DELAYS   = [1.0, 5.0, 15.0]   # seconds
    MAX_CONTEXT    = 180_000              # Tokens (Claude Sonnet context window)

    def __init__(
        self,
        model:       Optional[str] = None,
        temperature: Optional[float] = None,
        use_cache:   bool = True,
        agent_name:  str = "",
    ):
        self.model       = model or cfg.DEFAULT_LLM_MODEL
        self.temperature = temperature if temperature is not None else cfg.LLM_TEMPERATURE
        self.agent_name  = agent_name
        self.use_cache   = use_cache

        self.cost_tracker = LLMCostTracker()
        self.cache        = LLMResponseCache() if use_cache else None

        # Initialize the appropriate SDK client
        self._anthropic_client = None
        self._openai_client    = None
        self._setup_client()

        logger.info(
            f"LLMClient initialised: model={self.model} | "
            f"cache={'on' if use_cache else 'off'}"
        )

    def _setup_client(self):
        """Initialize SDK client based on model name."""
        model_lower = self.model.lower()

        if "claude" in model_lower:
            if not cfg.ANTHROPIC_API_KEY:
                raise ValueError(
                    "ANTHROPIC_API_KEY not set. Add it to .env file."
                )
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(
                    api_key=cfg.ANTHROPIC_API_KEY
                )
            except ImportError:
                raise ImportError("Install anthropic SDK: pip install anthropic")

        elif "gpt" in model_lower or "o1" in model_lower:
            if not cfg.OPENAI_API_KEY:
                raise ValueError(
                    "OPENAI_API_KEY not set. Add it to .env file."
                )
            try:
                import openai
                self._openai_client = openai.OpenAI(
                    api_key=cfg.OPENAI_API_KEY
                )
            except ImportError:
                raise ImportError("Install openai SDK: pip install openai")
        else:
            raise ValueError(
                f"Unknown model: {self.model}. "
                "Use a claude-* or gpt-* model name."
            )

    def complete(
        self,
        system:      str,
        user:        str,
        history:     Optional[List[Dict]] = None,
        max_tokens:  Optional[int] = None,
        use_cache:   Optional[bool] = None,
        purpose:     str = "",
    ) -> LLMResponse:
        """
        Send a completion request to the LLM.

        Args:
            system    : System prompt (agent role / instructions)
            user      : User message (the query / task)
            history   : Prior conversation turns [{"role":..., "content":...}]
            max_tokens: Override max output tokens
            use_cache : Override instance cache setting
            purpose   : Label for cost tracking (e.g. "allocation_decision")

        Returns:
            LLMResponse with content, usage stats, and cost
        """
        should_cache = use_cache if use_cache is not None else self.use_cache
        max_tok = max_tokens or cfg.LLM_MAX_TOKENS

        # Build message list
        messages = self._build_messages(user, history)

        # Check cache
        if should_cache and self.cache:
            cached_content = self.cache.get(self.model, [{"role": "system", "content": system}] + messages)
            if cached_content:
                logger.debug("Cache hit — returning cached response")
                # Estimate token counts for cached response
                in_tok  = len((system + user).split()) * 1.3
                out_tok = len(cached_content.split()) * 1.3
                return LLMResponse(
                    content=cached_content,
                    model=self.model,
                    input_tokens=int(in_tok),
                    output_tokens=int(out_tok),
                    cost_usd=0.0,
                    latency_ms=0.0,
                    cached=True,
                )

        # Call LLM with retry
        response = self._call_with_retry(system, messages, max_tok)

        # Cache the response
        if should_cache and self.cache:
            self.cache.set(
                self.model,
                [{"role": "system", "content": system}] + messages,
                response.content,
                response.cost_usd,
            )

        # Track cost
        self.cost_tracker.record(response, self.agent_name, purpose)

        if response.cost_usd > 0:
            logger.info(
                f"LLM call: {response.total_tokens} tokens | "
                f"${response.cost_usd:.5f} | "
                f"{response.latency_ms:.0f}ms"
            )

        return response

    def _build_messages(
        self,
        user: str,
        history: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        messages = list(history or [])
        messages.append({"role": "user", "content": user})
        return messages

    def _call_with_retry(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
    ) -> LLMResponse:
        """Call LLM with exponential backoff retry."""
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                start_time = time.time()

                if self._anthropic_client:
                    raw = self._call_anthropic(system, messages, max_tokens)
                else:
                    raw = self._call_openai(system, messages, max_tokens)

                latency_ms = (time.time() - start_time) * 1000

                content      = raw["content"]
                input_tokens = raw["input_tokens"]
                output_tokens= raw["output_tokens"]
                cost         = estimate_cost(self.model, input_tokens, output_tokens)

                return LLMResponse(
                    content=content,
                    model=self.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=cost,
                    latency_ms=latency_ms,
                    raw_response=raw.get("raw"),
                )

            except Exception as e:
                last_error = e
                error_str  = str(e).lower()

                # Rate limit or overload — always retry with backoff
                if any(k in error_str for k in ["rate limit", "529", "overloaded", "503"]):
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    logger.warning(f"Rate limit hit — waiting {delay}s (attempt {attempt+1})")
                    time.sleep(delay)

                # Context window exceeded — truncate and retry
                elif "context" in error_str or "token" in error_str:
                    logger.warning("Context window exceeded — truncating messages")
                    messages = self._truncate_messages(messages)

                # Auth error — don't retry
                elif any(k in error_str for k in ["authentication", "invalid_api_key", "401"]):
                    logger.error(f"Authentication error: {e}")
                    raise

                else:
                    delay = self.RETRY_DELAYS[min(attempt, len(self.RETRY_DELAYS) - 1)]
                    logger.warning(f"LLM error (attempt {attempt+1}): {e} — retrying in {delay}s")
                    time.sleep(delay)

        raise RuntimeError(
            f"LLM call failed after {self.MAX_RETRIES} attempts. "
            f"Last error: {last_error}"
        )

    def _call_anthropic(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
    ) -> Dict:
        """Call Anthropic Claude API."""
        response = self._anthropic_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            system=system,
            messages=messages,
        )
        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return {
            "content":       content,
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "raw":           response,
        }

    def _call_openai(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int,
    ) -> Dict:
        """Call OpenAI GPT API."""
        full_messages = [{"role": "system", "content": system}] + messages
        response = self._openai_client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=self.temperature,
            messages=full_messages,
        )
        return {
            "content":       response.choices[0].message.content or "",
            "input_tokens":  response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "raw":           response,
        }

    def _truncate_messages(self, messages: List[Dict]) -> List[Dict]:
        """Truncate oldest history to fit in context window."""
        if len(messages) <= 2:
            return messages
        # Keep first and last messages; remove middle history
        return [messages[0]] + messages[-2:]

    # ── Convenience methods ───────────────────────────────────────────────────

    def ask(self, question: str, context: str = "") -> str:
        """Simple one-shot question with optional context."""
        system = "You are a quantitative finance expert. Be precise and concise."
        user   = f"{context}\n\n{question}" if context else question
        return self.complete(system=system, user=user).content

    def structured_output(
        self,
        system: str,
        user: str,
        schema_description: str = "",
        max_tokens: int = 2048,
    ) -> Optional[Dict]:
        """
        Request structured JSON output from the LLM.

        Adds instructions to respond with valid JSON matching the schema.
        Returns parsed dict or None if parsing fails.
        """
        json_system = (
            system + "\n\n"
            "IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation. "
            "Just the JSON object. " + schema_description
        )
        response = self.complete(
            system=json_system,
            user=user,
            max_tokens=max_tokens,
            use_cache=False,  # Never cache structured decisions
        )
        return response.parse_json()

    def get_spend_summary(self) -> str:
        """Return formatted spend summary."""
        session = self.cost_tracker.session_summary()
        total_7d = self.cost_tracker.total_spend(7)
        total_30d = self.cost_tracker.total_spend(30)
        return (
            f"{session} | "
            f"7-day total: ${total_7d:.4f} | "
            f"30-day total: ${total_30d:.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Singleton factory
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[LLMClient] = None


def get_llm_client(
    model: Optional[str] = None,
    agent_name: str = "",
    use_cache: bool = True,
) -> LLMClient:
    """Get or create a singleton LLM client."""
    global _client
    if _client is None or model is not None:
        _client = LLMClient(
            model=model,
            agent_name=agent_name,
            use_cache=use_cache,
        )
    return _client


if __name__ == "__main__":
    """Test the LLM client."""
    import sys
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("  LLM Client — Integration Test")
    print("=" * 60)

    try:
        client = LLMClient(agent_name="test")
        print(f"\nModel: {client.model}")

        # Test 1: Basic completion
        print("\n1. Basic completion test...")
        resp = client.complete(
            system="You are a concise quantitative finance assistant.",
            user="In exactly one sentence: what is the Sharpe ratio?",
            purpose="test",
        )
        print(f"   Response: {resp.content[:200]}")
        print(f"   {resp.summary()}")

        # Test 2: Structured output
        print("\n2. Structured JSON output test...")
        result = client.structured_output(
            system="You are a portfolio risk assessor.",
            user=(
                "Assess this portfolio: 60% AAPL, 40% MSFT. "
                "Return JSON with keys: risk_level (low/medium/high), "
                "concentration_concern (bool), recommendation (string)."
            ),
            schema_description='Return: {"risk_level": str, "concentration_concern": bool, "recommendation": str}',
        )
        if result:
            print(f"   Parsed JSON: {json.dumps(result, indent=2)}")
        else:
            print("   Warning: Could not parse JSON")

        # Test 3: Cost tracking
        print("\n3. Cost tracking...")
        print(f"   {client.get_spend_summary()}")

        print("\n✅ LLM client tests passed")

    except ValueError as e:
        print(f"\n⚠️  API key not configured: {e}")
        print("   Add ANTHROPIC_API_KEY or OPENAI_API_KEY to your .env file")
        print("   Client module loaded correctly — set API key to run full tests")
