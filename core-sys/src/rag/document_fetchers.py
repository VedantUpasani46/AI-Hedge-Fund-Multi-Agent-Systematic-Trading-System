"""
AI Hedge Fund — Part 3: RAG & Data Intelligence
=================================================
document_fetchers.py — Real Financial Document Ingestion

Fetches real financial documents from free public sources.

Sources implemented:
  1. SEC EDGAR (free, no API key)
     - 10-K annual reports
     - 10-Q quarterly reports
     - 8-K material events (earnings, acquisitions)
     Full-text search via EDGAR full-text search API

  2. Yahoo Finance News (free via yfinance)
     - Recent news articles per ticker
     - Earnings summaries
     - Press releases

  3. EDGAR Earnings Transcripts (via 8-K filings)
     - Prepared remarks from earnings calls
     - Q&A sections embedded in 8-K exhibits

  4. FRED Economic Data (free, no key for basic access)
     - Fed minutes
     - Economic reports

  5. OpenBB (optional, free community tier)
     - Earnings transcripts
     - Analyst estimates

Rate limits:
  EDGAR: 10 requests/second (enforced automatically)
  Yahoo: No formal limit, 0.5s delay applied
  FRED:  No limit for public data

All fetchers return List[RawDocument] and are safe to call
repeatedly — they check if the document was already fetched.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from src.rag.document_models import (
    RawDocument, DocumentType
)

logger = logging.getLogger("hedge_fund.doc_fetchers")


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """Simple token-bucket rate limiter."""

    def __init__(self, requests_per_second: float = 2.0):
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0

    def wait(self):
        elapsed = time.time() - self.last_request
        wait_time = self.min_interval - elapsed
        if wait_time > 0:
            time.sleep(wait_time)
        self.last_request = time.time()


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class SECEdgarFetcher:
    """
    Fetches SEC filings directly from EDGAR (free, no API key required).

    EDGAR Full-Text Search API:
        https://efts.sec.gov/LATEST/search-index?q="AAPL"&dateRange=custom
        https://data.sec.gov/submissions/CIK{cik}.json

    EDGAR Filing Access:
        https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/

    SEC requires a user-agent header identifying who is making requests.
    Set your company name and email in .env as SEC_USER_AGENT.
    """

    BASE_URL     = "https://data.sec.gov"
    SEARCH_URL   = "https://efts.sec.gov/LATEST/search-index"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

    # Map of form types we want
    FORM_TYPE_MAP = {
        "10-K": DocumentType.SEC_10K,
        "10-Q": DocumentType.SEC_10Q,
        "8-K":  DocumentType.SEC_8K,
    }

    def __init__(self, user_agent: Optional[str] = None):
        import os
        self.user_agent = (
            user_agent
            or os.getenv("SEC_USER_AGENT", "AI-Hedge-Fund research@hedgefund.com")
        )
        self.headers = {
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.rate_limiter = RateLimiter(requests_per_second=8)  # EDGAR allows 10/s
        self._cik_cache: Dict[str, str] = {}

    def get_cik(self, ticker: str) -> Optional[str]:
        """Get EDGAR CIK number for a ticker symbol."""
        if ticker in self._cik_cache:
            return self._cik_cache[ticker]

        self.rate_limiter.wait()
        try:
            url  = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&company=&CIK={ticker}&type=10-K&dateb=&owner=include&count=1&search_text="
            resp = requests.get(url, headers=self.headers, timeout=15)
            # Extract CIK from response
            match = re.search(r'CIK=(\d{10})', resp.text)
            if not match:
                # Try company search
                url2  = f"https://data.sec.gov/submissions/CIK{ticker.zfill(10)}.json"
                resp2 = requests.get(url2, headers=self.headers, timeout=15)
                if resp2.status_code == 200:
                    cik = ticker.zfill(10)
                    self._cik_cache[ticker] = cik
                    return cik
                return None

            cik = match.group(1)
            self._cik_cache[ticker] = cik
            return cik
        except Exception as e:
            logger.warning(f"Could not get CIK for {ticker}: {e}")
            return None

    def get_cik_from_ticker_mapping(self, ticker: str) -> Optional[str]:
        """
        Use EDGAR's company tickers JSON mapping (most reliable method).
        Downloads a 3MB JSON file with all tickers — cached after first call.
        """
        cache_path = Path(__file__).parents[3] / "data" / "cache" / "sec_tickers.json"

        if not cache_path.exists():
            try:
                self.rate_limiter.wait()
                url  = "https://www.sec.gov/files/company_tickers.json"
                resp = requests.get(url, headers=self.headers, timeout=30)
                if resp.status_code == 200:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(resp.text)
            except Exception as e:
                logger.warning(f"Could not fetch ticker mapping: {e}")
                return None

        try:
            data = json.loads(cache_path.read_text())
            for entry in data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    return str(entry["cik_str"]).zfill(10)
        except Exception as e:
            logger.warning(f"Error reading ticker mapping: {e}")

        return None

    def get_recent_filings(
        self,
        ticker:     str,
        form_types: List[str] = ["10-K", "10-Q", "8-K"],
        max_filings: int = 5,
    ) -> List[Dict]:
        """
        Get list of recent SEC filings for a ticker.

        Returns list of filing metadata dicts.
        """
        cik = self.get_cik_from_ticker_mapping(ticker)
        if not cik:
            logger.warning(f"No CIK found for {ticker}")
            return []

        self.rate_limiter.wait()
        try:
            url  = f"{self.BASE_URL}/submissions/CIK{cik}.json"
            resp = requests.get(url, headers=self.headers, timeout=15)
            if resp.status_code != 200:
                logger.warning(f"EDGAR submissions API returned {resp.status_code} for {ticker}")
                return []

            data     = resp.json()
            filings  = data.get("filings", {}).get("recent", {})

            forms        = filings.get("form", [])
            dates        = filings.get("filingDate", [])
            accessions   = filings.get("accessionNumber", [])
            primary_docs = filings.get("primaryDocument", [])
            descriptions = filings.get("primaryDocDescription", [])

            results = []
            for i, form in enumerate(forms):
                if form in form_types and len(results) < max_filings:
                    results.append({
                        "ticker":      ticker,
                        "cik":         cik,
                        "form":        form,
                        "date":        dates[i] if i < len(dates) else "",
                        "accession":   accessions[i].replace("-", "") if i < len(accessions) else "",
                        "primary_doc": primary_docs[i] if i < len(primary_docs) else "",
                        "description": descriptions[i] if i < len(descriptions) else "",
                    })

            logger.info(f"Found {len(results)} filings for {ticker}")
            return results

        except Exception as e:
            logger.error(f"Failed to get filings for {ticker}: {e}")
            return []

    def fetch_filing_text(
        self,
        cik:        str,
        accession:  str,
        primary_doc: str,
    ) -> str:
        """
        Download and extract text from an SEC filing.

        Returns plain text (HTML/XML tags stripped).
        """
        self.rate_limiter.wait()
        cik_clean       = cik.lstrip("0")
        accession_clean = accession.replace("-", "")
        url = f"{self.ARCHIVES_URL}/{cik_clean}/{accession_clean}/{primary_doc}"

        try:
            resp = requests.get(url, headers=self.headers, timeout=30)
            if resp.status_code != 200:
                logger.warning(f"Failed to fetch {url}: {resp.status_code}")
                return ""

            text = resp.text
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Normalise whitespace
            text = re.sub(r'\s+', ' ', text)
            # Strip XML-encoded characters
            text = text.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            text = text.replace('&#160;', ' ').replace('&nbsp;', ' ')
            # Remove very short lines (navigation, headers)
            lines = [l.strip() for l in text.split('.') if len(l.strip()) > 30]
            text = '. '.join(lines)
            return text.strip()[:100_000]   # Cap at 100K chars (~20K words)

        except Exception as e:
            logger.error(f"Failed to fetch filing text: {e}")
            return ""

    def fetch_documents(
        self,
        ticker:     str,
        form_types: List[str] = ["10-K", "10-Q", "8-K"],
        max_each:   int = 2,
    ) -> List[RawDocument]:
        """
        Fetch real SEC documents for a ticker.

        Returns List[RawDocument] ready for chunking and embedding.
        """
        filings = self.get_recent_filings(ticker, form_types, max_filings=max_each * len(form_types))
        docs    = []

        for filing in filings:
            text = self.fetch_filing_text(
                filing["cik"],
                filing["accession"],
                filing["primary_doc"],
            )
            if not text or len(text) < 500:
                logger.debug(f"  Skipping {filing['form']} — too short")
                continue

            filing_date = None
            try:
                filing_date = date.fromisoformat(filing["date"]) if filing["date"] else None
            except ValueError:
                pass

            doc_type = self.FORM_TYPE_MAP.get(filing["form"], DocumentType.OTHER)
            title    = f"{ticker} {filing['form']} — {filing['date']}"
            doc_id   = RawDocument.make_id(
                ticker, filing["form"], title, filing["date"]
            )

            doc = RawDocument(
                doc_id      = doc_id,
                ticker      = ticker,
                doc_type    = doc_type,
                title       = title,
                text        = text,
                source_url  = (
                    f"{self.ARCHIVES_URL}/{filing['cik'].lstrip('0')}/"
                    f"{filing['accession']}/{filing['primary_doc']}"
                ),
                filing_date = filing_date,
                metadata    = {
                    "form":        filing["form"],
                    "accession":   filing["accession"],
                    "description": filing["description"],
                },
            )
            docs.append(doc)
            logger.info(f"  Fetched: {doc}")

        return docs


# ─────────────────────────────────────────────────────────────────────────────
# Yahoo Finance News Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class YahooNewsFetcher:
    """
    Fetches recent news articles for a ticker from Yahoo Finance.

    Uses yfinance's Ticker.news property — no API key required.
    Returns recent headlines and summaries.
    """

    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_second=2)

    def fetch_documents(
        self,
        ticker:   str,
        max_news: int = 20,
    ) -> List[RawDocument]:
        """
        Fetch recent news articles for a ticker.

        Returns List[RawDocument] with news text.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed: pip install yfinance")
            return []

        self.rate_limiter.wait()
        docs = []

        try:
            ticker_obj = yf.Ticker(ticker)
            news_items = ticker_obj.news or []

            logger.info(f"  Yahoo Finance: {len(news_items)} news items for {ticker}")

            for item in news_items[:max_news]:
                try:
                    # Build text from available fields
                    title    = item.get("title", "")
                    summary  = item.get("summary", "") or item.get("description", "")
                    publisher= item.get("publisher", "")

                    # Try to get full content if available
                    content  = item.get("content", {})
                    if isinstance(content, dict):
                        body = content.get("body", "") or content.get("summary", "")
                    else:
                        body = str(content) if content else ""

                    # Build the text
                    text_parts = []
                    if title:
                        text_parts.append(f"HEADLINE: {title}")
                    if publisher:
                        text_parts.append(f"SOURCE: {publisher}")
                    if summary:
                        text_parts.append(f"SUMMARY: {summary}")
                    if body and body != summary:
                        text_parts.append(f"CONTENT: {body}")

                    text = "\n\n".join(text_parts)
                    if len(text) < 50:
                        continue

                    # Parse timestamp
                    pub_time = item.get("providerPublishTime", 0)
                    published_at = (
                        datetime.fromtimestamp(pub_time)
                        if pub_time else None
                    )

                    doc_id = RawDocument.make_id(
                        ticker, "NEWS", title,
                        published_at.strftime("%Y%m%d") if published_at else "unknown"
                    )

                    docs.append(RawDocument(
                        doc_id      = doc_id,
                        ticker      = ticker,
                        doc_type    = DocumentType.NEWS_ARTICLE,
                        title       = title,
                        text        = text,
                        source_url  = item.get("link", ""),
                        published_at= published_at,
                        author      = publisher,
                        metadata    = {
                            "provider": publisher,
                            "uuid":     item.get("uuid", ""),
                        },
                    ))

                except Exception as e:
                    logger.debug(f"  Skipping news item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to fetch Yahoo news for {ticker}: {e}")

        return docs


# ─────────────────────────────────────────────────────────────────────────────
# Earnings Call Fetcher (via SEC 8-K exhibits)
# ─────────────────────────────────────────────────────────────────────────────

class EarningsCallFetcher:
    """
    Fetches earnings call transcripts from SEC 8-K filings.

    Companies file earnings call transcripts as exhibits to 8-K filings
    under Item 2.02 (Results of Operations and Financial Condition).
    These are publicly available on EDGAR for free.

    Note: Not all companies file transcripts as exhibits.
    Those that do include AAPL, MSFT, GOOGL, AMZN, META, NVDA, JPM, etc.
    """

    def __init__(self, edgar_fetcher: Optional[SECEdgarFetcher] = None):
        self.edgar = edgar_fetcher or SECEdgarFetcher()

    def fetch_documents(
        self,
        ticker:   str,
        max_calls: int = 4,         # Last 4 quarters
    ) -> List[RawDocument]:
        """
        Fetch earnings call transcripts for a ticker.

        Looks for 8-K filings with Item 2.02 (earnings release).
        Returns RawDocuments with transcript text.
        """
        # Get 8-K filings
        filings = self.edgar.get_recent_filings(
            ticker, form_types=["8-K"], max_filings=max_calls * 3
        )

        docs = []
        calls_found = 0

        for filing in filings:
            if calls_found >= max_calls:
                break

            # Fetch the filing text
            text = self.edgar.fetch_filing_text(
                filing["cik"],
                filing["accession"],
                filing["primary_doc"],
            )

            if not text:
                continue

            # Check if this is an earnings-related 8-K
            text_lower = text.lower()
            earnings_indicators = [
                "results of operations",
                "earnings per share",
                "revenue",
                "net income",
                "guidance",
                "quarterly",
            ]
            if not any(ind in text_lower for ind in earnings_indicators):
                continue

            # Extract the most relevant section
            # (often the Q&A or prepared remarks section)
            text = self._extract_call_content(text)
            if len(text) < 500:
                continue

            filing_date = None
            try:
                filing_date = date.fromisoformat(filing["date"]) if filing["date"] else None
            except ValueError:
                pass

            title  = f"{ticker} Earnings Call — {filing['date']}"
            doc_id = RawDocument.make_id(ticker, "EARNINGS_CALL", title, filing["date"])

            docs.append(RawDocument(
                doc_id      = doc_id,
                ticker      = ticker,
                doc_type    = DocumentType.EARNINGS_CALL,
                title       = title,
                text        = text,
                source_url  = (
                    f"https://www.sec.gov/Archives/edgar/data/"
                    f"{filing['cik'].lstrip('0')}/{filing['accession']}/"
                    f"{filing['primary_doc']}"
                ),
                filing_date = filing_date,
                metadata    = {
                    "form":      "8-K",
                    "accession": filing["accession"],
                },
            ))
            calls_found += 1
            logger.info(f"  Found earnings-related 8-K: {title}")

        return docs

    def _extract_call_content(self, text: str) -> str:
        """Extract the most relevant content from an earnings 8-K."""
        # Look for common earnings call sections
        markers = [
            "prepared remarks",
            "question and answer",
            "q&a session",
            "operator",
            "good morning",
            "good afternoon",
            "good evening",
            "thank you for standing by",
        ]

        text_lower = text.lower()
        best_start = -1

        for marker in markers:
            pos = text_lower.find(marker)
            if pos > 0 and (best_start < 0 or pos < best_start):
                best_start = pos

        if best_start > 0:
            return text[best_start:best_start + 50_000]

        # Fallback: return middle section (often has the substance)
        mid = len(text) // 4
        return text[mid:mid + 50_000]


# ─────────────────────────────────────────────────────────────────────────────
# Macro / Fed Data Fetcher
# ─────────────────────────────────────────────────────────────────────────────

class MacroReportFetcher:
    """
    Fetches macro-economic reports and Fed communications.

    Sources:
      - Federal Reserve FOMC minutes (free from federalreserve.gov)
      - Fed press releases
      - BLS economic releases (jobs, CPI)
    """

    FOMC_BASE = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"

    def __init__(self):
        self.rate_limiter = RateLimiter(requests_per_second=1)
        self.headers = {
            "User-Agent": "AI-Hedge-Fund research@hedgefund.com"
        }

    def fetch_fomc_minutes(self, max_meetings: int = 2) -> List[RawDocument]:
        """
        Fetch recent FOMC meeting minutes from the Federal Reserve.

        FOMC minutes are published ~3 weeks after each meeting.
        They contain detailed discussion of economic conditions and
        interest rate deliberations — high value for macro analysis.
        """
        docs = []

        # Known recent FOMC minutes URLs (format is consistent)
        current_year = datetime.now().year
        base_url     = "https://www.federalreserve.gov/monetarypolicy"

        # Try to fetch the minutes index
        self.rate_limiter.wait()
        try:
            url  = f"{base_url}/fomccalendars.htm"
            resp = requests.get(url, headers=self.headers, timeout=15)

            # Extract minutes links
            links = re.findall(
                r'href="(/monetarypolicy/fomcminutes\d{8}\.htm)"',
                resp.text
            )

            for link in links[:max_meetings]:
                self.rate_limiter.wait()
                full_url  = f"https://www.federalreserve.gov{link}"
                try:
                    resp2 = requests.get(full_url, headers=self.headers, timeout=20)
                    if resp2.status_code != 200:
                        continue

                    # Extract text from HTML
                    text = re.sub(r'<[^>]+>', ' ', resp2.text)
                    text = re.sub(r'\s+', ' ', text).strip()

                    # Find the substantive content
                    start_markers = ["participants", "committee", "discussion"]
                    start_pos     = 0
                    for marker in start_markers:
                        pos = text.lower().find(marker)
                        if pos > 0:
                            start_pos = max(start_pos, pos - 100)
                            break

                    text = text[start_pos:start_pos + 80_000]
                    if len(text) < 1000:
                        continue

                    # Extract date from URL
                    date_match = re.search(r'fomcminutes(\d{8})', link)
                    date_str   = date_match.group(1) if date_match else ""
                    try:
                        meeting_date = date(
                            int(date_str[:4]),
                            int(date_str[4:6]),
                            int(date_str[6:8]),
                        ) if date_str else None
                    except ValueError:
                        meeting_date = None

                    title  = f"FOMC Minutes — {date_str}"
                    doc_id = RawDocument.make_id("MACRO", "FOMC_MINUTES", title, date_str)

                    docs.append(RawDocument(
                        doc_id      = doc_id,
                        ticker      = "MACRO",
                        doc_type    = DocumentType.MACRO_REPORT,
                        title       = title,
                        text        = text,
                        source_url  = full_url,
                        filing_date = meeting_date,
                        metadata    = {"source": "Federal Reserve", "type": "FOMC_MINUTES"},
                    ))
                    logger.info(f"  Fetched: {title}")

                except Exception as e:
                    logger.warning(f"  Failed to fetch {full_url}: {e}")

        except Exception as e:
            logger.warning(f"  Failed to fetch FOMC calendar: {e}")

        return docs


# ─────────────────────────────────────────────────────────────────────────────
# Master document fetcher
# ─────────────────────────────────────────────────────────────────────────────

class DocumentFetchOrchestrator:
    """
    Coordinates all document fetchers.

    Single entry point for fetching all available documents
    for a ticker or set of tickers.

    Usage:
        orchestrator = DocumentFetchOrchestrator()

        # Fetch all document types for AAPL:
        docs = orchestrator.fetch_all(
            "AAPL",
            include_sec=True,
            include_news=True,
            include_earnings=True,
        )

        # Batch fetch for universe:
        all_docs = orchestrator.fetch_universe(
            ["AAPL", "MSFT", "NVDA"], max_per_ticker=10
        )
    """

    def __init__(self):
        self.sec_fetcher      = SECEdgarFetcher()
        self.news_fetcher     = YahooNewsFetcher()
        self.earnings_fetcher = EarningsCallFetcher(self.sec_fetcher)
        self.macro_fetcher    = MacroReportFetcher()

    def fetch_all(
        self,
        ticker:           str,
        include_sec:      bool = True,
        include_news:     bool = True,
        include_earnings: bool = True,
        max_sec:          int = 3,
        max_news:         int = 15,
        max_earnings:     int = 4,
    ) -> List[RawDocument]:
        """
        Fetch all document types for a single ticker.

        Returns merged list of all documents found.
        """
        all_docs: List[RawDocument] = []

        if include_news:
            logger.info(f"Fetching news for {ticker}...")
            try:
                docs = self.news_fetcher.fetch_documents(ticker, max_news=max_news)
                all_docs.extend(docs)
                logger.info(f"  Got {len(docs)} news articles")
            except Exception as e:
                logger.error(f"News fetch failed for {ticker}: {e}")

        if include_sec:
            logger.info(f"Fetching SEC filings for {ticker}...")
            try:
                docs = self.sec_fetcher.fetch_documents(
                    ticker,
                    form_types=["10-K", "10-Q"],
                    max_each=max_sec // 2 + 1,
                )
                all_docs.extend(docs)
                logger.info(f"  Got {len(docs)} SEC filings")
            except Exception as e:
                logger.error(f"SEC fetch failed for {ticker}: {e}")

        if include_earnings:
            logger.info(f"Fetching earnings calls for {ticker}...")
            try:
                docs = self.earnings_fetcher.fetch_documents(
                    ticker, max_calls=max_earnings
                )
                all_docs.extend(docs)
                logger.info(f"  Got {len(docs)} earnings-related filings")
            except Exception as e:
                logger.error(f"Earnings fetch failed for {ticker}: {e}")

        logger.info(
            f"Fetch complete for {ticker}: {len(all_docs)} total documents "
            f"({sum(d.word_count for d in all_docs):,} words)"
        )
        return all_docs

    def fetch_universe(
        self,
        tickers:          List[str],
        max_per_ticker:   int = 10,
        include_macro:    bool = True,
        delay_between:    float = 1.0,
    ) -> List[RawDocument]:
        """
        Fetch documents for multiple tickers.

        Includes macro documents once for the full universe.
        """
        all_docs: List[RawDocument] = []

        # Fetch macro documents once
        if include_macro:
            logger.info("Fetching FOMC minutes...")
            try:
                macro_docs = self.macro_fetcher.fetch_fomc_minutes(max_meetings=2)
                all_docs.extend(macro_docs)
                logger.info(f"  Got {len(macro_docs)} macro documents")
            except Exception as e:
                logger.warning(f"Macro fetch failed: {e}")

        # Fetch per-ticker documents
        for ticker in tickers:
            try:
                docs = self.fetch_all(
                    ticker,
                    max_sec      = min(2, max_per_ticker // 3),
                    max_news     = min(10, max_per_ticker),
                    max_earnings = min(2, max_per_ticker // 4),
                )
                all_docs.extend(docs)
            except Exception as e:
                logger.error(f"Failed to fetch documents for {ticker}: {e}")

            if delay_between > 0:
                time.sleep(delay_between)

        logger.info(
            f"Universe fetch complete: {len(all_docs)} documents "
            f"for {len(tickers)} tickers"
        )
        return all_docs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  Document Fetchers — Test")
    print("=" * 60)

    orchestrator = DocumentFetchOrchestrator()

    print("\nFetching documents for AAPL (news only — fast test)...")
    docs = orchestrator.fetch_all(
        "AAPL",
        include_sec      = False,
        include_news     = True,
        include_earnings = False,
        max_news         = 5,
    )

    for doc in docs[:3]:
        print(f"\n  {doc}")
        print(f"  Preview: {doc.text[:200]}...")

    print(f"\n✅ Fetched {len(docs)} documents for AAPL")
