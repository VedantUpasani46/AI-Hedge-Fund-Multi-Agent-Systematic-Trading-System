"""
AI Hedge Fund — Part 3: RAG & Data Intelligence
=================================================
document_models.py — Financial Document Data Structures

Every piece of text that flows through the RAG system
is represented as one of these document types.

The hierarchy:
    RawDocument     — Raw text from any source (SEC, news, transcript)
    DocumentChunk   — A slice of a RawDocument after chunking
    EmbeddedChunk   — A chunk with its vector embedding attached
    RetrievedChunk  — A chunk returned from a similarity search

Why chunking matters:
    LLM context windows are finite. A 10-K filing is 80,000+ words.
    You cannot fit it all in a single LLM call.
    Chunking splits documents into ~500-word overlapping segments.
    The RAG system retrieves only the most relevant chunks for each query.
    This is how you give the LLM access to unlimited document history.

Supported document types:
    SEC_10K          Annual report (risk factors, MD&A, financials)
    SEC_10Q          Quarterly report
    SEC_8K           Material event disclosure (earnings, acquisitions)
    EARNINGS_CALL    Earnings call transcript (CEO/CFO commentary)
    NEWS_ARTICLE     News article about a company or sector
    ANALYST_REPORT   Sell-side research note
    MACRO_REPORT     Fed minutes, economic reports
    PRESS_RELEASE    Company press release
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Any, Dict, List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Document types
# ─────────────────────────────────────────────────────────────────────────────

class DocumentType(str, Enum):
    SEC_10K        = "SEC_10K"
    SEC_10Q        = "SEC_10Q"
    SEC_8K         = "SEC_8K"
    EARNINGS_CALL  = "EARNINGS_CALL"
    NEWS_ARTICLE   = "NEWS_ARTICLE"
    ANALYST_REPORT = "ANALYST_REPORT"
    MACRO_REPORT   = "MACRO_REPORT"
    PRESS_RELEASE  = "PRESS_RELEASE"
    OTHER          = "OTHER"


class DocumentSentiment(str, Enum):
    VERY_POSITIVE = "VERY_POSITIVE"
    POSITIVE      = "POSITIVE"
    NEUTRAL       = "NEUTRAL"
    NEGATIVE      = "NEGATIVE"
    VERY_NEGATIVE = "VERY_NEGATIVE"
    UNKNOWN       = "UNKNOWN"


# ─────────────────────────────────────────────────────────────────────────────
# Raw document (before chunking)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    """
    A single financial document before processing.

    Created by document fetchers (SEC EDGAR, news scraper, etc.)
    and passed to the chunking pipeline.
    """
    doc_id:        str               # Unique identifier
    ticker:        str               # Primary ticker (e.g. "AAPL")
    doc_type:      DocumentType
    title:         str
    text:          str               # Full raw text
    source_url:    str = ""          # Where this was fetched from
    published_at:  Optional[datetime] = None
    filing_date:   Optional[date] = None
    period_of_report: Optional[date] = None   # For SEC filings: quarter/year end
    author:        str = ""
    metadata:      Dict[str, Any] = field(default_factory=dict)
    fetched_at:    datetime = field(default_factory=datetime.now)

    @classmethod
    def make_id(cls, ticker: str, doc_type: str, title: str, date_str: str) -> str:
        """Deterministic ID from content fingerprint."""
        raw = f"{ticker}_{doc_type}_{title}_{date_str}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16].upper()

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def char_count(self) -> int:
        return len(self.text)

    def to_dict(self) -> dict:
        return {
            "doc_id":           self.doc_id,
            "ticker":           self.ticker,
            "doc_type":         self.doc_type.value,
            "title":            self.title,
            "word_count":       self.word_count,
            "source_url":       self.source_url,
            "published_at":     self.published_at.isoformat() if self.published_at else None,
            "filing_date":      self.filing_date.isoformat() if self.filing_date else None,
            "fetched_at":       self.fetched_at.isoformat(),
            "metadata":         self.metadata,
        }

    def __repr__(self):
        return (
            f"RawDocument({self.ticker} | {self.doc_type.value} | "
            f"'{self.title[:50]}' | {self.word_count:,} words)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Document chunk (after chunking)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DocumentChunk:
    """
    A single chunk of a RawDocument.

    The atomic unit stored in the vector database.
    Each chunk is ~300-600 words with overlap to preserve context.
    """
    chunk_id:     str              # Unique chunk identifier
    doc_id:       str              # Parent document ID
    ticker:       str
    doc_type:     DocumentType
    title:        str              # Parent document title
    text:         str              # The actual chunk text
    chunk_index:  int              # Position in document (0-indexed)
    total_chunks: int              # Total chunks in parent document
    char_start:   int              # Character offset in original doc
    char_end:     int              # Character offset end
    published_at: Optional[datetime] = None
    filing_date:  Optional[date] = None
    metadata:     Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def make_id(cls, doc_id: str, chunk_index: int) -> str:
        return f"{doc_id}_C{chunk_index:04d}"

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def is_first_chunk(self) -> bool:
        return self.chunk_index == 0

    @property
    def is_last_chunk(self) -> bool:
        return self.chunk_index == self.total_chunks - 1

    @property
    def position_pct(self) -> float:
        """Where in the document this chunk sits (0=start, 1=end)."""
        if self.total_chunks <= 1:
            return 0.5
        return self.chunk_index / (self.total_chunks - 1)

    def to_chromadb_metadata(self) -> dict:
        """Convert to ChromaDB metadata dict (must be str/int/float/bool only)."""
        return {
            "chunk_id":     self.chunk_id,
            "doc_id":       self.doc_id,
            "ticker":       self.ticker,
            "doc_type":     self.doc_type.value,
            "title":        self.title[:200],
            "chunk_index":  self.chunk_index,
            "total_chunks": self.total_chunks,
            "word_count":   self.word_count,
            "published_at": self.published_at.isoformat() if self.published_at else "",
            "filing_date":  self.filing_date.isoformat() if self.filing_date else "",
            "position_pct": round(self.position_pct, 3),
        }

    def __repr__(self):
        return (
            f"Chunk({self.chunk_id} | {self.ticker} | "
            f"{self.doc_type.value} | "
            f"chunk {self.chunk_index+1}/{self.total_chunks} | "
            f"{self.word_count} words)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Retrieved chunk (from vector search)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A chunk returned from a vector similarity search.

    Includes the similarity score and can be formatted
    for inclusion in an LLM prompt.
    """
    chunk:          DocumentChunk
    similarity:     float          # Cosine similarity (0-1, higher = more relevant)
    rank:           int            # Position in results list (1-indexed)

    def to_context_string(self, include_metadata: bool = True) -> str:
        """Format for inclusion in LLM context."""
        lines = []
        if include_metadata:
            dt = self.chunk.published_at or self.chunk.filing_date
            date_str = dt.strftime("%Y-%m-%d") if dt else "Date unknown"
            lines.append(
                f"[Source: {self.chunk.ticker} {self.chunk.doc_type.value} | "
                f"{date_str} | Relevance: {self.similarity:.2f}]"
            )
        lines.append(self.chunk.text)
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"RetrievedChunk(rank={self.rank} sim={self.similarity:.3f} | "
            f"{self.chunk})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# RAG query and result
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RAGQuery:
    """A query to the RAG system."""
    query_text:      str
    ticker_filter:   Optional[List[str]] = None     # Restrict to these tickers
    doc_type_filter: Optional[List[DocumentType]] = None
    date_from:       Optional[datetime] = None
    date_to:         Optional[datetime] = None
    top_k:           int = 5                        # Number of chunks to retrieve
    min_similarity:  float = 0.30                   # Minimum relevance threshold


@dataclass
class RAGResult:
    """Result of a RAG query — retrieved chunks + LLM synthesis."""
    query:           RAGQuery
    retrieved:       List[RetrievedChunk]
    synthesis:       str              # LLM-generated answer using retrieved context
    sources_used:    List[str]        # doc_ids of sources cited
    total_tokens:    int = 0
    cost_usd:        float = 0.0
    latency_ms:      float = 0.0
    timestamp:       datetime = field(default_factory=datetime.now)

    @property
    def num_sources(self) -> int:
        return len(self.retrieved)

    def to_dict(self) -> dict:
        return {
            "query":       self.query.query_text,
            "synthesis":   self.synthesis,
            "num_sources": self.num_sources,
            "sources":     [r.chunk.doc_id for r in self.retrieved],
            "cost_usd":    self.cost_usd,
            "latency_ms":  self.latency_ms,
        }
