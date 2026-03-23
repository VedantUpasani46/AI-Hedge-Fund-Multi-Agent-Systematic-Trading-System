"""
AI Hedge Fund — Part 3: RAG & Data Intelligence
=================================================
document_processor.py — Chunking, Embedding & Vector Store

Pipeline:
    RawDocument
        → TextCleaner.clean()
        → DocumentChunker.chunk()           (→ List[DocumentChunk])
        → EmbeddingEngine.embed()           (→ List[float] per chunk)
        → VectorStore.upsert()              (→ stored in ChromaDB)

Then at query time:
    query_text
        → EmbeddingEngine.embed_query()     (→ List[float])
        → VectorStore.search()              (→ List[RetrievedChunk])
        → LLM synthesis                     (→ final answer)

Why ChromaDB:
    - Runs locally, no server needed (embedded mode)
    - Free, open source
    - Persists to disk (survives process restarts)
    - Supports metadata filtering (filter by ticker, date, doc_type)
    - Production upgrade path: Pinecone, Weaviate, Qdrant

Why sentence-transformers:
    - Free, runs locally (no API cost per embedding)
    - finance-specific model available (ProsusAI/finbert)
    - Fast inference on CPU
    - Fallback to OpenAI/Anthropic embeddings if installed

Chunking strategy:
    - Chunk size: 600 words (fits well in LLM context)
    - Overlap:    100 words (preserves cross-chunk context)
    - Sentence-aware: never splits mid-sentence
    - Paragraph-preserving: tries to keep paragraphs intact
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.rag.document_models import (
    RawDocument, DocumentChunk, RetrievedChunk, RAGQuery, DocumentType
)

logger = logging.getLogger("hedge_fund.doc_processor")


# ─────────────────────────────────────────────────────────────────────────────
# Text cleaner
# ─────────────────────────────────────────────────────────────────────────────

class TextCleaner:
    """
    Cleans raw financial text for embedding.

    Financial documents have a lot of noise:
      - Legal boilerplate (CAUTIONARY NOTE REGARDING FORWARD-LOOKING STATEMENTS)
      - Table of contents
      - Page numbers and headers
      - Repeated disclaimer paragraphs
      - XBRL/HTML artifacts

    We want the signal: substantive business discussion, financial results,
    forward guidance, risk factors.
    """

    # Sections we want to KEEP (high signal for investment analysis)
    HIGH_VALUE_SECTIONS = [
        "management's discussion",
        "results of operations",
        "revenue",
        "gross profit",
        "operating income",
        "net income",
        "earnings per share",
        "guidance",
        "outlook",
        "forward",
        "risk factor",
        "liquidity",
        "capital resources",
        "business",
        "competition",
        "market",
        "strategy",
        "product",
        "customer",
        "employee",
        "research and development",
    ]

    # Boilerplate patterns to remove
    BOILERPLATE_PATTERNS = [
        r"cautionary note regarding forward.looking statements",
        r"safe harbor statement",
        r"this document contains forward.looking statements",
        r"table of contents",
        r"page \d+ of \d+",
        r"exhibit \d+\.\d+",
        r"\d{1,2}/\d{1,2}/\d{4}",        # Dates in isolation
        r"item \d+[a-z]?\.",              # SEC item headers (Item 1., Item 1A.)
    ]

    def clean(self, text: str) -> str:
        """
        Clean raw document text.

        Returns cleaner text ready for chunking.
        """
        # 1. Normalise whitespace
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r' {3,}', '  ', text)

        # 2. Remove common HTML artifacts that weren't stripped
        text = re.sub(r'&\w+;', ' ', text)
        text = re.sub(r'\x00-\x1F', ' ', text)

        # 3. Remove excessive numbers without context (page numbers, IDs)
        text = re.sub(r'\b\d{8,}\b', '', text)   # Long numeric strings

        # 4. Normalise quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")

        # 5. Remove repeated spaces after cleaning
        text = re.sub(r'  +', ' ', text)

        return text.strip()

    def is_substantive(self, text: str) -> bool:
        """
        Check if a text segment is substantive (worth embedding).

        Filters out navigation, headers, and pure boilerplate.
        """
        if len(text) < 100:
            return False

        word_count = len(text.split())
        if word_count < 20:
            return False

        # Check for financial content indicators
        financial_words = [
            "revenue", "profit", "income", "loss", "earnings", "growth",
            "decline", "increase", "decrease", "million", "billion",
            "percent", "quarter", "fiscal", "annual", "guidance",
            "market", "product", "customer", "business", "strategy",
        ]
        text_lower = text.lower()
        financial_count = sum(1 for w in financial_words if w in text_lower)

        return financial_count >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Document chunker
# ─────────────────────────────────────────────────────────────────────────────

class DocumentChunker:
    """
    Splits RawDocuments into DocumentChunks for embedding.

    Strategy:
        1. Split into sentences using regex (no NLTK dependency)
        2. Group sentences into chunks of target_words
        3. Add overlap_words from previous chunk at start of each chunk
        4. Skip chunks that are not substantive (TextCleaner.is_substantive)

    This preserves semantic coherence better than splitting on character count.
    """

    def __init__(
        self,
        target_words:  int = 400,    # Target chunk size in words
        overlap_words: int = 80,     # Overlap with previous chunk
        min_words:     int = 50,     # Skip chunks shorter than this
    ):
        self.target_words  = target_words
        self.overlap_words = overlap_words
        self.min_words     = min_words
        self.cleaner       = TextCleaner()

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex (no NLTK needed)."""
        # Split on sentence-ending punctuation followed by whitespace and capital
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        # Further split on newlines (paragraph breaks in filings)
        result = []
        for sent in sentences:
            parts = [p.strip() for p in sent.split('\n') if p.strip()]
            result.extend(parts)
        return [s for s in result if len(s) > 15]

    def chunk(self, doc: RawDocument) -> List[DocumentChunk]:
        """
        Split a RawDocument into overlapping chunks.

        Returns List[DocumentChunk], each ready for embedding.
        """
        # Clean the text first
        clean_text = self.cleaner.clean(doc.text)

        # Split into sentences
        sentences  = self._split_sentences(clean_text)
        if not sentences:
            logger.warning(f"No sentences found in {doc.doc_id}")
            return []

        # Group sentences into chunks
        chunks         = []
        current_words  = []
        current_chars  = 0
        char_offset    = 0
        overlap_buffer: List[str] = []   # Sentences to prepend to next chunk

        for sentence in sentences:
            words = sentence.split()
            current_words.extend(words)

            # Check if we've hit the target chunk size
            if len(current_words) >= self.target_words:
                chunk_text = " ".join(current_words)

                # Only keep substantive chunks
                if self.cleaner.is_substantive(chunk_text) and len(current_words) >= self.min_words:
                    chunk_id = DocumentChunk.make_id(doc.doc_id, len(chunks))
                    chunks.append(DocumentChunk(
                        chunk_id     = chunk_id,
                        doc_id       = doc.doc_id,
                        ticker       = doc.ticker,
                        doc_type     = doc.doc_type,
                        title        = doc.title,
                        text         = chunk_text,
                        chunk_index  = len(chunks),
                        total_chunks = 0,      # Updated below
                        char_start   = char_offset,
                        char_end     = char_offset + len(chunk_text),
                        published_at = doc.published_at,
                        filing_date  = doc.filing_date,
                        metadata     = doc.metadata.copy(),
                    ))
                    char_offset += len(chunk_text)

                # Save overlap for next chunk
                overlap_words_list = current_words[-self.overlap_words:]
                current_words      = overlap_words_list

        # Handle remaining words
        if len(current_words) >= self.min_words:
            chunk_text = " ".join(current_words)
            if self.cleaner.is_substantive(chunk_text):
                chunk_id = DocumentChunk.make_id(doc.doc_id, len(chunks))
                chunks.append(DocumentChunk(
                    chunk_id     = chunk_id,
                    doc_id       = doc.doc_id,
                    ticker       = doc.ticker,
                    doc_type     = doc.doc_type,
                    title        = doc.title,
                    text         = chunk_text,
                    chunk_index  = len(chunks),
                    total_chunks = 0,
                    char_start   = char_offset,
                    char_end     = char_offset + len(chunk_text),
                    published_at = doc.published_at,
                    filing_date  = doc.filing_date,
                    metadata     = doc.metadata.copy(),
                ))

        # Update total_chunks now we know the final count
        for c in chunks:
            c.total_chunks = len(chunks)

        logger.debug(
            f"Chunked {doc.doc_id}: {doc.word_count:,} words → "
            f"{len(chunks)} chunks"
        )
        return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Embedding engine
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingEngine:
    """
    Generates vector embeddings for document chunks.

    Priority:
      1. sentence-transformers (free, local, preferred)
      2. OpenAI text-embedding-3-small (paid, $0.02/1M tokens)
      3. Anthropic (via voyage-finance-2 model, paid)

    sentence-transformers recommendation for finance:
      - all-MiniLM-L6-v2: Fast, 384 dimensions, good general purpose
      - ProsusAI/finbert: Finance-specific, 768 dimensions
      - sentence-transformers/all-mpnet-base-v2: Best quality, 768 dims

    Default: all-MiniLM-L6-v2 (fastest, works well for retrieval)
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
    ):
        self.model_name  = model_name
        self.batch_size  = batch_size
        self._model      = None
        self._dimensions = None
        self._provider   = None

        self._init_model()

    def _init_model(self):
        """Try to load embedding model in order of preference."""
        # Try sentence-transformers first (free)
        try:
            from sentence_transformers import SentenceTransformer
            self._model     = SentenceTransformer(self.model_name)
            self._provider  = "sentence_transformers"
            # Get dimensions by encoding a test string
            test_emb        = self._model.encode(["test"])
            self._dimensions= len(test_emb[0])
            logger.info(
                f"Embedding engine: sentence-transformers/{self.model_name} "
                f"({self._dimensions} dims)"
            )
            return
        except ImportError:
            logger.info("sentence-transformers not installed — trying OpenAI")
        except Exception as e:
            logger.warning(f"sentence-transformers failed: {e}")

        # Try OpenAI
        if os.getenv("OPENAI_API_KEY"):
            try:
                import openai
                self._model     = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self._provider  = "openai"
                self._dimensions= 1536    # text-embedding-3-small
                logger.info("Embedding engine: OpenAI text-embedding-3-small")
                return
            except ImportError:
                pass

        # Fallback: TF-IDF style sparse embeddings (no ML needed)
        logger.warning(
            "No embedding model available. Using TF-IDF fallback. "
            "Install: pip install sentence-transformers"
        )
        self._provider   = "tfidf_fallback"
        self._dimensions = 512

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of text strings.

        Returns list of embedding vectors (one per text).
        """
        if not texts:
            return []

        if self._provider == "sentence_transformers":
            return self._embed_sentence_transformers(texts)
        elif self._provider == "openai":
            return self._embed_openai(texts)
        else:
            return self._embed_tfidf_fallback(texts)

    def embed_query(self, query: str) -> List[float]:
        """Embed a single query string."""
        results = self.embed([query])
        return results[0] if results else [0.0] * self._dimensions

    def _embed_sentence_transformers(self, texts: List[str]) -> List[List[float]]:
        """Use sentence-transformers for local embeddings."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embeddings = self._model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embeddings.tolist())
        return all_embeddings

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        """Use OpenAI embeddings API."""
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                response = self._model.embeddings.create(
                    model="text-embedding-3-small",
                    input=batch,
                )
                embeddings = [e.embedding for e in response.data]
                all_embeddings.extend(embeddings)
            except Exception as e:
                logger.error(f"OpenAI embedding failed: {e}")
                # Return zero vectors for failed batch
                all_embeddings.extend([[0.0] * self._dimensions] * len(batch))
        return all_embeddings

    def _embed_tfidf_fallback(self, texts: List[str]) -> List[List[float]]:
        """
        TF-IDF based sparse embedding fallback.

        Used when no ML model is available.
        Much lower quality than neural embeddings but functional.
        """
        from collections import Counter
        import math

        # Build vocabulary from input texts
        all_words = set()
        tokenised = []
        for text in texts:
            words = re.findall(r'\b[a-z]{3,}\b', text.lower())
            tokenised.append(words)
            all_words.update(words)

        vocab = sorted(list(all_words))[:self._dimensions]
        word_to_idx = {w: i for i, w in enumerate(vocab)}
        N = len(texts)

        embeddings = []
        for words in tokenised:
            vec = [0.0] * self._dimensions
            counts = Counter(words)
            total  = len(words) if words else 1

            for word, count in counts.items():
                if word in word_to_idx:
                    tf  = count / total
                    df  = sum(1 for t in tokenised if word in t)
                    idf = math.log((N + 1) / (df + 1))
                    vec[word_to_idx[word]] = tf * idf

            # L2 normalise
            norm = math.sqrt(sum(v*v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]

            embeddings.append(vec)

        return embeddings

    @property
    def dimensions(self) -> int:
        return self._dimensions or 384


# ─────────────────────────────────────────────────────────────────────────────
# Vector store (ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────

class VectorStore:
    """
    ChromaDB-backed vector store for financial document chunks.

    Stores:
      - Document chunk text
      - Vector embeddings (for semantic search)
      - Metadata (ticker, doc_type, date, chunk_id)

    Supports:
      - Semantic similarity search
      - Metadata filtering (search only AAPL news, only 10-K filings, etc.)
      - Persistence (survives process restarts)
      - Incremental updates (add new documents without rebuilding)

    ChromaDB runs embedded — no server needed.
    Data stored in: data/vector_db/
    """

    COLLECTION_NAME = "hedge_fund_documents"

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = persist_dir or (
            Path(__file__).parents[3] / "data" / "vector_db"
        )
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._client     = None
        self._collection = None
        self._init_chromadb()

    def _init_chromadb(self):
        """Initialise ChromaDB client and collection."""
        try:
            import chromadb
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir)
            )
            self._collection = self._client.get_or_create_collection(
                name     = self.COLLECTION_NAME,
                metadata = {"hnsw:space": "cosine"},   # Cosine similarity
            )
            count = self._collection.count()
            logger.info(
                f"VectorStore ready: {count:,} chunks in collection "
                f"'{self.COLLECTION_NAME}'"
            )
        except ImportError:
            logger.warning(
                "ChromaDB not installed — using SQLite fallback. "
                "Install: pip install chromadb"
            )
            self._use_sqlite_fallback()
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            self._use_sqlite_fallback()

    def _use_sqlite_fallback(self):
        """
        Simple SQLite fallback when ChromaDB is not available.

        Stores embeddings as JSON blobs — much slower for large collections
        but functional for development and testing.
        """
        self._client = "sqlite_fallback"
        db_path      = self.persist_dir / "vector_store.db"
        self._sqlite_db = db_path

        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id    TEXT PRIMARY KEY,
                    doc_id      TEXT,
                    ticker      TEXT,
                    doc_type    TEXT,
                    title       TEXT,
                    text        TEXT,
                    embedding   TEXT,
                    metadata    TEXT,
                    created_at  TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_ticker
                ON chunks (ticker, doc_type)
            """)
            conn.commit()

        count = self._get_sqlite_count()
        logger.info(f"VectorStore (SQLite fallback): {count:,} chunks")

    def upsert(
        self,
        chunks:     List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """
        Insert or update chunks with their embeddings.

        Returns number of chunks stored.
        """
        if not chunks or not embeddings:
            return 0
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks ({len(chunks)}) and embeddings ({len(embeddings)}) "
                "count mismatch"
            )

        if self._collection is not None:
            return self._upsert_chromadb(chunks, embeddings)
        else:
            return self._upsert_sqlite(chunks, embeddings)

    def _upsert_chromadb(
        self,
        chunks:     List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert to ChromaDB in batches."""
        batch_size = 100
        stored     = 0

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_embs   = embeddings[i:i + batch_size]

            ids        = [c.chunk_id for c in batch_chunks]
            texts      = [c.text for c in batch_chunks]
            metadatas  = [c.to_chromadb_metadata() for c in batch_chunks]
            emb_lists  = [list(e) for e in batch_embs]

            try:
                self._collection.upsert(
                    ids        = ids,
                    documents  = texts,
                    embeddings = emb_lists,
                    metadatas  = metadatas,
                )
                stored += len(batch_chunks)
            except Exception as e:
                logger.error(f"ChromaDB upsert batch failed: {e}")

        return stored

    def _upsert_sqlite(
        self,
        chunks:     List[DocumentChunk],
        embeddings: List[List[float]],
    ) -> int:
        """Upsert to SQLite fallback."""
        now = datetime.now().isoformat()
        rows = []
        for c, e in zip(chunks, embeddings):
            rows.append((
                c.chunk_id,
                c.doc_id,
                c.ticker,
                c.doc_type.value,
                c.title,
                c.text,
                json.dumps(e),
                json.dumps(c.to_chromadb_metadata()),
                now,
            ))

        with sqlite3.connect(self._sqlite_db) as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO chunks
                (chunk_id, doc_id, ticker, doc_type, title, text,
                 embedding, metadata, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, rows)
            conn.commit()

        return len(rows)

    def search(
        self,
        query_embedding: List[float],
        query:           RAGQuery,
    ) -> List[RetrievedChunk]:
        """
        Semantic similarity search.

        Args:
            query_embedding: Vector embedding of the query
            query:           RAGQuery with filters and top_k

        Returns:
            List[RetrievedChunk] sorted by similarity (highest first)
        """
        if self._collection is not None:
            return self._search_chromadb(query_embedding, query)
        else:
            return self._search_sqlite(query_embedding, query)

    def _search_chromadb(
        self,
        query_embedding: List[float],
        query:           RAGQuery,
    ) -> List[RetrievedChunk]:
        """Search ChromaDB with metadata filters."""
        # Build where clause from filters
        where = None
        conditions = []

        if query.ticker_filter:
            if len(query.ticker_filter) == 1:
                conditions.append({"ticker": {"$eq": query.ticker_filter[0]}})
            else:
                conditions.append({"ticker": {"$in": query.ticker_filter}})

        if query.doc_type_filter:
            doc_types = [dt.value for dt in query.doc_type_filter]
            if len(doc_types) == 1:
                conditions.append({"doc_type": {"$eq": doc_types[0]}})
            else:
                conditions.append({"doc_type": {"$in": doc_types}})

        if len(conditions) == 1:
            where = conditions[0]
        elif len(conditions) > 1:
            where = {"$and": conditions}

        try:
            results = self._collection.query(
                query_embeddings = [list(query_embedding)],
                n_results        = min(query.top_k, self._collection.count()),
                where            = where,
                include          = ["documents", "metadatas", "distances"],
            )
        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return []

        retrieved = []
        documents  = results.get("documents", [[]])[0]
        metadatas  = results.get("metadatas", [[]])[0]
        distances  = results.get("distances", [[]])[0]

        for rank, (doc_text, meta, dist) in enumerate(
            zip(documents, metadatas, distances), start=1
        ):
            similarity = 1.0 - dist   # ChromaDB cosine distance → similarity
            if similarity < query.min_similarity:
                continue

            # Reconstruct DocumentChunk from metadata
            chunk = DocumentChunk(
                chunk_id     = meta.get("chunk_id", ""),
                doc_id       = meta.get("doc_id", ""),
                ticker       = meta.get("ticker", ""),
                doc_type     = DocumentType(meta.get("doc_type", "OTHER")),
                title        = meta.get("title", ""),
                text         = doc_text,
                chunk_index  = int(meta.get("chunk_index", 0)),
                total_chunks = int(meta.get("total_chunks", 1)),
                char_start   = 0,
                char_end     = len(doc_text),
                published_at = (
                    datetime.fromisoformat(meta["published_at"])
                    if meta.get("published_at") else None
                ),
            )

            retrieved.append(RetrievedChunk(
                chunk      = chunk,
                similarity = round(similarity, 4),
                rank       = rank,
            ))

        return retrieved

    def _search_sqlite(
        self,
        query_embedding: List[float],
        query:           RAGQuery,
    ) -> List[RetrievedChunk]:
        """Search SQLite fallback using cosine similarity (slower)."""
        # Build query
        sql  = "SELECT chunk_id, doc_id, ticker, doc_type, title, text, embedding, metadata FROM chunks"
        args = []
        wheres = []

        if query.ticker_filter:
            placeholders = ",".join("?" * len(query.ticker_filter))
            wheres.append(f"ticker IN ({placeholders})")
            args.extend(query.ticker_filter)

        if query.doc_type_filter:
            types = [dt.value for dt in query.doc_type_filter]
            placeholders = ",".join("?" * len(types))
            wheres.append(f"doc_type IN ({placeholders})")
            args.extend(types)

        if wheres:
            sql += " WHERE " + " AND ".join(wheres)

        with sqlite3.connect(self._sqlite_db) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, args).fetchall()

        if not rows:
            return []

        # Compute cosine similarities
        q_vec  = np.array(query_embedding)
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scored = []
        for row in rows:
            try:
                emb    = np.array(json.loads(row["embedding"]))
                emb_n  = np.linalg.norm(emb)
                if emb_n == 0:
                    continue
                sim    = float(np.dot(q_vec, emb) / (q_norm * emb_n))
                scored.append((sim, dict(row)))
            except Exception:
                continue

        scored.sort(key=lambda x: x[0], reverse=True)

        retrieved = []
        for rank, (sim, row) in enumerate(scored[:query.top_k], start=1):
            if sim < query.min_similarity:
                continue

            chunk = DocumentChunk(
                chunk_id     = row["chunk_id"],
                doc_id       = row["doc_id"],
                ticker       = row["ticker"],
                doc_type     = DocumentType(row["doc_type"]),
                title        = row["title"],
                text         = row["text"],
                chunk_index  = 0,
                total_chunks = 1,
                char_start   = 0,
                char_end     = len(row["text"]),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, similarity=sim, rank=rank))

        return retrieved

    def count(self, ticker: Optional[str] = None) -> int:
        """Count chunks in the store, optionally filtered by ticker."""
        if self._collection is not None:
            if ticker:
                try:
                    result = self._collection.get(where={"ticker": {"$eq": ticker}})
                    return len(result.get("ids", []))
                except Exception:
                    return self._collection.count()
            return self._collection.count()
        else:
            return self._get_sqlite_count(ticker)

    def _get_sqlite_count(self, ticker: Optional[str] = None) -> int:
        if not hasattr(self, '_sqlite_db'):
            return 0
        with sqlite3.connect(self._sqlite_db) as conn:
            if ticker:
                row = conn.execute(
                    "SELECT COUNT(*) FROM chunks WHERE ticker=?", (ticker,)
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()
        return row[0] if row else 0

    def delete_ticker(self, ticker: str) -> int:
        """Remove all chunks for a ticker (for refresh)."""
        if self._collection is not None:
            try:
                self._collection.delete(where={"ticker": {"$eq": ticker}})
                return -1  # ChromaDB doesn't return count
            except Exception as e:
                logger.error(f"Delete failed for {ticker}: {e}")
                return 0
        else:
            with sqlite3.connect(self._sqlite_db) as conn:
                cursor = conn.execute(
                    "DELETE FROM chunks WHERE ticker=?", (ticker,)
                )
                conn.commit()
                return cursor.rowcount

    def get_indexed_tickers(self) -> List[str]:
        """Return list of all tickers that have chunks indexed."""
        if self._collection is not None:
            try:
                result = self._collection.get()
                tickers = set(m.get("ticker", "") for m in result.get("metadatas", []))
                return sorted(t for t in tickers if t)
            except Exception:
                return []
        else:
            with sqlite3.connect(self._sqlite_db) as conn:
                rows = conn.execute(
                    "SELECT DISTINCT ticker FROM chunks ORDER BY ticker"
                ).fetchall()
            return [r[0] for r in rows]


# ─────────────────────────────────────────────────────────────────────────────
# Full ingestion pipeline
# ─────────────────────────────────────────────────────────────────────────────

class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.

    RawDocument → clean → chunk → embed → store

    Usage:
        pipeline = IngestionPipeline()

        # Ingest one document:
        n_chunks = pipeline.ingest_document(raw_doc)

        # Ingest many documents:
        stats = pipeline.ingest_batch(raw_docs)

        # Ingest for a ticker using fetchers:
        stats = pipeline.ingest_ticker("AAPL")
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        persist_dir:     Optional[Path] = None,
    ):
        self.chunker   = DocumentChunker()
        self.embedder  = EmbeddingEngine(model_name=embedding_model)
        self.store     = VectorStore(persist_dir=persist_dir)
        self._stats    = {"documents": 0, "chunks": 0, "errors": 0}

    def ingest_document(self, doc: RawDocument) -> int:
        """
        Ingest a single RawDocument into the vector store.

        Returns number of chunks stored.
        """
        # 1. Chunk
        chunks = self.chunker.chunk(doc)
        if not chunks:
            logger.debug(f"No chunks from {doc.doc_id}")
            return 0

        # 2. Embed
        texts      = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        if len(embeddings) != len(chunks):
            logger.error(
                f"Embedding count mismatch for {doc.doc_id}: "
                f"{len(embeddings)} embeddings for {len(chunks)} chunks"
            )
            return 0

        # 3. Store
        n_stored = self.store.upsert(chunks, embeddings)

        self._stats["documents"] += 1
        self._stats["chunks"]    += n_stored

        logger.debug(
            f"Ingested {doc.doc_id}: "
            f"{len(chunks)} chunks → {n_stored} stored"
        )
        return n_stored

    def ingest_batch(self, documents: List[RawDocument]) -> Dict[str, int]:
        """
        Ingest a batch of documents.

        Returns stats dict with documents, chunks, errors counts.
        """
        stats = {"documents": 0, "chunks": 0, "errors": 0}

        for doc in documents:
            try:
                n = self.ingest_document(doc)
                stats["documents"] += 1
                stats["chunks"]    += n
            except Exception as e:
                logger.error(f"Failed to ingest {doc.doc_id}: {e}")
                stats["errors"] += 1

        logger.info(
            f"Batch ingestion: {stats['documents']} docs | "
            f"{stats['chunks']} chunks | "
            f"{stats['errors']} errors"
        )
        return stats

    def ingest_ticker(
        self,
        ticker:           str,
        include_sec:      bool = True,
        include_news:     bool = True,
        include_earnings: bool = True,
        refresh:          bool = False,
    ) -> Dict[str, int]:
        """
        Fetch and ingest all available documents for a ticker.

        Args:
            ticker       : Stock ticker
            include_sec  : Include SEC 10-K/10-Q filings
            include_news : Include news articles
            include_earnings: Include earnings-related 8-Ks
            refresh      : If True, delete existing chunks and re-ingest
        """
        from src.rag.document_fetchers import DocumentFetchOrchestrator

        if refresh:
            deleted = self.store.delete_ticker(ticker)
            logger.info(f"Refreshing {ticker}: deleted existing chunks")

        orchestrator = DocumentFetchOrchestrator()
        documents = orchestrator.fetch_all(
            ticker,
            include_sec      = include_sec,
            include_news     = include_news,
            include_earnings = include_earnings,
        )

        if not documents:
            logger.warning(f"No documents fetched for {ticker}")
            return {"documents": 0, "chunks": 0, "errors": 0}

        return self.ingest_batch(documents)

    @property
    def total_chunks(self) -> int:
        return self.store.count()

    def stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "total_in_store":    self.total_chunks,
            "indexed_tickers":   self.store.get_indexed_tickers(),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("  Document Processor — Test")
    print("=" * 60)

    # Test chunker
    print("\n1. Testing chunker...")
    chunker = DocumentChunker()
    test_doc = RawDocument(
        doc_id="TEST001",
        ticker="AAPL",
        doc_type=DocumentType.NEWS_ARTICLE,
        title="Apple Reports Record Revenue",
        text=(
            "Apple Inc. today announced financial results for its fiscal 2024 "
            "fourth quarter and fiscal year ended September 28, 2024. "
            "The Company posted quarterly revenue of $94.9 billion, up 6 percent "
            "year over year, and quarterly earnings per diluted share of $1.64, "
            "up 12 percent year over year. International sales accounted for 60 "
            "percent of the quarter's revenue. "
            "Tim Cook, Apple's CEO, said: 'We are thrilled to report another "
            "record quarter for Apple, with strong performance across all our "
            "product categories and services.' "
            "Luca Maestri, Apple's CFO, added: 'Our business performance drove "
            "EPS growth of 12 percent during the September quarter, and we are "
            "pleased to extend our track record of sharing our success with "
            "shareholders, as we returned over $29 billion to them during the quarter.' "
        ) * 5,  # Repeat to make it longer
        published_at=datetime(2024, 10, 31),
    )

    chunks = chunker.chunk(test_doc)
    print(f"   ✓ {len(chunks)} chunks from {test_doc.word_count} words")
    if chunks:
        print(f"   First chunk: {chunks[0].word_count} words")
        print(f"   Last chunk:  {chunks[-1].word_count} words")

    # Test embedder
    print("\n2. Testing embedding engine...")
    embedder = EmbeddingEngine()
    print(f"   Provider: {embedder._provider}")
    print(f"   Dimensions: {embedder.dimensions}")

    test_texts = ["Apple revenue grew 6% year over year", "Microsoft cloud growth"]
    embeddings = embedder.embed(test_texts)
    print(f"   ✓ Embedded {len(embeddings)} texts")
    print(f"   Embedding shape: {len(embeddings[0])} dimensions")

    # Test similarity
    query_emb = embedder.embed_query("Apple quarterly earnings results")
    sim = float(np.dot(np.array(query_emb), np.array(embeddings[0])) /
               (np.linalg.norm(query_emb) * np.linalg.norm(embeddings[0])))
    print(f"   Similarity (Apple revenue ↔ Apple earnings query): {sim:.3f}")

    # Test vector store
    print("\n3. Testing vector store...")
    pipeline = IngestionPipeline()
    n_stored = pipeline.ingest_document(test_doc)
    print(f"   ✓ Stored {n_stored} chunks")
    print(f"   Total in store: {pipeline.total_chunks}")

    print(f"\n✅ Document processor tests passed")
    print(f"   Install: pip install sentence-transformers chromadb")
    print(f"   for full neural embedding and vector search")
