"""Okapi BM25 Lexical Keyword Search Index.

Provides fast, zero-dependency lexical keyword retrieval to complement dense
vector embeddings for hybrid search and reciprocal rank fusion (RRF).
"""

from collections import Counter
import math
import re
from typing import Any, Dict, List, Optional, Tuple

from app.vectorstore.base import DocumentChunk, SearchResult


STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had",
    "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself",
    "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours",
    "ourselves", "out", "over", "own", "s", "same", "she", "should", "so", "some", "such", "t", "than",
    "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your", "yours",
}


class BM25Index:
    """Okapi BM25 index built over a collection of DocumentChunk objects."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.corpus_tokens: List[List[str]] = []
        self.doc_count: int = 0
        self.avgdl: float = 0.0
        self.doc_lens: List[int] = []
        self.dfs: Counter = Counter()
        self.idfs: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str, filter_stopwords: bool = True) -> List[str]:
        """Extracts lowercase alphanumeric tokens for term matching."""
        if not text:
            return []
        tokens = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", text.lower())
        if filter_stopwords:
            tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokens

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Indexes a list of DocumentChunks for BM25 retrieval."""
        self.chunks = list(chunks)
        self.corpus_tokens = [self.tokenize(c.text) for c in self.chunks]
        self.doc_count = len(self.chunks)
        self.doc_lens = [len(tokens) for tokens in self.corpus_tokens]
        self.avgdl = (sum(self.doc_lens) / max(1, self.doc_count)) if self.doc_count > 0 else 0.0

        self.dfs = Counter()
        for tokens in self.corpus_tokens:
            self.dfs.update(set(tokens))

        # Standard Lucene-style Robertson-Spärck Jones IDF
        self.idfs = {}
        for term, freq in self.dfs.items():
            # math.log((N - n + 0.5) / (n + 0.5) + 1.0)
            self.idfs[term] = math.log((self.doc_count - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(
        self,
        query: str,
        k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Performs lexical BM25 scoring across indexed chunks."""
        if not self.chunks or not query.strip():
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores: List[Tuple[float, int]] = []

        for idx, (chunk, doc_tokens) in enumerate(zip(self.chunks, self.corpus_tokens)):
            # Check metadata filter if specified
            if filter_dict:
                match = True
                for f_key, f_val in filter_dict.items():
                    if getattr(chunk, f_key, None) != f_val:
                        match = False
                        break
                if not match:
                    continue

            # Calculate BM25 score
            doc_len = self.doc_lens[idx]
            tf = Counter(doc_tokens)
            score = 0.0

            for q_term in query_tokens:
                if q_term in tf:
                    f = tf[q_term]
                    idf = self.idfs.get(q_term, 0.0)
                    denom = f + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1.0, self.avgdl)))
                    score += idf * (f * (self.k1 + 1.0)) / max(1e-6, denom)

            if score > 0.0:
                scores.append((score, idx))

        # Sort descending by score
        scores.sort(key=lambda x: x[0], reverse=True)

        results: List[SearchResult] = []
        for score, idx in scores[:k]:
            results.append(SearchResult(chunk=self.chunks[idx], score=float(score)))

        return results
