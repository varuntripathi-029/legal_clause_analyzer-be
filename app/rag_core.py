"""
LegalRAGPipeline — Core retrieval-augmented generation engine.

Responsibilities:
  1. Load and embed a legal knowledge base into a FAISS vector index.
  2. Split raw contract text into individual clauses.
  3. Retrieve the most relevant legal provisions for each clause.
  4. Call Google Gemini (1.5 Flash) to produce structured JSON analysis.
"""

from __future__ import annotations

from collections import defaultdict
import json
import logging
import re
from pathlib import Path
from typing import Any, TypedDict

import faiss
from google import genai
from google.genai import types
import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas import ClauseAnalysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_EMBEDDING_DIM = 384
_GEMINI_MODEL_NAME = "gemini-3-flash-preview"
_KB_PATH = Path(__file__).resolve().parent.parent / "data" / "legal_kb.json"
_CASE_LAW_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.json"
_RETRIEVAL_SLICES: tuple[tuple[str, int], ...] = (
    ("constitution", 2),
    ("case_law", 2),
    ("statute", 1),
)
_BALANCED_RESULT_COUNT = sum(limit for _, limit in _RETRIEVAL_SLICES)
_ARTICLE_MATCH_BOOST = 0.10
_CASE_LAW_BOOST = 0.04
_MAX_CALIBRATED_CONFIDENCE = 0.92
_MAX_LOW_INFORMATION_CONFIDENCE = 0.62
_MAX_NO_VIOLATION_CONFIDENCE = 0.86


class Document(TypedDict):
    """Minimal document payload stored in the retrieval corpus."""

    text: str
    metadata: dict[str, Any]


def load_case_laws(file_path: str) -> list[Document]:
    """Load structured case-law entries into indexable retrieval documents."""
    with open(file_path, encoding="utf-8") as fh:
        raw_cases: list[dict[str, Any]] = json.load(fh)

    documents: list[Document] = []
    for case in raw_cases:
        embedding_text = str(case.get("embedding_text", "")).strip()
        if not embedding_text:
            logger.warning("Skipping case-law entry without embedding_text: %s", case.get("id"))
            continue

        documents.append(
            {
                "text": embedding_text,
                "metadata": {
                    "type": "case_law",
                    "case_name": case.get("case_name", "Unknown Case"),
                    "articles": case.get("articles", []),
                    "year": case.get("year"),
                },
            }
        )

    logger.info("Loaded %d case-law documents from %s", len(documents), file_path)
    return documents


class LegalRAGPipeline:
    """End-to-end RAG pipeline for legal contract analysis.

    Lifecycle:
        1. ``__init__`` loads the embedding model, FAISS index, and Gemini
           client.  This is called **once** at application startup.
        2. ``split_clauses`` breaks a contract into clause-level chunks.
        3. ``retrieve_context`` fetches the closest knowledge-base entries.
        4. ``analyze_clause`` asks Gemini to evaluate a clause against the
           retrieved context and returns a ``ClauseAnalysis`` object.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------
    def __init__(self, gemini_api_key: str) -> None:
        """Initialise all pipeline components.

        Args:
            gemini_api_key: API key for the Google Generative AI service.
        """
        logger.info("Loading embedding model: %s …", _EMBEDDING_MODEL_NAME)
        self.embedder = SentenceTransformer(_EMBEDDING_MODEL_NAME)
        self.doc_ids_by_type: dict[str, list[int]] = {}
        self.indices_by_type: dict[str, faiss.IndexFlatL2] = {}
        self.article_labels: list[str] = []
        self.article_profile_embeddings = np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        # Build the FAISS index from the seed knowledge base.
        self.kb_documents: list[Document] = self._load_knowledge_base()
        self.index = self._build_faiss_index()

        # Configure Google Generative AI (Gemini).
        self.client = genai.Client(api_key=gemini_api_key)
        self.model_name = _GEMINI_MODEL_NAME
        logger.info("LegalRAGPipeline ready ✓")

    # ------------------------------------------------------------------
    # Knowledge-base helpers
    # ------------------------------------------------------------------
    def _load_knowledge_base(self) -> list[Document]:
        """Load the JSON knowledge base from disk."""
        with open(_KB_PATH, encoding="utf-8") as fh:
            base_documents: list[Document] = json.load(fh)

        documents = [self._normalize_document(doc) for doc in base_documents]

        if _CASE_LAW_PATH.exists():
            case_documents = [
                self._normalize_document(doc)
                for doc in load_case_laws(str(_CASE_LAW_PATH))
            ]
            documents.extend(case_documents)
            logger.info(
                "Loaded %d base documents and %d case-law documents",
                len(base_documents),
                len(case_documents),
            )
        else:
            logger.warning("Case-law dataset not found at %s; continuing without it", _CASE_LAW_PATH)

        logger.info("Loaded %d total documents into the retrieval corpus", len(documents))
        return documents

    def _build_faiss_index(self) -> faiss.IndexFlatL2:
        """Embed every KB document and add the vectors to a flat L2 index."""
        texts = [doc["text"] for doc in self.kb_documents]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True)
        embeddings_np = np.array(embeddings, dtype=np.float32)

        index = faiss.IndexFlatL2(_EMBEDDING_DIM)
        index.add(embeddings_np)
        self._build_type_indices(embeddings_np)
        self._build_article_profiles()
        logger.info("FAISS index built — %d vectors, dim=%d", index.ntotal, _EMBEDDING_DIM)
        return index

    @staticmethod
    def _normalize_doc_type(doc_type: str | None) -> str:
        """Map legacy document labels into the retrieval categories."""
        if doc_type == "fundamental_right":
            return "constitution"
        return doc_type or "unknown"

    @staticmethod
    def _canonicalize_article_label(label: Any) -> str | None:
        """Convert article labels into a consistent comparison key."""
        if label is None:
            return None

        match = re.search(
            r"(?i)(?:article\s*)?(\d+[A-Za-z]?(?:\([^)]*\))*)",
            str(label),
        )
        if not match:
            return None
        return match.group(1).strip().upper()

    @classmethod
    def _extract_articles_from_text(cls, text: str) -> list[str]:
        """Extract article references mentioned directly in free text."""
        matches = re.findall(
            r"(?i)article\s+(\d+[A-Za-z]?(?:\([^)]*\))*)",
            text,
        )
        articles = [
            article
            for article in (
                cls._canonicalize_article_label(match)
                for match in matches
            )
            if article
        ]
        return list(dict.fromkeys(articles))

    @classmethod
    def _normalize_articles(cls, articles: list[Any] | None) -> list[str]:
        """Normalize article lists while preserving input order."""
        normalized = [
            article
            for article in (
                cls._canonicalize_article_label(item)
                for item in (articles or [])
            )
            if article
        ]
        return list(dict.fromkeys(normalized))

    @classmethod
    def _get_document_articles(cls, metadata: dict[str, Any]) -> list[str]:
        """Return the article tags associated with a document."""
        explicit_articles = cls._normalize_articles(metadata.get("articles"))
        if explicit_articles:
            return explicit_articles
        return cls._extract_articles_from_text(str(metadata.get("reference", "")))

    def _normalize_document(self, doc: Document) -> Document:
        """Enrich documents with retrieval-friendly metadata fields."""
        metadata = dict(doc.get("metadata", {}))
        normalized_type = self._normalize_doc_type(metadata.get("type"))
        if metadata.get("type") != normalized_type:
            metadata["legacy_type"] = metadata.get("type")
        metadata["type"] = normalized_type
        metadata["normalized_type"] = normalized_type

        articles = self._get_document_articles(metadata)
        if articles:
            metadata["articles"] = articles

        return {
            "text": doc["text"],
            "metadata": metadata,
        }

    def _build_type_indices(self, embeddings: np.ndarray) -> None:
        """Build type-specific FAISS views for multi-source retrieval."""
        self.doc_ids_by_type = {}
        self.indices_by_type = {}

        for doc_type, _ in _RETRIEVAL_SLICES:
            doc_ids = [
                idx
                for idx, doc in enumerate(self.kb_documents)
                if doc["metadata"].get("normalized_type") == doc_type
            ]
            type_index = faiss.IndexFlatL2(_EMBEDDING_DIM)
            if doc_ids:
                type_index.add(embeddings[doc_ids])

            self.doc_ids_by_type[doc_type] = doc_ids
            self.indices_by_type[doc_type] = type_index
            logger.info("Prepared %d %s documents for filtered retrieval", len(doc_ids), doc_type)

    def _build_article_profiles(self) -> None:
        """Precompute article embeddings used for lightweight query classification."""
        article_buckets: defaultdict[str, list[str]] = defaultdict(list)

        for doc in self.kb_documents:
            metadata = doc["metadata"]
            for article in self._get_document_articles(metadata):
                snippets = article_buckets[article]
                snippets.append(doc["text"])
                if case_name := metadata.get("case_name"):
                    snippets.append(str(case_name))
                if reference := metadata.get("reference"):
                    snippets.append(str(reference))

        self.article_labels = sorted(article_buckets)
        if not self.article_labels:
            self.article_profile_embeddings = np.empty((0, _EMBEDDING_DIM), dtype=np.float32)
            return

        article_texts = [
            f"Article {article}\n" + "\n".join(dict.fromkeys(article_buckets[article]))
            for article in self.article_labels
        ]
        article_embeddings = self.embedder.encode(article_texts, normalize_embeddings=True)
        self.article_profile_embeddings = np.array(article_embeddings, dtype=np.float32)

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        """Convert L2 distance into a sortable relevance score."""
        return 1.0 / (1.0 + max(distance, 0.0))

    @staticmethod
    def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        """Clamp a numeric value to the requested range."""
        return max(lower, min(upper, float(value)))

    @classmethod
    def _article_family(cls, label: Any) -> str | None:
        """Reduce article labels to a stable family key such as ``19``."""
        canonical = cls._canonicalize_article_label(label)
        if not canonical:
            return None

        match = re.match(r"(\d+[A-Za-z]?)", canonical)
        return match.group(1) if match else canonical

    @staticmethod
    def _canonicalize_section_label(label: Any) -> str | None:
        """Convert section labels into a comparable key."""
        if label is None:
            return None

        match = re.search(r"(?i)section\s+(\d+[A-Za-z]?)", str(label))
        if not match:
            return None
        return match.group(1).strip().upper()

    @classmethod
    def _extract_sections_from_text(cls, text: str) -> list[str]:
        """Extract statutory section references from free text."""
        matches = re.findall(r"(?i)section\s+(\d+[A-Za-z]?)", text)
        sections = [match.strip().upper() for match in matches if match]
        return list(dict.fromkeys(sections))

    @staticmethod
    def _safe_confidence_value(value: Any) -> float:
        """Parse a model-provided confidence score safely."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.5

    @staticmethod
    def _support_ratio(
        reported_items: set[str],
        retrieved_items: set[str],
        neutral_when_empty: float,
    ) -> float:
        """Measure how well reported authorities are supported by retrieved context."""
        if not reported_items:
            return neutral_when_empty
        if not retrieved_items:
            return 0.0
        return len(reported_items & retrieved_items) / len(reported_items)

    @staticmethod
    def _is_low_information_clause(clause: str) -> bool:
        """Identify very short or header-like clauses that deserve lower confidence."""
        words = re.findall(r"\b[A-Za-z][A-Za-z\-]*\b", clause)
        if len(words) <= 5:
            return True

        header_tokens = {
            "agreement",
            "employee",
            "signature",
            "signatory",
            "authorized",
            "company",
            "employer",
        }
        header_hits = sum(word.casefold() in header_tokens for word in words)
        return len(words) <= 10 and header_hits >= max(2, len(words) // 3)

    def _get_known_case_names(self) -> set[str]:
        """Return all case names currently loaded into the corpus."""
        return {
            str(doc["metadata"]["case_name"])
            for doc in self.kb_documents
            if doc["metadata"].get("normalized_type") == "case_law"
            and doc["metadata"].get("case_name")
        }

    def _collect_retrieved_authorities(self, context: list[dict[str, Any]]) -> dict[str, set[str]]:
        """Summarize the authorities present in the retrieved context."""
        authorities = {
            "article_families": set(),
            "sections": set(),
            "case_names": set(),
            "types": set(),
        }

        for ctx in context:
            metadata = ctx["metadata"]
            if doc_type := metadata.get("normalized_type") or metadata.get("type"):
                authorities["types"].add(str(doc_type))

            for article in self._get_document_articles(metadata):
                if family := self._article_family(article):
                    authorities["article_families"].add(family)

            if section := self._canonicalize_section_label(metadata.get("reference")):
                authorities["sections"].add(section)

            if case_name := metadata.get("case_name"):
                authorities["case_names"].add(str(case_name))

        return authorities

    def _collect_reported_authorities(self, analysis_data: dict[str, Any]) -> dict[str, set[str]]:
        """Summarize the authorities cited in the model output."""
        applicable_laws = analysis_data.get("applicable_laws", []) or []
        potential_violation = analysis_data.get("potential_violation", {}) or {}
        violation_entries = potential_violation.get("articles", []) or []
        legal_reasoning = str(analysis_data.get("legal_reasoning", ""))

        combined_segments = [
            *(str(item) for item in violation_entries),
            *(str(item) for item in applicable_laws),
            legal_reasoning,
        ]
        combined_text = "\n".join(segment for segment in combined_segments if segment)

        explicit_article_families = {
            family
            for family in (
                self._article_family(segment)
                for segment in combined_segments
            )
            if family
        }
        article_families = {
            family
            for family in (
                self._article_family(article)
                for article in self._extract_articles_from_text(combined_text)
            )
            if family
        }
        article_families.update(explicit_article_families)

        section_labels = set(self._extract_sections_from_text(combined_text))
        section_labels.update(
            section
            for section in (
                self._canonicalize_section_label(segment)
                for segment in combined_segments
            )
            if section
        )

        lower_text = combined_text.casefold()
        case_names = {
            case_name
            for case_name in self._get_known_case_names()
            if case_name.casefold() in lower_text
        }

        cited_types = set()
        if article_families:
            cited_types.add("constitution")
        if section_labels:
            cited_types.add("statute")
        if case_names:
            cited_types.add("case_law")

        return {
            "article_families": article_families,
            "sections": section_labels,
            "case_names": case_names,
            "types": cited_types,
        }

    def _calculate_retrieval_strength(self, context: list[dict[str, Any]]) -> float:
        """Estimate confidence support from vector-search quality."""
        if not context:
            return 0.0

        base_scores = [
            self._distance_to_score(float(ctx.get("distance", 1.0)))
            for ctx in context
        ]
        weights = [1.0 / (idx + 1) for idx in range(len(base_scores))]
        weighted_average = sum(score * weight for score, weight in zip(base_scores, weights)) / sum(weights)
        return self._clamp(weighted_average)

    def _calibrate_confidence(
        self,
        clause: str,
        context: list[dict[str, Any]],
        analysis_data: dict[str, Any],
    ) -> float:
        """Replace raw model confidence with an evidence-capped calibrated score."""
        raw_confidence = self._clamp(
            self._safe_confidence_value(analysis_data.get("confidence_score")),
        )
        retrieval_strength = self._calculate_retrieval_strength(context)
        retrieved = self._collect_retrieved_authorities(context)
        reported = self._collect_reported_authorities(analysis_data)

        article_support = self._support_ratio(
            reported["article_families"],
            retrieved["article_families"],
            neutral_when_empty=0.70,
        )
        section_support = self._support_ratio(
            reported["sections"],
            retrieved["sections"],
            neutral_when_empty=0.70,
        )
        case_support = self._support_ratio(
            reported["case_names"],
            retrieved["case_names"],
            neutral_when_empty=0.55 if retrieved["case_names"] else 0.75,
        )
        authority_support = (article_support + section_support + case_support) / 3.0
        source_usage = self._support_ratio(
            reported["types"],
            retrieved["types"],
            neutral_when_empty=0.65,
        )

        evidence_cap = (
            0.28
            + (0.34 * retrieval_strength)
            + (0.23 * authority_support)
            + (0.15 * source_usage)
        )

        potential_violation = analysis_data.get("potential_violation", {}) or {}
        is_violation = bool(potential_violation.get("is_violation"))

        if reported["case_names"] - retrieved["case_names"]:
            evidence_cap = min(evidence_cap, 0.45)
        if not context:
            evidence_cap = min(evidence_cap, 0.35)
        if is_violation and not (reported["article_families"] or reported["sections"] or reported["case_names"]):
            evidence_cap = min(evidence_cap, 0.50)
        if self._is_low_information_clause(clause):
            evidence_cap = min(evidence_cap, _MAX_LOW_INFORMATION_CONFIDENCE)
        elif not is_violation:
            evidence_cap = min(evidence_cap, _MAX_NO_VIOLATION_CONFIDENCE)

        evidence_cap = min(evidence_cap, _MAX_CALIBRATED_CONFIDENCE)
        blended_confidence = min(
            evidence_cap,
            (0.40 * raw_confidence) + (0.60 * evidence_cap),
        )
        return round(self._clamp(blended_confidence, lower=0.05, upper=_MAX_CALIBRATED_CONFIDENCE), 2)

    def _search_by_type(
        self,
        query_vec: np.ndarray,
        doc_type: str,
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Run FAISS search over one filtered document subset."""
        doc_ids = self.doc_ids_by_type.get(doc_type, [])
        if not doc_ids:
            return []

        search_k = min(top_k, len(doc_ids))
        distances, indices = self.indices_by_type[doc_type].search(query_vec, search_k)

        results: list[dict[str, Any]] = []
        for distance, local_idx in zip(distances[0], indices[0]):
            if local_idx == -1:
                continue

            doc_index = doc_ids[local_idx]
            document = self.kb_documents[doc_index]
            results.append(
                {
                    "doc_index": doc_index,
                    "text": document["text"],
                    "metadata": document["metadata"],
                    "distance": float(distance),
                    "base_score": self._distance_to_score(float(distance)),
                    "retrieval_type": document["metadata"].get("normalized_type", doc_type),
                }
            )
        return results

    def _classify_articles(self, query: str) -> dict[str, Any]:
        """Estimate the most relevant constitutional article for a query."""
        explicit_articles = self._extract_articles_from_text(query)
        if explicit_articles:
            top_article = explicit_articles[0]
            return {
                "top_article": top_article,
                "article_probabilities": {
                    article: 1.0 if article == top_article else 0.0
                    for article in self.article_labels
                },
            }

        if self.article_profile_embeddings.size == 0:
            return {"top_article": None, "article_probabilities": {}}

        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)[0]
        similarities = np.dot(self.article_profile_embeddings, query_vec)
        top_index = int(np.argmax(similarities))

        return {
            "top_article": self.article_labels[top_index],
            "article_probabilities": {
                article: float((score + 1.0) / 2.0)
                for article, score in zip(self.article_labels, similarities)
            },
        }

    def _get_top_article(self, classification: dict[str, Any]) -> str | None:
        """Extract the highest-confidence article from classifier output."""
        if top_article := self._canonicalize_article_label(classification.get("top_article")):
            return top_article

        article_probabilities = classification.get("article_probabilities", {})
        if article_probabilities:
            best_label = max(
                article_probabilities.items(),
                key=lambda item: float(item[1]),
            )[0]
            return self._canonicalize_article_label(best_label)

        articles = classification.get("articles")
        if isinstance(articles, list) and articles:
            return self._canonicalize_article_label(articles[0])
        return None

    def _balance_results(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep multi-source results diverse without sacrificing relevance."""
        ranked_candidates = sorted(
            candidates,
            key=lambda item: item["adjusted_score"],
            reverse=True,
        )

        selected: list[dict[str, Any]] = []
        selected_ids: set[int] = set()
        type_counts: defaultdict[str, int] = defaultdict(int)

        for required_type in ("constitution", "case_law"):
            best_candidate = next(
                (
                    candidate
                    for candidate in ranked_candidates
                    if candidate["retrieval_type"] == required_type
                    and candidate["doc_index"] not in selected_ids
                ),
                None,
            )
            if best_candidate is None:
                continue

            selected.append(best_candidate)
            selected_ids.add(best_candidate["doc_index"])
            type_counts[required_type] += 1

        while len(selected) < _BALANCED_RESULT_COUNT:
            remaining = [
                candidate
                for candidate in ranked_candidates
                if candidate["doc_index"] not in selected_ids
            ]
            if not remaining:
                break

            remaining_types = {candidate["retrieval_type"] for candidate in remaining}
            missing_types = [
                doc_type
                for doc_type, _ in _RETRIEVAL_SLICES
                if type_counts[doc_type] == 0 and doc_type in remaining_types
            ]
            pool = (
                [candidate for candidate in remaining if candidate["retrieval_type"] in missing_types]
                if missing_types
                else remaining
            )
            next_candidate = min(
                pool,
                key=lambda candidate: (
                    type_counts[candidate["retrieval_type"]],
                    -candidate["adjusted_score"],
                ),
            )

            selected.append(next_candidate)
            selected_ids.add(next_candidate["doc_index"])
            type_counts[next_candidate["retrieval_type"]] += 1

        return sorted(selected, key=lambda item: item["adjusted_score"], reverse=True)

    def smart_retrieve(self, query: str, classification: dict[str, Any]) -> list[dict[str, Any]]:
        """Retrieve and rerank a balanced mix of constitutional, statutory, and case-law context."""
        query_vec = self.embedder.encode([query], normalize_embeddings=True).astype(np.float32)
        top_article = self._get_top_article(classification)

        candidates: list[dict[str, Any]] = []
        for doc_type, limit in _RETRIEVAL_SLICES:
            candidates.extend(self._search_by_type(query_vec, doc_type, limit))

        for candidate in candidates:
            adjusted_score = candidate["base_score"]
            if top_article and top_article in self._get_document_articles(candidate["metadata"]):
                adjusted_score += _ARTICLE_MATCH_BOOST
            if candidate["retrieval_type"] == "case_law":
                adjusted_score += _CASE_LAW_BOOST
            candidate["adjusted_score"] = adjusted_score

        balanced_results = self._balance_results(candidates)
        return [
            {
                "text": candidate["text"],
                "metadata": candidate["metadata"],
                "distance": candidate["distance"],
            }
            for candidate in balanced_results
        ]

    # ------------------------------------------------------------------
    # Step 1 — Clause splitting
    # ------------------------------------------------------------------
    @staticmethod
    def split_clauses(text: str) -> list[str]:
        """Split raw contract text into individual clauses.

        Strategy (ordered by priority):
          • Numbered clauses   — ``1.``, ``2.``, ``(a)``, ``(i)`` etc.
          • Lettered sub-items — ``a.``, ``b.`` at the start of a line.
          • Sentence boundaries — as a fallback, split on sentence-ending
            punctuation followed by whitespace.

        Returns:
            A list of non-empty, stripped clause strings.
        """
        # Try numbered / lettered clause splitting first.
        pattern = r"""
            (?:                          # clause-number variants
                ^\s*\d+[\.\)]\s          # 1. or 1)
              | ^\s*\([a-zA-Z0-9]+\)\s   # (a) or (1)
              | ^\s*[a-zA-Z][\.\)]\s      # a. or a)
            )
        """
        splits = re.split(pattern, text, flags=re.MULTILINE | re.VERBOSE)
        # Filter out empty artefacts produced by the split.
        clauses = [c.strip() for c in splits if c and c.strip()]

        # Fallback: if we only got a single block, split on sentence endings.
        if len(clauses) <= 1:
            clauses = [
                s.strip()
                for s in re.split(r"(?<=[.;])\s+", text)
                if s.strip()
            ]

        return clauses

    # ------------------------------------------------------------------
    # Step 2 — Retrieval
    # ------------------------------------------------------------------
    def retrieve_context(
        self,
        clause: str,
        top_k: int = _BALANCED_RESULT_COUNT,
    ) -> list[dict[str, Any]]:
        """Retrieve the most relevant legal context for a clause.

        Args:
            clause: The clause text to search for.
            top_k: Number of results to return after balancing the merged set.

        Returns:
            A list of dicts, each containing ``text``, ``metadata``, and
            ``distance`` (L2 distance from the query vector).
        """
        classification = self._classify_articles(clause)
        return self.smart_retrieve(clause, classification)[:top_k]

    # ------------------------------------------------------------------
    # Step 3 — LLM analysis
    # ------------------------------------------------------------------
    def _build_prompt(self, clause: str, context: list[dict[str, Any]]) -> str:
        """Construct the Gemini prompt with retrieved legal context."""
        context_block = "\n\n".join(
            f"[{i + 1}] {ctx['text']}  "
            f"({self._format_context_label(ctx['metadata'])})"
            for i, ctx in enumerate(context)
        )

        return f"""You are an expert Indian legal analyst. Analyze the following employment contract clause for potential violations of the Constitution of India (Part III — Fundamental Rights) and the Indian Contract Act, 1872.

### Relevant Legal Provisions (Retrieved Context)
{context_block}

### Contract Clause to Analyze
\"{clause}\"

### Instructions
1. Summarize the clause in plain English.
2. Determine whether it potentially violates any fundamental right or statutory provision.
3. List all applicable articles / sections.
4. Provide step-by-step legal reasoning using only the retrieved context.
5. Assign a risk level: High, Medium, Low, or None.
6. Assign a preliminary confidence score between 0.0 and 1.0 conservatively.
7. Never return 1.0.
8. Use 0.90 to 0.99 only when the conclusion is directly supported by multiple retrieved authorities with little ambiguity.
9. Use 0.70 to 0.89 when support is strong but some inference is required.
10. Use 0.40 to 0.69 when support is partial, mixed, or the clause is vague.
11. Use 0.10 to 0.39 when the retrieved support is weak or uncertain.
12. Use retrieved case laws as precedents when available.
13. Always cite exact retrieved case names explicitly in applicable_laws and/or legal_reasoning.
14. Do not generate, infer, or hallucinate case names that are not present in the retrieved context.

### Required JSON Output Format
{{
    "clause_summary": "<string>",
    "potential_violation": {{
        "is_violation": <boolean>,
        "articles": ["<string>", ...]
    }},
    "applicable_laws": ["<string>", ...],
    "legal_reasoning": "<string>",
    "risk_level": "<High|Medium|Low|None>",
    "confidence_score": <float>
}}

Respond with ONLY the JSON object. Do NOT include any text outside the JSON."""

    def _format_context_label(self, metadata: dict[str, Any]) -> str:
        """Format a human-readable citation label for retrieved context."""
        if metadata.get("normalized_type") == "case_law":
            case_name = metadata.get("case_name", "Unknown Case")
            year = metadata.get("year", "N/A")
            return f"Case Law: {case_name} ({year})"

        return (
            f"Source: {metadata.get('source', 'N/A')}, "
            f"Reference: {metadata.get('reference', 'N/A')}"
        )

    async def analyze_clause(self, clause: str) -> ClauseAnalysis:
        """Run the full RAG pipeline for a single clause.

        1. Retrieve relevant context from the FAISS index.
        2. Build the LLM prompt.
        3. Call Gemini and parse the JSON response into a ``ClauseAnalysis``.
        """
        context = self.retrieve_context(clause)
        prompt = self._build_prompt(clause, context)

        logger.debug("Sending clause to Gemini: %.80s…", clause)
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                )
            )
        except Exception as e:
            logger.error("Error from Gemini API: %s", e)
            raise

        try:
            raw: dict[str, Any] = json.loads(response.text)
            raw["confidence_score"] = self._calibrate_confidence(clause, context, raw)
            return ClauseAnalysis(**raw)
        except Exception as json_e:
            logger.error("Failed to parse Gemini response as JSON: %s. Raw response: %s", json_e, response.candidates)
            raise ValueError(f"Invalid JSON from LLM: {str(json_e)}") from json_e

    # ------------------------------------------------------------------
    # Step 4 — Chat Session
    # ------------------------------------------------------------------
    def create_chat_session(self, contract_text: str) -> Any:
        """Initialize an ongoing chat session pre-loaded with the contract."""
        system_instruction = (
            "You are an expert Indian legal assistant. The user has uploaded an "
            "employment contract. Answer any questions about the contract. "
            f"Here is the contract text:\n\n{contract_text}"
        )
        return self.client.aio.chats.create(
            model=self.model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.3,
            )
        )
