import json
import logging
import re
from pathlib import Path
from typing import AsyncGenerator

import numpy as np
import faiss
from fastembed import TextEmbedding
from groq import AsyncGroq

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
_EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_EMBEDDING_DIM = 768  # bge-base-en-v1.5 produces 768-dimensional embeddings
_GENERATIVE_MODEL_NAME = "openai/gpt-oss-120b"

# --- Retrieval confidence thresholds (L2 distance) ---
# bge embeddings are normalized, so L2² ≈ 2(1 - cosine).
# L2² ≤ 0.30 → cosine ≥ 0.85 → High
# 0.30 < L2² ≤ 0.50 → cosine 0.75-0.85 → Medium
# L2² > 0.50 → cosine < 0.75 → Low
_RETRIEVAL_HIGH_THRESHOLD = 0.30
_RETRIEVAL_MEDIUM_THRESHOLD = 0.50

# --- Patterns for contract preprocessing (Item 3) ---
_SIGNATURE_PATTERNS = [
    r"(?i)^\s*employee\s*signature\s*[:._\s-]*$",
    r"(?i)^\s*authorized\s*signatory\s*[:._\s-]*$",
    r"(?i)^\s*employer\s*signature\s*[:._\s-]*$",
    r"(?i)^\s*witness\s*[:._\s-]*.*$",
    r"(?i)^\s*signature\s*[:._\s-]*$",
    r"(?i)^\s*sign\s*[:._\s-]*$",
    r"(?i)^\s*date\s*[:._\s-]*$",
    r"(?i)^\s*name\s*[:._\s-]*$",
    r"(?i)^\s*designation\s*[:._\s-]*$",
    r"(?i)^\s*employee\s*id\s*[:._\s-]*$",
    r"(?i)^\s*place\s*[:._\s-]*$",
    r"(?i)^\s*\[?\s*seal\s*\]?\s*$",
    r"(?i)^\s*\[?\s*stamp\s*\]?\s*$",
    r"(?i)^\s*company\s*seal\s*$",
]

_NOISE_PATTERNS = [
    r"(?i)^\s*page\s+\d+\s*(of\s+\d+)?\s*$",            # Page 1 of 5
    r"^\s*-?\s*\d+\s*-?\s*$",                             # Isolated page numbers
    r"(?i)^\s*confidential\s*$",                           # Confidential headers
    r"(?i)^\s*private\s*&?\s*confidential\s*$",
    r"(?i)^\s*strictly\s+confidential\s*$",
    r"(?i)^\s*for\s+internal\s+use\s+only\s*$",
    r"(?i)^\s*draft\s*$",
    r"(?i)^\s*_{3,}\s*$",                                  # Underline-only lines
    r"^\s*[-=]{3,}\s*$",                                   # Separator lines
]


class LegalRAGPipeline:
    def __init__(self, groq_api_key: str):
        """Initializes the RAG Pipeline with fastembed (local CPU) and Groq (LPU)."""
        logger.info("Initializing Legal RAG Pipeline (fastembed + Groq)...")

        # Groq async client for blazing-fast LLM inference
        self.groq_client = AsyncGroq(api_key=groq_api_key)
        self.model_name = _GENERATIVE_MODEL_NAME

        # Local CPU embedding model — no network calls needed
        logger.info("Loading local embedding model: %s", _EMBEDDING_MODEL_NAME)
        self.embedding_model = TextEmbedding(model_name=_EMBEDDING_MODEL_NAME)
        logger.info("Embedding model loaded successfully.")

        self.kb_documents = []
        self.index = None

        logger.info("RAG Pipeline object created. Call await startup() to initialize.")

    async def startup(self):
        """Async initialization for the RAG Pipeline."""
        self._load_knowledge_base()
        await self._build_faiss_index()
        logger.info("RAG Pipeline initialized successfully.")

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate vector embeddings locally on CPU using fastembed."""
        if not texts:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        embeddings = list(self.embedding_model.embed(texts))
        return np.array(embeddings, dtype=np.float32)

    def _load_knowledge_base(self):
        """Loads JSON seed data from the data directory."""
        data_dir = Path("data")

        try:
            with open(data_dir / "legal_kb.json", "r", encoding="utf-8") as f:
                self.kb_documents.extend(json.load(f))

            with open(data_dir / "cases.json", "r", encoding="utf-8") as f:
                self.kb_documents.extend(json.load(f))

        except FileNotFoundError as e:
            logger.warning(f"Knowledge base file missing: {e}. Index will be empty.")

        self.kb_documents = [doc for doc in self.kb_documents if doc.get("text", "").strip()]

    async def _build_faiss_index(self):
        """Embeds the knowledge base and builds the FAISS vector search index."""
        if not self.kb_documents:
            return

        texts = [doc.get("text", "") for doc in self.kb_documents]

        logger.info(f"Generating embeddings for {len(texts)} KB documents locally via fastembed...")
        embeddings_np = self._generate_embeddings(texts)

        self.index = faiss.IndexFlatL2(_EMBEDDING_DIM)
        self.index.add(embeddings_np)

        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    # ------------------------------------------------------------------
    # Document Preprocessing (Item 3)
    # ------------------------------------------------------------------
    @staticmethod
    def preprocess_contract_text(text: str) -> str:
        """Remove non-contractual noise from raw document text.

        Strips signature blocks, page headers/footers, page numbers,
        company seals, blank OCR fragments, and isolated labels that
        should never reach the LLM for analysis.
        """
        lines = text.split("\n")
        cleaned: list[str] = []
        all_patterns = _SIGNATURE_PATTERNS + _NOISE_PATTERNS

        for line in lines:
            stripped = line.strip()

            # Skip blank or near-blank lines (< 3 meaningful chars)
            meaningful = re.sub(r"[^a-zA-Z0-9]", "", stripped)
            if len(meaningful) < 3:
                continue

            # Skip lines matching any noise/signature pattern
            if any(re.match(pat, stripped) for pat in all_patterns):
                continue

            cleaned.append(line)

        result = "\n".join(cleaned).strip()

        # Remove trailing signature blocks (everything after the last
        # substantive clause, identified by common signature-area markers)
        sig_block_pattern = r"(?i)\n\s*(IN\s+WITNESS\s+WHEREOF|SIGNED\s+AND\s+DELIVERED|EXECUTED\s+ON|FOR\s+AND\s+ON\s+BEHALF)\b.*$"
        result = re.sub(sig_block_pattern, "", result, flags=re.DOTALL).strip()

        return result

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------
    async def smart_retrieve(self, query: str, top_k: int = 3) -> tuple[list[dict], float]:
        """Embeds the query and searches the FAISS index for the most relevant context."""
        if self.index is None or self.index.ntotal == 0:
            return [], 2.0

        query_vec = self._generate_embeddings([query])
        distances, indices = self.index.search(query_vec, top_k)

        results = []
        best_l2_distance = float(distances[0][0]) if len(distances[0]) > 0 else 2.0

        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.kb_documents):
                doc = self.kb_documents[idx].copy()
                doc["_distance"] = float(dist)
                results.append(doc)

        return results, best_l2_distance

    @staticmethod
    def _compute_retrieval_confidence(best_l2_distance: float, context_docs: list[dict]) -> dict:
        """Convert raw L2 distance into a qualitative retrieval confidence object.

        Returns a dict with 'level' and 'reason' suitable for the
        RetrievalConfidence schema.
        """
        num_docs = len(context_docs)

        if best_l2_distance <= _RETRIEVAL_HIGH_THRESHOLD:
            level = "High"
            reason = (
                f"Retrieved {num_docs} highly relevant legal provisions "
                f"with strong semantic match."
            )
        elif best_l2_distance <= _RETRIEVAL_MEDIUM_THRESHOLD:
            level = "Medium"
            reason = (
                f"Retrieved {num_docs} moderately relevant legal provisions. "
                f"The clause may touch on areas not fully covered by the knowledge base."
            )
        else:
            level = "Low"
            reason = (
                f"Retrieved {num_docs} legal provisions with limited semantic match. "
                f"The clause may address areas outside the current knowledge base coverage."
            )

        return {"level": level, "reason": reason}

    # ------------------------------------------------------------------
    # Contract Summary (Item 1 — prerequisite)
    # ------------------------------------------------------------------
    async def generate_contract_summary(self, contract_text: str) -> str:
        """Generate a concise 3-5 sentence summary of the entire contract.

        This summary is provided to the LLM during each clause analysis
        so it understands the full contract context.
        """
        try:
            response = await self.groq_client.chat.completions.create(
                model=_GENERATIVE_MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Indian employment law analyst. "
                            "Summarize the following employment contract in 3-5 sentences. "
                            "Focus on: parties involved, key obligations, compensation structure, "
                            "restrictive covenants, and any unusual or one-sided provisions."
                        ),
                    },
                    {"role": "user", "content": contract_text},
                ],
                temperature=0.1,
                max_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Failed to generate contract summary: {e}")
            return "Unable to generate contract summary."

    # ------------------------------------------------------------------
    # Context-Aware Clause Analysis (Items 1, 2, 4, 5, 6, 9)
    # ------------------------------------------------------------------
    async def analyze_clause(
        self,
        clause_text: str,
        clause_index: int,
        total_clauses: int,
        contract_summary: str,
        previous_analyses: list[dict],
        all_clauses: list[str],
    ) -> dict:
        """Analyze a clause with full contract context, cross-clause awareness,
        categorized law output, calibrated confidence, and retrieval explanations.
        """
        from app.schemas import ClauseAnalysis

        # 1. Retrieve specific legal context (local embedding + FAISS, ~15ms)
        context_docs, best_l2_distance = await self.smart_retrieve(clause_text, top_k=3)
        context_text = "\n\n".join([
            f"Source: {doc.get('title', 'Unknown')}\nText: {doc.get('text', '')}"
            for doc in context_docs
        ])

        # 2. Compute qualitative retrieval confidence (Item 2)
        retrieval_conf = self._compute_retrieval_confidence(best_l2_distance, context_docs)

        # 3. Build previous-clause context for cross-clause reasoning (Item 1)
        prev_context = ""
        if previous_analyses:
            prev_summaries = []
            for i, prev in enumerate(previous_analyses):
                prev_summaries.append(
                    f"Clause {i + 1}: {prev.get('clause_summary', 'N/A')} "
                    f"[Risk: {prev.get('risk_level', 'N/A')}]"
                )
            prev_context = "\n".join(prev_summaries)

        # 4. Build all-clauses reference list
        clauses_list = "\n".join(
            f"Clause {i + 1}: {c[:120]}{'...' if len(c) > 120 else ''}"
            for i, c in enumerate(all_clauses)
        )

        # 5. Build the comprehensive system prompt
        json_schema = json.dumps(ClauseAnalysis.model_json_schema(), indent=2)

        system_instruction = (
            "You are an expert Indian Corporate and Constitutional Lawyer.\n\n"
            "Analyze the provided employment contract clause for violations of the "
            "Constitution of India (Fundamental Rights), the Indian Contract Act, 1872, "
            "and other applicable Indian labour laws.\n\n"
            "CONTEXT AWARENESS (CRITICAL):\n"
            "- Do NOT analyze this clause in complete isolation.\n"
            "- Consider the clause in the context of the entire employment agreement.\n"
            "- Previous or subsequent clauses may strengthen, weaken, or alter the legal interpretation.\n"
            "- If another clause materially affects your reasoning, explicitly mention the relationship in cross_clause_notes.\n\n"
            "APPLICABLE LAWS (CRITICAL — Item 4):\n"
            "- Only include constitutional provisions, statutes, labour laws, or judicial precedents "
            "that are genuinely applicable to this clause.\n"
            "- If no specific legal authority directly applies, return empty lists rather than "
            "guessing or selecting loosely related provisions.\n"
            "- Categorize each cited authority into: constitutional_articles, statutory_provisions, "
            "labour_laws, or judicial_precedents.\n\n"
            "CONFIDENCE SCORE (Item 6):\n"
            "- The confidence_score should reflect: retrieval quality, legal clarity, "
            "certainty of the cited legal authority, and ambiguity of the contractual language.\n"
            "- If legal interpretation depends on judicial discretion, lower the confidence.\n"
            "- Do NOT assign confidence scores arbitrarily or cluster them around 70-80.\n"
            "- llm_certainty should be your honest assessment of how certain you are about "
            "the legal principles applied.\n\n"
            "RETRIEVAL REASONING (Item 9):\n"
            "- For each retrieved legal source used, provide a concise one-line explanation "
            "of why it was relevant (e.g. \"Matched 'non-compete' with Section 27 (restraint of trade).\").\n"
            "- Place these in the retrieval_reasoning array.\n\n"
            "CROSS-CLAUSE NOTES (Item 1):\n"
            "- If this clause independently violates the law, state so.\n"
            "- If it reinforces another clause, creates an unlawful combination, "
            "or forms part of a broader restrictive pattern, note that.\n"
            "- Example: 'This clause alone may appear valid, however when combined with "
            "Clause 3 it effectively restrains employment.'\n\n"
            f"You MUST return a valid JSON object matching this exact schema:\n{json_schema}\n\n"
            "Return ONLY the JSON object, no other text."
        )

        # 6. Build the user prompt with full context
        prompt_parts = [
            f"CONTRACT SUMMARY:\n{contract_summary}",
            f"\nALL CLAUSES IN THIS CONTRACT:\n{clauses_list}",
        ]
        if prev_context:
            prompt_parts.append(f"\nPREVIOUSLY ANALYZED CLAUSES:\n{prev_context}")
        prompt_parts.append(f"\nRETRIEVED LEGAL CONTEXT:\n{context_text}")
        prompt_parts.append(
            f"\nCLAUSE TO ANALYZE (Clause {clause_index + 1} of {total_clauses}):\n{clause_text}"
        )

        prompt = "\n".join(prompt_parts)

        # 7. Call Groq LPU
        try:
            response = await self.groq_client.chat.completions.create(
                model=_GENERATIVE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )

            result = json.loads(response.choices[0].message.content)

            # Override retrieval_confidence with our computed value (Item 2)
            result["retrieval_confidence"] = retrieval_conf

            # Hybrid Confidence Formulation (Item 6)
            retrieval_score = max(0.0, min(1.0, 1.0 - (best_l2_distance / 2.0)))
            llm_cert_val = result.get("llm_certainty", 50)
            llm_cert_float = max(0.0, min(1.0, float(llm_cert_val) / 100.0))
            hybrid_float = (retrieval_score * 0.4) + (llm_cert_float * 0.6)

            result["llm_certainty"] = int(llm_cert_val)
            result["confidence_score"] = int(hybrid_float * 100)

            return result

        except Exception as e:
            logger.error(f"Error during Groq analysis: {e}")
            return {
                "clause_summary": "Error analyzing clause.",
                "potential_violation": {"is_violation": False, "articles": []},
                "applicable_laws": {
                    "constitutional_articles": [],
                    "statutory_provisions": [],
                    "labour_laws": [],
                    "judicial_precedents": [],
                },
                "legal_reasoning": f"Analysis failed due to an internal error: {str(e)}",
                "risk_level": "None",
                "llm_certainty": 0,
                "confidence_score": 0,
                "retrieval_confidence": {"level": "Low", "reason": "Analysis failed."},
                "retrieval_reasoning": [],
                "cross_clause_notes": [],
            }

    # ------------------------------------------------------------------
    # Executive Summary (Items 7 & 8)
    # ------------------------------------------------------------------
    async def generate_executive_summary(
        self,
        contract_summary: str,
        clause_analyses: list[dict],
        all_clauses: list[str],
    ) -> dict:
        """Generate a contract-level executive summary after all clause analyses.

        This identifies cross-clause interactions, systemic risks, and provides
        actionable recommendations.
        """
        from app.schemas import ExecutiveSummary

        # Build a compact representation of all clause analyses
        analyses_text = []
        for i, analysis in enumerate(clause_analyses):
            clause_ref = f"Clause {i + 1}"
            analyses_text.append(
                f"{clause_ref}:\n"
                f"  Summary: {analysis.get('clause_summary', 'N/A')}\n"
                f"  Risk: {analysis.get('risk_level', 'N/A')}\n"
                f"  Violation: {analysis.get('potential_violation', {}).get('is_violation', False)}\n"
                f"  Cross-clause notes: {analysis.get('cross_clause_notes', [])}\n"
            )

        json_schema = json.dumps(ExecutiveSummary.model_json_schema(), indent=2)

        system_instruction = (
            "You are an expert Indian Corporate and Constitutional Lawyer.\n\n"
            "Generate an executive summary of the entire employment contract based on "
            "the individual clause analyses provided below.\n\n"
            "CROSS-CLAUSE REASONING (Item 8 — CRITICAL):\n"
            "- Explicitly identify interactions between clauses.\n"
            "- Examples:\n"
            "  • Certificate retention + employment bond together create coercive employment restrictions.\n"
            "  • Non-compete + certificate retention together significantly restrict occupational freedom.\n"
            "  • Dispute waiver + unilateral CEO decision together substantially reduce access to legal remedies.\n"
            "  • Multiple discriminatory provisions may indicate a broader pattern of unfair employment practices.\n"
            "- Reason about combined effects rather than treating each clause independently.\n\n"
            "The report should:\n"
            "- Summarize the contract as a whole.\n"
            "- Identify patterns across multiple clauses.\n"
            "- Explain how clauses interact.\n"
            "- Highlight systemic risks.\n"
            "- Provide actionable recommendations before signing.\n\n"
            f"You MUST return a valid JSON object matching this exact schema:\n{json_schema}\n\n"
            "Return ONLY the JSON object, no other text."
        )

        prompt = (
            f"CONTRACT SUMMARY:\n{contract_summary}\n\n"
            f"ALL CLAUSE ANALYSES:\n{''.join(analyses_text)}"
        )

        try:
            response = await self.groq_client.chat.completions.create(
                model=_GENERATIVE_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
            )

            return json.loads(response.choices[0].message.content)

        except Exception as e:
            logger.error(f"Error generating executive summary: {e}")
            return {
                "overall_risk_score": 0,
                "overall_risk_level": "None",
                "critical_violations": 0,
                "moderate_concerns": 0,
                "unenforceable_clauses": [],
                "constitutional_issues": [],
                "most_serious_issue": None,
                "cross_clause_observations": [],
                "recommended_actions": [
                    f"Executive summary generation failed: {str(e)}. "
                    "Please review the individual clause analyses."
                ],
            }

    # ------------------------------------------------------------------
    # Chat Streaming
    # ------------------------------------------------------------------
    async def stream_chat(
        self,
        query: str,
        history: list[dict],
        system_instruction: str,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat response via Groq with RAG-augmented context."""
        context_docs, _ = await self.smart_retrieve(query, top_k=3)
        context_text = "\n\n".join([
            f"Source: {doc.get('title', 'Unknown')}\nText: {doc.get('text', '')}"
            for doc in context_docs
        ]) if context_docs else "No specific legal context found for this query."

        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": f"[RETRIEVED LEGAL CONTEXT]\n{context_text}"},
            {"role": "assistant", "content": "I have noted the relevant legal provisions. How can I help?"},
        ]

        for msg in history:
            messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": query})

        stream = await self.groq_client.chat.completions.create(
            model=_GENERATIVE_MODEL_NAME,
            messages=messages,
            temperature=0.3,
            stream=True,
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content

    # ------------------------------------------------------------------
    # Clause Splitting
    # ------------------------------------------------------------------
    @staticmethod
    def split_clauses(text: str) -> list[str]:
        """Split raw contract text into individual clauses."""
        # Try numbered / lettered clause splitting first.
        pattern = r"""
            (?:                          # clause-number variants
                ^\s*\d+[\.\\)]\s          # 1. or 1)
              | ^\s*\([a-zA-Z0-9]+\)\s   # (a) or (1)
              | ^\s*[a-zA-Z][\.\\)]\s      # a. or a)
            )
        """
        splits = re.split(pattern, text, flags=re.MULTILINE | re.VERBOSE)
        clauses = [c.strip() for c in splits if c and c.strip()]

        # Fallback: if we only got a single block, split on sentence endings.
        if len(clauses) <= 1:
            clauses = [
                s.strip()
                for s in re.split(r"(?<=[.;])\s+", text)
                if s.strip()
            ]

        return clauses