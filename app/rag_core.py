import json
import logging
import re
from pathlib import Path
import numpy as np
import faiss
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# --- Configuration Constants ---
_EMBEDDING_MODEL_NAME = "gemini-embedding-001"
_EMBEDDING_DIM = 768  # Gemini embeddings set to 768-dimensional using config
_GENERATIVE_MODEL_NAME = "gemini-2.5-flash"

class LegalRAGPipeline:
    def __init__(self, api_key: str):
        """Initializes the RAG Pipeline with the Google GenAI client and FAISS index."""
        logger.info("Initializing Legal RAG Pipeline via Gemini API...")
        
        # Initialize the new Google GenAI client
        self.client = genai.Client(api_key=api_key)
        self.model_name = _GENERATIVE_MODEL_NAME
        
        self.kb_documents = []
        self.index = None
        
        # Load and build the knowledge base upon startup
        self._load_knowledge_base()
        self._build_faiss_index()
        
        logger.info("RAG Pipeline initialized successfully.")

    def _generate_embeddings(self, texts: list[str]) -> np.ndarray:
        """Generate vector embeddings using the Gemini API in batches."""
        if not texts:
            return np.empty((0, _EMBEDDING_DIM), dtype=np.float32)

        all_embeddings = []
        batch_size = 100  # Gemini allows batching multiple texts per request
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Call the Gemini Embedding API
            response = self.client.models.embed_content(
                model=_EMBEDDING_MODEL_NAME,
                contents=batch,
                config=types.EmbedContentConfig(output_dimensionality=_EMBEDDING_DIM)
            )
            
            # Extract and append the vector values
            all_embeddings.extend([emb.values for emb in response.embeddings])
            
        return np.array(all_embeddings, dtype=np.float32)

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
            
        # Ensure we only retain valid documents to prevent empty string API crashes.
        self.kb_documents = [doc for doc in self.kb_documents if doc.get("text", "").strip()]

    def _build_faiss_index(self):
        """Embeds the knowledge base and builds the FAISS vector search index."""
        if not self.kb_documents:
            return
            
        texts = [doc.get("text", "") for doc in self.kb_documents]
        
        logger.info(f"Generating embeddings for {len(texts)} KB documents via Gemini API...")
        embeddings_np = self._generate_embeddings(texts)
        
        # Build the CPU-optimized FAISS index
        self.index = faiss.IndexFlatL2(_EMBEDDING_DIM)
        self.index.add(embeddings_np)
        
        logger.info(f"FAISS index built with {self.index.ntotal} vectors.")

    def smart_retrieve(self, query: str, top_k: int = 3) -> tuple[list[dict], float]:
        """Embeds the query and searches the FAISS index for the most relevant context."""
        if self.index is None or self.index.ntotal == 0:
            return [], 2.0
            
        # Generate embedding for the single search query
        query_vec = self._generate_embeddings([query])
        
        # Search the FAISS index
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        best_l2_distance = float(distances[0][0]) if len(distances[0]) > 0 else 2.0
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.kb_documents):
                doc = self.kb_documents[idx].copy()
                doc["_distance"] = float(dist)
                results.append(doc)
                
        return results, best_l2_distance

    async def analyze_clause(self, clause_text: str) -> dict:
        """Main pipeline entry: Retrieves context and asks Gemini to analyze the clause."""
        # 1. Retrieve specific legal context
        context_docs, best_l2_distance = self.smart_retrieve(clause_text, top_k=3)
        context_text = "\n\n".join([
            f"Source: {doc.get('title', 'Unknown')}\nText: {doc.get('text', '')}"
            for doc in context_docs
        ])
        
        # 2. Build the LLM prompt
        system_instruction = (
            "You are an expert Indian Corporate and Constitutional Lawyer. "
            "Analyze the provided employment contract clause for violations of the "
            "Constitution of India (Fundamental Rights) or the Indian Contract Act, 1872. "
            "Use the provided Legal Context to ground your reasoning.\n\n"
            "CRITICAL RULES FOR CONFIDENCE SCORE:\n"
            "- Extract an honest internal probability of certainty about the legal principles applied.\n"
            "- Return a structured JSON response containing: "
            "clause_summary (string), potential_violation (object with is_violation boolean and articles string list), applicable_laws (string list), legal_reasoning (string), risk_level (Low, Medium, High, None), llm_certainty (integer 0-100), and confidence_score (integer 0-100)."
        )
        
        prompt = f"LEGAL CONTEXT:\n{context_text}\n\nCLAUSE TO ANALYZE:\n{clause_text}"

        from app.schemas import ClauseAnalysis
        
        # 3. Call the Generative Model
        try:
            response = await self.client.aio.models.generate_content(
                model=_GENERATIVE_MODEL_NAME,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=ClauseAnalysis,
                    temperature=0.1,
                )
            )
            
            # Parse JSON
            result = json.loads(response.text)
            
            # Hybrid Confidence Formulation
            retrieval_score = max(0.0, min(1.0, 1.0 - (best_l2_distance / 2.0)))
            
            llm_cert_val = result.get("llm_certainty", 50)
            llm_cert_float = max(0.0, min(1.0, float(llm_cert_val) / 100.0))
            
            hybrid_float = (retrieval_score * 0.4) + (llm_cert_float * 0.6)
            
            result["retrieval_match"] = int(retrieval_score * 100)
            result["llm_certainty"] = int(llm_cert_val)
            result["confidence_score"] = int(hybrid_float * 100)
                
            return result
            
        except Exception as e:
            logger.error(f"Error during Gemini analysis: {e}")
            return {
                "clause_summary": "Error analyzing clause.",
                "potential_violation": {"is_violation": False, "articles": []},
                "applicable_laws": [],
                "legal_reasoning": f"Analysis failed due to an internal error: {str(e)}",
                "risk_level": "None",
                "llm_certainty": 0,
                "retrieval_match": 0,
                "confidence_score": 0
            }

    @staticmethod
    def split_clauses(text: str) -> list[str]:
        """Split raw contract text into individual clauses."""
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