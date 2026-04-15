"""
Pydantic models for the Legal Contract Analyzer API.

Defines strict request/response schemas for contract analysis,
including clause-level violation reporting and risk scoring.
"""

from pydantic import BaseModel, Field


class ContractRequest(BaseModel):
    """Incoming request containing raw contract text to analyze."""

    contract_text: str = Field(
        ...,
        min_length=10,
        description="Raw text of the employment contract to analyze."
    )


class PotentialViolation(BaseModel):
    """Describes whether a clause violates constitutional or statutory provisions."""

    is_violation: bool = Field(
        ...,
        description="True if the clause potentially violates a legal provision.",
    )
    articles: list[str] = Field(
        default_factory=list,
        description="List of constitutional articles or statutory sections violated."
    )


class ClauseAnalysis(BaseModel):
    """Full analytical breakdown of a single contract clause."""

    clause_summary: str = Field(
        ...,
        description="Plain-English summary of what the clause states.",
    )
    potential_violation: PotentialViolation = Field(
        ...,
        description="Violation assessment for this clause.",
    )
    applicable_laws: list[str] = Field(
        default_factory=list,
        description="Laws, articles, or sections relevant to this clause.",
    )
    legal_reasoning: str = Field(
        ...,
        description="Step-by-step legal reasoning explaining the analysis.",
    )
    risk_level: str = Field(
        ...,
        description="Risk classification: High, Medium, Low, or None."
    )
    confidence_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Final hybrid confidence score from 0 to 100.",
    )
    llm_certainty: int = Field(
        ...,
        description="Confidence of the LLM in its specific reasoning (0-100).",
    )
    retrieval_match: int = Field(
        ...,
        description="Distance-based retrieval match strength (0-100).",
    )


class ClauseResponse(BaseModel):
    """Response object wrapping a clause with its analysis."""

    clause: str = Field(
        ...,
        description="The original clause text extracted from the contract.",
    )
    analysis: ClauseAnalysis = Field(
        ...,
        description="Detailed legal analysis of this clause.",
    )


class ContractAnalysisResponse(BaseModel):
    """Response object containing analysis results and a chat session ID."""
    
    session_id: str = Field(
        ...,
        description="A unique UUID for the conversation thread to use in follow-up chat.",
    )
    results: list[ClauseResponse] = Field(
        ...,
        description="List of legal violation analysis per clause.",
    )


# ---------------------------------------------------------------------------
# Additional schemas for frontend-ready routes
# ---------------------------------------------------------------------------

class SingleClauseRequest(BaseModel):
    """Request to analyze a single clause (without splitting)."""

    clause_text: str = Field(
        ...,
        min_length=5,
        description="A single contract clause to analyze.",
    )


class ChatRequest(BaseModel):
    """Request to send a message within an active document chat session."""

    session_id: str = Field(..., description="The session ID returned from /api/analyze")
    message: str = Field(..., description="User's chat message about the contract.")


class ChatResponse(BaseModel):
    """Response containing the AI's reply to the chat message."""
    
    response: str



class ClausePreviewResponse(BaseModel):
    """Response showing how a contract will be split into clauses."""

    total_clauses: int = Field(
        ...,
        description="Total number of clauses extracted.",
    )
    clauses: list[str] = Field(
        ...,
        description="List of extracted clause texts.",
    )


class KBDocumentResponse(BaseModel):
    """A single knowledge-base document."""

    text: str
    metadata: dict


class KBStatsResponse(BaseModel):
    """Statistics about the loaded knowledge base."""

    total_documents: int
    sources: dict[str, int] = Field(
        ...,
        description="Count of documents per source.",
    )
    types: dict[str, int] = Field(
        ...,
        description="Count of documents per type.",
    )


class PipelineStatusResponse(BaseModel):
    """Detailed status of the RAG pipeline."""

    status: str
    embedding_model: str
    llm_model: str
    faiss_index_size: int
    kb_documents_loaded: int

