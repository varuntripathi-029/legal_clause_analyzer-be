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


class ApplicableLaws(BaseModel):
    """Categorized legal provisions relevant to a clause."""

    constitutional_articles: list[str] = Field(
        default_factory=list,
        description="Constitutional articles (e.g. 'Article 14', 'Article 19(1)(g)').",
    )
    statutory_provisions: list[str] = Field(
        default_factory=list,
        description="Statutory sections (e.g. 'Section 27, Indian Contract Act, 1872').",
    )
    labour_laws: list[str] = Field(
        default_factory=list,
        description="Labour law provisions (e.g. 'Industrial Disputes Act, 1947').",
    )
    judicial_precedents: list[str] = Field(
        default_factory=list,
        description="Relevant case law (e.g. 'Niranjan Shankar Golikari v. Century Spinning').",
    )


class RetrievalConfidence(BaseModel):
    """Qualitative confidence level for the RAG retrieval step."""

    level: str = Field(
        ...,
        description="Qualitative confidence: 'High', 'Medium', or 'Low'.",
    )
    reason: str = Field(
        ...,
        description="Brief explanation of why retrieval confidence is at this level.",
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
    applicable_laws: ApplicableLaws = Field(
        default_factory=ApplicableLaws,
        description="Categorized laws, articles, or sections relevant to this clause. Empty categories if none genuinely apply.",
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
        description="Final hybrid confidence score (0-100). Reflects retrieval quality, legal clarity, authority certainty, and contractual ambiguity.",
    )
    llm_certainty: int = Field(
        ...,
        ge=0,
        le=100,
        description="Calibrated LLM confidence (0-100). Lower if interpretation depends on judicial discretion.",
    )
    retrieval_confidence: RetrievalConfidence = Field(
        ...,
        description="Qualitative retrieval confidence level with explanation.",
    )
    retrieval_reasoning: list[str] = Field(
        default_factory=list,
        description="Concise explanations of why specific legal sources were retrieved.",
    )
    cross_clause_notes: list[str] = Field(
        default_factory=list,
        description="How this clause interacts with, reinforces, or conflicts with other clauses.",
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


class DocumentUploadResponse(BaseModel):
    """Response after document upload and text extraction."""

    extraction_id: str = Field(
        ...,
        description="UUID to reference this extraction in follow-up requests.",
    )
    extracted_text: str = Field(
        ...,
        description="Text extracted from the uploaded document (may need user review).",
    )
    document_type: str = Field(
        ...,
        description="Classification: plain_text | searchable_pdf | scanned_pdf | image.",
    )
    page_count: int = Field(
        ...,
        ge=1,
        description="Number of pages in the document.",
    )
    ocr_confidence: int | None = Field(
        default=None,
        description="Average OCR confidence (0-100). None if no OCR was performed.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="User-facing warnings (e.g. low OCR confidence).",
    )


class AnalyzeFromTextRequest(BaseModel):
    """Request to run legal analysis on user-confirmed / edited text."""

    contract_text: str = Field(
        ...,
        min_length=10,
        description="Confirmed contract text to analyze (may have been edited after OCR).",
    )


class PipelineStatusResponse(BaseModel):
    """Detailed status of the RAG pipeline."""

    status: str
    embedding_model: str
    llm_model: str
    faiss_index_size: int
    kb_documents_loaded: int


# ---------------------------------------------------------------------------
# Executive Summary (generated after all clause analyses)
# ---------------------------------------------------------------------------

class MostSeriousIssue(BaseModel):
    """The single most concerning issue found in the contract."""

    clause: str = Field(
        ...,
        description="Reference to the clause (e.g. 'Clause 2').",
    )
    reason: str = Field(
        ...,
        description="Why this is the most serious issue.",
    )


class ExecutiveSummary(BaseModel):
    """Contract-level executive summary generated after all clause analyses."""

    overall_risk_score: int = Field(
        ...,
        ge=0,
        le=100,
        description="Aggregate risk score for the entire contract (0-100).",
    )
    overall_risk_level: str = Field(
        ...,
        description="Aggregate risk level: High, Medium, Low, or None.",
    )
    critical_violations: int = Field(
        ...,
        ge=0,
        description="Number of clauses with critical (High risk) violations.",
    )
    moderate_concerns: int = Field(
        ...,
        ge=0,
        description="Number of clauses with moderate (Medium risk) concerns.",
    )
    unenforceable_clauses: list[str] = Field(
        default_factory=list,
        description="List of clause references likely unenforceable under Indian law.",
    )
    constitutional_issues: list[str] = Field(
        default_factory=list,
        description="Constitutional articles implicated across the contract.",
    )
    most_serious_issue: MostSeriousIssue | None = Field(
        default=None,
        description="The single most concerning issue in the contract.",
    )
    cross_clause_observations: list[str] = Field(
        default_factory=list,
        description="Observations about how clauses interact, reinforce, or combine to create risks.",
    )
    recommended_actions: list[str] = Field(
        default_factory=list,
        description="Actionable recommendations before signing the contract.",
    )
