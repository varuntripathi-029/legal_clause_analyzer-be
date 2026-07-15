import unittest
from app.rag_core import LegalRAGPipeline
from app.schemas import (
    ClauseAnalysis,
    ExecutiveSummary,
    ApplicableLaws,
    RetrievalConfidence,
    MostSeriousIssue,
    PotentialViolation,
)


class PreprocessingAndSchemasTests(unittest.TestCase):
    def test_preprocess_contract_text_removes_signatures_and_noise(self):
        raw_text = """
Page 1 of 5
CONFIDENTIAL

1. Position and Duties
The Employee shall serve as Senior Engineer and perform duties faithfully.

Employee Signature: ____________
Authorized Signatory: ____________
[SEAL]

2. Compensation
The Employee will receive a salary as outlined in Schedule A.

Page 2 of 5
----------------------------------------
IN WITNESS WHEREOF, the parties hereto have signed on this date.
Employee Signature
Witness: John Doe
"""
        cleaned = LegalRAGPipeline.preprocess_contract_text(raw_text)

        # Ensure genuine clauses are preserved
        self.assertIn("1. Position and Duties", cleaned)
        self.assertIn("The Employee shall serve as Senior Engineer and perform duties faithfully.", cleaned)
        self.assertIn("2. Compensation", cleaned)
        self.assertIn("The Employee will receive a salary as outlined in Schedule A.", cleaned)

        # Ensure headers, footers, and signature noise are removed
        self.assertNotIn("Page 1 of 5", cleaned)
        self.assertNotIn("CONFIDENTIAL", cleaned)
        self.assertNotIn("Employee Signature", cleaned)
        self.assertNotIn("Authorized Signatory", cleaned)
        self.assertNotIn("[SEAL]", cleaned)
        self.assertNotIn("IN WITNESS WHEREOF", cleaned)

    def test_compute_retrieval_confidence_levels(self):
        high = LegalRAGPipeline._compute_retrieval_confidence(0.20, [{"text": "doc1"}, {"text": "doc2"}])
        self.assertEqual(high["level"], "High")

        medium = LegalRAGPipeline._compute_retrieval_confidence(0.40, [{"text": "doc1"}])
        self.assertEqual(medium["level"], "Medium")

        low = LegalRAGPipeline._compute_retrieval_confidence(0.65, [])
        self.assertEqual(low["level"], "Low")

    def test_clause_analysis_schema_validation(self):
        data = {
            "clause_summary": "Summary of clause",
            "potential_violation": {"is_violation": True, "articles": ["Article 14"]},
            "applicable_laws": {
                "constitutional_articles": ["Article 14"],
                "statutory_provisions": [],
                "labour_laws": [],
                "judicial_precedents": [],
            },
            "legal_reasoning": "Reasoning text",
            "risk_level": "High",
            "confidence_score": 85,
            "llm_certainty": 80,
            "retrieval_confidence": {"level": "High", "reason": "Strong match"},
            "retrieval_reasoning": ["Matched equality clause with Article 14."],
            "cross_clause_notes": ["Combined with Clause 3, creates restrictive covenant."],
        }
        analysis = ClauseAnalysis.model_validate(data)
        self.assertEqual(analysis.confidence_score, 85)
        self.assertEqual(analysis.applicable_laws.constitutional_articles, ["Article 14"])
        self.assertEqual(analysis.retrieval_confidence.level, "High")
        self.assertEqual(analysis.cross_clause_notes[0], "Combined with Clause 3, creates restrictive covenant.")

    def test_executive_summary_schema_validation(self):
        data = {
            "overall_risk_score": 82,
            "overall_risk_level": "High",
            "critical_violations": 4,
            "moderate_concerns": 1,
            "unenforceable_clauses": ["Clause 2", "Clause 4"],
            "constitutional_issues": ["Article 14", "Article 19(1)(g)"],
            "most_serious_issue": {
                "clause": "Clause 2",
                "reason": "Post-employment non-compete likely void under Section 27.",
            },
            "cross_clause_observations": [
                "Certificate retention + employment bond together create coercive employment restrictions."
            ],
            "recommended_actions": [
                "Negotiate deletion of Clause 2."
            ],
        }
        summary = ExecutiveSummary.model_validate(data)
        self.assertEqual(summary.overall_risk_score, 82)
        self.assertEqual(summary.most_serious_issue.clause, "Clause 2")
        self.assertIn("Certificate retention", summary.cross_clause_observations[0])


if __name__ == "__main__":
    unittest.main()
