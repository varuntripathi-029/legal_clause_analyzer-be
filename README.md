---
title: Legal RAG Assistant
emoji: ⚖️
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---
# Legal Contract Analyzer — Backend

A RAG-powered FastAPI service that takes Indian employment contracts (PDF), splits them into clauses, and checks each one against the Constitution of India and the Indian Contract Act, 1872. It returns risk levels, confidence scores, legal reasoning, and lets you chat with the analyzed document afterwards.

**Stack:** FastAPI · Gemini 2.5 Flash · FAISS · Redis · Docker

---

## Problems I Ran Into (and How I Fixed Them)

---

### 1. Splitting Contracts Into Clauses Is Harder Than It Sounds

**The problem:** I assumed I could just split on sentences or newlines. Nope. Real employment contracts have nested numbering like `1.`, `(a)`, `(ii)`, sub-clauses inside sub-clauses, and zero consistency between different PDFs. Sentence splitting was giving me broken clauses — half a legal thought in one chunk, the other half in the next.

**What I did:** Wrote a multi-pattern regex that tries numbered/lettered clause markers first (`1.`, `(a)`, `a)`, etc.). If that fails (like when the contract is just a wall of text), it falls back to splitting on sentence-ending punctuation (`.` and `;`). It's not perfect, but it handles the majority of real-world contracts I tested with.

---

### 2. Building the Legal Knowledge Base Was a Manual Grind

**The problem:** Constitutional articles, Contract Act sections, and Supreme Court judgments all come in completely different formats — some from PDFs, some from legal websites, some from bare text files. There was no clean dataset I could just plug in.

**What I did:** Manually extracted the relevant provisions and converted everything into a common JSON format. Each entry has `text`, `source`, `reference` (like "Section 27" or "Article 19"), and `type` (statute vs case law). Not glamorous, but it gave me full control over what goes into the vector index.

**Another problem with case law:** Supreme Court judgments can be hundreds of pages long. If I indexed them as-is, retrieval would pull back huge irrelevant paragraphs instead of the actual legal principle I needed.

**Fix:** I hand-picked landmark cases relevant to employment rights (like *Vishaka v. State of Rajasthan*, *Olga Tellis v. BMC*, *Niranjan Shankar Golikari*), extracted just the key holdings and constitutional principles, and stored them as short, citation-ready chunks. Much better retrieval quality.

---

### 3. Contract Clauses Don't Talk Like the Constitution

**The problem:** A non-compete clause in a contract will never say "this violates Article 19(1)(g) — freedom of profession." It'll say something like "the employee shall not engage in any competing business for 2 years after termination." Keyword-based search was basically useless here.

**What I did:** Used semantic embeddings (Gemini's `gemini-embedding-001`) so the system retrieves based on *meaning*, not exact words. A clause about restricting future employment now correctly pulls up Article 19(1)(g) and Section 27 of the Contract Act, even though they share zero keywords.

---

### 4. Retrieval: Too Little Context vs. Too Much

**The problem:** When I set `top_k=1`, the system would miss important legal references. When I cranked it up to `top_k=10`, the LLM got flooded with irrelevant context, slowed down, and started giving wishy-washy answers.

**What I did:** Settled on `top_k=3` after testing. Three relevant documents give enough legal grounding without turning the prompt into a wall of text. The FAISS L2 distance also helps — if the top result is already far away, the system knows retrieval confidence is low.

---

### 5. LLM Confidence Alone Is Not Trustworthy

**The problem:** I initially asked Gemini to self-report a confidence score. It would say "95% confident" on answers where the retrieved context was barely relevant. Pure LLM confidence is just vibes.

**What I did:** Built a hybrid confidence formula that blends two signals:
- **Retrieval match** (40% weight) — based on how close the FAISS L2 distance was. Closer = more relevant context was found.
- **LLM certainty** (60% weight) — the model's self-reported confidence, but now anchored by real retrieval quality.

This way, if the model is confident but the retrieval was weak, the final score drops. If both are strong, the score is high. Much more calibrated than trusting either signal alone.

---

### 6. The 10-Second Wait Was Killing the UX

**The problem:** When a contract has 15-20 clauses and each one needs an embedding call + LLM generation, processing them one-by-one took 10-15 seconds. The user just stared at a loading spinner with no feedback.

**What I did two things:**

**Concurrency:** Wrapped all clause analyses in `asyncio.create_task()` and used `asyncio.Semaphore(5)` to process up to 5 clauses in parallel. The semaphore prevents hitting Gemini's rate limits while still being way faster than sequential.

**SSE Streaming:** Switched from returning one big JSON blob to streaming results via Server-Sent Events (`text/event-stream`). Now clauses pop up on the frontend one-by-one as they finish. First result shows up in under a second instead of waiting for all 15 to complete. Used `asyncio.as_completed()` so faster clauses arrive first regardless of their position in the document.

---

### 7. Chat Sessions Kept Disappearing

**The problem:** After analyzing a contract, users want to ask follow-up questions ("what does clause 5 mean?", "is this enforceable?"). I initially stored the conversation state in Python memory, but it disappeared whenever the server restarted or if I ran multiple workers.

**What I did:** Moved session state to Redis. Each analysis creates a `session_id`, and the full Gemini chat history (serialized as JSON) gets stored with a 1-hour TTL. To prevent Redis from filling up, I track sessions in a Sorted Set (ZSET) ordered by timestamp and auto-evict the oldest ones when it hits 250 sessions.

For local development without Redis, there's an automatic in-memory fallback with the same interface — so the app works with or without Redis configured.

---

### 8. Getting the LLM to Return Consistent JSON

**The problem:** Gemini would sometimes return markdown, sometimes JSON with extra fields, sometimes JSON with missing fields. The frontend needs a predictable schema every time.

**What I did:** Used Gemini's `response_mime_type="application/json"` with a `response_schema` set to my Pydantic model (`ClauseAnalysis`). This forces the model to output JSON that matches my exact schema — `clause_summary`, `potential_violation`, `applicable_laws`, `legal_reasoning`, `risk_level`, `confidence_score`. No more parsing surprises.

---

### 9. Deploying on Free Infrastructure Was Its Own Battle

**The problem:** Hugging Face Spaces gives you a free Docker container, but RAM is limited, startup times matter, and some Python packages are huge. I also hit issues with:
- `Invalid Host header` errors (the HF proxy uses a different hostname than `localhost`)
- Port mismatches between the Dockerfile and HF's expected `app_port`

**What I did:**
- Added `*.hf.space` to `TrustedHostMiddleware` allowed hosts
- Matched the Dockerfile `EXPOSE` and `CMD` port to `7860` (what HF expects)
- Used `python:3.11-slim` as the base image to keep the container small
- Set `--workers 1` since the free tier doesn't have enough RAM for multiple Uvicorn workers

---

### 10. Balancing Cost, Speed, and Accuracy

**The problem:** Gemini embeddings and reasoning give great quality but every API call costs time and adds external dependency. For a portfolio project, I needed it to be fast enough to demo live without awkward pauses.

**What I explored:**
- Looked into replacing Gemini embeddings with local models like `BAAI/bge-small-en-v1.5` (runs on CPU, no API call needed, cuts embedding latency to near-zero)
- Considered Groq-hosted open-source LLMs for faster inference
- Ended up sticking with Gemini for quality, but the architecture is designed so swapping the embedding model or LLM is a one-file change in `rag_core.py`

---

## Running It Locally

```bash
# Clone and set up
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate
pip install -r requirements.txt

# Create .env with your Gemini API key
echo "GEMINI_API_KEY=your_key_here" > .env

# Start
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger docs at `http://localhost:8000/docs`.

## Docker

```bash
docker build -t legal-rag-backend .
docker run -p 7860:7860 --env-file .env legal-rag-backend
```

---

## API Endpoints

| Method | Path | What it does |
|--------|------|-------------|
| `POST` | `/api/analyze` | Upload a PDF → get clause-by-clause legal analysis streamed via SSE |
| `POST` | `/api/preview` | Upload a PDF → see how it splits into clauses (no LLM call) |
| `POST` | `/api/analyze/clause` | Send a single clause text → get analysis |
| `POST` | `/api/chat` | Ask follow-up questions about an analyzed contract |
| `GET` | `/api/status` | Pipeline readiness, model info, FAISS index size |
| `GET` | `/api/kb` | List all documents in the legal knowledge base |
| `GET` | `/api/kb/stats` | Stats about KB sources and document types |
| `GET` | `/` | Health check |