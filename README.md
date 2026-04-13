# Legal Clause Analyzer Backend

FastAPI backend for clause-level legal analysis using:

- clause splitting
- FAISS-backed retrieval
- Gemini reasoning
- persistent per-document chat sessions
- constitutional, statutory, and case-law retrieval

## Production Notes

- This service currently keeps chat sessions in process memory.
- Run with a single worker unless you replace session storage with an external shared store.
- Set explicit `CORS_ORIGINS` and `TRUSTED_HOSTS` in production.
- Do not commit a real `.env` file.

## Environment Variables

Copy `.env.example` to `.env` and update the values:

```env
GEMINI_API_KEY=your_gemini_api_key_here
ENVIRONMENT=production
DEBUG=false
CORS_ORIGINS=https://your-frontend.example.com
TRUSTED_HOSTS=your-backend.example.com
```

Important runtime settings:

- `ANALYSIS_CONCURRENCY`: maximum concurrent clause analyses
- `MAX_PDF_SIZE_BYTES`: upload size limit
- `MAX_CLAUSES_PER_DOCUMENT`: protects the service from oversized documents
- `SESSION_TTL_SECONDS`: chat session expiry time
- `MAX_CHAT_SESSIONS`: in-memory session cap

## Local Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker Run

```bash
docker build -t legal-clause-analyzer-be .
docker run --env-file .env -p 8000:8000 legal-clause-analyzer-be
```

## Verification

```powershell
python -m py_compile app\rag_core.py app\settings.py app\session_store.py main.py
python -m unittest discover -s tests
```
