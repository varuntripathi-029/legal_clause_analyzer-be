# Legal Contract Analyzer (RAG Backend)

This repository contains the backend service for the Legal Contract Analyzer, a Retrieval-Augmented Generation (RAG) API built with FastAPI and Google Gemini. The system is designed to parse Indian employment contracts, isolate individual clauses, and evaluate them for potential violations of the Constitution of India (specifically Fundamental Rights) and the Indian Contract Act, 1872.

## Core Capabilities

* **Document Parsing & Clause Extraction:** Intelligently processes uploaded PDF contracts, segmenting the raw text into distinct, analyzable legal clauses.
* **Context-Aware Retrieval (RAG):** Utilizes `BAAI/bge-small-en-v1.5` embeddings alongside a FAISS vector database to retrieve highly relevant legal precedents, including Constitutional Articles, Statutes, and Case Law.
* **Generative AI Evaluation:** Leverages Google's Gemini model to evaluate each clause against the retrieved legal context. The system returns a calibrated risk level, a confidence score, and detailed, step-by-step legal reasoning.
* **Persistent Document Chat:** Maintains stateful, multi-turn chat sessions using Redis, allowing users to ask follow-up questions and interact directly with the analyzed contract.
* **Cloud-Native Architecture:** Designed for robust cloud deployment. By externalizing the chat state to Redis, the application supports safe multi-worker scaling.
* **Optimized Containerization:** Features a production-ready Docker environment explicitly configured to utilize CPU-optimized PyTorch binaries, ensuring a lightweight footprint and rapid build times.

## Technology Stack

* **Application Framework:** FastAPI, Uvicorn
* **Artificial Intelligence:** Google GenAI SDK (Gemini), Sentence-Transformers
* **Vector Search:** FAISS (CPU)
* **State Management:** Redis (with an automatic in-memory fallback for local development)
* **Document Processing:** PyPDF
* **Infrastructure:** Docker

---

## Installation & Deployment

### Prerequisites
Before running the application, please ensure you have the following installed:
* Python 3.11 or higher
* Docker (recommended for production deployment)
* A valid Google Gemini API Key

### 1. Environment Configuration
Clone the repository and create a `.env` file in the root directory to store your environment variables:

```env
ENVIRONMENT=development
DEBUG=True
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Leave blank to use the in-memory session manager for local testing
REDIS_URL=redis://localhost:6379/0
REDIS_KEY_PREFIX=legal-clause-analyzer
```

### 2. Local Development setup
To run the application locally without Docker, it is recommended to use a Python virtual environment.

```bash
# Create and activate the virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Install the required dependencies
pip install -r requirements.txt

# Initialize the server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Once the server is running, the API will be accessible at `http://localhost:8000`. You may view and test the endpoints using the interactive Swagger documentation at `http://localhost:8000/docs`.

### 3. Docker Deployment
This project includes a Dockerfile optimized for CPU environments, keeping the final image lightweight and cost-effective for standard cloud hosting.

```bash
# Build the Docker image
docker build -t legal-rag-backend .

# Run the container, passing the environment variables
docker run -p 8000:8000 --env-file .env legal-rag-backend
```

---

## Application Programming Interface (API)

The backend exposes several RESTful endpoints to handle document analysis and user interaction.

### Analysis & Processing
* `POST /api/analyze` — Accepts a PDF contract upload. The system extracts the text, splits it into clauses, executes the RAG pipeline on each segment, and returns a comprehensive legal analysis alongside a unique `session_id`.
* `POST /api/preview` — Accepts a PDF upload to preview the clause extraction process without triggering the LLM analysis.
* `POST /api/analyze/clause` — Accepts a single text string for direct evaluation against the knowledge base.

### Document Chat
* `POST /api/chat` — Accepts a user query and a `session_id`, allowing the user to converse with the LLM regarding the specific contract analyzed during that session.

### System Health & Knowledge Base
* `GET /` — Standard health check and liveness probe.
* `GET /api/status` — Returns the current readiness of the RAG pipeline, the loaded embedding models, and the FAISS index size.
* `GET /api/kb` — Retrieves a list of all legal documents currently embedded within the local knowledge base.
* `GET /api/kb/stats` — Provides aggregate statistics regarding the sources and types of legal texts in the knowledge base.
