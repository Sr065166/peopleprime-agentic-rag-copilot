# PeoplePrime Technologies — Enterprise HR Policy & Employee Support Agentic RAG Copilot

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.0%2B-orange.svg)](https://github.com/langchain-ai/langgraph)
[![Pinecone](https://img.shields.io/badge/Pinecone-v10-blueviolet.svg)](https://www.pinecone.io/)
[![Groq](https://img.shields.io/badge/Groq-API-F55036.svg)](https://groq.com/)
[![Embeddings](https://img.shields.io/badge/all--MiniLM--L6--v2-384--d-yellow.svg)](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

An enterprise-grade, deterministic **Agentic RAG Copilot** built for **PeoplePrime Technologies** (a fictional enterprise of 5,000 employees).

The Copilot resolves employee inquiries across company leave, remote-work guidelines, attendance, payroll, benefits, onboarding, offboarding, and conduct policies. It prioritizes private company documents, grades retrieved evidence using Groq LLM, falls back conditionally to web search (Tavily) when private knowledge is insufficient, rewrites queries within bounded retries (max 1), and logs all execution traces to an ACID-compliant SQLite audit store.

> **Disclaimer**: All company policies, documents, and rules described herein are strictly **fictional** and created solely for portfolio demonstration and system evaluation purposes. Never treat generated responses as legal or real-world employment guidance.

---

## 1. System Architecture

```text
                                 [ Employee Web UI ]
                       (HTML5 / CSS3 Dark / Vanilla ES6 JavaScript)
                                          │
                                          ▼ POST /api/chat
                                   [ FastAPI API ]
                                          │
                                          ▼
                         [ LangGraph Agentic Controller ]
                                          │
               ┌──────────────────────────┴──────────────────────────┐
               ▼ (1. Intent Classifier)                              ▼ (Greetings / Chit-chat)
        [ Policy Query ]                                     [ Direct Friendly Answer ]
               │                                                (Zero External API Calls)
               ▼ (2. Private Retrieval)
   [ Pinecone Vector DB (384-d) ]
   (Local Hugging Face all-MiniLM-L6-v2)
               │
               ▼ (3. Grade Evidence: GOOD vs WEAK)
       ┌───────┴───────┐
       ▼ (GOOD)        ▼ (WEAK)
[ Grounded KB Answer ] [ 4. Tavily Web Fallback ]
                       │
                       ▼ (5. Grade Web Evidence: GOOD vs WEAK)
               ┌───────┴───────┐
               ▼ (GOOD)        ▼ (WEAK & Retry < 1)
        [ Web Answer ]   [ 6. Query Rewrite & Retry ]
        (With Disclaimer)      │
                               ▼
                         [ 7. Insufficient Evidence Fallback ]
```

### Core Architectural Decisions & Invariants:
1. **Private KB Priority**: Internal HR knowledge base is always queried first. Public search is unreachable when internal evidence is graded `GOOD`.
2. **Local Hugging Face Embeddings**: Document and query embeddings are generated locally using `sentence-transformers/all-MiniLM-L6-v2` (output dimension: **384**). Text is never sent to external embedding APIs.
3. **Pinecone Index Integrity**: Enforces dimension **384** with `cosine` metric under namespace `peopleprime-policies`.
4. **LLM Inference**: Groq API (`langchain-groq`) using default model `openai/gpt-oss-120b` with strict temperature (`0.0`). Alternate models such as `llama-3.3-70b-versatile` require confirming organization-level access and service-tier rate limits.
5. **Deterministic Greeting Fast-Path**: Instant greetings (`hello`, `hi`, `good morning`) return directly without invoking Groq, Pinecone, or Tavily.
6. **Bounded Retries**: Maximum retry limit strictly capped at 1 to prevent infinite graph execution loops.
7. **Audit & Governance**: Parameterized SQLite logging (`data/audit.db`) records timestamps, question hashes (SHA-256), workflow stages, evidence grades, source types, retry counts, latency, and final outcomes.

---

## 2. Project Layout

```text
solution/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py              # REST endpoints: /api/health, /api/chat, /api/documents/ingest
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Pydantic BaseSettings & env resolution
│   │   └── logging.py             # Structured logging setup
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── state.py               # Typed AgentState & Pydantic decision models
│   │   ├── embeddings.py          # Local all-MiniLM-L6-v2 singleton provider (384-d)
│   │   ├── vectorstore.py         # Pinecone index verification, upsert & similarity search
│   │   └── workflow.py            # LangGraph StateGraph nodes, edges & routing logic
│   ├── services/
│   │   ├── __init__.py
│   │   ├── audit.py               # SQLite audit log storage & hashing
│   │   ├── ingestion.py           # Multi-format document parser (PDF, DOCX, TXT, MD) & chunking
│   │   └── web_search.py          # Tavily search fallback integration
│   └── main.py                    # FastAPI root application & lifecycle initialization
├── static/
│   ├── css/style.css              # Dark-mode dashboard stylesheet & metric cards
│   └── js/app.js                  # Frontend client: chat stream, live trace & upload modal
├── templates/
│   └── index.html                 # Semantic HTML5 3-panel UI shell
├── tests/
│   ├── __init__.py
│   ├── test_health.py             # Health check & home page template tests
│   ├── test_embeddings.py         # Local Hugging Face embedding dimension & norm tests
│   ├── test_ingestion.py          # Multi-format parsing & deterministic chunking tests
│   ├── test_vectorstore.py        # Pinecone index verification & retrieval tests
│   ├── test_ingest_api.py         # Ingestion endpoint authorization & size validation tests
│   ├── test_workflow.py           # LangGraph workflow branch tests (KB, Web, Rewrite, Greetings)
│   ├── test_chat_api.py           # Chat API endpoint validation & error tests
│   ├── test_audit.py              # SQLite operational auditing & question hash tests
│   └── test_frontend_integration.py # Zero-secret checks, asset delivery & safe URL tests
├── Dockerfile                     # Multi-stage production container configuration
├── .dockerignore
├── .gitignore                     # Git protection for .env, sqlite DB, and caches
├── .env.example                   # Secure configuration template
├── requirements.txt               # Application dependencies
└── run.py                         # Development server launcher
```

---

## 3. Local Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.14 (64-bit)
- Git
- Virtual environment tool (`venv` or `conda`)

### Step 1: Clone & Navigate
```powershell
cd "c:\Users\Sahil Raj\Documents\Agentic-RAG-FDE\solution"
```

### Step 2: Create & Activate Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

Edit `.env` and fill in your actual credentials:
```env
APP_NAME="PeoplePrime HR Policy Agentic RAG Copilot"
APP_ENV="development"

# LLM Provider (Groq API)
GROQ_API_KEY="gsk_your_groq_api_key_here"
GROQ_MODEL="openai/gpt-oss-120b"
# Note: Alternate models like "llama-3.3-70b-versatile" depend on your Groq organization tier and current rate limits.

# Vector Database (Pinecone)
PINECONE_API_KEY="pcsk_your_pinecone_api_key_here"
PINECONE_INDEX_NAME="peopleprime-hr-kb-384"
PINECONE_NAMESPACE="peopleprime-policies"
PINECONE_CLOUD="aws"
PINECONE_REGION="us-east-1"

# Local Embedding Model (384 Dimensions)
EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION=384

# Web Search API (Tavily Fallback)
TAVILY_API_KEY="tvly-your_tavily_api_key_here"

# Retrieval & Workflow
TOP_K=4
MAX_RETRIES=1

# Security
ADMIN_API_KEY="your-strong-admin-token"
```

---

## 4. Running Automated Tests

All tests run locally using deterministic mocks for cloud services (Pinecone, Groq, Tavily), enabling offline verification:

```powershell
# Run the complete test suite (67 passed tests):
python -m pytest tests -v
```

---

## 5. Running the Application

### Development Server:
```powershell
python run.py
# Or directly via uvicorn:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Open **`http://localhost:8000`** in your browser.
Open **`http://localhost:8000/docs`** for interactive Swagger API documentation.

---

## 6. Document Ingestion

Authorized administrators can index policy documents (`.pdf`, `.docx`, `.txt`, `.md`) into Pinecone:

### Option A: Using the Web UI
1. Click **+ Upload Policy Document** in the navigation sidebar.
2. Enter your `ADMIN_API_KEY`.
3. Choose a document (e.g. `PeoplePrime_Employee_Handbook.pdf`).
4. Click **Index Document**.

### Option B: Using cURL or PowerShell
```powershell
curl -X POST "http://localhost:8000/api/documents/ingest" `
  -H "x-admin-key: your-strong-admin-token" `
  -F "file=@c:\path\to\PeoplePrime_Handbook.pdf"
```

---

## 7. Cloud Deployment Guide: Render Free Web Service ($0 Budget)

This project includes a native `render.yaml` blueprint and `Dockerfile.render` designed specifically for **Render Free Web Service** with zero hosting costs.

### Render Free Tier Runtime Characteristics & Constraints:
- **Cost**: **$0.00** (Free compute plan, zero paid add-ons, no credit card required).
- **RAM Limit**: **512 MB**.
- **Ephemeral Filesystem**: SQLite audit logs (`data/audit.db`) are ephemeral and reset when the free container spins down or redeploys.
- **Spin-down & Cold Starts**: Render Free services spin down after 15 minutes of inactivity. The first incoming request will experience a cold-start delay (~50–60 seconds) while the container boots and PyTorch initializes. Subsequent warm requests process with normal low latency.
- **Single Worker**: Configured with `--workers 1` to stay within memory limits.

### Step-by-Step Render Deployment:
1. **Push to GitHub**:
   Ensure all changes are pushed to your repository `https://github.com/Sr065166/peopleprime-agentic-rag-copilot`.
2. **Connect to Render**:
   - Go to the [Render Dashboard](https://dashboard.render.com/).
   - Click **New +** -> **Blueprint**.
   - Connect your GitHub repository `peopleprime-agentic-rag-copilot`.
   - Render detects `render.yaml` automatically and configures the `peopleprime-hr-copilot` service on the **Free** plan.
3. **Configure Encrypted Environment Secrets**:
   Under the service environment settings, supply your secure API keys (never commit these to Git):
   - `GROQ_API_KEY`: `gsk_...`
   - `PINECONE_API_KEY`: `pcsk_...`
   - `TAVILY_API_KEY`: `tvly-...`
   - `ADMIN_API_KEY`: `your-admin-token`
4. **Deploy**:
   - Click **Apply**.
   - Render builds the container using `Dockerfile.render`, checks `/api/health`, and provisions a free HTTPS domain (e.g. `https://peopleprime-agentic-rag-copilot.onrender.com`).

---

### Deployment Target B: DigitalOcean Droplet (Docker / Linux VM)

1. **Create an Ubuntu Droplet**:
   - 2GB RAM / 1 vCPU ($12/mo) or higher.
   - Install Docker & Docker Compose:
     ```bash
     sudo apt-get update
     sudo apt-get install -y docker.io
     sudo systemctl enable --now docker
     ```

2. **Clone Repository & Set Environment**:
     ```bash
     git clone <YOUR_REPO_URL>
     cd solution
     cp .env.example .env
     nano .env   # Enter production keys
     ```

3. **Build & Run Docker Container**:
     ```bash
     docker build -t peopleprime-copilot .
     docker run -d --name hr-copilot -p 80:8000 --env-file .env peopleprime-copilot
     ```

4. **Verify Health Endpoint**:
     ```bash
     curl http://localhost/api/health
     # {"status":"ok","service":"PeoplePrime HR Policy Agentic RAG Copilot",...}
     ```

---

## 8. Verification Matrix: Mocked vs. Live Integrations

| Subsystem | Unit & Test Suite State | Live Production Verification |
|---|---|---|
| **API Health & Endpoints** | Verified (`test_health.py`, `test_chat_api.py`) | Verified via local FastAPI & Uvicorn runner (`/api/health`). |
| **Embeddings (all-MiniLM-L6-v2)** | Verified live locally (`test_embeddings.py`) | Real local inference generates 384-dimensional normalized vectors. |
| **Pinecone Vectorstore** | Mocked in automated suite (`test_vectorstore.py`) | **Live Verified**: Connected to Pinecone index `peopleprime-hr-kb-384`, indexed 18 policy chunks, and performed nearest-neighbor similarity search. |
| **Groq LLM Reasoning** | Mocked in automated suite (`test_workflow.py`) | **Live Verified**: Evaluated private evidence as `GOOD` and synthesized grounded policy responses. |
| **Tavily Web Search** | Mocked in automated suite (`test_workflow.py`) | Fallback available via `tavily-python` when `kb_grade == 'WEAK'`. |
| **SQLite Auditing** | Verified (`test_audit.py`) | Live database created at `data/audit.db` logging request hashes and timings. |

---

## 9. Security & Governance Checklist
- [x] Zero hardcoded secrets in source code, templates, or JavaScript client.
- [x] `.env` and `data/*.db` excluded from source control via `.gitignore`.
- [x] Admin authorization enforced via `X-Admin-Key` header on document uploads.
- [x] Upload sanitization and file size limits (15MB maximum; PDF, DOCX, TXT, MD only).
- [x] Prompt injection mitigation: Retrieved chunks are treated as passive data blocks, never executable system instructions.
- [x] SHA-256 question hashing in audit logs to protect employee query confidentiality.
