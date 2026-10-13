# Implementation Plan: Enterprise HR Policy & Employee Support Agentic RAG Copilot

## Phase Overview

### Phase 0: Discovery, Architecture & Specifications (Current Phase)
- [x] Inspect reference repository architecture, prompts, state graphs, ingestion logic, and UI design.
- [x] Analyze environment constraints: Python runtime, dependency compatibility (`langchain-groq`, `sentence-transformers`, `pinecone`).
- [x] Formulate complete system specification (`solution/PROJECT_SPEC.md`).
- [x] Detail execution roadmap and test matrix (`solution/IMPLEMENTATION_PLAN.md`).

---

### Phase 1: Foundation & Application Shell (Current Phase Target)
- [ ] Establish solution directory structure:
  ```text
  solution/
  ├── app/
  │   ├── api/
  │   │   ├── __init__.py
  │   │   └── routes.py
  │   ├── core/
  │   │   ├── __init__.py
  │   │   ├── config.py
  │   │   └── logging.py
  │   ├── rag/
  │   │   ├── __init__.py
  │   │   ├── state.py
  │   │   ├── vectorstore.py
  │   │   └── workflow.py
  │   ├── services/
  │   │   ├── __init__.py
  │   │   ├── audit.py
  │   │   └── ingestion.py
  │   └── main.py
  ├── data/
  │   └── sample_kb/
  ├── static/
  │   ├── css/style.css
  │   └── js/app.js
  ├── templates/
  │   └── index.html
  ├── tests/
  │   ├── __init__.py
  │   ├── conftest.py
  │   ├── test_health.py
  │   ├── test_ingestion.py
  │   ├── test_workflow_mocked.py
  │   └── test_audit.py
  ├── .env.example
  ├── .gitignore
  ├── Dockerfile
  ├── requirements.txt
  ├── run.py
  ├── PROJECT_SPEC.md
  └── IMPLEMENTATION_PLAN.md
  ```
- [ ] Configure environment and settings (`app/core/config.py`):
  - Support `GROQ_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`, `ADMIN_API_KEY`.
  - Set default embedding model to `sentence-transformers/all-MiniLM-L6-v2` (dimension 384).
  - Set default LLM to `llama-3.3-70b-versatile` / `llama-3.1-8b-instant`.
- [ ] Set up clean `.env.example` and production `.gitignore`.
- [ ] Implement initial `FastAPI` application with `/api/health` and HTML shell mounting.
- [ ] Create initial test suite for health check and configuration integrity.
- [ ] Verify test suite and basic service launch.

---

### Phase 2: Ingestion Pipeline & Local Embeddings
- [ ] Implement multi-format document parser (`app/services/ingestion.py`):
  - Support PDF (`PyPDFLoader` or `pypdf`), DOCX (`python-docx`), TXT, Markdown (`TextLoader`).
  - Metadata preservation (filename, chunk start index, page number).
  - Configurable chunk size (900 chars) and overlap (120 chars).
- [ ] Implement local Hugging Face embedding provider (`app/rag/vectorstore.py`):
  - Load `sentence-transformers/all-MiniLM-L6-v2` with deterministic dimension check (384).
  - Pinecone index lifecycle verification (`ensure_index`).
- [ ] Build authorized `/api/ingest` endpoint with admin authorization header and file validation.
- [ ] Build standalone sample KB seeding script (`ingest_sample_kb.py`).
- [ ] Add unit tests for document loading, chunking, and mock embedding integration.

---

### Phase 3: Agentic RAG Workflow (LangGraph & Groq)
- [ ] Define state schemas (`app/rag/state.py`):
  - `AgentState`: `question`, `current_query`, `kb_docs`, `kb_grade`, `web_results`, `web_grade`, `answer`, `source_used`, `retry_count`, `trace`, `citations`.
  - Pydantic decision models: `RouteDecision` (`kb` | `direct`), `EvidenceGrade` (`good` | `weak`).
- [ ] Implement LangGraph nodes (`app/rag/workflow.py`):
  - `route_question`: Direct response for greetings/chit-chat vs. private KB.
  - `retrieve_kb`: Pinecone similarity search.
  - `grade_kb`: LLM structured grading of private chunks.
  - `search_web`: Tavily query fallback when KB is weak.
  - `grade_web`: LLM structured grading of external search results.
  - `rewrite_query`: Query expansion with HR context. Bounded by `max_retries = 1`.
  - `generate_from_kb`: Grounded generation strictly citing company docs.
  - `generate_from_web`: Grounded generation with explicit external disclaimer.
  - `insufficient`: Graceful fallback explaining that evidence was insufficient.
- [ ] Integrate SQLite auditing (`app/services/audit.py`):
  - Record execution duration, decision route, retry count, evidence grades.
- [ ] Add unit and mock integration tests covering all graph branches:
  - Direct greeting path.
  - Private KB success path.
  - Private KB weak -> Web success path.
  - Private KB weak -> Web weak -> Query rewrite -> Retry path.
  - Insufficient evidence final fallback.

---

### Phase 4: UI & Frontend Experience
- [ ] Build responsive dark-mode web interface (`templates/index.html`, `static/css/style.css`, `static/js/app.js`):
  - Three-column layout: Workflow overview & KB upload modal, Chat stream, Live decision trace & citation inspector.
  - Visual badges for `PRIVATE KB`, `WEB SEARCH`, `DIRECT CHAT`, `INSUFFICIENT EVIDENCE`.
  - Interactive file upload modal with admin key input and real-time progress.
  - Preset questions for one-click employee testing.

---

### Phase 5: Containerization, Final Verification & Documentation
- [ ] Multi-stage Dockerfile optimized for Python + Hugging Face embeddings caching.
- [ ] Comprehensive verification of all test suites.
- [ ] Final handoff documentation with exact launch and operational instructions.
