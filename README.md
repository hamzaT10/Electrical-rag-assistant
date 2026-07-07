# Electrical RAG Assistant

Production-style Retrieval-Augmented Generation platform for technical electrical
documentation. The system ingests PDF documents, extracts text with OCR fallback,
indexes chunks in a vector database, retrieves relevant context, and generates
source-grounded answers through an OpenAI-compatible LLM runtime.

This public repository contains the application code, tests, Docker deployment
assets, CI/CD workflows, evaluation framework, and observability configuration.
The private document corpus is intentionally not included.

![Electrical RAG Assistant architecture](docs/assets/architecture-generated.png)

## Highlights

- FastAPI backend for chat, streaming responses, document upload, ingestion jobs,
  chat history, health checks, and Prometheus metrics.
- RAG pipeline with PDF extraction, OCR fallback, chunking, multilingual
  embeddings, Qdrant vector search, metadata filters, context selection, and
  optional cross-encoder reranking.
- PostgreSQL for users, chat sessions, messages, retrieved sources, documents,
  and ingestion job state.
- Redis for response caching, rate limiting, Celery broker, and Celery result
  backend.
- Celery worker for asynchronous PDF ingestion and incremental indexing.
- Frontend chat interface with source citations, streaming answer support,
  document upload, and document filtering.
- Evaluation runner for retrieval quality, answer fact coverage, refusal quality,
  latency, MRR, and failed-case analysis.
- Production observability with structured JSON logs, request IDs, Prometheus,
  Grafana dashboards, and optional Langfuse traces.
- GitHub Actions CI/CD for linting, tests, Docker build verification, and
  container image publishing.

## Architecture

The application is organized as a production RAG service rather than a notebook
prototype.

1. Users upload or select PDF documents from the frontend.
2. FastAPI stores document metadata and creates ingestion jobs in PostgreSQL.
3. Celery workers extract PDF text, apply OCR fallback when needed, chunk the
   content, generate embeddings, and upsert vectors into Qdrant.
4. Chat requests are rate-limited through Redis and persisted in PostgreSQL.
5. The retriever embeds the user question, searches Qdrant, applies metadata
   filters, optionally reranks candidates, and selects a compact context.
6. The LLM provider calls an OpenAI-compatible runtime such as Ollama, OpenAI-compatible LLM runtime,
   or another hosted compatible endpoint.
7. Answers, citations, timings, cache outcomes, and source metadata are returned
   to the frontend and recorded through logs, metrics, and optional traces.

## Tech Stack

| Layer | Technology |
| --- | --- |
| API | FastAPI, Pydantic, Uvicorn |
| Frontend | HTML, CSS, JavaScript, Nginx |
| RAG orchestration | LangChain-compatible components, custom service layer |
| Embeddings | SentenceTransformers multilingual MiniLM |
| Vector database | Qdrant |
| Reranking | Cross-encoder `ms-marco-MiniLM-L-6-v2` |
| LLM runtime | OpenAI-compatible local or hosted endpoint |
| Database | PostgreSQL, SQLAlchemy, Alembic |
| Cache and broker | Redis |
| Async workers | Celery |
| Observability | Prometheus, Grafana, Langfuse, structured JSON logs |
| Deployment | Docker, Docker Compose, GitHub Actions, GHCR |
| Quality | Pytest, Ruff |

## Repository Layout

```text
src/electrical_rag/
  api/              FastAPI app, routes, schemas, middleware
  cache/            Redis cache integration
  core/             Settings and shared configuration
  db/               SQLAlchemy models, sessions, repositories
  evaluation/       Benchmark runner and quality metrics
  observability/    Logging, metrics, request context, tracing
  providers/        OpenAI-compatible LLM provider
  rag/              Ingestion, Qdrant store, retriever, reranker, prompting
  security/         Redis-backed rate limiting
  services/         Application service orchestration
  workers/          Celery app and ingestion tasks

frontend/           Static chat UI served by Nginx
monitoring/         Prometheus and Grafana provisioning
alembic/            Database migrations
evaluation/         Benchmark inputs and documentation
tests/              Unit and integration tests
docs/               Architecture and production learning notes
.github/workflows/ CI/CD pipelines
```

## Data Policy

The public repository does not include private PDFs, extracted text, vector
indexes, uploaded files, `.env` files, or generated evaluation reports.

Use your own documents locally by placing them under a local `Data/` directory or
by uploading PDFs through the UI. The directory is ignored by Git.

Ignored runtime/private artifacts include:

- `Data/`
- `vectorstore/`
- `.env`
- `evaluation/results/`
- `outputs/`
- local virtual environments
- local model caches

## Configuration

Copy the example environment file:

```powershell
Copy-Item .env.example .env
```

Important settings:

```env
VECTOR_BACKEND=qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=electrical_rag_chunks
DATABASE_URL=postgresql+psycopg://electrical_rag:electrical_rag@localhost:5432/electrical_rag
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
EMBEDDING_MODEL_NAME=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
LLM_BASE_URL=http://localhost:1234/v1
LLM_MODEL=openai-compatible-model
ENABLE_RERANKER=false
ENABLE_LANGFUSE=false
```

The `LLM_*` names are compatibility settings for any OpenAI-compatible
runtime. They can point to Ollama, OpenAI-compatible LLM runtime, vLLM, or a hosted compatible API.

## Run With Docker Compose

Start an OpenAI-compatible LLM server first, then run:

```powershell
docker compose up -d --build
```

Open:

- Frontend: `http://localhost:3000`
- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3001`

Useful logs:

```powershell
docker compose logs -f rag-api
docker compose logs -f rag-worker
docker compose logs -f qdrant
```

## Local Development

Install dependencies:

```powershell
pip install -r requirements-app.txt
pip install -r requirements-ocr.txt
```

Set the package path:

```powershell
$env:PYTHONPATH="src"
```

Run database migrations:

```powershell
alembic upgrade head
```

Run the API:

```powershell
uvicorn electrical_rag.api:app --app-dir src --reload
```

Run a worker:

```powershell
celery -A electrical_rag.workers.celery_app worker --loglevel=info
```

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Runtime readiness, vector backend, LLM status, embedding status |
| `GET /meta` | Active model and retrieval configuration |
| `GET /metrics` | Prometheus metrics |
| `POST /chat` | Non-streaming RAG answer |
| `POST /chat/stream` | Streaming RAG answer |
| `GET /chat/sessions` | Chat session history |
| `GET /chat/sessions/{session_id}` | Messages and retrieved sources |
| `POST /documents/upload` | Upload PDF and enqueue ingestion |
| `GET /documents` | List indexed/uploaded documents |
| `POST /ingestion/jobs` | Create ingestion job |
| `GET /ingestion/jobs/{job_id}` | Check ingestion status |

## Evaluation

Run the benchmark:

```powershell
python -m electrical_rag.evaluation.runner
```

The runner measures:

- retrieval hit rate
- mean reciprocal rank
- expected source rank
- answer keyword pass rate
- fact pass rate
- average fact coverage
- no-answer/refusal quality
- top retrieval score
- retrieval latency
- total request latency
- failed-case diagnostics

Evaluation reports are generated locally under `evaluation/results/` and are not
committed.

## Observability

The API emits structured JSON logs with request IDs and RAG timing fields.
Prometheus metrics cover:

- HTTP request rate, latency, and status codes
- Redis cache hits and misses
- rate-limit decisions
- retrieval latency
- reranking latency
- time to first token
- LLM generation latency
- context size and chunk counts
- RAG success and refusal outcomes

Grafana dashboards are provisioned from `monitoring/grafana/`.

Langfuse tracing is optional and disabled by default. When enabled, each RAG
request can be inspected as nested observations for retrieval, context selection,
generation, and cache behavior. Content capture remains off by default to avoid
storing private questions or document text.

## CI/CD

GitHub Actions verify the project with:

```powershell
ruff check src tests
pytest -q
docker compose config --quiet
```

The CD workflow can publish API and frontend images to GitHub Container Registry
on pushes to `main`.

## Portfolio Summary

This project demonstrates production AI engineering skills across:

- RAG architecture
- vector databases
- asynchronous ingestion
- database modeling
- Redis caching and rate limiting
- LLM provider abstraction
- evaluation and failed-case analysis
- observability and tracing
- Docker deployment
- CI/CD automation

