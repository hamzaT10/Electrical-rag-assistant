# Embedding Model Startup and Cache

## 1. Three Separate Operations

```text
download model files -> load model into RAM -> embed text
```

The Hugging Face cache stores model artifacts on disk. It does not store PDF vectors,
questions, or answers.

| Component | Stores |
|---|---|
| Hugging Face cache | Embedding model weights, tokenizer, and configuration |
| Process RAM | The loaded model used by FastAPI or Celery |
| Qdrant | Document chunk vectors and payload metadata |
| Redis | Temporary completed answer payloads |

## 2. Docker Lifecycle

The API and Celery worker share the named Docker volume:

```text
huggingface_cache:/models/huggingface
```

Both services set:

```env
HF_HOME=/models/huggingface
```

On the first start, Sentence Transformers downloads model files into this volume. A
container recreation reuses the files instead of downloading them again.

The volume only persists disk files. Each process must still load the model from disk
into its own RAM.

## 3. API Startup Sequence

With the production defaults:

```env
PRELOAD_RAG_SERVICE=true
WARMUP_EMBEDDING_MODEL=true
```

the API performs:

```text
start process
  -> construct RAGService and VectorRetriever
  -> load embedding model into RAM
  -> verify the Qdrant collection
  -> embed "electrical_rag embedding model warmup"
  -> validate that a non-empty vector was returned
  -> record its vector dimension
  -> set electrical_rag_rag_service_ready = 1
  -> finish FastAPI startup
```

Uvicorn does not accept traffic until the lifespan startup finishes. Docker also gives
the API a health-check start period, and the frontend waits for the API to become
healthy.

## 4. Why the Warmup Vector Is Discarded

The warmup text exists only to execute the model once. It initializes the model's
runtime path and proves that embedding works.

Every real question is still embedded when it arrives:

```text
new question -> already loaded model -> query vector -> Qdrant
```

The warmup vector is not stored in Qdrant or Redis.

## 5. Readiness Inspection

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Relevant fields:

```json
{
  "status": "ready",
  "rag_service_ready": true,
  "rag_service_error": null,
  "rag_service_startup_seconds": 4.2,
  "embedding_dimension": 384
}
```

Inspect structured startup logs:

```powershell
docker compose logs rag-api |
  Select-String "rag_service_initialized"
```

Inspect Prometheus:

```promql
electrical_rag_rag_service_ready
```

```promql
electrical_rag_rag_component_initialization_seconds_sum
```

Components include:

- `retriever_initialization`
- `embedding_warmup`
- `rag_service_total`

## 6. First Start Versus Restart

First start with an empty volume:

```text
download + disk load + RAM initialization + warmup
```

Later container recreation:

```text
disk load + RAM initialization + warmup
```

Startup can still take several seconds because loading into RAM cannot be skipped. The
important behavior is that this cost happens before readiness, not inside the first
user request.

## 7. Model Compatibility

Ingestion and question answering must use the same embedding model. Qdrant document
vectors and query vectors must share dimensions and embedding space.

Changing `EMBEDDING_MODEL_NAME` requires reindexing into a new or rebuilt Qdrant
collection. Reusing old vectors with a different query model can produce invalid search
results even when both models happen to output the same dimension.
