# Langfuse Distributed Tracing

## 1. Why This Exists

Prometheus, logs, evaluation, and Langfuse answer different questions:

| Signal | Main question |
|---|---|
| Prometheus/Grafana | Is the system healthy across many requests? |
| JSON logs | What events happened for one request? |
| Langfuse trace | Which RAG stage consumed time or failed? |
| Evaluation benchmark | Did a change improve answer quality? |

Langfuse is optional. Chat, ingestion, metrics, and logging continue to work when it is
disabled, unreachable, or incorrectly configured.

## 2. Electrical RAG Trace Structure

One chat request creates this hierarchy:

```text
electrical-rag-chat (chain)
├── chat-cache-lookup (span)
├── retrieve-context (retriever)
│   └── rerank-candidates (retriever, when enabled)
├── select-context (span)
└── generate-answer (generation)
```

A cache hit produces only the root and cache observations because retrieval and LLM
generation are intentionally skipped.

Each trace includes operational metadata such as:

- request ID and sync/stream mode
- document scope
- cache result
- retrieval source names and scores
- selected chunk count and context size
- model and temperature
- TTFT for streaming generation
- citation count

## 3. Create a Langfuse Project

Use either Langfuse Cloud or a separately self-hosted Langfuse deployment. Create a
project and obtain its public and secret API keys.

Do not commit keys. Put them only in the ignored `.env` file or a production secrets
manager.

```env
ENABLE_LANGFUSE=true
LANGFUSE_PUBLIC_KEY=pk-lf-your-public-key
LANGFUSE_SECRET_KEY=sk-lf-your-secret-key
LANGFUSE_BASE_URL=https://cloud.langfuse.com
LANGFUSE_ENVIRONMENT=development
LANGFUSE_SAMPLE_RATE=1.0
LANGFUSE_CAPTURE_CONTENT=false
APP_RELEASE=local
```

For a self-hosted instance, `LANGFUSE_BASE_URL` must be reachable from `rag-api`. A
service on the Windows host is normally addressed from Docker with
`http://host.docker.internal:<port>`.

## 4. Privacy Mode

The default is:

```env
LANGFUSE_CAPTURE_CONTENT=false
```

In this mode, traces include lengths and operational metadata but not complete:

- questions
- prompts and retrieved chunk text
- model answers

For example:

```json
{
  "question_chars": 42
}
```

Set `LANGFUSE_CAPTURE_CONTENT=true` only for controlled development data. Production
systems should add redaction, retention, access-control, and data-processing policies
before exporting content.

## 5. Sampling

Development can trace every request:

```env
LANGFUSE_SAMPLE_RATE=1.0
```

High-volume production systems can sample fewer requests:

```env
LANGFUSE_SAMPLE_RATE=0.2
```

This exports approximately 20 percent of traces. Prometheus metrics still cover all
requests.

## 6. Start and Test

Rebuild the API because Langfuse adds a Python dependency:

```powershell
docker compose up -d --build rag-api
```

Verify activation:

```powershell
Invoke-RestMethod http://localhost:8000/meta |
  Select-Object langfuse_enabled, langfuse_capture_content
```

Ask one question through the frontend or API. Then inspect the API log:

```powershell
docker compose logs rag-api |
  Select-String "langfuse_trace_created"
```

The event contains `request_id` and `langfuse_trace_id`. Search for that trace ID in the
Langfuse project.

## 7. How Correlation Works

Electrical RAG derives the Langfuse trace ID deterministically from `X-Request-ID`. This connects:

```text
browser X-Request-ID
  -> JSON log request_id
  -> Langfuse trace ID
  -> nested RAG observations
```

Normal chat also propagates the PostgreSQL user and chat-session IDs. Streaming
currently uses the demo user because stream persistence and authentication are later
production steps.

## 8. Failure Isolation

Trace exporting happens asynchronously. If Langfuse cannot initialize, start an
observation, update it, close it, or flush it, Electrical RAG writes a warning and continues.

Tracing must never become a dependency of the answer path:

```text
Langfuse available   -> answer + trace
Langfuse unavailable -> answer + warning log
```

## 9. Interview Explanation

> I instrumented each RAG request as a Langfuse trace with nested cache, retrieval,
> reranking, context-selection, and LLM-generation observations. I correlated traces
> with API request IDs, propagated user/session attributes, disabled content capture by
> default, added sampling controls, and isolated exporter failures so observability
> could not interrupt the user request.
