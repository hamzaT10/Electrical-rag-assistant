# Production Observability

## 1. What Observability Means

Observability answers three different questions:

| Signal | Question | Electrical RAG implementation |
|---|---|---|
| Logs | What happened to one request? | Structured JSON events with `request_id` |
| Metrics | Is the system healthy across many requests? | Prometheus counters, gauges, and histograms |
| Traces | Which distributed operation caused the delay? | Future OpenTelemetry/Langfuse step |

Evaluation and observability are complementary. Evaluation measures answer quality on a
controlled benchmark. Observability measures real runtime behavior, traffic, latency,
cache usage, and failures.

## 2. Request Lifecycle

```text
Client
  -> Observability middleware creates or validates X-Request-ID
  -> FastAPI route
  -> Redis cache
  -> Qdrant retrieval
  -> optional cross-encoder reranking
  -> LLM generation
  -> response or streamed response
  -> middleware records final status and end-to-end duration
```

The middleware measures until the final response body is sent. This matters for
`/chat/stream`: measuring only until response headers are created would hide the actual
generation duration.

## 3. Structured Logs

Production defaults to JSON:

```json
{
  "timestamp": "2026-07-03T09:00:00+00:00",
  "level": "INFO",
  "logger": "electrical_rag.services.qa_service",
  "event": "rag_request_completed",
  "request_id": "46b6b38d7fae45ffaca7a17331deba13",
  "mode": "stream",
  "retrieval_seconds": 0.08,
  "llm_seconds": 1.72,
  "total_seconds": 1.84,
  "chunk_count": 5
}
```

The application logs lengths and operational metadata, not the complete user question
or answer. This reduces accidental exposure of user data.

Every HTTP response includes `X-Request-ID`. A frontend or gateway may provide a valid
ID, and the API will preserve it. Invalid or excessively long values are replaced.

Inspect API logs:

```powershell
docker compose logs -f rag-api
```

Filter one request:

```powershell
docker compose logs rag-api | Select-String "46b6b38d7fae45ffaca7a17331deba13"
```

## 4. Prometheus Metrics

Read the metrics endpoint:

```powershell
Invoke-WebRequest http://localhost:8000/metrics |
  Select-Object -ExpandProperty Content
```

Main metric families:

| Metric | Type | Meaning |
|---|---|---|
| `electrical_rag_http_requests_total` | Counter | Requests by method, route template, and status |
| `electrical_rag_http_request_duration_seconds` | Histogram | Complete HTTP response latency |
| `electrical_rag_http_requests_in_progress` | Gauge | Requests currently executing |
| `electrical_rag_chat_cache_events_total` | Counter | Redis cache hits and misses |
| `electrical_rag_rag_requests_total` | Counter | Answered and no-context RAG requests |
| `electrical_rag_rag_stage_duration_seconds` | Histogram | Retrieval, reranking, LLM, TTFT, and total latency |
| `electrical_rag_rag_retrieved_chunks` | Histogram | Number of chunks sent to the LLM |
| `electrical_rag_rag_component_initialization_seconds` | Histogram | Retriever load, embedding warmup, and total startup duration |
| `electrical_rag_rag_service_ready` | Gauge | Whether the in-process RAG service is initialized |

Useful PromQL examples:

```promql
sum(rate(electrical_rag_http_requests_total[5m])) by (route, status)
```

```promql
histogram_quantile(
  0.95,
  sum(rate(electrical_rag_http_request_duration_seconds_bucket[5m])) by (le, route)
)
```

```promql
sum(rate(electrical_rag_chat_cache_events_total{result="hit"}[5m]))
/
sum(rate(electrical_rag_chat_cache_events_total[5m]))
```

```promql
histogram_quantile(
  0.95,
  sum(rate(electrical_rag_rag_stage_duration_seconds_bucket{stage="ttft"}[5m])) by (le)
)
```

## 5. Cardinality Rule

Prometheus labels must have a small, bounded set of possible values. The implementation
uses route templates such as `/ingestion/jobs/{job_id}`, not raw URLs.

Never use these values as metric labels:

- request IDs
- user IDs
- document IDs
- filenames
- questions
- error messages

Those values create unbounded time series and can exhaust monitoring storage. Put them
in structured logs or traces instead.

## 6. Configuration

```env
APP_LOG_LEVEL=INFO
APP_LOG_FORMAT=json
ENABLE_METRICS=true
```

Use `APP_LOG_FORMAT=text` for easier local reading. Keep JSON in container or cloud
deployments where a log platform parses fields.

## 7. Start the Monitoring Stack

Set a local Grafana password in `.env`:

```env
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=change-me
```

Use a stronger secret outside local development.

Build the API and start the monitoring services:

```powershell
docker compose up -d --build rag-api prometheus grafana
```

Check container status:

```powershell
docker compose ps
```

The services are available at:

| Service | URL | Purpose |
|---|---|---|
| API metrics | `http://localhost:8000/metrics` | Raw metrics from one API process |
| Prometheus | `http://localhost:9090` | Scraping, storage, and PromQL |
| Grafana | `http://localhost:3001` | Dashboard visualization |

## 8. Verify Prometheus

Open `http://localhost:9090/targets`. The `electrical-rag-api` target should be `UP`.

You can also query Prometheus from PowerShell:

```powershell
Invoke-RestMethod `
  "http://localhost:9090/api/v1/query?query=up%7Bjob%3D%22electrical-rag-api%22%7D"
```

The result should contain a value of `1`.

If the target is down:

```powershell
docker compose logs prometheus
docker compose logs rag-api
```

## 9. Generate Observable Traffic

Metrics do not describe traffic that has not happened. Send several lightweight API
requests:

```powershell
1..5 | ForEach-Object {
  Invoke-RestMethod http://localhost:8000/meta | Out-Null
}
```

Then ask one normal chat question and repeat it once. The first request should normally
produce a cache miss and full RAG execution; the repeated request should produce a cache
hit:

```powershell
$body = '{"question":"What is the power meter?"}'

Invoke-RestMethod http://localhost:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body

Invoke-RestMethod http://localhost:8000/chat `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Wait at least one 15-second scrape interval before checking rate-based panels.

## 10. Use the Grafana Dashboard

1. Open `http://localhost:3001`.
2. Sign in with the credentials from `.env`.
3. Open `Dashboards -> Electrical RAG -> Electrical RAG Overview`.
4. Set the time range to `Last 1 hour`.

The dashboard is provisioned automatically and contains:

- API request rate
- Redis cache hit ratio
- P95 API latency
- requests currently in progress
- traffic by route and HTTP status
- P95 retrieval, reranking, LLM, TTFT, and total latency
- answered versus no-context RAG outcomes
- median context chunk count

## 11. Debugging Workflow

Use this sequence when a user reports a problem:

1. Check Grafana to determine whether the issue affects one request or the system.
2. Identify the slow or failing route and time window.
3. Obtain the user's `X-Request-ID`.
4. Filter API logs using that ID.
5. Inspect retrieval, reranking, LLM, source, cache, and status fields.
6. Reproduce the case in the evaluation benchmark if it is a quality problem.

This creates a professional feedback loop:

```text
Grafana detects a pattern
  -> Prometheus provides aggregated evidence
  -> request ID locates structured logs
  -> logs identify the failing stage
  -> evaluation verifies the eventual fix
```

## 12. Current Boundary

Prometheus and Grafana provide local metrics collection and dashboards. Production
deployment still needs authentication, TLS, durable backup/retention policy, alert
delivery, and access control around both monitoring services.

The next observability extension is distributed tracing with OpenTelemetry or Langfuse,
followed by alert rules based on measured service-level objectives.
