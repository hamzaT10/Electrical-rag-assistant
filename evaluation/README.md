# Evaluation Pipeline

This folder contains a benchmark-driven evaluation setup for the RAG system.

The public repository includes a small generic benchmark so the runner, metrics,
and report format can be tested without publishing the private document corpus.
For real projects, replace `benchmark_questions.json` with cases from your own
validated documents.

## Files

- `benchmark_questions.json`: sanitized example benchmark cases.
- `results/`: generated locally and ignored by Git.

## Metrics

The runner separates three evaluation layers:

- retrieval quality
- answer quality
- no-answer behavior

Tracked metrics include:

- retrieval hit rate
- retrieval MRR
- expected source rank
- top, average, and minimum retrieval score
- keyword-based answer pass rate
- fact-based answer pass rate
- average fact coverage
- no-answer pass rate
- retrieval latency
- total answer latency

## Expected Facts

`expected_facts` checks whether an answer contains required technical facts while
allowing accepted wording variants.

Example:

```json
{
  "name": "mentions_function_code",
  "accepted_terms": ["function code"]
}
```

This is more useful than exact string matching because technical answers can be
correct while using slightly different wording.

## Run

```powershell
$env:PYTHONPATH="src"
python -m electrical_rag.evaluation.runner
```

Reports are written to `evaluation/results/` and should stay local.

## Production Workflow

1. Build a benchmark from validated documents.
2. Run the benchmark after retrieval, prompt, reranker, or model changes.
3. Inspect failed cases and separate retrieval failures from generation failures.
4. Tune retrieval thresholds, metadata filters, prompt rules, or model settings.
5. Track quality and latency over time.
