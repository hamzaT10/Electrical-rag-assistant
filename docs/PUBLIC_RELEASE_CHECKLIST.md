# Public Release Checklist

Use this checklist before publishing the recruiter-facing repository.

## Keep

- `src/`
- `tests/`
- `frontend/`
- `alembic/`
- `monitoring/`
- `docs/`
- `evaluation/README.md`
- `evaluation/benchmark_questions.json`
- `.github/workflows/`
- `Dockerfile`
- `docker-compose.yml`
- `.env.example`
- `pyproject.toml`
- `requirements-app.txt`
- `requirements-ocr.txt`

## Exclude

- `Data/`
- `vectorstore/`
- `.env`
- `.env.*` except `.env.example`
- `evaluation/results/`
- `outputs/`
- local notebooks with outputs
- local virtual environments such as `.venv/` and `Llm_assis/`
- local IDE/cache folders
- private notes

## Verify

Run these commands before the first public commit:

```powershell
git status --short
rg -n --hidden "api[_-]?key|secret|password|token|bearer|private_key|BEGIN PRIVATE|LANGFUSE_SECRET_KEY|OPENAI_API_KEY" .
python -m ruff check src tests
python -m pytest -q
docker compose config --quiet
```

## First Public Commit

Recommended first commit message:

```text
Initial public release: production RAG assistant
```

Recommended repository description:

```text
Production-style RAG assistant for technical electrical PDFs with FastAPI, Qdrant, PostgreSQL, Redis, Celery, Docker, evaluation, and observability.
```

## Notes

The public repository should start from a clean history. Keep the private
repository as the full learning and development history. Continue future work in
the public repository with normal commits after the first clean release.
