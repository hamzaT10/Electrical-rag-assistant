from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from electrical_rag.core.messages import NO_RELEVANT_CONTEXT_ANSWER
from electrical_rag.core.settings import Settings


@dataclass
class ExpectedFact:
    name: str
    accepted_terms: list[str]


@dataclass
class BenchmarkCase:
    case_id: str
    question: str
    should_answer: bool = True
    expected_source: str | None = None
    expected_sources: list[str] | None = None
    expected_source_aliases: list[str] | None = None
    expected_answer_contains: list[str] | None = None
    expected_facts: list[ExpectedFact] | None = None
    topic: str | None = None
    language: str | None = None
    notes: str | None = None


@dataclass
class BenchmarkResult:
    case_id: str
    question: str
    answer: str
    citations: list[dict[str, object]]
    retrieved_sources: list[str]
    retrieved_scores: list[float]
    retrieval_score_top: float | None
    retrieval_score_avg: float | None
    retrieval_score_min: float | None
    retrieval_seconds: float
    total_seconds: float
    retrieval_hit: bool | None
    answer_keyword_pass: bool | None
    fact_pass: bool | None
    fact_pass_rate: float | None
    missing_facts: list[str]
    no_answer_pass: bool | None
    should_answer: bool = True
    expected_source_rank: int | None = None
    expected_source: str | None = None
    expected_sources: list[str] | None = None
    expected_source_aliases: list[str] | None = None
    expected_answer_contains: list[str] | None = None
    expected_facts: list[ExpectedFact] | None = None
    topic: str | None = None
    language: str | None = None


@dataclass
class BenchmarkReport:
    generated_at_utc: str
    benchmark_path: str
    settings: dict[str, object]
    summary: dict[str, object]
    failed_cases: list[dict[str, object]]
    results: list[BenchmarkResult]


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    raw_items = json.loads(path.read_text(encoding="utf-8"))
    cases: list[BenchmarkCase] = []
    for item in raw_items:
        expected_facts = item.pop("expected_facts", None)
        if expected_facts is not None:
            item["expected_facts"] = [
                ExpectedFact(**fact) for fact in expected_facts
            ]
        cases.append(BenchmarkCase(**item))
    return cases


def _normalize_text(value: str) -> str:
    return " ".join(value.lower().split())


def _source_matches(retrieved_source: str, expected_source: str) -> bool:
    retrieved = retrieved_source.lower()
    expected = expected_source.lower()
    return retrieved == expected or retrieved.endswith(f"/{expected}") or retrieved.endswith(
        f"\\{expected}"
    )


def _expected_source_candidates(
    expected_source: str | None,
    expected_sources: list[str] | None = None,
    expected_source_aliases: list[str] | None = None,
) -> list[str]:
    candidates: list[str] = []
    if expected_sources:
        candidates.extend(expected_sources)
    elif expected_source:
        candidates.append(expected_source)
    if expected_source_aliases:
        candidates.extend(expected_source_aliases)
    return candidates


def _answer_keyword_pass(answer: str, expected_keywords: list[str] | None) -> bool | None:
    if not expected_keywords:
        return None
    normalized_answer = _normalize_text(answer)
    return all(_normalize_text(keyword) in normalized_answer for keyword in expected_keywords)


def _fact_matches(answer: str, fact: ExpectedFact) -> bool:
    normalized_answer = _normalize_text(answer)
    return any(_normalize_text(term) in normalized_answer for term in fact.accepted_terms)


def _fact_pass(answer: str, expected_facts: list[ExpectedFact] | None) -> bool | None:
    if not expected_facts:
        return None
    return all(_fact_matches(answer, fact) for fact in expected_facts)


def _fact_pass_rate(answer: str, expected_facts: list[ExpectedFact] | None) -> float | None:
    if not expected_facts:
        return None
    passed_count = sum(1 for fact in expected_facts if _fact_matches(answer, fact))
    return round(passed_count / len(expected_facts), 3)


def _missing_facts(answer: str, expected_facts: list[ExpectedFact] | None) -> list[str]:
    if not expected_facts:
        return []
    return [fact.name for fact in expected_facts if not _fact_matches(answer, fact)]


def _missing_keywords(answer: str, expected_keywords: list[str] | None) -> list[str]:
    if not expected_keywords:
        return []
    normalized_answer = _normalize_text(answer)
    return [
        keyword
        for keyword in expected_keywords
        if _normalize_text(keyword) not in normalized_answer
    ]


def _retrieval_hit(
    retrieved_sources: list[str],
    expected_source: str | None,
    expected_sources: list[str] | None = None,
    expected_source_aliases: list[str] | None = None,
) -> bool | None:
    rank = _expected_source_rank(
        retrieved_sources,
        expected_source,
        expected_sources,
        expected_source_aliases,
    )
    if rank is None:
        expected_items = _expected_source_candidates(
            expected_source,
            expected_sources,
            expected_source_aliases,
        )
        return None if not expected_items else False
    return True


def _expected_source_rank(
    retrieved_sources: list[str],
    expected_source: str | None,
    expected_sources: list[str] | None = None,
    expected_source_aliases: list[str] | None = None,
) -> int | None:
    expected_items = _expected_source_candidates(
        expected_source,
        expected_sources,
        expected_source_aliases,
    )
    if not expected_items:
        return None
    for index, source in enumerate(retrieved_sources, start=1):
        if any(_source_matches(source, expected_source) for expected_source in expected_items):
            return index
    return None


def _score_summary(scores: list[float]) -> tuple[float | None, float | None, float | None]:
    if not scores:
        return None, None, None
    return max(scores), sum(scores) / len(scores), min(scores)


def _is_no_answer(answer: str, citations: list[dict[str, object]]) -> bool:
    normalized_answer = _normalize_text(answer)
    no_answer_markers = [
        _normalize_text(NO_RELEVANT_CONTEXT_ANSWER),
        "not present in the provided documents",
        "not found in the indexed documents",
        "cannot provide an answer",
        "could not find",
    ]
    return not citations or any(marker in normalized_answer for marker in no_answer_markers)


def _no_answer_pass(
    should_answer: bool,
    answer: str,
    citations: list[dict[str, object]],
) -> bool | None:
    no_answer = _is_no_answer(answer, citations)
    if should_answer:
        return not no_answer
    return no_answer


def compute_summary(results: list[BenchmarkResult]) -> dict[str, object]:
    retrieval_checks = [
        result.retrieval_hit for result in results if result.retrieval_hit is not None
    ]
    ranked_retrieval_checks = [
        result
        for result in results
        if result.retrieval_hit is not None
    ]
    answer_checks = [
        result.answer_keyword_pass for result in results if result.answer_keyword_pass is not None
    ]
    fact_checks = [
        result.fact_pass for result in results if result.fact_pass is not None
    ]
    fact_rates = [
        result.fact_pass_rate for result in results if result.fact_pass_rate is not None
    ]
    no_answer_checks = [
        result.no_answer_pass for result in results if result.no_answer_pass is not None
    ]
    top_scores = [
        result.retrieval_score_top
        for result in results
        if result.retrieval_score_top is not None
    ]
    avg_retrieval = (
        sum(result.retrieval_seconds for result in results) / len(results)
        if results
        else 0.0
    )
    avg_total = sum(result.total_seconds for result in results) / len(results) if results else 0.0
    ranked_source_count = sum(
        1 for result in ranked_retrieval_checks if result.expected_source_rank is not None
    )

    return {
        "cases": len(results),
        "failed_cases": len(build_failed_case_summary(results)),
        "avg_retrieval_seconds": round(avg_retrieval, 3),
        "avg_total_seconds": round(avg_total, 3),
        "retrieval_hit_rate": (
            round(sum(1 for item in retrieval_checks if item) / len(retrieval_checks), 3)
            if retrieval_checks
            else None
        ),
        "retrieval_mrr": (
            round(
                sum(
                    1 / result.expected_source_rank
                    if result.expected_source_rank is not None
                    else 0
                    for result in ranked_retrieval_checks
                )
                / len(ranked_retrieval_checks),
                3,
            )
            if ranked_retrieval_checks
            else None
        ),
        "avg_expected_source_rank": (
            round(
                sum(
                    result.expected_source_rank
                    for result in ranked_retrieval_checks
                    if result.expected_source_rank is not None
                )
                / ranked_source_count,
                3,
            )
            if ranked_source_count
            else None
        ),
        "answer_keyword_pass_rate": (
            round(sum(1 for item in answer_checks if item) / len(answer_checks), 3)
            if answer_checks
            else None
        ),
        "fact_pass_rate": (
            round(sum(1 for item in fact_checks if item) / len(fact_checks), 3)
            if fact_checks
            else None
        ),
        "avg_fact_coverage": (
            round(sum(fact_rates) / len(fact_rates), 3) if fact_rates else None
        ),
        "no_answer_pass_rate": (
            round(sum(1 for item in no_answer_checks if item) / len(no_answer_checks), 3)
            if no_answer_checks
            else None
        ),
        "avg_top_retrieval_score": (
            round(sum(top_scores) / len(top_scores), 4) if top_scores else None
        ),
    }


def _expected_source_items(result: BenchmarkResult) -> list[str]:
    return _expected_source_candidates(
        result.expected_source,
        result.expected_sources,
        result.expected_source_aliases,
    )


def build_failed_case_summary(results: list[BenchmarkResult]) -> list[dict[str, object]]:
    failed_cases: list[dict[str, object]] = []

    for result in results:
        reasons: list[str] = []
        if result.retrieval_hit is False:
            reasons.append("expected_source_not_retrieved")
        if result.answer_keyword_pass is False:
            reasons.append("answer_keywords_missing")
        if result.fact_pass is False:
            reasons.append("expected_facts_missing")
        if result.no_answer_pass is False:
            reasons.append(
                "answered_when_should_refuse"
                if not result.should_answer
                else "refused_when_should_answer"
            )

        if not reasons:
            continue

        failed_cases.append(
            {
                "case_id": result.case_id,
                "topic": result.topic,
                "language": result.language,
                "question": result.question,
                "reasons": reasons,
                "expected_sources": _expected_source_items(result),
                "expected_source_rank": result.expected_source_rank,
                "retrieved_sources": result.retrieved_sources,
                "retrieved_scores": result.retrieved_scores,
                "retrieval_score_top": result.retrieval_score_top,
                "retrieval_score_avg": result.retrieval_score_avg,
                "retrieval_score_min": result.retrieval_score_min,
                "should_answer": result.should_answer,
                "expected_answer_contains": result.expected_answer_contains or [],
                "missing_keywords": _missing_keywords(
                    result.answer,
                    result.expected_answer_contains,
                ),
                "expected_facts": [
                    asdict(fact) for fact in result.expected_facts or []
                ],
                "missing_facts": result.missing_facts,
                "fact_pass_rate": result.fact_pass_rate,
            }
        )

    return failed_cases


def run_benchmark(benchmark_path: Path, results_dir: Path | None = None) -> BenchmarkReport:
    from electrical_rag.services.qa_service import RAGService

    settings = Settings()
    service = RAGService(settings)
    cases = load_benchmark(benchmark_path)
    results: list[BenchmarkResult] = []

    for case in cases:
        retrieval_start = time.perf_counter()
        raw_results = service.retriever.search(case.question, settings.retrieval_top_k)
        retrieval_end = time.perf_counter()

        chunks = [service._to_chunk(item) for item in raw_results]
        retrieved_scores = [round(float(score), 4) for _, score in raw_results]
        score_top, score_avg, score_min = _score_summary(retrieved_scores)

        total_start = time.perf_counter()
        answer, citations = service.ask(case.question)
        total_end = time.perf_counter()

        retrieved_sources = [chunk.source for chunk in chunks]
        retrieval_hit = _retrieval_hit(
            retrieved_sources,
            case.expected_source,
            case.expected_sources,
            case.expected_source_aliases,
        )
        expected_source_rank = _expected_source_rank(
            retrieved_sources,
            case.expected_source,
            case.expected_sources,
            case.expected_source_aliases,
        )
        answer_keyword_pass = _answer_keyword_pass(answer, case.expected_answer_contains)
        fact_pass = _fact_pass(answer, case.expected_facts)
        fact_pass_rate = _fact_pass_rate(answer, case.expected_facts)
        missing_facts = _missing_facts(answer, case.expected_facts)
        no_answer_pass = _no_answer_pass(case.should_answer, answer, citations)

        results.append(
            BenchmarkResult(
                case_id=case.case_id,
                question=case.question,
                answer=answer,
                citations=citations,
                retrieved_sources=retrieved_sources,
                retrieved_scores=retrieved_scores,
                retrieval_score_top=round(score_top, 4) if score_top is not None else None,
                retrieval_score_avg=round(score_avg, 4) if score_avg is not None else None,
                retrieval_score_min=round(score_min, 4) if score_min is not None else None,
                retrieval_seconds=round(retrieval_end - retrieval_start, 3),
                total_seconds=round(total_end - total_start, 3),
                retrieval_hit=retrieval_hit,
                expected_source_rank=expected_source_rank,
                answer_keyword_pass=answer_keyword_pass,
                fact_pass=fact_pass,
                fact_pass_rate=fact_pass_rate,
                missing_facts=missing_facts,
                no_answer_pass=no_answer_pass,
                should_answer=case.should_answer,
                expected_source=case.expected_source,
                expected_sources=case.expected_sources,
                expected_source_aliases=case.expected_source_aliases,
                expected_answer_contains=case.expected_answer_contains,
                expected_facts=case.expected_facts,
                topic=case.topic,
                language=case.language,
            )
        )

    report = BenchmarkReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        benchmark_path=str(benchmark_path),
        settings={
            "embedding_model_name": settings.embedding_model_name,
            "retrieval_top_k": settings.retrieval_top_k,
            "rag_min_retrieval_score": settings.rag_min_retrieval_score,
            "enable_reranker": settings.enable_reranker,
            "reranker_model_name": (
                settings.reranker_model_name if settings.enable_reranker else None
            ),
            "chunk_size": settings.chunk_size,
            "chunk_overlap": settings.chunk_overlap,
            "llm_model": settings.llm_model,
        },
        summary=compute_summary(results),
        failed_cases=build_failed_case_summary(results),
        results=results,
    )

    if results_dir is not None:
        results_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = results_dir / f"report_{timestamp}.json"
        payload = asdict(report)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Electrical RAG benchmark pipeline.")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("evaluation/benchmark_questions.json"),
        help="Path to the benchmark cases JSON file.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("evaluation/results"),
        help="Directory where JSON reports will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_benchmark(args.benchmark_path, args.results_dir)
    print(
        json.dumps(
            {
                "summary": report.summary,
                "failed_cases": report.failed_cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
