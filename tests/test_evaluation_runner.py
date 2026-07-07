from pathlib import Path

from electrical_rag.evaluation.runner import (
    BenchmarkResult,
    ExpectedFact,
    _expected_source_rank,
    _fact_pass,
    _no_answer_pass,
    _retrieval_hit,
    build_failed_case_summary,
    compute_summary,
    load_benchmark,
)


def test_load_benchmark_reads_cases() -> None:
    path = Path("evaluation/benchmark_questions.json")
    cases = load_benchmark(path)

    assert len(cases) >= 1
    assert cases[0].case_id
    assert cases[0].question


def test_compute_summary_aggregates_metrics() -> None:
    results = [
        BenchmarkResult(
            case_id="a",
            question="q1",
            answer="a1",
            citations=[],
            retrieved_sources=["doc-a.pdf"],
            retrieved_scores=[0.8],
            retrieval_score_top=0.8,
            retrieval_score_avg=0.8,
            retrieval_score_min=0.8,
            retrieval_seconds=0.2,
            total_seconds=1.2,
            retrieval_hit=True,
            answer_keyword_pass=True,
            fact_pass=True,
            fact_pass_rate=1.0,
            missing_facts=[],
            no_answer_pass=True,
            expected_source_rank=1,
            expected_source="doc-a.pdf",
            expected_facts=[
                ExpectedFact(name="has_modbus", accepted_terms=["modbus"])
            ],
        ),
        BenchmarkResult(
            case_id="b",
            question="q2",
            answer="a2",
            citations=[],
            retrieved_sources=["doc-b.pdf"],
            retrieved_scores=[0.2],
            retrieval_score_top=0.2,
            retrieval_score_avg=0.2,
            retrieval_score_min=0.2,
            retrieval_seconds=0.4,
            total_seconds=1.8,
            retrieval_hit=False,
            answer_keyword_pass=True,
            fact_pass=False,
            fact_pass_rate=0.5,
            missing_facts=["has_bacnet"],
            no_answer_pass=True,
            expected_source_rank=None,
            expected_source="target.pdf",
            expected_facts=[
                ExpectedFact(name="has_modbus", accepted_terms=["modbus"]),
                ExpectedFact(name="has_bacnet", accepted_terms=["bacnet"]),
            ],
        ),
    ]

    summary = compute_summary(results)

    assert summary["cases"] == 2
    assert summary["failed_cases"] == 1
    assert summary["avg_retrieval_seconds"] == 0.3
    assert summary["avg_total_seconds"] == 1.5
    assert summary["retrieval_hit_rate"] == 0.5
    assert summary["retrieval_mrr"] == 0.5
    assert summary["avg_expected_source_rank"] == 1.0
    assert summary["answer_keyword_pass_rate"] == 1.0
    assert summary["fact_pass_rate"] == 0.5
    assert summary["avg_fact_coverage"] == 0.75
    assert summary["no_answer_pass_rate"] == 1.0
    assert summary["avg_top_retrieval_score"] == 0.5


def test_retrieval_hit_accepts_multiple_expected_sources() -> None:
    retrieved_sources = ["IEEE-std-519-1992-harmonic-limits.pdf"]

    assert _retrieval_hit(
        retrieved_sources,
        expected_source=None,
        expected_sources=[
            "IEEE_519_Harmonic_standard.pdf",
            "IEEE-std-519-1992-harmonic-limits.pdf",
        ],
    )


def test_retrieval_hit_accepts_source_aliases() -> None:
    retrieved_sources = ["Standards/Dépliant power meter.pdf"]

    assert _retrieval_hit(
        retrieved_sources,
        expected_source=None,
        expected_sources=["Depliant power meter.pdf"],
        expected_source_aliases=["Dépliant power meter.pdf"],
    )
    assert (
        _expected_source_rank(
            retrieved_sources,
            expected_source=None,
            expected_sources=["Depliant power meter.pdf"],
            expected_source_aliases=["Dépliant power meter.pdf"],
        )
        == 1
    )


def test_expected_source_rank_returns_one_based_rank() -> None:
    retrieved_sources = ["wrong.pdf", "Standards/target.pdf", "other.pdf"]

    assert _expected_source_rank(
        retrieved_sources,
        expected_source="target.pdf",
    ) == 2


def test_no_answer_pass_handles_refusal_and_answer_cases() -> None:
    assert _no_answer_pass(
        should_answer=False,
        answer="I could not find this in the indexed documents.",
        citations=[],
    )
    assert _no_answer_pass(
        should_answer=True,
        answer="The power meter is a portable analyzer.",
        citations=[{"source": "power-meter.pdf"}],
    )
    assert not _no_answer_pass(
        should_answer=False,
        answer="France won a tournament.",
        citations=[{"source": "power-meter.pdf"}],
    )


def test_fact_pass_accepts_term_variants() -> None:
    facts = [
        ExpectedFact(name="interface_rs485", accepted_terms=["rs-485", "rs485"]),
        ExpectedFact(name="supports_modbus", accepted_terms=["modbus", "modbus tcp"]),
    ]

    assert _fact_pass(
        "The module provides RS485 communication and supports Modbus TCP.",
        facts,
    )


def test_failed_case_summary_explains_retrieval_and_answer_failures() -> None:
    results = [
        BenchmarkResult(
            case_id="modbus-001",
            question="What is a Modbus function code?",
            answer="A function code specifies an action.",
            citations=[],
            retrieved_sources=["wrong.pdf"],
            retrieved_scores=[0.12],
            retrieval_score_top=0.12,
            retrieval_score_avg=0.12,
            retrieval_score_min=0.12,
            retrieval_seconds=0.1,
            total_seconds=2.0,
            retrieval_hit=False,
            answer_keyword_pass=False,
            fact_pass=False,
            fact_pass_rate=0.5,
            missing_facts=["has_data_field"],
            no_answer_pass=False,
            expected_source_rank=None,
            expected_source="modbus.pdf",
            expected_source_aliases=["intro_modbustcp.pdf"],
            expected_answer_contains=["function code", "data field"],
            expected_facts=[
                ExpectedFact(name="has_function_code", accepted_terms=["function code"]),
                ExpectedFact(name="has_data_field", accepted_terms=["data field"]),
            ],
            topic="modbus",
            language="en",
        )
    ]

    failed_cases = build_failed_case_summary(results)

    assert failed_cases == [
        {
            "case_id": "modbus-001",
            "topic": "modbus",
            "language": "en",
            "question": "What is a Modbus function code?",
            "reasons": [
                "expected_source_not_retrieved",
                "answer_keywords_missing",
                "expected_facts_missing",
                "refused_when_should_answer",
            ],
            "expected_sources": ["modbus.pdf", "intro_modbustcp.pdf"],
            "expected_source_rank": None,
            "retrieved_sources": ["wrong.pdf"],
            "retrieved_scores": [0.12],
            "retrieval_score_top": 0.12,
            "retrieval_score_avg": 0.12,
            "retrieval_score_min": 0.12,
            "should_answer": True,
            "expected_answer_contains": ["function code", "data field"],
            "missing_keywords": ["data field"],
            "expected_facts": [
                {
                    "name": "has_function_code",
                    "accepted_terms": ["function code"],
                },
                {
                    "name": "has_data_field",
                    "accepted_terms": ["data field"],
                },
            ],
            "missing_facts": ["has_data_field"],
            "fact_pass_rate": 0.5,
        }
    ]
