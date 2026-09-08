from pathlib import Path

from scripts.benchmark_conversation import (
    assert_minimums,
    benchmark,
    load_dataset,
    load_document,
    render_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET = PROJECT_ROOT / "data/eval/conversation_benchmark.json"
DOCUMENT = PROJECT_ROOT / "data/demo/sample_employee_handbook.md"


def test_portfolio_dataset_is_versioned_unique_and_fictional() -> None:
    dataset = load_dataset(DATASET)
    case_ids = [case["id"] for case in dataset["cases"]]

    assert dataset["schema_version"] == "1.0"
    assert len(case_ids) >= 10
    assert len(case_ids) == len(set(case_ids))
    assert "fictional" in DOCUMENT.read_text(encoding="utf-8").lower()


def test_sample_handbook_builds_stable_benchmark_sections() -> None:
    chunks, sections = load_document(DOCUMENT)

    assert len(chunks) == 6
    assert set(sections) >= {
        "parental_maternity_leave",
        "business_travel_expenses",
        "remote_work",
        "annual_leave_probation",
        "vpn_access_recovery",
    }
    assert all(chunk.metadata.document_id == "northstar-handbook-2026" for chunk in chunks)


def test_conversation_benchmark_meets_ci_quality_gate_without_ollama() -> None:
    result = benchmark(DATASET, DOCUMENT, with_generation=False)

    assert_minimums(result)
    assert result["systems"]["improved"]["retrieval_hit_at_3"] == 1.0
    assert result["systems"]["improved"]["retrieval_policy_accuracy"] == 1.0
    assert result["systems"]["improved"]["retrieval_hit_at_3"] >= result["systems"]["baseline"][
        "retrieval_hit_at_3"
    ]

    report = render_report(result)
    assert "Retrieval hit@3" in report
    assert "Not measured" in report
    assert "Case-level evidence" in report
