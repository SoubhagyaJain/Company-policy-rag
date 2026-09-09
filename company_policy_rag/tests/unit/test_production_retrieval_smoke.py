from __future__ import annotations

from scripts.production_retrieval_smoke import DEFAULT_DATASET, load_dataset, run_smoke


def test_public_retrieval_dataset_is_well_formed() -> None:
    payload, cases = load_dataset(DEFAULT_DATASET)

    assert payload["document"] == "data/demo/sample_employee_handbook.md"
    assert len(cases) >= 8
    assert len({case.id for case in cases}) == len(cases)


def test_production_retrieval_smoke_meets_committed_floors() -> None:
    result = run_smoke()

    assert result["chunk_count"] >= 6
    assert result["metrics"]["hit_at_k"] >= result["minimums"]["hit_at_k"]
    assert (
        result["metrics"]["mean_reciprocal_rank"]
        >= result["minimums"]["mean_reciprocal_rank"]
    )
