"""Incomplete downloads must not advertise the vision checkpoint as ready."""

import json

import pytest

from backend.vision import hf_vision_client as vision


@pytest.mark.parametrize("checkpoint", ["missing", "empty", "single", "missing_shard", "sharded"])
def test_checkpoint_readiness_requires_complete_weights(tmp_path, monkeypatch, checkpoint):
    monkeypatch.setattr(vision.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(vision, "_runtime_imported", False)
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    if checkpoint in {"empty", "single"}:
        (tmp_path / "model.safetensors").write_bytes(b"weights" if checkpoint == "single" else b"")
    elif checkpoint in {"missing_shard", "sharded"}:
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer.one": "part1.safetensors", "layer.two": "part2.safetensors"}}),
            encoding="utf-8",
        )
        (tmp_path / "part1.safetensors").write_bytes(b"weights")
        if checkpoint == "sharded":
            (tmp_path / "part2.safetensors").write_bytes(b"weights")

    ready, message = vision.HFVisionClient(tmp_path).readiness()

    assert ready is (checkpoint in {"single", "sharded"})
    if not ready:
        assert "missing or incomplete" in message
