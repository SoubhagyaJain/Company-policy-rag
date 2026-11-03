import importlib
import json
import struct
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from company_policy_rag.src.finetuning import merger, gguf_exporter


def fake_hf(monkeypatch, *, fail=False):
    def save_tokenizer(path):
        path = Path(path)
        (path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "ORIGINAL", "eos_token": "custom"}))
        (path / "tokenizer.json").write_text("{}")
    def save_model(path, **kwargs):
        path = Path(path)
        (path / "config.json").write_text(json.dumps({"model_type": "qwen2", "hidden_size": 96, "num_hidden_layers": 2}))
        (path / "model.safetensors").write_bytes(b"TEST_ONLY_WEIGHTS")
    model = SimpleNamespace(save_pretrained=save_model)
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos", save_pretrained=save_tokenizer)
    load = MagicMock(side_effect=RuntimeError("merge failed")) if fail else MagicMock(return_value=model)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer)))
    monkeypatch.setitem(sys.modules, "peft", SimpleNamespace(PeftModel=SimpleNamespace(
        from_pretrained=lambda *a, **kw: SimpleNamespace(merge_and_unload=lambda **kw: model))))


def test_merge_failure_cannot_publish_mock_weights(tmp_path, monkeypatch):
    fake_hf(monkeypatch, fail=True)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    with pytest.raises(merger.ModelMergeError):
        merger.merge_lora_weights("test/base", adapter, tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_successful_merge_preserves_real_exporter_metadata(tmp_path, monkeypatch):
    fake_hf(monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    output = Path(merger.merge_lora_weights("test/base", adapter, tmp_path / "output"))
    assert json.loads((output / "config.json").read_text())["hidden_size"] == 96
    assert json.loads((output / "tokenizer_config.json").read_text())["chat_template"] == "ORIGINAL"


def test_export_without_converter_fails_without_output(tmp_path, monkeypatch):
    monkeypatch.setattr(gguf_exporter, "find_llama_cpp_tools", lambda *a: {"convert_script": None, "quantize_bin": None})
    with pytest.raises(gguf_exporter.GGUFExportError):
        gguf_exporter.convert_to_gguf(tmp_path, tmp_path / "output.gguf")
    assert not (tmp_path / "output.gguf").exists()


def test_existing_artifacts_are_never_replaced(tmp_path, monkeypatch):
    fake_hf(monkeypatch)
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    (output / "config.json").write_text("original")
    with pytest.raises((FileExistsError, merger.ModelMergeError)):
        merger.merge_lora_weights("test/base", adapter, output)
    assert (output / "config.json").read_text() == "original"


def test_registration_failure_is_a_nonzero_exit(tmp_path, monkeypatch):
    module = importlib.import_module("company_policy_rag.scripts.export_and_register_ollama")
    monkeypatch.setattr(module, "register_model_in_ollama", lambda **kw: False)
    gguf = tmp_path / "existing.gguf"
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + b"fixture" * 50)
    assert module.main(["--skip-merge", "--skip-quant", "--gguf_path", str(gguf), "--output_dir", str(tmp_path)]) == 1
