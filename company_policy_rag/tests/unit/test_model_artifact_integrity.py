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
        (path / "config.json").write_text(
            json.dumps({"model_type": "qwen2", "hidden_size": 96, "num_hidden_layers": 2})
        )
        (path / "model.safetensors").write_bytes(b"TEST_ONLY_WEIGHTS")

    model = SimpleNamespace(save_pretrained=save_model)
    tokenizer = SimpleNamespace(pad_token="pad", eos_token="eos", save_pretrained=save_tokenizer)
    load = MagicMock(side_effect=RuntimeError("merge failed")) if fail else MagicMock(return_value=model)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: tokenizer),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "peft",
        SimpleNamespace(
            PeftModel=SimpleNamespace(
                from_pretrained=lambda *a, **kw: SimpleNamespace(merge_and_unload=lambda **kw: model)
            )
        ),
    )


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
    monkeypatch.setattr(
        gguf_exporter, "find_llama_cpp_tools", lambda *a: {"convert_script": None, "quantize_bin": None}
    )
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


@pytest.mark.parametrize("registered", [False, True])
def test_pipeline_requires_registration_and_verification(tmp_path, monkeypatch, registered):
    module = importlib.import_module("company_policy_rag.scripts.run_finetune_pipeline")
    monkeypatch.setattr(module, "register_model_in_ollama", lambda **kw: registered)
    monkeypatch.setattr(module, "verify_model_registered", lambda *a, **kw: False)
    gguf = tmp_path / "qwen2.5-coder-7b-policy-q4-k-m.gguf"
    gguf.write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 1, 1) + b"fixture" * 50)
    assert module.main(["--skip-train", "--skip-merge", "--skip-quant", "--output_dir", str(tmp_path)]) == 1


def test_converter_success_without_output_is_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        gguf_exporter,
        "find_llama_cpp_tools",
        lambda *a: {"convert_script": Path("convert.py"), "quantize_bin": Path("quantize")},
    )
    monkeypatch.setattr(
        gguf_exporter, "subprocess", SimpleNamespace(run=lambda *a, **kw: SimpleNamespace(returncode=0))
    )
    with pytest.raises(gguf_exporter.GGUFExportError):
        gguf_exporter.convert_to_gguf(tmp_path, tmp_path / "missing.gguf")
    assert not (tmp_path / "missing.gguf").exists()


@pytest.mark.parametrize("body", [b"", b"[]", b'{"status":"loading"}', b'{"error":"unsuccessful"}',
                                  b'{"status":"success"}\n{"error":"failed"}'])
def test_registration_requires_explicit_completion(tmp_path, monkeypatch, body):
    from company_policy_rag.src.finetuning import ollama_registrar
    response = MagicMock()
    response.status = 200
    response.read.return_value = body
    response.__enter__.return_value = response
    monkeypatch.setattr(ollama_registrar, "urlopen", lambda *a, **kw: response)
    modelfile = tmp_path / "Modelfile"
    modelfile.write_text("FROM fixture.gguf")
    assert not ollama_registrar.register_model_api("test", modelfile, stream=True)


def test_verification_cannot_accept_an_unrelated_model_tag(monkeypatch):
    from company_policy_rag.src.finetuning import ollama_registrar

    monkeypatch.setattr(
        ollama_registrar, "probe_ollama_tags",
        lambda **kw: (True, ["policy:old", "localhost:5000/team/policy:latest"], None),
    )
    assert not ollama_registrar.verify_model_registered("policy:new")
    assert not ollama_registrar.verify_model_registered("policy")
    assert ollama_registrar.verify_model_registered("localhost:5000/team/policy")
