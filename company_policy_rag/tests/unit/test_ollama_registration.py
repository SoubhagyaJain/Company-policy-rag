"""
Unit tests for Ollama Model Registration Utility (CLI & REST API), Model Inspection, and Pipeline CLI.

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2. Model Merging, GGUF Export & Ollama Registration)
- PROJECT.md (§ Architecture, Feature Inventory F2.4, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F2.4 & Tier 1/2 coverage)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError
import pytest

# Ensure project root and company_policy_rag root are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
WORKSPACE_ROOT = PROJECT_ROOT.parent
for p in [str(PROJECT_ROOT), str(WORKSPACE_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# ── Dynamic Import with Robust Multi-Tier Fallback ──────────────────────────
try:
    from company_policy_rag.src.finetuning.ollama_registrar import (
        OllamaRegistrar,
        find_ollama_binary,
        get_model_details,
        probe_ollama_tags,
        register_model_api,
        register_model_cli,
        register_model_in_ollama,
        verify_model_registered,
    )
    from company_policy_rag.scripts.run_finetune_pipeline import (
        build_arg_parser as build_pipeline_arg_parser,
        main as pipeline_main,
    )
except ImportError:
    from src.finetuning.ollama_registrar import (
        OllamaRegistrar,
        find_ollama_binary,
        get_model_details,
        probe_ollama_tags,
        register_model_api,
        register_model_cli,
        register_model_in_ollama,
        verify_model_registered,
    )
    from scripts.run_finetune_pipeline import (
        build_arg_parser as build_pipeline_arg_parser,
        main as pipeline_main,
    )


# ── Test Suites ─────────────────────────────────────────────────────────────

class TestOllamaRegistrationCLI:
    """Tests for Ollama CLI registration commands and error handling."""

    @patch("subprocess.run")
    def test_cli_registration_success(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """CLI invocation succeeds with returncode 0."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")

        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "create", "qwen2.5-coder-7b-policy", "-f", str(modelfile)],
            returncode=0,
            stdout="transferring model data\nsuccess",
            stderr="",
        )

        ok = register_model_cli("qwen2.5-coder-7b-policy", modelfile)
        assert ok is True
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_cli_registration_failure_exit_code(self, mock_run: MagicMock, tmp_path: Path) -> None:
        """CLI failure with non-zero exit code returns False without raising unhandled exception."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")

        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["ollama", "create"],
            output="",
            stderr="Error: parsing modelfile failed",
        )

        ok = register_model_cli("qwen2.5-coder-7b-policy", modelfile)
        assert ok is False

    def test_cli_registration_missing_modelfile(self) -> None:
        """Non-existent modelfile path raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            register_model_cli("qwen2.5-coder-7b-policy", "/non/existent/Modelfile")

    def test_cli_registration_empty_model_name(self, tmp_path: Path) -> None:
        """Empty model name raises ValueError."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")
        with pytest.raises(ValueError, match="model_name"):
            register_model_cli("", modelfile)


class TestOllamaRegistrationAPI:
    """Tests for Ollama REST API registration payload, headers, and HTTP responses."""

    @patch("urllib.request.urlopen")
    def test_api_registration_success(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """REST API registration POST /api/create succeeds with status 200 and success status."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf\nPARAMETER temperature 0.1", encoding="utf-8")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"status": "success"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        ok = register_model_api("qwen2.5-coder-7b-policy", modelfile, ollama_url="http://localhost:11434")
        assert ok is True

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url == "http://localhost:11434/api/create"
        assert req.get_header("Content-type") == "application/json"
        body = json.loads(req.data.decode("utf-8"))
        assert body["name"] == "qwen2.5-coder-7b-policy"
        assert "FROM ./test.gguf" in body["modelfile"]

    @patch("urllib.request.urlopen")
    def test_api_registration_server_offline(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """When Ollama server is offline (URLError / ConnectionRefused), returns False gracefully."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")

        mock_urlopen.side_effect = URLError("Connection refused [Errno 111]")
        ok = register_model_api("qwen2.5-coder-7b-policy", modelfile, ollama_url="http://localhost:11434")
        assert ok is False

    @patch("urllib.request.urlopen")
    def test_api_registration_server_500_error(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """When Ollama server returns HTTP 500 error, returns False gracefully."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")

        mock_urlopen.side_effect = HTTPError(
            url="http://localhost:11434/api/create",
            code=500,
            msg="Internal Server Error",
            hdrs={},  # type: ignore
            fp=None,  # type: ignore
        )
        ok = register_model_api("qwen2.5-coder-7b-policy", modelfile)
        assert ok is False


class TestUnifiedOllamaRegistrar:
    """Tests for unified registration fallback, registrar class wrapper, and utilities."""

    @patch("urllib.request.urlopen")
    def test_unified_registrar_via_api(self, mock_urlopen: MagicMock, tmp_path: Path) -> None:
        """Unified registrar succeeds via REST API when Ollama is online."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")

        mock_response = MagicMock()
        mock_response.status = 200
        mock_response.read.return_value = json.dumps({"status": "success"}).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        registrar = OllamaRegistrar(ollama_url="http://localhost:11434")
        assert registrar.register("qwen2.5-coder-7b-policy", modelfile) is True

    @patch("subprocess.run")
    @patch("urllib.request.urlopen")
    def test_unified_registrar_fallback_to_cli(self, mock_urlopen: MagicMock, mock_run: MagicMock, tmp_path: Path) -> None:
        """Unified registrar falls back to CLI when API endpoint returns connection error."""
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM ./test.gguf", encoding="utf-8")

        mock_urlopen.side_effect = URLError("Connection refused")
        mock_run.return_value = subprocess.CompletedProcess(
            args=["ollama", "create"],
            returncode=0,
            stdout="success",
            stderr="",
        )

        ok = register_model_in_ollama("qwen2.5-coder-7b-policy", modelfile, prefer_api=True)
        assert ok is True
        mock_run.assert_called_once()


class TestOllamaInspectionAndProbing:
    """Tests for probe_ollama_tags, verify_model_registered, get_model_details, and find_ollama_binary."""

    @patch("urllib.request.urlopen")
    def test_probe_ollama_tags_online(self, mock_urlopen: MagicMock) -> None:
        """probe_ollama_tags successfully extracts model list when online."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        payload = {
            "models": [
                {"name": "qwen2.5-coder-7b-policy:latest"},
                {"name": "llama3:latest"},
            ]
        }
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        is_online, models, err = probe_ollama_tags()
        assert is_online is True
        assert "qwen2.5-coder-7b-policy:latest" in models
        assert "llama3:latest" in models
        assert err is None

    @patch("urllib.request.urlopen")
    def test_probe_ollama_tags_offline(self, mock_urlopen: MagicMock) -> None:
        """probe_ollama_tags returns False, empty list, and error string when offline."""
        mock_urlopen.side_effect = URLError("Connection refused")

        is_online, models, err = probe_ollama_tags()
        assert is_online is False
        assert models == []
        assert err is not None
        assert "Connection refused" in err

    @patch("company_policy_rag.src.finetuning.ollama_registrar.probe_ollama_tags")
    def test_verify_model_registered_variants(self, mock_probe: MagicMock) -> None:
        """verify_model_registered handles exact tag match, :latest suffix, base name match, and missing tag."""
        mock_probe.return_value = (True, ["qwen2.5-coder-7b-policy:latest", "mistral:7b"], None)

        # 1. Exact match
        assert verify_model_registered("qwen2.5-coder-7b-policy:latest") is True
        # 2. Match without :latest suffix
        assert verify_model_registered("qwen2.5-coder-7b-policy") is True
        # 3. Base name match
        assert verify_model_registered("mistral") is True
        # 4. Not found
        assert verify_model_registered("non-existent-model") is False

    @patch("company_policy_rag.src.finetuning.ollama_registrar.probe_ollama_tags")
    def test_verify_model_registered_offline(self, mock_probe: MagicMock) -> None:
        """verify_model_registered returns False if probe indicates daemon is offline."""
        mock_probe.return_value = (False, [], "Offline")
        assert verify_model_registered("qwen2.5-coder-7b-policy") is False

    @patch("urllib.request.urlopen")
    def test_get_model_details_success(self, mock_urlopen: MagicMock) -> None:
        """get_model_details parses /api/show response successfully."""
        mock_resp = MagicMock()
        mock_resp.status = 200
        payload = {
            "parameters": "num_ctx 8192\ntemperature 0.1",
            "template": "{{ .Prompt }}",
            "details": {"format": "gguf", "family": "qwen2"},
        }
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        info = get_model_details("qwen2.5-coder-7b-policy")
        assert info.get("parameters") == "num_ctx 8192\ntemperature 0.1"
        assert info.get("details", {}).get("family") == "qwen2"

    @patch("urllib.request.urlopen")
    def test_get_model_details_error_resilience(self, mock_urlopen: MagicMock) -> None:
        """get_model_details returns empty dict on HTTP error or exception."""
        mock_urlopen.side_effect = URLError("Server timeout")
        info = get_model_details("qwen2.5-coder-7b-policy")
        assert info == {}

    def test_find_ollama_binary_custom_and_path(self) -> None:
        """find_ollama_binary returns valid custom path, PATH result, or fallback."""
        # With valid custom path
        with patch("shutil.which", return_value="/custom/bin/ollama"):
            bin_path = find_ollama_binary("/custom/bin/ollama")
            assert bin_path == "/custom/bin/ollama"

        # Lookup via system PATH
        with patch("shutil.which", side_effect=lambda x: "/usr/bin/ollama" if x == "ollama" else None):
            bin_path = find_ollama_binary()
            assert bin_path == "/usr/bin/ollama"

        # Fallback when not found anywhere
        with patch("shutil.which", return_value=None), patch("os.environ.get", return_value=""):
            bin_path = find_ollama_binary()
            assert bin_path == "ollama"

    @patch("company_policy_rag.src.finetuning.ollama_registrar.probe_ollama_tags")
    @patch("company_policy_rag.src.finetuning.ollama_registrar.verify_model_registered")
    @patch("company_policy_rag.src.finetuning.ollama_registrar.get_model_details")
    def test_ollama_registrar_class_methods(
        self,
        mock_details: MagicMock,
        mock_verify: MagicMock,
        mock_probe: MagicMock,
    ) -> None:
        """OllamaRegistrar class methods delegate correctly to module functions."""
        mock_probe.return_value = (True, ["model:latest"], None)
        mock_verify.return_value = True
        mock_details.return_value = {"model": "info"}

        registrar = OllamaRegistrar(ollama_url="http://localhost:11434")
        assert registrar.is_available() is True
        assert registrar.is_registered("model:latest") is True
        assert registrar.get_info("model:latest") == {"model": "info"}


class TestRunFinetunePipelineCLI:
    """Unit tests for company_policy_rag/scripts/run_finetune_pipeline.py CLI."""

    def test_pipeline_arg_parser_defaults(self) -> None:
        """build_pipeline_arg_parser configures default arguments correctly."""
        parser = build_pipeline_arg_parser()
        args = parser.parse_args([])

        assert args.model_name_or_path == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert args.output_dir == "./outputs/pipeline_run"
        assert args.ollama_model_name == "qwen2.5-coder-7b-policy"
        assert args.gguf_quantization == "Q4_K_M"
        assert args.num_ctx == 8192
        assert args.temperature == 0.1
        assert args.dry_run is False
        assert args.smoke_test is False
        assert args.skip_train is False
        assert args.skip_merge is False
        assert args.skip_quant is False
        assert args.skip_register is False

    def test_pipeline_arg_parser_custom_args(self) -> None:
        """build_pipeline_arg_parser parses custom arguments correctly."""
        parser = build_pipeline_arg_parser()
        args = parser.parse_args([
            "--dataset_path", "./data/custom.jsonl",
            "--gguf_quantization", "Q8_0",
            "--ollama_model_name", "qwen-pipeline-policy",
            "--num_ctx", "4096",
            "--dry-run",
            "--smoke-test",
            "--skip-train",
        ])

        assert args.dataset_path == "./data/custom.jsonl"
        assert args.gguf_quantization == "Q8_0"
        assert args.ollama_model_name == "qwen-pipeline-policy"
        assert args.num_ctx == 4096
        assert args.dry_run is True
        assert args.smoke_test is True
        assert args.skip_train is True

    def test_pipeline_main_dry_run_success(self) -> None:
        """pipeline_main with --dry-run executes successfully and returns 0."""
        rc = pipeline_main(["--dry-run"])
        assert rc == 0

    def test_pipeline_main_dry_run_with_dataset(self) -> None:
        """pipeline_main with --dry-run and valid dataset loads and validates successfully."""
        sample_dataset = FIXTURES_DIR / "alpaca_sample.json"
        rc = pipeline_main([
            "--dry-run",
            "--dataset_path", str(sample_dataset),
            "--gguf_quantization", "FP16",
        ])
        assert rc == 0
