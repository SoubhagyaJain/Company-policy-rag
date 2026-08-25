"""Ollama Model Registration and Verification Utility (CLI & REST API).

Provides dual-channel model registration into local Ollama storage:
- REST API channel: POST /api/create with JSON payload
- CLI channel: subprocess invocation of `ollama create <name> -f <modelfile>`
- Unified fallback orchestrator and tag verification

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2. Model Merging, GGUF Export & Ollama Registration)
- PROJECT.md (§ Architecture, Feature Inventory F2.4, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F2.4 & Tier 1/2 coverage)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class OllamaRegistrarError(Exception):
    """Base exception for Ollama registration failures."""
    pass


class OllamaServiceUnavailableError(OllamaRegistrarError):
    """Raised when Ollama daemon cannot be reached via REST API or CLI."""
    pass


class OllamaRegistrationFailedError(OllamaRegistrarError):
    """Raised when model registration command or API returns failure."""
    pass


def find_ollama_binary(custom_path: Optional[str] = None) -> str:
    """Find ollama executable across custom path, PATH, and standard Windows/Unix locations."""
    if custom_path and shutil.which(custom_path):
        return custom_path

    which_ollama = shutil.which("ollama")
    if which_ollama:
        return which_ollama

    # Windows standard paths
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            candidate = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
            if candidate.is_file():
                return str(candidate)

    return "ollama"


def register_model_cli(
    model_name: str,
    modelfile_path: Union[str, Path],
    binary: str = "ollama",
    timeout: float = 300.0,
) -> bool:
    """Register model via Ollama CLI command: ollama create <name> -f <modelfile>.

    Args:
        model_name: Target model tag (e.g. 'qwen2.5-coder-7b-policy').
        modelfile_path: Path to the generated Modelfile.
        binary: Path or name of ollama executable.
        timeout: Maximum seconds to wait for creation.

    Returns:
        bool: True if registration succeeded, False otherwise.

    Raises:
        ValueError: If model_name is empty.
        FileNotFoundError: If modelfile_path does not exist.
    """
    if not model_name or not str(model_name).strip():
        raise ValueError("Model name cannot be empty (model_name)")

    path = Path(modelfile_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelfile not found: {path}")

    resolved_bin = find_ollama_binary(binary)
    cmd = [resolved_bin, "create", model_name.strip(), "-f", str(path)]

    logger.info("Executing Ollama CLI registration: %s", " ".join(cmd))
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=timeout,
        )
        logger.info("Ollama CLI registration output: %s", res.stdout)
        return res.returncode == 0
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("Ollama CLI registration failed: %s", exc)
        return False


def register_model_api(
    model_name: str,
    modelfile_path: Union[str, Path],
    ollama_url: str = "http://localhost:11434",
    timeout: float = 300.0,
    stream: bool = False,
) -> bool:
    """Register model via Ollama REST API endpoint: POST /api/create.

    Args:
        model_name: Target model tag (e.g. 'qwen2.5-coder-7b-policy').
        modelfile_path: Path to the generated Modelfile.
        ollama_url: Base URL of local Ollama instance.
        timeout: HTTP request timeout in seconds.
        stream: Whether to stream creation progress.

    Returns:
        bool: True if registration succeeded with status 200, False otherwise.

    Raises:
        ValueError: If model_name is empty.
        FileNotFoundError: If modelfile_path does not exist.
    """
    if not model_name or not str(model_name).strip():
        raise ValueError("Model name cannot be empty (model_name)")

    path = Path(modelfile_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelfile not found: {path}")

    modelfile_content = path.read_text(encoding="utf-8")
    url = ollama_url.rstrip("/") + "/api/create"
    payload = json.dumps({
        "name": model_name.strip(),
        "modelfile": modelfile_content,
        "stream": stream,
    }).encode("utf-8")

    logger.info("Sending Ollama REST API create request to %s for model '%s'", url, model_name)
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")

    try:
        with urlopen(req, timeout=timeout) as response:
            if response.status == 200:
                raw_body = response.read().decode("utf-8")
                if not raw_body.strip():
                    return True
                try:
                    resp_data = json.loads(raw_body)
                    if isinstance(resp_data, dict):
                        return resp_data.get("status") == "success" or "error" not in resp_data
                    return True
                except json.JSONDecodeError:
                    return "success" in raw_body.lower()
            return False
    except (HTTPError, URLError, TimeoutError, OSError) as exc:
        logger.warning("Ollama API registration failed: %s", exc)
        return False


def register_model_in_ollama(
    model_name: str,
    modelfile_path: Union[str, Path],
    ollama_url: str = "http://localhost:11434",
    prefer_api: bool = True,
    timeout: float = 300.0,
    binary: str = "ollama",
) -> bool:
    """Unified entrypoint: attempts primary channel first, falling back to secondary channel.

    Args:
        model_name: Target model tag (e.g. 'qwen2.5-coder-7b-policy').
        modelfile_path: Path to the generated Modelfile.
        ollama_url: Base URL of local Ollama instance.
        prefer_api: If True, tries REST API before CLI; if False, tries CLI before REST API.
        timeout: Operation timeout in seconds.
        binary: Path or name of ollama executable for CLI fallback.

    Returns:
        bool: True if registration succeeded via either method, False otherwise.
    """
    if not model_name or not str(model_name).strip():
        raise ValueError("Model name cannot be empty (model_name)")

    path = Path(modelfile_path)
    if not path.exists():
        raise FileNotFoundError(f"Modelfile not found: {path}")

    if prefer_api:
        logger.info("Attempting primary registration via REST API...")
        ok = register_model_api(model_name, path, ollama_url=ollama_url, timeout=timeout)
        if ok:
            logger.info("Model '%s' successfully registered in Ollama via REST API.", model_name)
            return True
        logger.warning("REST API registration unsuccessful, falling back to Ollama CLI...")
        ok_cli = register_model_cli(model_name, path, binary=binary, timeout=timeout)
        if ok_cli:
            logger.info("Model '%s' successfully registered in Ollama via CLI.", model_name)
            return True
        return False
    else:
        logger.info("Attempting primary registration via CLI...")
        ok = register_model_cli(model_name, path, binary=binary, timeout=timeout)
        if ok:
            logger.info("Model '%s' successfully registered in Ollama via CLI.", model_name)
            return True
        logger.warning("CLI registration unsuccessful, falling back to REST API...")
        ok_api = register_model_api(model_name, path, ollama_url=ollama_url, timeout=timeout)
        if ok_api:
            logger.info("Model '%s' successfully registered in Ollama via REST API.", model_name)
            return True
        return False


def probe_ollama_tags(
    ollama_url: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> Tuple[bool, List[str], Optional[str]]:
    """Probe Ollama daemon and retrieve list of available model tags.

    Returns:
        (is_online: bool, model_names: List[str], error_message: Optional[str])
    """
    url = ollama_url.rstrip("/") + "/api/tags"
    req = Request(url, headers={"Accept": "application/json"}, method="GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                data = json.loads(resp.read().decode("utf-8"))
                models = [m["name"] for m in data.get("models", []) if "name" in m]
                return True, models, None
            return False, [], f"HTTP {resp.status}"
    except Exception as exc:
        return False, [], str(exc)


def verify_model_registered(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    timeout: float = 5.0,
) -> bool:
    """Verify whether model tag is present in Ollama's local manifest."""
    is_online, models, _ = probe_ollama_tags(ollama_url=ollama_url, timeout=timeout)
    if not is_online:
        return False

    clean_target = model_name.strip()
    target_base = clean_target.split(":")[0]

    for m in models:
        if m == clean_target:
            return True
        if m == f"{clean_target}:latest":
            return True
        if m.split(":")[0] == target_base:
            return True

    return False


def get_model_details(
    model_name: str,
    ollama_url: str = "http://localhost:11434",
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Retrieve detailed model parameters, template, and system prompt from /api/show."""
    url = ollama_url.rstrip("/") + "/api/show"
    payload = json.dumps({"name": model_name.strip()}).encode("utf-8")
    req = Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
            return {}
    except Exception:
        return {}


class OllamaRegistrar:
    """Class wrapper encapsulating Ollama registration, probing, and verification."""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        prefer_api: bool = True,
        timeout: float = 300.0,
        binary: str = "ollama",
    ) -> None:
        self.ollama_url = ollama_url.rstrip("/")
        self.prefer_api = prefer_api
        self.timeout = timeout
        self.binary = binary

    def register(
        self,
        model_name: str,
        modelfile_path: Union[str, Path],
        prefer_api: Optional[bool] = None,
    ) -> bool:
        """Register model in Ollama."""
        pref = prefer_api if prefer_api is not None else self.prefer_api
        return register_model_in_ollama(
            model_name=model_name,
            modelfile_path=modelfile_path,
            ollama_url=self.ollama_url,
            prefer_api=pref,
            timeout=self.timeout,
            binary=self.binary,
        )

    def is_available(self) -> bool:
        """Check if Ollama service is reachable."""
        is_online, _, _ = probe_ollama_tags(self.ollama_url, timeout=5.0)
        return is_online

    def is_registered(self, model_name: str) -> bool:
        """Check if model is registered in Ollama."""
        return verify_model_registered(model_name, self.ollama_url, timeout=5.0)

    def get_info(self, model_name: str) -> Dict[str, Any]:
        """Get model details from Ollama."""
        return get_model_details(model_name, self.ollama_url, timeout=8.0)


def main() -> None:
    """CLI entrypoint for standalone Ollama registrar."""
    parser = argparse.ArgumentParser(description="Register a Modelfile with local Ollama.")
    parser.add_argument("--model_name", "-n", type=str, default="qwen2.5-coder-7b-policy", help="Model name tag.")
    parser.add_argument("--modelfile_path", "-f", type=str, required=True, help="Path to Modelfile.")
    parser.add_argument("--ollama_url", type=str, default="http://localhost:11434", help="Ollama server URL.")
    parser.add_argument("--prefer_cli", action="store_true", help="Prefer CLI over REST API.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Creation timeout in seconds.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    success = register_model_in_ollama(
        model_name=args.model_name,
        modelfile_path=args.modelfile_path,
        ollama_url=args.ollama_url,
        prefer_api=not args.prefer_cli,
        timeout=args.timeout,
    )
    if success:
        print(f"Model '{args.model_name}' successfully registered in Ollama.")
        sys.exit(0)
    else:
        print(f"Failed to register model '{args.model_name}' in Ollama.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
