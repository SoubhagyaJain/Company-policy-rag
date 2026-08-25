"""Ollama Modelfile Generation Engine for Fine-Tuned Enterprise Qwen Models.

Configures:
- ChatML prompt template for system/user/assistant turns
- Enterprise stop tokens (<|im_start|>, <|im_end|>, <|endoftext|>)
- Production hyperparameters (num_ctx 8192, temperature 0.1, top_p 0.95, repeat_penalty 1.1)
- Enterprise policy system prompt injection
- POSIX forward-slash path normalization for Windows and Linux/Docker

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ R2. Model Merging, GGUF Export & Ollama Registration)
- PROJECT.md (§ Architecture, Feature Inventory F2.3, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory F2.3 & Tier 1/2 coverage)
"""

from __future__ import annotations

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

DEFAULT_ENTERPRISE_SYSTEM_PROMPT = (
    "You are an enterprise company policy and software development assistant for Company Policy and Engineering Guidelines.\n"
    "Your role is to provide precise, faithful, and well-cited answers to employee questions.\n"
    "Adhere strictly to corporate policies, cite official policy document identifiers, and never hallucinate internal rules."
)

CHATML_TEMPLATE = (
    '"""{{ if .System }}<|im_start|>system\n'
    '{{ .System }}<|im_end|>\n'
    '{{ end }}{{ if .Prompt }}<|im_start|>user\n'
    '{{ .Prompt }}<|im_end|>\n'
    '{{ end }}<|im_start|>assistant\n'
    '{{ .Response }}<|im_end|>"""'
)

DEFAULT_STOP_TOKENS = [
    "<|im_end|>",
    "<|endoftext|>",
]

DEFAULT_PARAMETERS: Dict[str, Any] = {
    "num_ctx": 8192,
    "temperature": 0.1,
    "top_p": 0.95,
    "repeat_penalty": 1.1,
}


def normalize_gguf_path(gguf_path: Union[str, Path]) -> str:
    """Normalize GGUF path with POSIX forward slashes for cross-platform Ollama compatibility."""
    path_str = str(gguf_path).strip()
    if not path_str:
        raise ValueError("gguf_path cannot be empty")
    return path_str.replace("\\", "/")


def generate_modelfile(
    gguf_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    system_prompt: Optional[str] = None,
    num_ctx: int = 8192,
    temperature: float = 0.1,
    stop_tokens: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate an optimized Ollama Modelfile string and optionally write to disk.

    Args:
        gguf_path: Path to the converted GGUF model binary.
        output_path: Optional output file path where Modelfile will be saved.
        system_prompt: Custom system prompt string. Falls back to enterprise default if None or empty.
        num_ctx: Ollama context window token size (default: 8192).
        temperature: Sampling temperature (default: 0.1).
        stop_tokens: List of stop token strings.
        parameters: Optional dictionary of additional Ollama parameter directives.

    Returns:
        The generated Modelfile content string.
    """
    normalized_from = normalize_gguf_path(gguf_path)

    # Fall back to default if system_prompt is None or whitespace-only
    sys_prompt = system_prompt.strip() if system_prompt and system_prompt.strip() else DEFAULT_ENTERPRISE_SYSTEM_PROMPT

    stops = stop_tokens if stop_tokens is not None else DEFAULT_STOP_TOKENS

    merged_params = dict(DEFAULT_PARAMETERS)
    merged_params["num_ctx"] = num_ctx
    merged_params["temperature"] = temperature
    if parameters:
        merged_params.update(parameters)

    lines = [
        f"FROM {normalized_from}",
        "",
        f"TEMPLATE {CHATML_TEMPLATE}",
        "",
    ]

    for stop in stops:
        if stop and str(stop).strip():
            lines.append(f'PARAMETER stop "{stop}"')

    for param_name, param_val in merged_params.items():
        if param_name != "stop":
            lines.append(f"PARAMETER {param_name} {param_val}")

    lines.extend([
        "",
        f'SYSTEM """{sys_prompt}"""',
        "",
    ])

    content = "\n".join(lines)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")
        logger.info("Generated Ollama Modelfile written to: %s", out)

    return content


def parse_modelfile(modelfile_path: Union[str, Path]) -> Dict[str, Any]:
    """Parse an existing Ollama Modelfile into a structured dictionary."""
    path = Path(modelfile_path)
    if not path.is_file():
        raise FileNotFoundError(f"Modelfile not found: {path}")

    text = path.read_text(encoding="utf-8")
    parsed: Dict[str, Any] = {
        "from": None,
        "template": None,
        "parameters": {},
        "stop_tokens": [],
        "system": None,
    }

    # Extract FROM
    from_match = re.search(r"^FROM\s+(.+)$", text, re.MULTILINE)
    if from_match:
        parsed["from"] = from_match.group(1).strip()

    # Extract TEMPLATE
    template_match = re.search(r'TEMPLATE\s+"""(.*?)"""', text, re.DOTALL)
    if template_match:
        parsed["template"] = template_match.group(1).strip()

    # Extract PARAMETER lines
    for line in text.splitlines():
        line_clean = line.strip()
        if line_clean.startswith("PARAMETER"):
            parts = line_clean.split(maxsplit=2)
            if len(parts) >= 3:
                key = parts[1]
                val = parts[2].strip().strip('"')
                if key == "stop":
                    parsed["stop_tokens"].append(val)
                else:
                    parsed["parameters"][key] = val

    # Extract SYSTEM
    sys_match = re.search(r'SYSTEM\s+"""(.*?)"""', text, re.DOTALL)
    if sys_match:
        parsed["system"] = sys_match.group(1).strip()

    return parsed


class ModelfileGenerator:
    """Configurable generator class for Ollama Modelfile production."""

    def __init__(
        self,
        system_prompt: Optional[str] = None,
        num_ctx: int = 8192,
        temperature: float = 0.1,
        stop_tokens: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.system_prompt = system_prompt
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.stop_tokens = stop_tokens
        self.parameters = parameters or {}

    def generate(
        self,
        gguf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate Ollama Modelfile for given GGUF file."""
        return generate_modelfile(
            gguf_path=gguf_path,
            output_path=output_path,
            system_prompt=self.system_prompt,
            num_ctx=self.num_ctx,
            temperature=self.temperature,
            stop_tokens=self.stop_tokens,
            parameters=self.parameters,
        )

    def parse(self, modelfile_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse Modelfile at path."""
        return parse_modelfile(modelfile_path)


def main() -> None:
    """CLI entrypoint for standalone Modelfile generator."""
    parser = argparse.ArgumentParser(description="Generate Ollama Modelfile for quantized GGUF model.")
    parser.add_argument("--gguf_path", "-g", type=str, required=True, help="Path to .gguf binary file.")
    parser.add_argument("--output_path", "-o", type=str, default="./Modelfile", help="Output Modelfile path.")
    parser.add_argument("--system_prompt", "-s", type=str, default=None, help="Custom system prompt.")
    parser.add_argument("--num_ctx", type=int, default=8192, help="Context window size.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Sampling temperature.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    content = generate_modelfile(
        gguf_path=args.gguf_path,
        output_path=args.output_path,
        system_prompt=args.system_prompt,
        num_ctx=args.num_ctx,
        temperature=args.temperature,
    )
    print(f"Modelfile successfully written to: {args.output_path}")


if __name__ == "__main__":
    main()
