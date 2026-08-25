"""Fine-Tuning, Model Merging, GGUF Export, and Ollama Registration Subsystem.

Provides:
- Multi-format dataset ingestion, ChatML normalization, validation splitting
- LoRA/QLoRA training with completion masking and perplexity logging
- LoRA adapter weight merging into standalone FP16 HuggingFace format
- GGUF conversion with multi-tier fallback (llama.cpp -> gguf-py -> binary simulation)
- Ollama Modelfile generation with ChatML template and enterprise parameters
- Direct local Ollama storage registration via REST API and CLI

Authoritative Reference:
- ORIGINAL_REQUEST.md (§ Requirements R1, R2, R3, R4)
- PROJECT.md (§ Architecture, Feature Inventory F1.1-F2.4, Interface Contracts)
- TEST_INFRA.md (§ Feature Inventory & Test Matrix)
"""

from __future__ import annotations

from .dataset_loader import (
    DatasetEmptyError,
    DatasetFormatValidationError,
    DatasetLoaderError,
    DatasetValidationError,
    compute_dataset_statistics,
    detect_format,
    load_dataset_from_file,
    normalize_record,
    profile_dataset_tokens,
    sanitize_messages,
    split_dataset,
    validate_dataset_records,
)
from .gguf_exporter import (
    GGUFExportError,
    GGUFExporter,
    GGUFValidationResult,
    SUPPORTED_QUANTIZATIONS,
    convert_to_gguf,
    find_llama_cpp_tools,
    normalize_quantization,
    validate_gguf_file,
)
from .merger import (
    AdapterNotFoundError,
    BaseModelNotFoundError,
    MergeConfig,
    MergeOutput,
    MergeValidationError,
    ModelMergeError,
    ModelMerger,
    cleanup_memory,
    merge_lora_weights,
    resolve_device,
    resolve_torch_dtype,
)
from .modelfile_generator import (
    CHATML_TEMPLATE,
    DEFAULT_ENTERPRISE_SYSTEM_PROMPT,
    DEFAULT_PARAMETERS,
    DEFAULT_STOP_TOKENS,
    ModelfileGenerator,
    generate_modelfile,
    normalize_gguf_path,
    parse_modelfile,
)
from .ollama_registrar import (
    OllamaRegistrar,
    OllamaRegistrarError,
    OllamaRegistrationFailedError,
    OllamaServiceUnavailableError,
    get_model_details,
    probe_ollama_tags,
    register_model_api,
    register_model_cli,
    register_model_in_ollama,
    verify_model_registered,
)
from .trainer import (
    FineTuneConfig,
    MetricsLoggingCallback,
    TrainingMetricsCallback,
    TrainingOutput,
    calculate_perplexity,
    create_completion_collator,
    get_completion_data_collator,
    get_lora_model,
    get_tokenizer,
    setup_model,
    setup_peft_config,
    setup_tokenizer,
    train_lora,
)

__all__ = [
    # Dataset Ingestion & Hygiene (M1)
    "DatasetEmptyError",
    "DatasetFormatValidationError",
    "DatasetLoaderError",
    "DatasetValidationError",
    "compute_dataset_statistics",
    "detect_format",
    "load_dataset_from_file",
    "normalize_record",
    "profile_dataset_tokens",
    "sanitize_messages",
    "split_dataset",
    "validate_dataset_records",
    # Trainer & PEFT LoRA (M1)
    "FineTuneConfig",
    "MetricsLoggingCallback",
    "TrainingMetricsCallback",
    "TrainingOutput",
    "calculate_perplexity",
    "create_completion_collator",
    "get_completion_data_collator",
    "get_lora_model",
    "get_tokenizer",
    "setup_model",
    "setup_peft_config",
    "setup_tokenizer",
    "train_lora",
    # LoRA Weight Merger (M2)
    "AdapterNotFoundError",
    "BaseModelNotFoundError",
    "MergeConfig",
    "MergeOutput",
    "MergeValidationError",
    "ModelMergeError",
    "ModelMerger",
    "cleanup_memory",
    "merge_lora_weights",
    "resolve_device",
    "resolve_torch_dtype",
    # GGUF Exporter & Quantization (M2)
    "GGUFExportError",
    "GGUFExporter",
    "GGUFValidationResult",
    "SUPPORTED_QUANTIZATIONS",
    "convert_to_gguf",
    "find_llama_cpp_tools",
    "normalize_quantization",
    "validate_gguf_file",
    # Ollama Modelfile Generator (M2)
    "CHATML_TEMPLATE",
    "DEFAULT_ENTERPRISE_SYSTEM_PROMPT",
    "DEFAULT_PARAMETERS",
    "DEFAULT_STOP_TOKENS",
    "ModelfileGenerator",
    "generate_modelfile",
    "normalize_gguf_path",
    "parse_modelfile",
    # Ollama Registrar (M2)
    "OllamaRegistrar",
    "OllamaRegistrarError",
    "OllamaRegistrationFailedError",
    "OllamaServiceUnavailableError",
    "get_model_details",
    "probe_ollama_tags",
    "register_model_api",
    "register_model_cli",
    "register_model_in_ollama",
    "verify_model_registered",
]

__version__ = "0.2.0"
