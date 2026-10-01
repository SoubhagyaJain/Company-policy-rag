from __future__ import annotations

import gc
import io
import importlib.util
import subprocess
import threading
from pathlib import Path
from typing import Any

from PIL import Image

torch: Any = None
AutoProcessor: Any = None
_VisionModelClass: Any = None
_runtime_imported = False
_runtime_error: str | None = None
_runtime_lock = threading.Lock()

from backend.utils.logging import logger
from src.config import settings


def _ensure_runtime() -> tuple[bool, str | None]:
    """Import the heavy ML runtime only when visual inference is requested."""
    global torch, AutoProcessor, _VisionModelClass
    global _runtime_imported, _runtime_error

    if _runtime_imported:
        return _runtime_error is None, _runtime_error
    with _runtime_lock:
        if _runtime_imported:
            return _runtime_error is None, _runtime_error
        try:
            import torch as torch_module
            from transformers import AutoProcessor as processor_class

            try:
                from transformers import Qwen3VLForConditionalGeneration as model_class
            except ImportError:  # pragma: no cover - older transformers fallback
                from transformers import AutoModelForImageTextToText as model_class

            torch = torch_module
            AutoProcessor = processor_class
            _VisionModelClass = model_class
            _runtime_error = None
        except ImportError as exc:  # pragma: no cover - minimal deployments
            _runtime_error = str(exc)
        finally:
            _runtime_imported = True
    return _runtime_error is None, _runtime_error


def gpu_memory_mb(device_index: int = 0) -> dict[str, Any] | None:
    """Device-wide VRAM use as the driver sees it, or None without nvidia-smi.

    Needs no torch import, so status pages can call it without loading the runtime.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        name, total, used, free = (part.strip() for part in completed.stdout.strip().splitlines()[0].split(","))
        return {"name": name, "total_mb": float(total), "used_mb": float(used), "free_mb": float(free)}
    except Exception:
        return None


def _driver_free_gb(device_index: int) -> float | None:
    """Device-wide free VRAM in GiB as the driver sees it, or None without nvidia-smi."""
    memory = gpu_memory_mb(device_index)
    return None if memory is None else memory["free_mb"] / 1024


class HFVisionClient:
    """Lazy, bounded local Qwen3-VL client.

    Loading is delayed until a query actually needs visual understanding. This
    prevents ordinary text RAG requests from paying the model's cold-start and
    avoids competing with the Ollama text model for GPU memory at API startup.
    """

    _instance: "HFVisionClient | None" = None
    _instance_lock = threading.Lock()

    def __init__(self, model_path: str | Path | None = None) -> None:
        configured_path = model_path or getattr(settings, "vision_model_path", None)
        self.model_path = Path(configured_path or (Path.home() / "Qwen3-VL-2B-Instruct"))
        self.model: Any = None
        self.processor: Any = None
        self.device = "cpu"
        self._load_lock = threading.Lock()
        self._load_error: str | None = None

    @classmethod
    def get_instance(cls) -> "HFVisionClient":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @property
    def is_loaded(self) -> bool:
        return self.model is not None and self.processor is not None

    def readiness(self) -> tuple[bool, str]:
        """Check dependencies and local weights without loading ~4 GB of weights."""
        if (
            importlib.util.find_spec("torch") is None
            or importlib.util.find_spec("transformers") is None
        ):
            return False, "PyTorch and a Qwen3-VL-compatible Transformers build are required."
        if _runtime_imported and _runtime_error:
            return False, f"Qwen3-VL runtime import failed: {_runtime_error}"
        if not self.model_path.is_dir():
            return False, f"Qwen3-VL model directory was not found: {self.model_path}"
        if not (self.model_path / "config.json").is_file():
            return False, f"Qwen3-VL config.json is missing from: {self.model_path}"
        if self._load_error:
            return False, f"Qwen3-VL failed to load: {self._load_error}"
        state = "loaded" if self.is_loaded else "available (lazy load)"
        return True, f"HF Vision model {self.model_path.name} is {state}."

    def gpu_free_gb(self) -> float | None:
        """Free VRAM in GiB, or None when there is no usable CUDA device.

        Readiness probes call this before any load, so it imports the runtime
        itself; otherwise ``torch`` is still None and a GPU host reports "cpu".
        """
        runtime_ready, _ = _ensure_runtime()
        if not runtime_ready or not torch.cuda.is_available():
            return None
        try:
            free_bytes, _ = torch.cuda.mem_get_info()
            device_index = torch.cuda.current_device()
        except Exception as exc:
            logger.warning("Could not inspect free GPU memory; using CPU for Qwen3-VL: %s", exc)
            return None
        free_gb = free_bytes / (1024**3)
        # Under Windows WDDM torch reports this process's own budget and does not see
        # VRAM held by other processes such as Ollama, so trust the lower figure.
        driver_free_gb = _driver_free_gb(device_index)
        return free_gb if driver_free_gb is None else min(free_gb, driver_free_gb)

    def has_gpu_headroom(self) -> bool:
        free_gb = self.gpu_free_gb()
        return free_gb is not None and free_gb >= float(getattr(settings, "vision_min_gpu_free_gb", 2.0))

    def _choose_device(self) -> str:
        free_gb = self.gpu_free_gb()
        if free_gb is None:
            return "cpu"
        required_gb = float(getattr(settings, "vision_min_gpu_free_gb", 2.0))
        if free_gb < required_gb:
            logger.warning(
                "Only %.2f GiB GPU memory is free; loading Qwen3-VL on CPU "
                "instead of triggering slow GPU/CPU offload (minimum %.2f GiB).",
                free_gb,
                required_gb,
            )
            return "cpu"
        return "cuda"

    def unload(self) -> None:
        """Drop the loaded weights and hand their VRAM back to the driver."""
        with self._load_lock:
            if not self.is_loaded:
                return
            was_cuda = self.device == "cuda"
            self.model = None
            self.processor = None
            self.device = "cpu"
            gc.collect()
            if was_cuda:
                torch.cuda.empty_cache()
            logger.info("HF Vision Model unloaded.")

    def load_model(self) -> bool:
        if self.is_loaded:
            return True

        with self._load_lock:
            if self.is_loaded:
                return True
            runtime_ready, runtime_error = _ensure_runtime()
            if not runtime_ready:
                self._load_error = runtime_error or "Unable to import the Qwen3-VL runtime."
                logger.error(self._load_error)
                return False
            ready, message = self.readiness()
            if not ready:
                self._load_error = message
                logger.error(message)
                return False

            self.device = self._choose_device()
            logger.info("Loading HF Vision Model from %s on %s", self.model_path, self.device)

            quantization_config = None
            dtype: Any = "auto"
            if self.device == "cuda":
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                    logger.info("Using 4-bit Qwen3-VL quantization on CUDA.")
                except (ImportError, RuntimeError) as exc:
                    logger.info("4-bit Qwen3-VL quantization unavailable: %s", exc)

            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    local_files_only=True,
                )
                model_kwargs: dict[str, Any] = {
                    "trust_remote_code": True,
                    "local_files_only": True,
                    "low_cpu_mem_usage": True,
                    "dtype": dtype,
                }
                if self.device == "cuda":
                    model_kwargs["device_map"] = "auto"
                if quantization_config is not None:
                    model_kwargs["quantization_config"] = quantization_config

                self.model = _VisionModelClass.from_pretrained(
                    self.model_path,
                    **model_kwargs,
                )
                if self.device == "cpu":
                    self.model.to("cpu")
                self.model.eval()
                self._load_error = None
                logger.info("HF Vision Model loaded successfully on %s.", self.device)
                return True
            except Exception as exc:
                self.model = None
                self.processor = None
                self._load_error = str(exc)
                logger.exception("Failed to load HF Vision Model: %s", exc)
                return False

    def execute(
        self,
        prompt: str,
        image_bytes: bytes,
        *,
        timeout: float | None = None,
        max_new_tokens: int | None = None,
    ) -> str:
        if not image_bytes:
            raise ValueError("Image bytes cannot be empty for vision completion.")
        if not self.load_model():
            raise RuntimeError(self._load_error or "HF Vision Model is not loaded.")

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # This is the native Qwen3-VL processor path. It avoids separately
        # templating and then re-tokenizing the same prompt.
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", self.device)
        inputs = inputs.to(model_device)

        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": int(
                max_new_tokens
                if max_new_tokens is not None
                else getattr(settings, "vision_num_predict", 160)
            ),
            "do_sample": False,
            "use_cache": True,
        }
        if timeout is not None and timeout > 0:
            # Transformers checks max_time during generation and returns the
            # tokens produced so far, providing a real upper bound for inference.
            generation_kwargs["max_time"] = max(1.0, float(timeout))

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, **generation_kwargs)

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return output_text.strip()
