"""Serving Engine backed by vLLM for high-performance LLM inference."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════
# SECTION 1: Request/Response Dataclasses
# ═══════════════════════════════════════════════


@dataclass
class GenerationRequest:
    """Input parameters for text generation.

    Attributes:
        prompt: The input text to generate from.
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (0.0 = greedy, 1.0 = creative).
        top_p: Nucleus sampling threshold.
        top_k: Only consider the top-k most likely tokens. 0 = disabled.
        stop: List of strings that stop generation when encountered.
        stream: Whether to stream tokens one at a time.
        repetition_penalty: Penalty for repeating tokens (1.0 = no penalty).
    """

    prompt: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    stop: list[str] = field(default_factory=list)
    stream: bool = False
    repetition_penalty: float = 1.0


@dataclass
class GenerationResponse:
    """Output from text generation.

    Attributes:
        text: The generated text (not including the prompt).
        finish_reason: Why generation stopped ("stop", "length", or "error").
        usage: Token usage statistics.
        latency_ms: Time taken for generation in milliseconds.
        model_name: Which model produced this response.
    """

    text: str
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0
    model_name: str = ""


# ═══════════════════════════════════════════════
# SECTION 2: ServingEngine Class
# ═══════════════════════════════════════════════


class ServingEngine:
    """High-performance LLM serving engine backed by vLLM.

    Manages the lifecycle of a vLLM engine instance with fallback to
    HuggingFace generate() when vLLM is unavailable.

    Usage:
        engine = ServingEngine()
        engine.load_model("./outputs/sft/final", config)

        response = engine.generate(GenerationRequest(prompt="Hello!"))
        print(response.text)

        async for chunk in engine.generate_stream(request):
            print(chunk.text, end="", flush=True)
    """

    def __init__(self) -> None:
        """Initialize the serving engine.

        No model is loaded yet — call load_model() to load a model.
        """
        self._engine: Any | None = None
        self._hf_model: Any | None = None
        self._hf_tokenizer: Any | None = None
        self._model_name: str = ""
        self._is_ready: bool = False
        self._using_vllm: bool = False
        self._config: Any | None = None

    @property
    def is_ready(self) -> bool:
        """Whether the engine has a model loaded and is ready to serve."""
        return self._is_ready

    @property
    def model_name(self) -> str:
        """Name/path of the currently loaded model."""
        return self._model_name

    def load_model(self, model_path: str, config: Any = None) -> None:
        """Load a model into the serving engine.

        Attempts to use vLLM first (for production performance).
        Falls back to HuggingFace if vLLM is not available.

        Args:
            model_path: Path to model weights (local dir, HF Hub ID, or S3 URI).
            config: ServingConfig instance from config/models.py.

        Raises:
            FileNotFoundError: If model_path doesn't exist (for local paths).
            RuntimeError: If neither vLLM nor HuggingFace can load the model.
        """
        self._model_name = model_path
        self._config = config

        max_model_len = getattr(config, "max_model_len", 2048) if config else 2048
        gpu_memory_utilization = (
            getattr(config, "gpu_memory_utilization", 0.9) if config else 0.9
        )

        if self._try_load_vllm(model_path, max_model_len, gpu_memory_utilization):
            self._using_vllm = True
            self._is_ready = True
            logger.info(f"Model loaded with vLLM: {model_path}")
            return

        logger.info("vLLM not available. Falling back to HuggingFace generate().")
        self._try_load_huggingface(model_path)
        self._using_vllm = False
        self._is_ready = True
        logger.info(f"Model loaded with HuggingFace: {model_path}")

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text synchronously (returns full response at once).

        Args:
            request: GenerationRequest with prompt and sampling parameters.

        Returns:
            GenerationResponse with generated text and metadata.

        Raises:
            RuntimeError: If no model is loaded (call load_model() first).
        """
        if not self._is_ready:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start_time = time.perf_counter()

        if self._using_vllm:
            response = self._generate_vllm(request)
        else:
            response = self._generate_huggingface(request)

        response.latency_ms = (time.perf_counter() - start_time) * 1000.0
        response.model_name = self._model_name

        return response

    async def generate_stream(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationResponse]:
        """Generate text as an async stream (token by token).

        Args:
            request: GenerationRequest with prompt and sampling parameters.

        Yields:
            GenerationResponse objects, each containing a text chunk.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if not self._is_ready:
            raise RuntimeError("No model loaded. Call load_model() first.")

        start_time = time.perf_counter()

        if self._using_vllm:
            async for chunk in self._stream_vllm(request):
                chunk.model_name = self._model_name
                yield chunk
        else:
            response = self._generate_huggingface(request)
            words = response.text.split(" ")
            for i, word in enumerate(words):
                chunk_text = word if i == 0 else " " + word
                is_last = i == len(words) - 1
                yield GenerationResponse(
                    text=chunk_text,
                    finish_reason="stop" if is_last else "",
                    model_name=self._model_name,
                    latency_ms=(time.perf_counter() - start_time) * 1000.0,
                )

    # ─── Private: vLLM backend ────────────────────────────────────

    def _try_load_vllm(
        self, model_path: str, max_model_len: int, gpu_memory_utilization: float
    ) -> bool:
        """Attempt to load the model using vLLM.

        Returns:
            True if vLLM loaded successfully, False otherwise.
        """
        try:
            from vllm import LLM, SamplingParams  # noqa: F401

            self._engine = LLM(
                model=model_path,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=True,
            )
            return True

        except ImportError:
            logger.debug("vLLM not installed. Install with: pip install vllm")
            return False
        except Exception as e:
            logger.warning(f"Failed to load model with vLLM: {e}")
            return False

    def _generate_vllm(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using the vLLM engine (synchronous).

        Args:
            request: Generation parameters.

        Returns:
            GenerationResponse with the full generated text.
        """
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
            stop=request.stop or None,
            repetition_penalty=request.repetition_penalty,
        )

        outputs = self._engine.generate([request.prompt], sampling_params)

        if outputs and outputs[0].outputs:
            output = outputs[0].outputs[0]
            return GenerationResponse(
                text=output.text,
                finish_reason=output.finish_reason or "stop",
                usage={
                    "prompt_tokens": len(outputs[0].prompt_token_ids),
                    "completion_tokens": len(output.token_ids),
                    "total_tokens": len(outputs[0].prompt_token_ids) + len(output.token_ids),
                },
            )

        return GenerationResponse(text="", finish_reason="error")

    async def _stream_vllm(
        self, request: GenerationRequest
    ) -> AsyncIterator[GenerationResponse]:
        """Stream generation using vLLM's async engine.

        Args:
            request: Generation parameters.

        Yields:
            GenerationResponse chunks with partial text.
        """
        try:
            from vllm import SamplingParams

            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else -1,
                stop=request.stop or None,
                repetition_penalty=request.repetition_penalty,
            )

            # Fall back to sync generation + simulated streaming
            # (True streaming requires AsyncLLMEngine setup)
            outputs = self._engine.generate([request.prompt], sampling_params)

            if outputs and outputs[0].outputs:
                text = outputs[0].outputs[0].text
                yield GenerationResponse(
                    text=text,
                    finish_reason=outputs[0].outputs[0].finish_reason or "stop",
                    usage={
                        "prompt_tokens": len(outputs[0].prompt_token_ids),
                        "completion_tokens": len(outputs[0].outputs[0].token_ids),
                    },
                )

        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield GenerationResponse(text="", finish_reason="error")

    # ─── Private: HuggingFace fallback ────────────────────────────

    def _try_load_huggingface(self, model_path: str) -> None:
        """Load model using HuggingFace transformers (fallback).

        Args:
            model_path: Path to model weights or HuggingFace Hub ID.

        Raises:
            RuntimeError: If HuggingFace can't load the model either.
        """
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True
            )
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
            )
            if self._hf_tokenizer.pad_token is None:
                self._hf_tokenizer.pad_token = self._hf_tokenizer.eos_token

        except Exception as e:
            raise RuntimeError(
                f"Failed to load model with both vLLM and HuggingFace: {e}"
            ) from e

    def _generate_huggingface(self, request: GenerationRequest) -> GenerationResponse:
        """Generate using HuggingFace transformers (fallback).

        Args:
            request: Generation parameters.

        Returns:
            GenerationResponse with the full generated text.
        """
        import torch

        inputs = self._hf_tokenizer(
            request.prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self._hf_model.device)

        prompt_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature if request.temperature > 0 else 1.0,
                top_p=request.top_p,
                top_k=request.top_k if request.top_k > 0 else 50,
                do_sample=request.temperature > 0,
                repetition_penalty=request.repetition_penalty,
            )

        generated_ids = outputs[0][prompt_length:]
        generated_text = self._hf_tokenizer.decode(
            generated_ids, skip_special_tokens=True
        )

        finish_reason = "length" if len(generated_ids) >= request.max_tokens else "stop"

        return GenerationResponse(
            text=generated_text,
            finish_reason=finish_reason,
            usage={
                "prompt_tokens": prompt_length,
                "completion_tokens": len(generated_ids),
                "total_tokens": prompt_length + len(generated_ids),
            },
        )
