"""Base helpers for running local Hugging Face causal language models."""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerBase


@dataclass(slots=True)
class ModelConfig:
    """Common knobs for any locally hosted causal LM."""

    model_name: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    device: Optional[str] = None
    device_map: Optional[str] = None
    batch_size: int = 1
    is_reasoning_model: bool = False


@dataclass(slots=True)
class ModelGeneration:
    """Structured output from the model runner."""

    prompt: str
    completion: str
    prompt_tokens_num: int
    completion_tokens_num: int


class LocalModelRunner:
    """
    Thin wrapper around transformers to keep prompt/response handling uniform.
    """

    def __init__(self, config: ModelConfig):
        """Load model and tokenizer according to ModelConfig"""
        self.config = config

        multi_gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
        device_map = config.device_map
        if device_map is None and config.device is None and multi_gpu_available:
            device_map = "auto"

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        if device_map:
            primary_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(primary_device_str)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=device_map,
            )  # type: ignore
        else:
            primary_device_str = (
                config.device if config.device else (
                    "cuda" if torch.cuda.is_available() else "cpu")
            )
            self.device = torch.device(primary_device_str)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name
            ).to(self.device)  # type: ignore

        self.device_map = device_map
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer = tokenizer

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self._is_unloaded = False

    def unload(self) -> None:
        """Free model/tokenizer memory so another model can be loaded."""
        if self._is_unloaded:
            return

        try:
            if self.model is not None:
                if self.device_map:  # accelerate manages model placement, just drop reference
                    model_ref = self.model
                    self.model = None
                    del model_ref
                else:
                    try:
                        self.model.to("cpu")  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    model_ref = self.model
                    self.model = None
                    del model_ref

            if self.tokenizer is not None:
                self.tokenizer = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            gc.collect()
        finally:
            self._is_unloaded = True

    def generate(self, prompt: str) -> ModelGeneration:
        """Run generation on the underlying model and decode the completion."""

        if self._is_unloaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Cannot generate with an unloaded model runner.")

        model = self.model
        tokenizer = self.tokenizer

        # Encode prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # Run generation
        generation = model.generate(  # type: ignore
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0.0,
            pad_token_id=tokenizer.pad_token_id,
        )

        # Decode completion
        prompt_length = inputs["input_ids"].shape[-1]  # type: ignore
        completion_ids = generation[0, prompt_length:]
        completion = tokenizer.decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return ModelGeneration(
            prompt=prompt,
            completion=completion,
            prompt_tokens_num=prompt_length,
            completion_tokens_num=len(completion_ids)
        )
