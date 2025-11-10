"""Base helpers for running local Hugging Face causal language models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


@dataclass(slots=True)
class ModelConfig:
    """Common knobs for any locally hosted causal LM."""

    model_name: str
    max_new_tokens: int = 1024
    temperature: float = 0.0
    device: Optional[str] = None
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

        self.device = torch.device(
            config.device if config.device else (
                "cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name).to(self.device)  # type: ignore

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(self, prompt: str) -> ModelGeneration:
        """Run generation on the underlying model and decode the completion."""

        # Encode prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Run generation
        generation = self.model.generate(
            **inputs,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.temperature > 0.0,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Decode completion
        prompt_length = inputs["input_ids"].shape[-1]
        completion_ids = generation[0, prompt_length:]
        completion = self.tokenizer.decode(
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
