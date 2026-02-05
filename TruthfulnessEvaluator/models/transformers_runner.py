"""Transformers-backed model runner."""

from __future__ import annotations

import gc
import json
from dataclasses import asdict
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from .base import ModelConfig, ModelGeneration


class TransformersModelRunner:
    """Thin wrapper around transformers to keep prompt/response handling uniform."""

    def __init__(self, config: ModelConfig):
        self.config = config
        multi_gpu_available = torch.cuda.is_available() and torch.cuda.device_count() > 1
        device_map = config.device_map
        if device_map is None and config.device is None and multi_gpu_available:
            device_map = "auto"

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[PreTrainedTokenizerBase] = None

        if device_map:  # Try to apply multi-device device map
            primary_device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(primary_device_str)
            self.model = AutoModelForCausalLM.from_pretrained(
                config.model_name,
                device_map=device_map,
            )  # type: ignore
        else:  # Else load to specified device
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

        # Extra configuration for tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.tokenizer.padding_side = "left"  # type: ignore

        # Set model to eval mode
        self.model.eval()  # type: ignore

        self._is_unloaded = False
        self._log_configuration()

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
        generations = self.generate_batch([prompt])
        return generations[0]

    def generate_batch(self, prompts: List[str]) -> List[ModelGeneration]:
        """Run batched generation and decode completions for each prompt."""
        if not prompts:
            return []
        if self._is_unloaded or self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Cannot generate with an unloaded model runner.")

        model = self.model
        tokenizer = self.tokenizer

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)
        attention_mask = inputs["attention_mask"]
        prompt_lengths = attention_mask.sum(  # type: ignore[call-arg]
            dim=-1).tolist()

        with torch.inference_mode():
            generation = model.generate(  # type: ignore
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                do_sample=self.config.temperature > 0.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=None if self.config.ignore_eos else tokenizer.eos_token_id,
            )

        generations: List[ModelGeneration] = []
        for idx, prompt in enumerate(prompts):
            prompt_length = int(prompt_lengths[idx])
            completion_ids = generation[idx, prompt_length:]
            completion = tokenizer.decode(
                completion_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            generations.append(ModelGeneration(
                prompt=prompt,
                completion=completion,
                prompt_tokens_num=prompt_length,
                completion_tokens_num=int(len(completion_ids)),
            ))
        return generations

    def _log_configuration(self) -> None:
        config_payload = asdict(self.config)
        print("[model] transformers runner initialized with config:")
        print(json.dumps(config_payload, indent=2, sort_keys=True))
