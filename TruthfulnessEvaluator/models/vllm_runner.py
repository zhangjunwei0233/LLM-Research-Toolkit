"""vLLM-backed model runner used when engine=vllm."""

from __future__ import annotations

import gc
from typing import List, Optional

import torch
from vllm import LLM, SamplingParams

from .base import ModelConfig, ModelGeneration


class VLLMModelRunner:
    """Wrapper around vLLM's high-throughput engine."""

    def __init__(self, config: ModelConfig):
        self.config = config
        tensor_parallel = (
            config.vllm_tensor_parallel_size
            if config.vllm_tensor_parallel_size is not None
            else max(1, torch.cuda.device_count() or 1)
        )
        self._llm: Optional[LLM] = LLM(
            model=config.model_name,
            tensor_parallel_size=tensor_parallel,
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_num_seqs=config.vllm_max_num_seqs,
            max_model_len=config.vllm_max_model_len,
            swap_space=config.vllm_swap_space,
        )
        self._is_unloaded = False

    def unload(self) -> None:
        if self._is_unloaded:
            return
        try:
            if self._llm is not None:
                llm_ref = self._llm
                self._llm = None
                del llm_ref
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        finally:
            self._is_unloaded = True

    def generate(self, prompt: str) -> ModelGeneration:
        generations = self.generate_batch([prompt])
        return generations[0]

    def generate_batch(self, prompts: List[str]) -> List[ModelGeneration]:
        if not prompts:
            return []
        if self._is_unloaded or self._llm is None:
            raise RuntimeError("Cannot generate with an unloaded vLLM runner.")

        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        outputs = self._llm.generate(prompts, sampling_params)
        generations: List[ModelGeneration] = []
        for prompt, result in zip(prompts, outputs):
            if not result.outputs:
                generations.append(ModelGeneration(
                    prompt=prompt,
                    completion="",
                    prompt_tokens_num=len(
                        result.prompt_token_ids),  # type: ignore
                    completion_tokens_num=0,
                ))
                continue

            completion_text = result.outputs[0].text if result.outputs else ""
            completion_tokens = (
                len(result.outputs[0].token_ids) if result.outputs else 0
            )
            prompt_tokens = len(result.prompt_token_ids)  # type: ignore
            generations.append(ModelGeneration(
                prompt=prompt,
                completion=completion_text,
                prompt_tokens_num=prompt_tokens,
                completion_tokens_num=completion_tokens,
            ))
        return generations
