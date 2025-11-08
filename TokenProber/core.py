from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from typing import Optional, Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import load_probe_config
from .cli import resolve_user_choice
from .render import supports_color_output, render_session_view, render_final_view


def _compute_probabilities(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Convert logits into a probability distribution under a given temperature."""
    if temperature <= 0:
        raise ValueError("Temperature must be strictly positive.")
    calibrated = logits / temperature
    return torch.softmax(calibrated, dim=-1)


def main(argv: Optional[Iterable[str]] = None) -> int:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Interactive token probability explorer.")
    parser.add_argument(
        "--config",
        default=None,
        help=f"Yaml path supplying a ProbeConfig",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    # Load and resolve configuration
    config = load_probe_config(args.config)
    print("Loaded probe configuration:")
    for key, value in asdict(config).items():
        print(f"  {key}: {value}")

    model_path = config.model_path
    prompt = config.prompt
    device = config.resolve_device()
    dtype = config.resolve_dtype()
    dtype_kwargs = {} if dtype in (None, "auto") else {"torch_dtype": dtype}
    max_new_tokens = config.max_new_tokens
    temperature = config.temperature
    top_k_display = config.top_k_display

    enable_color = supports_color_output()

    # Load model
    print(f"\nLoading tokenizer and model from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, **dtype_kwargs)

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)  # pyright: ignore[reportArgumentType]
    model.eval()  # Set to inference mode

    print(f"Successfully loaded model: {model_path}")

    # Start generation
    with torch.no_grad():
        # Initialization
        encoded = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model(**encoded, use_cache=True)
        logits = outputs.logits[:, -1, :].squeeze(0)
        past_key_values = outputs.past_key_values  # For KV cache

        # Global buffers
        gen_id_so_far: List[int] = encoded["input_ids"].tolist()[0]
        gen_prob_so_far: List[float] = [-1 for _ in range(len(gen_id_so_far))]

        prompt_id_len = len(gen_id_so_far)

        step = 0
        while step < max_new_tokens:
            probabilities = _compute_probabilities(logits, temperature)
            # Find k biggest probabilities
            k = min(top_k_display, probabilities.numel())
            top_probs, top_indices = torch.topk(probabilities, k)

            # Unpdate render display
            render_session_view(tokenizer, step, gen_id_so_far,
                                gen_prob_so_far, top_indices, top_probs, enable_color)

            # resolve and execute user choice
            decision = resolve_user_choice(tokenizer, top_indices, step)
            match decision.action:
                case "token":
                    if decision.token_id is None:
                        print("Selection did not resolve a token, try again")
                        continue

                    choice = decision.token_id
                    choice_prob = float(probabilities[choice].item())

                    # Append token to generated_so_far
                    gen_prob_so_far.append(choice_prob)
                    gen_id_so_far.append(choice)

                    # Continue to generate
                    new_input_id = torch.tensor([[choice]], device=device)

                    outputs = model(
                        input_ids=new_input_id,
                        use_cache=True,
                        past_key_values=past_key_values
                    )

                    step += 1

                case "rollback":
                    if len(gen_id_so_far) <= prompt_id_len:
                        print("No tokens to roll back.")
                        continue

                    gen_id_so_far.pop()
                    gen_prob_so_far.pop()

                    # Re-send inputs to model
                    gen_str_so_far = tokenizer.decode(
                        gen_id_so_far, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                    encoded = tokenizer(
                        gen_str_so_far, return_tensors="pt").to(device)
                    outputs = model(**encoded, use_cache=True)

                    step -= 1

                case "stop":
                    break

                case _:
                    raise NotImplementedError(
                        f"Unknown user action: {decision.action}")

            logits = outputs.logits[:, -1, :].squeeze(0)
            past_key_values = outputs.past_key_values

    render_final_view(tokenizer, step, max_new_tokens,
                      gen_id_so_far, gen_prob_so_far, enable_color)

    return 0
