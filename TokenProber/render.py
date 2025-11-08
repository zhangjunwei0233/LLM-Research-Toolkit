import os
import sys
import torch
from typing import List

ANSI_RESET = "\033[0m"
PROBABILITY_COLOR_BINS = (
    (0.80, "\033[0;92m"),  # bright green
    (0.50, "\033[0;32m"),  # green
    (0.20, "\033[0;33m"),  # yellow
    (0.10, "\033[0;31m"),  # red
    (0.00, "\033[0;91m"),  # bright red
    (-1.0, "\033[0;90m"),  # dim grey
)


def supports_color_output() -> bool:
    """Return True when stdout can render ANSI colors."""
    if os.environ.get("NO_COLOR"):
        return False
    stream = getattr(sys, "stdout", None)
    if stream is None or not stream.isatty():
        return False

    if os.name != "nt":
        return True
    else:
        return any(os.environ.get(var) for var in ("ANSICON", "WT_SESSION", "TERM_PROGRAM", "TERM"))


def _clear_terminal():
    """Clear terminal window to keep the interacitve region compact."""
    if os.name == "nt":
        os.system("cls")
    else:
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()


def _colorize_token_text(text: str, probability: float, enable_color: bool) -> str:
    """Wrap token text in a color code chosen by probability mass."""
    if not enable_color:
        return text
    for threshold, color_code in PROBABILITY_COLOR_BINS:
        if probability >= threshold:
            return f"{color_code}{text}{ANSI_RESET}"
    raise RuntimeError(
        f"No color to assign to token with probability: {probability}")


def _render_gen_so_far(
        tokenizer,
        gen_id_so_far: List[int],
        gen_prob_so_far: List[float],
        enable_color: bool
) -> str:
    """Render decoded tokens stitched together, coloring each by probability."""
    assert len(gen_id_so_far) == len(
        gen_prob_so_far), "current generated id num doesn't match up with current prob num."

    pieces: List[str] = []
    for token_id, probability in zip(gen_id_so_far, gen_prob_so_far):
        token_text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        pieces.append(_colorize_token_text(
            token_text, probability, enable_color))
    return "".join(pieces) or ""


def _format_token(tokenizer, token_id: int) -> str:
    """Return a printable representation of an individual token."""
    text = tokenizer.convert_ids_to_tokens(int(token_id))

    # When conversion fails, fall back to decoding without cleanup so we retain prefixes
    if text is None or text == tokenizer.unk_token:
        text = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    if not text:
        text = str(token_id)

    # Replace whitespace characters for readability
    text = (
        text.replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
        .replace("\u202f", "\\u202f")
    )
    return text


def _render_prob_distribution(
    tokenizer,
    top_indices: torch.Tensor,
    top_probs: torch.Tensor
) -> str:
    """Return a formatted probability table for the current position."""
    lines: List[str] = []

    top_mass = top_probs.sum().item()
    header = f"Candidate tokens (top-{len(top_probs)} mass {top_mass:.4f})"
    lines.append(header)
    lines.append("-" * 40)
    lines.append(f"{'Idx':>3}{'Prob%':>7}{'Token Id':>9} Token")
    lines.append("-" * 40)

    indices = top_indices.detach().cpu().tolist()
    probs = top_probs.detach().cpu().tolist()

    for idx, (token_id, prob) in enumerate(zip(indices, probs), start=1):
        token_repr = _format_token(tokenizer, token_id)
        lines.append(f"{idx:>3}{prob * 100:7.2f}{token_id:9d} {token_repr}")

    remaining = max(0.0, 1.0 - top_mass)
    if remaining > 0:
        lines.append(
            f" ... Remaining probability mass: {remaining * 100:0.4f}%")

    return "\n".join(lines)


def render_session_view(
    tokenizer,
    step: int,
    gen_id_so_far: List[int],
    gen_prob_so_far: List[float],
    top_indices: torch.Tensor,
    top_probs: torch.Tensor,
    enable_color: bool
) -> None:
    """Render the session overview, replacing the prior output."""
    _clear_terminal()

    # Header
    lines = [
        f"Step {step + 1}",
        "",
    ]

    # Decoded output
    current_decoded_output = _render_gen_so_far(
        tokenizer, gen_id_so_far, gen_prob_so_far, enable_color)
    lines.extend(
        [
            "Current decoded output:",
            "=" * 40,
            current_decoded_output or "(empty)",
            "=" * 40,
        ]
    )

    # Prob distribution
    current_prob_distribution = _render_prob_distribution(
        tokenizer, top_indices, top_probs)
    lines.extend(
        [
            "",
            current_prob_distribution,
            "",
        ]
    )

    # Help message
    lines.append("Commands: <index> to accept, type a single token to force it, 'exit' to halt, '?' for instructions, <- to roll back, -> or ENTOR for top choice.")
    lines.append("")

    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()


def render_final_view(
        tokenizer,
        step: int,
        max_new_tokens: int,
        gen_id_so_far: List[int],
        gen_prob_so_far: List[float],
        enable_color: bool
) -> None:
    """Render the final view on closing."""
    _clear_terminal()

    # Close msg
    lines = [
        "Generation halted" if step < max_new_tokens else f"Has reached token limit: {max_new_tokens}",
        ""
    ]

    # Decoded output
    final_decoded_output = _render_gen_so_far(
        tokenizer, gen_id_so_far, gen_prob_so_far, enable_color)
    lines.extend(
        [
            "Final decoded output:",
            "=" * 40,
            final_decoded_output or "(empty)",
            "=" * 40
        ]
    )

    sys.stdout.write("\n".join(lines))
    sys.stdout.flush()
