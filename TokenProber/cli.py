from typing import NamedTuple, Optional, Tuple

import torch
import sys
import os
from contextlib import contextmanager


class UserDecision(NamedTuple):
    """Structured user decision from interactive prompt."""

    action: str  # token, rollback, stop
    token_id: Optional[int] = None


@contextmanager
def raw_mode(stream):
    """Temporarily switch stdin into raw mode to capture single keypress."""
    if os.name != "posix":
        yield
        return

    fd = stream.fileno()
    import termios
    import tty

    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)  # Disable line buffering and echo
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def read_user_cmd(prompt: str) -> Tuple[str, Optional[str]]:
    """
    Capture user input, supporting arrow-key shortcuts and raw characters.

    Supported cmds:
    - Direct \\n:  "accept_top"
    - Left Arrow:  "rollback"
    - Right Arrow: "accept_top"
    - Input text:  "text" (id selection, token input, other cmds, ...)
    """
    sys.stdout.write(prompt)
    sys.stdout.flush()

    if os.name == "nt":
        raise NotImplementedError(
            "The package does not support Windows systems now.")

    buffer: list[str] = []
    with raw_mode(sys.stdin):
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                continue
            if ch in ("\n", "\r"):
                sys.stdout.write("\n")
                sys.stdout.flush()
                text = "".join(buffer)
                return ("text", text) if buffer else ("accept_top", None)
            if ch == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            if ch == "\x7f":  # Backspace
                if buffer:
                    buffer.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
                continue
            if ch == "\x1b":  # Escape sequence prefix for arrow keys
                seq = sys.stdin.read(2)
                if seq == "[D":  # Left arrow
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return ("rollback", None)
                if seq == "[C":  # Right arrow
                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    return ("accept_top", None)
                continue

            buffer.append(ch)
            sys.stdout.write(ch)
            sys.stdout.flush()
    # Unreachable
    raise RuntimeError("Should not reach here.")


def resolve_user_choice(
        tokenizer,
        top_indices: torch.Tensor,
        step: int,
) -> UserDecision:
    """
    Obtain next action from user input.

    Supported actions:
    - token: selected token id
    - rollback
    - stop
    """
    while True:
        mode, raw_input = read_user_cmd(f"[step {step + 1}] Enter selection> ")

        # accept_top mode
        if mode == "accept_top":
            return UserDecision("token", int(top_indices[0].item()))

        # roll_back mode
        if mode == "rollback":
            return UserDecision("rollback", None)

        # Parse other text inputs
        if mode == "text":
            if not raw_input:
                raise RuntimeError("Text input with empty text body.")

            processed_input = raw_input.strip().lower()

            # id_selection mode
            if processed_input.isdigit():
                index = int(processed_input)
                if 1 <= index <= len(top_indices):
                    return UserDecision("token", int(top_indices[index - 1].item()))
                print(f"Index {index} is outside the displayed range.")
                continue

            # Other cmds
            if processed_input in {"/stop", "/quit", "quit", "exit"}:
                return UserDecision("stop", None)
            if processed_input in {"/help", "?"}:
                print(
                    "Enter a rank index, type a single token, press ENTER or -> for the top choices, or <- to roll back."
                )
                continue

            # Attempt to interpret the raw text as a token
            # Use raw_input to maintain input backspaces
            encoded = tokenizer.encode(raw_input, add_special_tokens=False)
            if len(encoded) == 1:
                return UserDecision("token", encoded[0])
            else:
                print("Input must resolve to exactly one token. Try a shorter fragment.")

        # Should not reach here
        raise NotImplementedError(f"Unkown input mode: {mode}")
