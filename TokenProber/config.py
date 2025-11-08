from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Any, Dict, Optional

from .utils import load_yaml_config, deep_merge


@dataclass
class ProbeConfig:
    """Runtime configuration for the token probing session."""

    model_path: str
    prompt: str
    device: Optional[str] = None
    torch_dtype: Optional[str] = "auto"
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k_display: int = 12

    def resolve_device(self) -> str:
        """Return the user-specified device or auto-detect a sensible default"""
        if self.device:
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if (
                getattr(torch.backends, "mps", None)
                and torch.backends.mps.is_available()
            ):
                return "mps"

        except Exception:
            pass
        return "cpu"

    def resolve_dtype(self):
        """Convert the configured dtype string into a torch dtype object."""
        if self.torch_dtype in (None, "auto"):
            return self.torch_dtype

        import torch

        valid = {
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float64": torch.float64,
            "fp64": torch.float64,
        }
        key = str(self.torch_dtype).lower()
        if key not in valid:
            raise ValueError(
                f"Unsupported torch_dtype '{self.torch_dtype}'. Expecting one of {sorted(valid)}"
            )
        return valid[key]


def _resolve_user_config_path(raw_path: Path | str) -> Path:
    """Resolve a user-supplied config reference into an absolute path."""
    candidate = Path(raw_path).expanduser()

    if candidate.exists():
        return candidate.resolve()

    if candidate.is_absolute():
        raise FileNotFoundError(f"Config file not found: {candidate}")

    search_roots = [
        Path.cwd() / "configs",
        Path(__file__).resolve().parent,
        Path(__file__).resolve().parent / "configs",
    ]
    seen: set[Path] = set()
    for root in search_roots:
        test_path = (root / candidate).resolve(strict=False)
        if test_path in seen:
            continue
        seen.add(test_path)
        if test_path.exists():
            return test_path

    raise FileNotFoundError(
        f"Config file not found: {candidate}. Checked {', '.join(str(p) for p in seen)}"
    )


def load_probe_config(user_path: Path | None = None) -> ProbeConfig:
    """Import the user-provided setup module and extract a ProbeConfig"""

    # Load default config
    default_path = Path(__file__).resolve().with_name("default_config.yaml")
    config_dict = load_yaml_config(default_path)

    # Override with user config
    if user_path:
        resolved_path = _resolve_user_config_path(user_path)
        user_cfg = load_yaml_config(resolved_path)
        config_dict = deep_merge(config_dict, user_cfg)

    # Filter out invalid keys
    valid_keys = {f.name for f in fields(ProbeConfig)}
    filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
    extra_keys = set(config_dict) - valid_keys
    if extra_keys:
        print(
            f"[config] Ignoring unknown keys in YAML: {', '.join(extra_keys)}")

    return ProbeConfig(**filtered)
