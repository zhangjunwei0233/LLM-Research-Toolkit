import yaml
from pathlib import Path
from typing import Dict, Any

def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Load a YAML config file and return a dict (empty if missing)"""
    if not path.exists():
        return {}
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise RuntimeError(f"Failed to parse YAML config at {path}") from e


def deep_merge(default: dict, overwrite: dict) -> dict:
    """Deep Merge two dicts to prevent key ignore"""
    result = default.copy()
    for k, v in overwrite.items():
        if (k in result and isinstance(result[k], dict) and isinstance(v, dict)):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v

    return result
