"""Configuration loading for PISDR. All experiment parameters live in YAML
configs so every reported number traces to a config file (see docs/03)."""
from pathlib import Path
import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "config.yaml"


def load_config(path=None):
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)
