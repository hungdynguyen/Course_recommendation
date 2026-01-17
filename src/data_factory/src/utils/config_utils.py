from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv

from ..settings import Settings


def load_config() -> Settings:
    """Load settings from default config file with environment variable substitution."""
    # Load .env file from project root
    env_path = Path(__file__).resolve().parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    
    config_path = Path(__file__).resolve().parent.parent.parent / "config" / "settings.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file: {config_path}")
    
    # Read raw YAML content
    raw_yaml = config_path.read_text(encoding="utf-8")
    
    # Replace environment variables ($VAR or ${VAR})
    expanded_yaml = _expand_env_vars(raw_yaml)
    
    # Parse with Settings
    import yaml
    config_dict = yaml.safe_load(expanded_yaml)
    
    # Temporarily write to Settings.load
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as tmp:
        yaml.dump(config_dict, tmp)
        tmp_path = tmp.name
    
    try:
        return Settings.load(tmp_path)
    finally:
        Path(tmp_path).unlink()


def _expand_env_vars(text: str) -> str:
    """Replace $VAR or ${VAR} with environment variable values."""
    def replacer(match):
        var_name = match.group(1) or match.group(2)
        return os.getenv(var_name, match.group(0))  # Keep original if not found
    
    # Match ${VAR} or $VAR
    pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)'
    return re.sub(pattern, replacer, text)
