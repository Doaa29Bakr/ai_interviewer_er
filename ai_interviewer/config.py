import json
import os
from pathlib import Path

# Path to the JSON file containing API keys
KEYS_FILE = Path(__file__).parent / "api_keys.json"

_keys = {}

def _load_keys():
    global _keys
    if KEYS_FILE.exists():
        try:
            with open(KEYS_FILE, "r") as f:
                _keys = json.load(f)
        except Exception as e:
            print(f"Error reading {KEYS_FILE}: {e}")

# Load keys on module import
_load_keys()

def get_key(key_name: str, default: str = "") -> str:
    """
    Get an API key or configuration value.
    First checks api_keys.json, then environment variables, then falls back to default.
    """
    if key_name in _keys:
        return _keys[key_name]
    
    val = os.getenv(key_name)
    if val is not None:
        return val
        
    return default

def get_key_int(key_name: str, default: str = "") -> int:
    return int(get_key(key_name, default))

def get_key_float(key_name: str, default: str = "") -> float:
    return float(get_key(key_name, default))
