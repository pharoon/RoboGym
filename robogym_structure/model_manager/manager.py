import os
import json
from datetime import datetime
from typing import List, Dict, Optional
from stable_baselines3 import PPO

MODELS_DIR = "trained_models"
METADATA_FILE = "model_metadata.json"

os.makedirs(MODELS_DIR, exist_ok=True)


def _get_metadata_path():
    return os.path.join(MODELS_DIR, METADATA_FILE)


def _load_metadata() -> Dict:
    path = _get_metadata_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {}


def _save_metadata(metadata: Dict):
    try:
        with open(_get_metadata_path(), "w") as f:
            json.dump(metadata, f, indent=4)
        print("[] Metadata saved.")
    except Exception as e:
        print(f"[!] Failed to save metadata: {e}")

def list_models() -> List[Dict]:
    metadata = _load_metadata()
    return [{"name": name, **details} for name, details in metadata.items()]


def save_model(model, model_name: str, algorithm="PPO", model_path: Optional[str] = None):
    """
    Saves the model and its metadata.
    """
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, f"{model_name}.zip")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    model.save(model_path)

    metadata = _load_metadata()
    metadata[model_name] = {
        "filename": os.path.basename(model_path),
        "algorithm": algorithm,
        "created_at": datetime.now().isoformat(),
        "path": model_path
    }
    _save_metadata(metadata)
    print(f"[] Model '{model_name}' saved and registered.")


def load_model(model_name: str, model_path: Optional[str] = None):
    """
    Loads the model by name.
    """
    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
        return PPO.load(model_path)
    
    metadata = _load_metadata()
    if model_name not in metadata:
        raise FileNotFoundError(f"No metadata found for model: {model_name}")

    default_path = metadata[model_name]["path"]
    if not os.path.exists(default_path):
        raise FileNotFoundError(f"Model file not found at: {default_path}")

    return PPO.load(default_path)


def delete_model(model_name: str, model_path: Optional[str] = None):
    """
    Deletes a saved model and updates the metadata.
    """
    if model_path:
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"[] Model file deleted at: {model_path}")
        else:
            print(f"[!] Model file not found at: {model_path}")
    
    metadata = _load_metadata()
    if model_name in metadata:
        default_path = metadata[model_name]["path"]
        if not model_path and os.path.exists(default_path):
            os.remove(default_path)
        del metadata[model_name]
        _save_metadata(metadata)
        print(f"[] Model '{model_name}' metadata deleted.")
    else:
        print(f"[!] Model '{model_name}' not found in metadata.")
