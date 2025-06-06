import os
import json
from datetime import datetime
from typing import List, Dict
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


def save_model(model, model_name: str, algorithm="PPO"):
    """
    Saves the model and its metadata.
    """
    filename = f"{model_name}.zip"
    path = os.path.join(MODELS_DIR, filename)
    model.save(path)

    metadata = _load_metadata()
    metadata[model_name] = {
        "filename": filename,
        "algorithm": algorithm,
        "created_at": datetime.now().isoformat(),
        "path": path
    }
    _save_metadata(metadata)
    print(f"[] Model '{model_name}' saved and registered.")


def load_model(model_name: str):
    print("model name is now ", model_name, flush=True)
    """
    Loads the model by name.
    """
    metadata = _load_metadata()
    # print("metadata is now ", metadata, flush=True)
    # print("metadata[omar]", metadata["Omar"])
    if model_name not in metadata:
        raise FileNotFoundError(f"No metadata found for model: {model_name}")

    model_path = metadata[model_name]["path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")

    return PPO.load(model_path)


def delete_model(model_name: str):
    """
    Deletes a saved model and updates the metadata.
    """
    metadata = _load_metadata()
    if model_name in metadata:
        model_path = metadata[model_name]["path"]
        if os.path.exists(model_path):
            os.remove(model_path)
        del metadata[model_name]
        _save_metadata(metadata)
        print(f"[] Model '{model_name}' deleted.")
    else:
        print(f"[!] Model '{model_name}' not found.")
