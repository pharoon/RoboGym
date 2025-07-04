import os
import json
from datetime import datetime
from typing import Dict, Optional
from stable_baselines3 import PPO


class ModelManager:
    def __init__(self, models_dir: str = "trained_models", metadata_file: str = "model_metadata.json"):
        self.models_dir = models_dir
        self.metadata_file = metadata_file
        os.makedirs(self.models_dir, exist_ok=True)

    def _get_metadata_path(self) -> str:
        return os.path.join(self.models_dir, self.metadata_file)

    def _load_metadata(self) -> Dict:
        path = self._get_metadata_path()
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, metadata: Dict):
        try:
            with open(self._get_metadata_path(), "w") as f:
                json.dump(metadata, f, indent=4)
            print("[✓] Metadata saved.")
        except Exception as e:
            print(f"[!] Failed to save metadata: {e}")

    def save_model(self, model, model_name: str, algorithm: str = "PPO", model_path: Optional[str] = None):
        """
        Save the model to disk and register it in metadata.
        """
        if model_path is None:
            model_path = os.path.join(self.models_dir, f"{model_name}.zip")

        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        model.save(model_path)

        metadata = self._load_metadata()
        metadata[model_name] = {
            "filename": os.path.basename(model_path),
            "algorithm": algorithm,
            "created_at": datetime.now().isoformat(),
            "path": model_path
        }
        self._save_metadata(metadata)
        print(f"Model '{model_name}' saved and registered.")

    def load_model(self, model_name: str, model_path: Optional[str] = None):
        """
        Load model from disk using metadata or given path.
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            return PPO.load(model_path)

        metadata = self._load_metadata()
        if model_name not in metadata:
            raise FileNotFoundError(f"No metadata found for model: {model_name}")

        default_path = metadata[model_name]["path"]
        if not os.path.exists(default_path):
            raise FileNotFoundError(f"Model file not found at: {default_path}")

        return PPO.load(default_path)

    def delete_model(self, model_name: str, model_path: Optional[str] = None):
        """
        Delete a model file and remove metadata entry.
        """
        if model_path and os.path.exists(model_path):
            os.remove(model_path)
            print(f"[✓] Model file deleted at: {model_path}")
        elif model_path:
            print(f"[!] Model file not found at: {model_path}")

        metadata = self._load_metadata()
        if model_name in metadata:
            default_path = metadata[model_name]["path"]
            if not model_path and os.path.exists(default_path):
                os.remove(default_path)
                print(f"[✓] Model file deleted at: {default_path}")
            del metadata[model_name]
            self._save_metadata(metadata)
            print(f"[✓] Model '{model_name}' metadata deleted.")
        else:
            print(f"[!] Model '{model_name}' not found in metadata.")
