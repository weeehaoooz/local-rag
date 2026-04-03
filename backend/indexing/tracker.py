import json
import os
import hashlib
from typing import Dict, Set


class IndexingTracker:
    """Tracks file hashes to support incremental indexing."""

    def __init__(self, state_file: str):
        self.state_file = state_file
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                    # Ensure both top-level keys exist
                    if "files" not in data:
                        # Legacy format: the whole dict is file hashes
                        data = {"files": data, "guardrail_hashes": {}}
                    data.setdefault("guardrail_hashes", {})
                    return data
            except Exception:
                pass
        return {"files": {}, "guardrail_hashes": {}}

    def save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=4)

    def get_file_hash(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def is_file_changed(self, file_path: str) -> bool:
        current_hash = self.get_file_hash(file_path)
        stored_hash = self.state["files"].get(file_path)
        return current_hash != stored_hash

    def update_file_hash(self, file_path: str):
        self.state["files"][file_path] = self.get_file_hash(file_path)

    def get_dirty_files(self, directory: str) -> Set[str]:
        dirty_files = set()
        for root, _, files in os.walk(directory):
            for file in files:
                if file.endswith('.txt') or file.endswith('.pdf'):
                    full_path = os.path.join(root, file)
                    if self.is_file_changed(full_path):
                        dirty_files.add(full_path)
        return dirty_files

    # --- Guardrail change tracking ---

    def is_guardrail_changed(self, category: str, current_hash: str) -> bool:
        return self.state["guardrail_hashes"].get(category) != current_hash

    def update_guardrail_hash(self, category: str, current_hash: str):
        self.state["guardrail_hashes"][category] = current_hash
