import json
import os
import hashlib
from pathlib import Path
from typing import Dict, List, Optional

class PaperCache:
    def __init__(self, cache_dir: str = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".arxai" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
    def _paper_path(self, doi: str) -> Path:
        paper_id = hashlib.md5(doi.encode()).hexdigest()
        return self.cache_dir / f"{paper_id}.json"
    
    def get_paper(self, doi: str) -> Optional[Dict]:
        path = self._paper_path(doi)
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def save_paper(self, doi: str, data: Dict):
        path = self._paper_path(doi)
        with open(path, 'w') as f:
            json.dump(data, f)
    
    def get_session(self, parent_doi: str) -> Optional[Dict]:
        session_id = hashlib.md5(parent_doi.encode()).hexdigest()
        path = self.cache_dir / f"session_{session_id}.json"
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return None
    
    def save_session(self, parent_doi: str, data: Dict):
        session_id = hashlib.md5(parent_doi.encode()).hexdigest()
        path = self.cache_dir / f"session_{session_id}.json"
        with open(path, 'w') as f:
            json.dump(data, f)

