import os
from dataclasses import dataclass


@dataclass
class Config:
    kb_path: str = os.getenv("NGO_KB_PATH", os.path.join(os.path.dirname(__file__), "..", "kb"))
    chunk_size: int = int(os.getenv("NGO_CHUNK_SIZE", 800))
    chunk_overlap: int = int(os.getenv("NGO_CHUNK_OVERLAP", 120))
    top_k: int = int(os.getenv("NGO_TOP_K", 5))

    @staticmethod
    def resolve_kb_path(path: str | None) -> str:
        if path:
            return os.path.abspath(path)
        return os.path.abspath(Config.kb_path)
