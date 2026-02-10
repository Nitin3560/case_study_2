from __future__ import annotations
import json
import os
from typing import Any


class JsonlLogger:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self) -> None:
        self.f.close()