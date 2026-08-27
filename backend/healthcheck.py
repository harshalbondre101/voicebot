from __future__ import annotations

import os
import sys
from pathlib import Path

import requests


def check_llama() -> bool:
    base_url = os.getenv("LLAMA_BASE_URL", "http://127.0.0.1:8080/v1").rstrip("/")
    try:
        response = requests.get(f"{base_url}/models", timeout=5)
        print(f"llama.cpp /models: {response.status_code}")
        return response.ok
    except Exception as exc:
        print(f"llama.cpp check failed: {exc}")
        return False


def check_orato_files() -> bool:
    model_dir = os.getenv("ORATO_MODEL_DIR", "")
    if not model_dir:
        print("ORATO_MODEL_DIR is empty; backend will download from Hugging Face at startup.")
        return True
    root = Path(model_dir)
    required = ["model.pt", "vocab.txt", "voices.json"]
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        print(f"Missing Orato files in {root}: {missing}")
        return False
    print(f"Orato files present: {root}")
    return True


def main() -> int:
    ok = True
    ok = check_llama() and ok
    ok = check_orato_files() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
