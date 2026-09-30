"""Thin wrapper: python scripts/01_prepare.py [--raw-dir data/raw] [--out-dir data/prepared]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.prepare import main  # noqa: E402

if __name__ == "__main__":
    main()
