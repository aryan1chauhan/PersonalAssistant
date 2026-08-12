"""
Root entrypoint for SecondSelf RAG Q&A Engine.
Delegates to src.ask.
"""

import sys
from pathlib import Path

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ask import cli_entrypoint

if __name__ == "__main__":
    cli_entrypoint()
