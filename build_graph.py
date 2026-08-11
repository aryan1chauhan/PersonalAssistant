"""
Root entrypoint for SecondSelf Knowledge Graph Builder.
Delegates to src.build_graph.
"""

import sys
from pathlib import Path

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.build_graph import cli_entrypoint

if __name__ == "__main__":
    cli_entrypoint()
