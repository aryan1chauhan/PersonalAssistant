"""
Root entrypoint for SecondSelf PARA Classifier.
Delegates to src.classify.
"""

import sys
from pathlib import Path

# Ensure workspace root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.classify import cli_entrypoint

if __name__ == "__main__":
    cli_entrypoint()
