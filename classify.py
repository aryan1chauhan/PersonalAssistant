"""
Root entrypoint for SecondSelf PARA Classifier.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    print("Classify module will be executed in Phase 2.")
