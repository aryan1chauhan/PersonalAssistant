import sys
from pathlib import Path

# convenient root wrapper
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.build_graph import cli_entrypoint

if __name__ == "__main__":
    cli_entrypoint()
