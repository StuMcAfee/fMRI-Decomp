"""Let pytest import mejoint straight from src/ (no pip install needed on the cluster)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
