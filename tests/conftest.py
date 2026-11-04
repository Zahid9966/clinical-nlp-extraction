import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw"
needs_data = pytest.mark.skipif(not (RAW / "mesh_disease_lexicon.json").exists(), reason="run `make data` first")
needs_models = pytest.mark.skipif(not (ROOT / "models" / "ner.crf").exists(), reason="run `make train` first")
