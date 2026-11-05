"""Run the EHR catboost refit or fixed-test evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.models.test.ehr.catboost.model import fit, predict
from src.models.test.ehr.cli import main

if __name__ == "__main__":
    raise SystemExit(main("catboost", fit, predict))
