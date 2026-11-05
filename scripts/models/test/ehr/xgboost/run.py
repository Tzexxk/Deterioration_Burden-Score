"""Run the EHR xgboost refit or fixed-test evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.models.test.ehr.cli import main
from src.models.test.ehr.xgboost.model import fit, predict

if __name__ == "__main__":
    raise SystemExit(main("xgboost", fit, predict))
