"""Fit configured XGBoost experiments on the five development folds."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.train.ehr.xgboost.training import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv", help="Fit each selected ID on five development folds")
    cv.add_argument("--config", default="configs/models/train/ehr/xgboost.json",
                    help="Project-relative XGBoost batch configuration")
    cv.add_argument("--id", help="Experiment ID from the configuration")
    args = parser.parse_args(argv)
    try:
        result = run_training(Layout(), args.config, args.id)
        print(json.dumps(result))
        return 2 if result["status"] == "failed" else 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: EHR model operation cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"EHR_MODEL_ERROR: {type(error).__name__}; check the accepted package and protocol", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
