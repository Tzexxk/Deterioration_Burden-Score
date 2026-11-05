"""Assess configured Ridge IDs on the frozen development validation folds."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.assessment.ehr.ridge.assessment import run_assessment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/assessment/ehr/ridge.json",
                        help="Project-relative Ridge assessment configuration")
    parser.add_argument("--id", help="Assess only this ID from the configuration")
    args = parser.parse_args(argv)
    try:
        layout = Layout()
        result = run_assessment(layout, args.config, args.id)
        print(json.dumps(result))
        return 2 if result["status"] == "failed" else 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: EHR assessment cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"EHR_ASSESSMENT_ERROR: {type(error).__name__}; check the package and training artifacts", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
