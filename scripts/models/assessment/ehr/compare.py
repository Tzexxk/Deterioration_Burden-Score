"""Compare all completed EHR assessments on paired development patients."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.common.paths import Layout, PipelineError
from src.models.assessment.ehr.comparison import run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="ID for the published comparison")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_comparison(Layout(), args.id)))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: EHR comparison cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"EHR_COMPARISON_ERROR: {type(error).__name__}; check the assessment outputs", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
