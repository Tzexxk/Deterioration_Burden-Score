"""Compare completed image assessments on paired development patients."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.common.paths import Layout, PipelineError
from src.models.assessment.image.comparison import run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True, help="ID for the published comparison")
    parser.add_argument("--assessment", action="append", metavar="MODEL/ID",
                        help="Select an assessment, repeat for multiple candidates (default: all completed)")
    args = parser.parse_args(argv)
    try:
        progress = partial(print, file=sys.stderr, flush=True)
        print(json.dumps(run_comparison(Layout(), args.id, assessment_ids=args.assessment, progress=progress)))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Image comparison cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"IMAGE_COMPARISON_ERROR: {type(error).__name__}; check the assessment outputs", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
