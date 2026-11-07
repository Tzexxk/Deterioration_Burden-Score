"""Compare explicitly selected fusion fixed-test evaluations."""

import argparse
import json
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.common.paths import Layout, PipelineError
from src.models.test.fusion.comparison import run_comparison


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id", required=True)
    parser.add_argument("--evaluation", action="append", required=True, metavar="FAMILY/MODEL/ID",
                        help="Select an evaluation, repeat for each entry in the comparison")
    args = parser.parse_args(argv)
    try:
        progress = partial(print, file=sys.stderr, flush=True)
        print(json.dumps(run_comparison(Layout(), args.id, evaluation_ids=args.evaluation, progress=progress)))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Fusion comparison cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
