"""Assess fixed score fusion from two completed development assessments."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.assessment.fusion.weighted_late.assessment import run_assessment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/models/assessment/fusion/weighted_late.json")
    parser.add_argument("--id", help="Assess only this configured ID")
    args = parser.parse_args(argv)
    try:
        result = run_assessment(Layout(), args.config, args.id)
        print(json.dumps(result))
        return 2 if result["status"] == "failed" else 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Fusion assessment cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
