"""Fit the base-model splits for nested learned fusion."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.train.fusion.learned_late.training import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cv = commands.add_parser("cv")
    cv.add_argument("--config", default="configs/models/train/fusion/learned_late.json")
    cv.add_argument("--id")
    args = parser.parse_args(argv)
    try:
        result = run_training(Layout(), args.config, args.id)
        print(json.dumps(result))
        return 2 if result["status"] == "failed" else 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Learned fusion cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
