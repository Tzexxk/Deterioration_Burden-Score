"""Train Swin-B on released development images."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.train.image.swin_b.training import run_training


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    cv = commands.add_parser("cv", help="Fit each selected ID on the five development folds")
    cv.add_argument("--id", help="Experiment ID from the configuration")
    cv.add_argument("--config", default="configs/models/train/image/swin_b.json",
                       help="Project-relative image training configuration")
    args = parser.parse_args(argv)
    try:
        result = run_training(Layout(), args.config, args.id)
        print(json.dumps(result))
        return 2 if result["status"] == "failed" else 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Image training cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
