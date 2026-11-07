"""Refit and evaluate a ResNet50 candidate on the fixed test split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.test.image.resnet50.workflow import run_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("refit", "evaluate"))
    parser.add_argument("--config", default="configs/models/test/image/resnet50.json",
                        help="Project-relative fixed-test configuration")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_test(Layout(), args.config, args.mode)))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Image fixed-test operation cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
