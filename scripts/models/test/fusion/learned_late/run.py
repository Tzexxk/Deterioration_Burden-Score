"""Freeze and evaluate the selected learned_late fusion candidate."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.common.paths import Layout, PipelineError
from src.models.test.fusion.workflow import run_test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    for mode in ("refit", "evaluate"):
        command = commands.add_parser(mode)
        command.add_argument("--config", default="configs/models/test/fusion/learned_late.json")
    args = parser.parse_args(argv)
    try:
        print(f"Fusion fixed test {args.mode}: checking frozen sources", file=sys.stderr, flush=True)
        print(json.dumps(run_test(Layout(), args.config, args.mode, "learned_late")))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Fusion test cancelled", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
