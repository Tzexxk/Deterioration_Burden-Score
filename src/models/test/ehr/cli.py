"""Command-line entry for fixed-test EHR experiments."""

import argparse
import json
import sys

from src.common.paths import Layout, PipelineError
from src.models.test.ehr.workflow import run_test


def main(model, fit, predict, argv=None):
    parser = argparse.ArgumentParser(description=f"Refit and evaluate an EHR {model} candidate")
    parser.add_argument("mode", choices=("refit", "evaluate"))
    parser.add_argument("--config", default=f"configs/models/test/ehr/{model}.json",
                        help="Project-relative fixed-test configuration")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_test(Layout(), args.config, model, args.mode, fit, predict)))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: EHR fixed-test operation cancelled", file=sys.stderr)
        return 130
