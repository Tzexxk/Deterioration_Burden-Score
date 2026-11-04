"""Publish formal 72-hour labels from the frozen Step05 audit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cohort.build_labels import build_burden_labels
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        build_burden_labels(context)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step06", "build")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step06", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref,
                          "details": manifest["details"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Label publication stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"LABEL_BUILD_ERROR: {type(error).__name__}; inspect the frozen audit and endpoint decision", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
