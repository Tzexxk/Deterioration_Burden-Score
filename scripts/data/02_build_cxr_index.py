"""Build a local CXR index using the current data run and raw-audit manifest."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cohort.build_cxr_index import build_cxr_index
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--imaging", help="Project-relative imaging configuration within the selected data run")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        imaging = "imaging/raw_audit.json"
        if args.imaging is not None:
            path = context.check_config_ref(context.layout.ref("project", args.imaging))
            imaging = path.relative_to(context.layout.path("project", context.directory)).as_posix()
        build_cxr_index(context, imaging)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step02", "build")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step02", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref,
                          "counts": manifest["details"]["counts"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Indexing stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print("INDEX_ERROR: " + type(error).__name__ + "; inspect index keys, schema and configured paths", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
