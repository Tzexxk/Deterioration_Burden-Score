"""Audit local raw data using the selected data-run configuration."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cohort.audit_raw_data import run_audit
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def selected_config(context, argument, default):
    if argument is None:
        return default
    ref = context.layout.ref("project", argument)
    path = context.check_config_ref(ref)
    base = context.layout.path("project", context.directory)
    return path.relative_to(base).as_posix()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--imaging", help="Optional project-relative config inside the selected data run")
    parser.add_argument("--ehr", help="Optional project-relative config inside the selected data run")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        run_audit(context, selected_config(context, args.imaging, "imaging/raw_audit.json"),
                  selected_config(context, args.ehr, "ehr/raw_audit.json"))
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step01", "audit", area="outputs")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step01", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref,
                          "counts": manifest["details"]["counts"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Audit stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print("AUDIT_ERROR: " + type(error).__name__ + "; inspect configuration and input schema", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
