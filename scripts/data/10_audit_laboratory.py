"""Audit laboratory evidence, apply explicit review, or freeze the reviewed dictionary."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError
from src.ehr.build_ehr_features import run_step10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--mode", choices=("audit", "review", "freeze"), required=True)
    parser.add_argument("--lab-config", help="Project-relative configuration within the selected data run; audit only")
    parser.add_argument("--reviews", nargs="+", help="Completed current-run review CSVs; review only")
    args = parser.parse_args()
    if args.lab_config and args.mode != "audit":
        parser.error("--lab-config applies only to audit; later modes use its configuration snapshot")
    if args.reviews and args.mode != "review":
        parser.error("--reviews applies only to review mode")
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        def local(path):
            target = context.check_config_ref(context.layout.ref("project", path))
            return target.relative_to(context.layout.path("project", context.directory)).as_posix()
        run_step10(context, args.mode, local(args.lab_config) if args.lab_config else "ehr/lab_processing.json",
                   [local(path) for path in args.reviews] if args.reviews else None)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step10", args.mode)
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step10", "mode": args.mode, "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref, "details": manifest["details"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Laboratory processing stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"LABORATORY_ERROR: {type(error).__name__}; inspect configuration, review and source diagnostics", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
