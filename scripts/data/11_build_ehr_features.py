"""Audit cached laboratory/static features or publish a checked EHR feature table."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError
from src.ehr.build_features import run_step11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--mode", choices=("audit", "build"), required=True)
    parser.add_argument("--feature-config", help="Project-relative current-run configuration; audit only")
    args = parser.parse_args()
    if args.feature_config and args.mode != "audit":
        parser.error("--feature-config applies only to audit; build uses its snapshot")
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        config_path = "ehr/features.json"
        if args.feature_config:
            target = context.check_config_ref(context.layout.ref("project", args.feature_config))
            config_path = target.relative_to(context.layout.path("project", context.directory)).as_posix()
        status = run_step11(context, args.mode, config_path)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step11", args.mode)
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete" if status == 0 else "blocked", "step": "step11", "mode": args.mode,
                          "data_run": context.key, "generation": manifest["generation"], "manifest": ref, "details": manifest["details"]}, ensure_ascii=False))
        return status
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: EHR feature processing stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"EHR_FEATURE_ERROR: {type(error).__name__}; inspect source diagnostics and feature configuration", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
