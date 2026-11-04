"""Audit 72-hour clinical evidence or freeze an explicitly reviewed audit."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cohort.audit_burden_labels import audit_burden_labels
from src.cohort.freeze_endpoint import freeze_endpoint
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--mode", choices=("audit", "freeze"), required=True)
    parser.add_argument("--review-policy", choices=("lenient", "strict"))
    parser.add_argument("--cohort", help="Project-relative rules configuration within the selected data run")
    parser.add_argument("--ehr", help="Project-relative clinical-source configuration within the selected data run")
    parser.add_argument("--imaging", help="Project-relative imaging configuration within the selected data run")
    parser.add_argument("--decision-review", help="Project-relative endpoint review within the selected data run; freeze only")
    args = parser.parse_args()
    if args.mode == "audit" and args.decision_review is not None:
        parser.error("--decision-review is available only in freeze mode")
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        def local(value, default):
            if value is None:
                return default
            path = context.check_config_ref(context.layout.ref("project", value))
            return path.relative_to(context.layout.path("project", context.directory)).as_posix()
        cohort = local(args.cohort, "cohort/burden_rules.json")
        if args.mode == "audit":
            audit_burden_labels(context, cohort,
                                local(args.ehr, "ehr/burden_sources.json"),
                                local(args.imaging, "imaging/raw_audit.json"), args.review_policy)
        else:
            if args.ehr or args.imaging:
                parser.error("freeze uses source configurations declared by the selected audit")
            freeze_endpoint(context, cohort,
                            local(args.decision_review, "cohort/reviews/endpoint_review.json"), args.review_policy)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step05", args.mode)
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step05", "mode": args.mode,
                          "data_run": context.key, "generation": manifest["generation"], "manifest": ref,
                          "review_policy": manifest["details"]["review_policy"], "details": manifest["details"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Clinical processing stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print("CLINICAL_AUDIT_ERROR: " + type(error).__name__ + "; inspect clinical tables, configuration and evidence reviews", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
