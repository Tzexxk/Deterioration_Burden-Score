"""Match selected CXR studies to admission intervals for the current data run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cohort.match_admissions import match_admissions
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--cohort", help="Project-relative admission-matching configuration within the selected data run")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        cohort = "cohort/admission_matching.json"
        if args.cohort is not None:
            path = context.check_config_ref(context.layout.ref("project", args.cohort))
            cohort = path.relative_to(context.layout.path("project", context.directory)).as_posix()
        match_admissions(context, cohort)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step04", "match")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step04", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref,
                          "counts": manifest["details"]["counts"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Admission matching stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print("ADMISSION_MATCH_ERROR: " + type(error).__name__ + "; inspect admission keys, intervals and selection schema", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
