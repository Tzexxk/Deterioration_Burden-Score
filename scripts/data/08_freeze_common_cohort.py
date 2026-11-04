"""Freeze the common cohort from formal labels and reviewed image QC."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from src.cohort.freeze_cohort import freeze_common_cohort
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    args=parser.parse_args()
    try:
        context=DataContext.load(Layout(),args.version,args.run)
        freeze_common_cohort(context)
        store=ManifestStore(context.layout,data=context)
        ref=store.manifest_ref("step08","freeze")
        manifest=store.validate(ref)
        print(json.dumps({"status":"complete","step":"step08","data_run":context.key,
                          "generation":manifest["generation"],"manifest":ref,"details":manifest["details"]},ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Cohort freezing stopped; rerun the same command to recover",file=sys.stderr)
        return 130
    except Exception as error:
        print(f"COHORT_FREEZE_ERROR: {type(error).__name__}; inspect formal labels, evidence and image QC",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
