"""Create a generation-bound patient-level train/validation/test split."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from src.cohort.create_patient_split import SPLIT_METHODS, create_patient_split
from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--split-config",help="Project-relative split configuration within the selected data run")
    parser.add_argument("--split-method", choices=SPLIT_METHODS, default="holdout",
                        help="Patient allocation method (default: holdout); must match split config")
    args=parser.parse_args()
    try:
        context=DataContext.load(Layout(),args.version,args.run)
        path="cohort/patient_split.json"
        if args.split_config is not None:
            target=context.check_config_ref(context.layout.ref("project",args.split_config))
            path=target.relative_to(context.layout.path("project",context.directory)).as_posix()
        create_patient_split(context,path,split_method=args.split_method)
        store=ManifestStore(context.layout,data=context)
        ref=store.manifest_ref("step09","split")
        manifest=store.validate(ref)
        print(json.dumps({"status":"complete","step":"step09","data_run":context.key,
                          "generation":manifest["generation"],"manifest":ref,"details":manifest["details"]},ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error),file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Patient splitting stopped; rerun the same command to recover",file=sys.stderr)
        return 130
    except Exception as error:
        print(f"PATIENT_SPLIT_ERROR: {type(error).__name__}; inspect frozen cohort, split policy and source isolation",file=sys.stderr)
        return 2


if __name__=="__main__":
    raise SystemExit(main())
