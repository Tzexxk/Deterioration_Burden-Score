"""Select one readable, reliably timed frontal CXR image per study."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.cohort.select_frontal_image import select_frontal_images
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
        select_frontal_images(context, imaging)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step03", "select")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step03", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref,
                          "counts": manifest["details"]["counts"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Selection stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print("SELECTION_ERROR: " + type(error).__name__ + "; inspect index schema, keys and image times", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
