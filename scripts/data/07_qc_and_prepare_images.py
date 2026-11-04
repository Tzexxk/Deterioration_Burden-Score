"""Prepare formal-label DICOMs and publish image quality-control results."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError
from src.imaging.prepare_images import prepare_images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--processing", help="Project-relative configuration within the selected data run")
    parser.add_argument("--reviews", help="Project-relative review CSV within the selected data run")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        def local(value, default):
            if value is None:
                return default
            path = context.check_config_ref(context.layout.ref("project", value))
            return path.relative_to(context.layout.path("project", context.directory)).as_posix()
        prepare_images(context, local(args.processing, "imaging/processing.json"), local(args.reviews, None))
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step07", "prepare")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step07", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref, "details": manifest["details"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Image preparation stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"IMAGE_PREPARATION_ERROR: {type(error).__name__}; inspect processing settings, review and source metadata", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
