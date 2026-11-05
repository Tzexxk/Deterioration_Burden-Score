"""Export the checked training directory and its portable ZIP."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError
from src.data.export_training_package import run_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--export-config", help="Project-relative configuration inside the selected data run")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        config_path = "export/training_package.json"
        if args.export_config:
            target = context.check_config_ref(context.layout.ref("project", args.export_config))
            config_path = target.relative_to(context.layout.path("project", context.directory)).as_posix()
        run_export(context, config_path)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step12", "export")
        manifest = store.validate(ref)
        print(json.dumps({"status": "complete", "step": "step12", "data_run": context.key,
                          "generation": manifest["generation"], "manifest": ref, "details": manifest["details"]}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Export stopped; rerun the same command to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"TRAINING_EXPORT_ERROR: {type(error).__name__}; inspect frozen inputs, pixels and package configuration", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
