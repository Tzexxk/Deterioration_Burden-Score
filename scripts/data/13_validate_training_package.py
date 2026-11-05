"""Independently reconstruct, validate and freeze the exported training data run."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.manifest import ManifestStore
from src.common.paths import Layout, PipelineError
from src.data.validate_training_package import run_validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--validation-config", help="Project-relative configuration inside the selected data run")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        config_path = "export/validation.json"
        if args.validation_config:
            path = context.check_config_ref(context.layout.ref("project", args.validation_config))
            config_path = path.relative_to(context.layout.path("project", context.directory)).as_posix()
        status = run_validation(context, config_path)
        store = ManifestStore(context.layout, data=context)
        ref = store.manifest_ref("step13", "validate")
        manifest = store.validate(ref)
        print(json.dumps({"status":"accepted" if status == 0 else "rejected", "step":"step13", "data_run":context.key,
                          "generation":manifest["generation"], "manifest":ref, "details":manifest["details"],
                          "data_run_released":(store.control / "release.json").is_file()}, ensure_ascii=False))
        return status
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Acceptance stopped; rerun the same command before release to recover", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"TRAINING_ACCEPTANCE_ERROR: {type(error).__name__}; inspect current-run acceptance diagnostics", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
