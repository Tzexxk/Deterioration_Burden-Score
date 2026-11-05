"""Generate, independently validate and freeze variants of a released image master."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.paths import Layout, PipelineError
from src.utils.build_image_variants import run_variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--config", default="configs/utils/image_variants.json",
                        help="Project-relative JSON path within configs/utils/")
    parser.add_argument("--mode", choices=("build", "validate"), default="build",
                        help="Build and freeze a new batch or revalidate an existing batch")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        print(json.dumps(run_variants(context, args.config, validate_only=args.mode == "validate")))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Image variant operation cancelled", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"IMAGE_VARIANT_ERROR: {type(error).__name__}; check NPY inputs and storage", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
