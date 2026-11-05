"""Copy selected files from an older released data run using a configuration list."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.config import DataContext, add_data_arguments
from src.common.paths import Layout, PipelineError
from src.utils.copy_data_artifacts import copy_data_artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_data_arguments(parser)
    parser.add_argument("--config", required=True, help="Project-relative JSON path within configs/utils/")
    args = parser.parse_args()
    try:
        context = DataContext.load(Layout(), args.version, args.run)
        print(json.dumps(copy_data_artifacts(context, args.config), ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Copy cancelled; retry to recover the transaction", file=sys.stderr)
        return 130
    except Exception as error:
        print("COPY_FAILED: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
