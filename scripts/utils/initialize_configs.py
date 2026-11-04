"""Initialize local configuration directories and files from public templates."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.paths import PipelineError
from src.utils.initialize_configs import initialize_configs


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        created = initialize_configs()
        print(json.dumps({"status": "initialized", "created": created}, ensure_ascii=False))
        return 0
    except PipelineError as error:
        print(str(error), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("INTERRUPTED: Initialization cancelled; owned outputs rolled back", file=sys.stderr)
        return 130
    except Exception as error:
        print("INITIALIZATION_FAILED: " + type(error).__name__, file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
