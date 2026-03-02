from __future__ import annotations

import argparse
import json
import traceback

from config import load_config
from pipeline import run_full_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TMS-Alpha multi-factor scoring + contribution analysis pipeline"
    )
    p.add_argument(
        "--config",
        required=True,
        help="Path to JSON config file",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    try:
        cfg = load_config(args.config)
        result = run_full_experiment(cfg)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print("\nOutput files:")
        for k, v in result["files"].items():
            if v:
                print(f"- {k}: {v}")
        return 0
    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

