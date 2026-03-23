from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
SIGNAL_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_full_factor_experiment_v6_20260320.py"
ACCOUNT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_buy_sell_model_account_v2_20260320.py"
SELL_MODEL_DIR = ROOT / "results" / "b1_sell_habit_experiment_v2_20260320_233121"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 全链路总控：先信号层，再账户层")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--file-limit", type=int, default=0)
    parser.add_argument("--topn-list", type=str, default="")
    parser.add_argument("--result-root", type=Path, default=ROOT / "results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if args.mode == "smoke" else "full"
    signal_dir = args.result_root / f"b1_full_factor_signal_v6_{suffix}_{ts}"
    account_dir = args.result_root / f"b1_buy_sell_model_account_v2_{suffix}_{ts}"
    chain_dir = args.result_root / f"b1_full_chain_v1_{suffix}_{ts}"
    chain_dir.mkdir(parents=True, exist_ok=True)

    write_json(
        chain_dir / "chain_progress.json",
        {
            "stage": "starting",
            "signal_dir": str(signal_dir),
            "account_dir": str(account_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    signal_cmd = [
        sys.executable,
        "-u",
        str(SIGNAL_SCRIPT),
        "--result-dir",
        str(signal_dir),
    ]
    if args.mode == "smoke":
        signal_cmd.extend(["--file-limit", str(args.file_limit or 300)])
    if args.topn_list:
        signal_cmd.extend(["--topn-list", args.topn_list])
    subprocess.run(signal_cmd, check=True)

    signal_progress = signal_dir / "progress.json"
    signal_selected = signal_dir / "final_test_selected_rows.csv"
    signal_summary = signal_dir / "summary.json"
    must_exist(signal_progress)
    must_exist(signal_selected)
    must_exist(signal_summary)
    progress_payload = json.loads(signal_progress.read_text(encoding="utf-8"))
    if progress_payload.get("stage") != "finished":
        raise RuntimeError(f"signal stage not finished: {progress_payload}")

    write_json(
        chain_dir / "chain_progress.json",
        {
            "stage": "signal_finished",
            "signal_dir": str(signal_dir),
            "account_dir": str(account_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    account_cmd = [
        sys.executable,
        "-u",
        str(ACCOUNT_SCRIPT),
        "--buy-signal-dir",
        str(signal_dir),
        "--sell-model-dir",
        str(SELL_MODEL_DIR),
        "--result-dir",
        str(account_dir),
    ]
    subprocess.run(account_cmd, check=True)

    account_progress = account_dir / "progress.json"
    account_summary = account_dir / "summary.json"
    account_results = account_dir / "account_results.csv"
    must_exist(account_progress)
    must_exist(account_summary)
    must_exist(account_results)
    account_payload = json.loads(account_progress.read_text(encoding="utf-8"))
    if account_payload.get("stage") != "finished":
        raise RuntimeError(f"account stage not finished: {account_payload}")

    write_json(
        chain_dir / "chain_progress.json",
        {
            "stage": "finished",
            "signal_dir": str(signal_dir),
            "account_dir": str(account_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
    )

    write_json(
        chain_dir / "summary.json",
        {
            "signal_dir": str(signal_dir),
            "account_dir": str(account_dir),
            "signal_summary": str(signal_summary),
            "account_summary": str(account_summary),
        },
    )


if __name__ == "__main__":
    main()
