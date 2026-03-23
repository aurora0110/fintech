from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
SIGNAL_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_cluster_subtype_experiment_v1_20260321.py"
ACCOUNT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_buy_sell_model_account_v2_20260320.py"
SELL_MODEL_DIR = ROOT / "results" / "b1_sell_habit_experiment_v2_20260320_233121"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def must_exist(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(str(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="B1 分簇子类型总控链")
    parser.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--file-limit", type=int, default=300)
    parser.add_argument("--topn-list", type=str, default="3,5,8,10")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = "smoke" if args.mode == "smoke" else "full"
    signal_dir = ROOT / "results" / f"b1_cluster_subtype_signal_v1_{suffix}_{ts}"
    account_dir = ROOT / "results" / f"b1_cluster_subtype_account_v1_{suffix}_{ts}"
    chain_dir = ROOT / "results" / f"b1_cluster_subtype_chain_v1_{suffix}_{ts}"
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
        "--topn-list",
        args.topn_list,
    ]
    if args.mode == "smoke":
        signal_cmd.extend(["--file-limit", str(args.file_limit)])
    subprocess.run(signal_cmd, check=True)

    signal_progress = signal_dir / "progress.json"
    signal_summary = signal_dir / "summary.json"
    signal_selected = signal_dir / "final_test_selected_rows.csv"
    must_exist(signal_progress)
    must_exist(signal_summary)
    must_exist(signal_selected)
    signal_payload = json.loads(signal_progress.read_text(encoding="utf-8"))
    if signal_payload.get("stage") != "finished":
        raise RuntimeError(f"signal stage not finished: {signal_payload}")

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
