from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path("/Users/lidongyang/Desktop/Qstrategy")
ACCOUNT_SCRIPT = ROOT / "utils" / "tmp" / "run_b1_buy_sell_model_account_v2_20260320.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="等待 B1 语义化信号层完成后自动启动账户层")
    parser.add_argument("--buy-signal-dir", type=Path, required=True)
    parser.add_argument("--sell-model-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def progress_finished(progress_path: Path) -> bool:
    if not progress_path.exists():
        return False
    try:
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return payload.get("stage") == "finished"


def main() -> None:
    args = parse_args()
    buy_dir = args.buy_signal_dir
    sell_dir = args.sell_model_dir
    progress_path = buy_dir / "progress.json"
    final_rows_path = buy_dir / "final_test_selected_rows.csv"

    print(f"[{datetime.now().isoformat(timespec='seconds')}] 开始等待买点结果完成: {buy_dir}")
    while True:
        if progress_finished(progress_path) and final_rows_path.exists():
            print(f"[{datetime.now().isoformat(timespec='seconds')}] 检测到买点结果完成，启动账户层回测")
            cmd = [
                sys.executable,
                str(ACCOUNT_SCRIPT),
                "--buy-signal-dir",
                str(buy_dir),
                "--sell-model-dir",
                str(sell_dir),
            ]
            subprocess.run(cmd, check=True)
            print(f"[{datetime.now().isoformat(timespec='seconds')}] 账户层回测完成")
            return
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    main()
