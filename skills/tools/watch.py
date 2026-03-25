#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from sync import PROJECT_SKILLS_DIR, collect_project_skill_dirs, skill_tree_hash, sync_all


DEFAULT_STATE_PATH = Path("/Users/lidongyang/Desktop/Qstrategy/results/skill_sync_watcher_state.json")


class SkillWatcher:
    def __init__(self, interval_sec: float, state_path: Path, dry_run: bool) -> None:
        self.interval_sec = interval_sec
        self.state_path = state_path
        self.dry_run = dry_run
        self.running = True
        self.last_snapshot: dict[str, str] = {}

    def install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_stop)
        signal.signal(signal.SIGTERM, self._handle_stop)

    def _handle_stop(self, signum, _frame) -> None:
        self.running = False
        self._write_state("stopping", reason=f"signal_{signum}")

    def _current_snapshot(self) -> dict[str, str]:
        snapshot: dict[str, str] = {}
        for skill_dir in collect_project_skill_dirs():
            snapshot[skill_dir.name] = skill_tree_hash(skill_dir)
        return snapshot

    def _write_state(self, stage: str, **extra: object) -> None:
        payload = {
            "stage": stage,
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "interval_sec": self.interval_sec,
            "dry_run": self.dry_run,
            "skill_count": len(self.last_snapshot),
        }
        payload.update(extra)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> None:
        self.install_signal_handlers()
        self.last_snapshot = self._current_snapshot()
        self._write_state("initial_sync")
        initial_results = sync_all(dry_run=self.dry_run)
        self._write_state("watching", last_actions=initial_results)
        print(f"[skill-sync] watching {PROJECT_SKILLS_DIR} interval={self.interval_sec}s dry_run={self.dry_run}", flush=True)

        while self.running:
            time.sleep(self.interval_sec)
            snapshot = self._current_snapshot()
            if snapshot == self.last_snapshot:
                self._write_state("watching", last_actions=[])
                continue
            self.last_snapshot = snapshot
            self._write_state("syncing")
            results = sync_all(dry_run=self.dry_run)
            print(f"[skill-sync] synced: {results}", flush=True)
            self._write_state("watching", last_actions=results)

        self._write_state("stopped")


def main() -> None:
    parser = argparse.ArgumentParser(description="监听 Qstrategy/skills 变化并自动同步到 ~/.codex/skills")
    parser.add_argument("--interval-sec", type=float, default=2.0, help="轮询间隔，默认 2 秒")
    parser.add_argument("--state-path", type=str, default=str(DEFAULT_STATE_PATH), help="状态文件路径")
    parser.add_argument("--dry-run", action="store_true", help="仅监听并输出，不真正写入 ~/.codex/skills")
    args = parser.parse_args()

    watcher = SkillWatcher(interval_sec=max(args.interval_sec, 0.5), state_path=Path(args.state_path), dry_run=args.dry_run)
    watcher.run()


if __name__ == "__main__":
    sys.exit(main())
