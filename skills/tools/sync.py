#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path
from typing import Iterable


# 项目内主版本 skill，只在这里维护。
PROJECT_SKILLS_DIR = Path("/Users/lidongyang/Desktop/Qstrategy/skills")
# Codex 实际读取的运行副本。
CODEX_SKILLS_DIR = Path("/Users/lidongyang/.codex/skills")
# 这些目录不是 skill 目录，不参与同步。
IGNORED_PROJECT_DIRS = {"__pycache__", "daily-stock-selection", "quant-backtest-research", "tools"}
LEGACY_MANAGED_NAMES = {
    "qstrategy-allocation-equal-topn",
    "qstrategy-allocation-fixed-small-position",
    "qstrategy-allocation-score-weighted",
    "qstrategy-b1-entry",
    "qstrategy-b1-template-fusion-entry",
    "qstrategy-b2-entry",
    "qstrategy-b3-entry",
    "qstrategy-backtest-conventions",
    "qstrategy-brick-entry",
    "qstrategy-brick-formal-best-entry",
    "qstrategy-brick-formal-exit",
    "qstrategy-brick-minute-feature-exit",
    "qstrategy-brick-relaxed-fusion-entry",
    "qstrategy-equal-weight-topn-allocation",
    "qstrategy-exit-fixed-tp-sl",
    "qstrategy-exit-minute-feature",
    "qstrategy-exit-model-plus-tp",
    "qstrategy-exit-partial-tp",
    "qstrategy-pin-entry",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_skill_files(skill_dir: Path) -> Iterable[Path]:
    for path in sorted(skill_dir.rglob("*")):
        if path.is_file():
            yield path


def skill_tree_hash(skill_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in iter_skill_files(skill_dir):
        relative = path.relative_to(skill_dir).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def collect_project_skill_dirs() -> list[Path]:
    dirs = []
    for path in sorted(PROJECT_SKILLS_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith(".") or path.name in IGNORED_PROJECT_DIRS:
            continue
        if not (path / "SKILL.md").exists():
            continue
        dirs.append(path)
    return dirs


def collect_codex_skill_dirs(managed_names: set[str]) -> list[Path]:
    dirs = []
    if not CODEX_SKILLS_DIR.exists():
        return dirs
    for path in sorted(CODEX_SKILLS_DIR.iterdir()):
        if not path.is_dir():
            continue
        if path.name.startswith("."):
            continue
        if path.name not in managed_names:
            continue
        dirs.append(path)
    return dirs


def sync_one(skill_dir: Path, dry_run: bool) -> dict[str, str]:
    target_dir = CODEX_SKILLS_DIR / skill_dir.name
    action = "unchanged"

    if not target_dir.exists():
        action = "create"
        if not dry_run:
            shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)
        return {"skill": skill_dir.name, "action": action}

    source_hash = skill_tree_hash(skill_dir)
    target_hash = skill_tree_hash(target_dir)
    if source_hash != target_hash:
        action = "update"
        if not dry_run:
            shutil.copytree(skill_dir, target_dir, dirs_exist_ok=True)
    return {"skill": skill_dir.name, "action": action}


def prune_deleted_skills(project_skill_dirs: list[Path], dry_run: bool) -> list[dict[str, str]]:
    project_names = {path.name for path in project_skill_dirs}
    managed_names = project_names | LEGACY_MANAGED_NAMES
    results: list[dict[str, str]] = []
    for target_dir in collect_codex_skill_dirs(managed_names):
        if target_dir.name in project_names:
            continue
        results.append({"skill": target_dir.name, "action": "delete"})
        if not dry_run:
            shutil.rmtree(target_dir)
    return results


def sync_all(dry_run: bool) -> list[dict[str, str]]:
    skill_dirs = collect_project_skill_dirs()
    if not skill_dirs:
        return []
    results = [sync_one(skill_dir, dry_run=dry_run) for skill_dir in skill_dirs]
    results.extend(prune_deleted_skills(skill_dirs, dry_run=dry_run))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="同步 Qstrategy 项目内主版本 skill 到 ~/.codex/skills")
    parser.add_argument("--dry-run", action="store_true", help="仅显示将要同步的内容，不真正写入")
    args = parser.parse_args()

    skill_dirs = collect_project_skill_dirs()
    if not skill_dirs:
        print("未找到可同步的 Qstrategy skill。")
        return

    print(f"项目主版本目录: {PROJECT_SKILLS_DIR}")
    print(f"Codex 运行目录: {CODEX_SKILLS_DIR}")
    print(f"dry_run={args.dry_run}")

    for result in sync_all(dry_run=args.dry_run):
        print(f"{result['action']}: {result['skill']}")


if __name__ == "__main__":
    main()
