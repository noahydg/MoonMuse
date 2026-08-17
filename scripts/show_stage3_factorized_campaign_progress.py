#!/usr/bin/env python3
"""Plain-language progress view for factorized-VA Stage 3 formal fine-tuning."""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def progress_bar(done: int, total: int, width: int = 24) -> str:
    done = min(max(done, 0), total)
    filled = int(width * done / max(total, 1))
    return "█" * filled + "░" * (width - filled)


def gpu_status() -> str:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "GPU 状态不可读"
    values = [part.strip() for part in result.stdout.splitlines()[0].split(",")] \
        if result.stdout.strip() else []
    return "利用率 {}%｜显存 {}/{} MiB".format(*values) if len(values) == 3 \
        else "GPU 状态不可读"


def parse_rows(campaign_dir: Path) -> list[dict]:
    latest = {}
    for line in read_text(campaign_dir / "run_dirs.tsv").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 4 and fields[0].isdigit():
            latest[int(fields[0])] = {
                "seed": int(fields[0]),
                "run_dir": Path(fields[1]),
                "source": fields[2],
                "state": fields[3],
            }
    return [latest.get(seed, {"seed": seed, "state": "WAITING"}) for seed in (42, 43, 44)]


def seed_progress(row: dict, max_epochs: int) -> tuple[str, int, str]:
    run_dir = row.get("run_dir")
    if row.get("state") == "PASS":
        return "通过 checkpoint 门禁", max_epochs, "done"
    if not run_dir:
        return "等待前一个 seed", 0, "waiting"
    status = read_text(run_dir / "status.txt")
    if status.startswith("STAGE3_FACTORIZED_FORMAL_PASS"):
        return "通过 checkpoint 门禁", max_epochs, "done"
    log = read_text(run_dir / "logs/training.log")
    epochs = [int(value) for value in re.findall(r"\[epoch\s+(\d+)\]", log)]
    if epochs:
        return "正式微调", min(max(epochs), max_epochs), "active"
    if (run_dir / "migrated_checkpoint_integrity.json").is_file():
        return "checkpoint 已迁移，准备训练", 0, "active"
    return "准备数据和迁移 checkpoint", 0, "active"


def build_dashboard(campaign_dir: Path, max_epochs: int = 30) -> str:
    current = read_text(campaign_dir / "current_stage.txt").strip() or "等待启动"
    final_status = read_text(campaign_dir / "status.txt").strip()
    lines = [
        "MoonMuse｜Stage 3 因子化 VA 正式微调看板",
        "更新时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "现在在做什么：" + current,
        "结论边界：正式微调仍不是音乐质量结论；之后必须跑固定同源反事实门禁",
        "GPU：" + gpu_status(),
        "",
    ]
    for row in parse_rows(campaign_dir):
        label, done, state = seed_progress(row, max_epochs)
        symbol = {"done": "✓", "active": "▶", "waiting": "○"}[state]
        lines.append(
            f"{symbol} seed {row['seed']}：{label} "
            f"{progress_bar(done, max_epochs)} {done}/{max_epochs} epochs"
        )
    lines.extend(["", f"campaign：{campaign_dir.resolve()}"])
    if final_status:
        lines.append("状态标记：" + final_status)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    if args.max_epochs < 1 or args.watch < 0:
        parser.error("epoch/watch values must be non-negative and epochs positive")
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(build_dashboard(args.campaign_dir, args.max_epochs), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
