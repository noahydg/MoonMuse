#!/usr/bin/env python3
"""Plain-language dashboard for the Stage 3 arousal-alignment smoke."""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.show_stage3_factorized_progress import gpu_status, progress_bar


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def read_json(path):
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_progress(log_text):
    steps = [int(value) for value in re.findall(r"\bstep = (\d+)", log_text)]
    alignment = [float(value) for value in re.findall(
        r"alignment_loss = ([-+0-9.eE]+)", log_text
    )]
    return max(steps, default=0), alignment[-1] if alignment else None


def build_dashboard(run_root, expected_batches=100):
    run_root = Path(run_root)
    log_text = read_text(run_root / "logs/training.log")
    batches, latest_alignment = parse_progress(log_text)
    batches = min(batches, expected_batches)
    updates = (batches + 1) // 2
    current_stage = read_text(run_root / "current_stage.txt").strip() or "等待开始"
    status = read_text(run_root / "status.txt").strip()
    audit = read_json(run_root / "arousal_supervision_audit.json")
    validation = read_json(run_root / "smoke_validation.json")

    if status.startswith("STAGE3_AROUSAL_ALIGNMENT_SMOKE_PASS") and validation.get("passed"):
        conclusion = "smoke 已通过：只证明新损失能训练，尚未证明音乐更好"
    elif batches:
        conclusion = "正在训练新目标：交叉熵 + 固定源反事实唤醒度对齐"
    elif audit:
        conclusion = "数据监督审计已完成，正在准备检查点/训练"
    else:
        conclusion = "正在执行训练前监督审计"

    lines = [
        "MoonMuse｜Stage 3 唤醒度对齐 smoke 看板",
        "更新时间：{}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        "",
        "现在在做什么：{}".format(current_stage),
        "证据边界：{}".format(conclusion),
        "GPU：{}".format(gpu_status()),
        "",
        "1. 监督审计：{}".format(
            "通过" if audit.get("recommended_intervention") == "explicit_counterfactual_arousal_alignment" else "未通过/未执行"
        ),
        "2. smoke 训练：{} {} / {} batches".format(
            progress_bar(batches, expected_batches), batches, expected_batches
        ),
        "   参数更新：{} / {}（梯度累计=2）".format(
            updates, (expected_batches + 1) // 2
        ),
        "   最近对齐损失：{}".format(
            "尚无" if latest_alignment is None else "{:.4f}".format(latest_alignment)
        ),
        "3. 最终 checkpoint 与 smoke 门禁：{}".format(
            "通过" if validation.get("passed") else "未通过/未执行"
        ),
        "",
        "运行目录：{}".format(run_root.resolve()),
    ]
    if status:
        lines.append("状态标记：{}".format(status))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root")
    parser.add_argument("--expected-batches", type=int, default=100)
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(build_dashboard(args.run_root, args.expected_batches), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
