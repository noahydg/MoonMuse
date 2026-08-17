#!/usr/bin/env python3
"""Combined formal-training and fixed-source progress for arousal alignment."""

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.show_stage3_factorized_campaign_progress import gpu_status, progress_bar, read_text


def rows(campaign_dir):
    latest = {}
    for line in read_text(Path(campaign_dir) / "run_dirs.tsv").splitlines()[1:]:
        fields = line.split("\t")
        if len(fields) == 4 and fields[0].isdigit():
            latest[int(fields[0])] = fields
    return latest


def seed_progress(fields, max_epochs):
    if not fields:
        return "等待前一个 seed", 0
    run_dir = Path(fields[1])
    if fields[3] == "PASS" or read_text(run_dir / "status.txt").startswith(
        "STAGE3_AROUSAL_ALIGNMENT_FORMAL_PASS"
    ):
        return "通过正式 checkpoint 门禁", max_epochs
    epochs = [int(value) for value in re.findall(
        r"\[epoch\s+(\d+)\]", read_text(run_dir / "logs/training.log")
    )]
    return ("唤醒度对齐正式微调", min(max(epochs), max_epochs)) if epochs \
        else ("准备监督审计/检查点", 0)


def count(root, pattern):
    return sum(path.is_file() for path in Path(root).glob(pattern))


def gate_label(output_root):
    path = Path(output_root) / "gate_c_evaluation.json"
    if not path.is_file():
        return "等待固定源客观门禁"
    try:
        passed = json.loads(path.read_text("utf-8")).get("passed")
    except (OSError, json.JSONDecodeError):
        return "门禁报告不可读"
    return "技术 Gate C 通过，等待盲听" if passed else "技术 Gate C 未通过，结果已保留"


def build_dashboard(campaign_dir, output_root, max_epochs=30):
    campaign_dir = Path(campaign_dir)
    output_root = Path(output_root)
    current = read_text(campaign_dir / "current_stage.txt").strip() or "等待启动"
    table = rows(campaign_dir)
    lines = [
        "MoonMuse｜Stage 3 唤醒度对齐正式训练与 Gate C 看板",
        "更新时间：" + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "现在在做什么：" + current,
        "结论边界：三 seed 训练完成后仍须相同12源生成与质量门禁",
        "GPU：" + gpu_status(),
        "",
        "一、三 seed 匹配正式训练",
    ]
    for seed in (42, 43, 44):
        label, done = seed_progress(table.get(seed), max_epochs)
        lines.append("seed {}：{} {} {}/{} epochs".format(
            seed, label, progress_bar(done, max_epochs), done, max_epochs
        ))
    events = count(output_root, "seed*/*/stage3_performance/*.txt")
    midis = count(output_root, "seed*/*/midi/*.mid")
    groups = max(0, len(read_text(output_root / "status.tsv").splitlines()) - 1)
    lines.extend([
        "",
        "二、固定12源反事实复测",
        "事件文本：{} {}/24".format(progress_bar(events, 24), events),
        "有效 MIDI：{} {}/24".format(progress_bar(midis, 24), midis),
        "生成组：{} {}/6".format(progress_bar(groups, 6), groups),
        "门禁：" + gate_label(output_root),
        "",
        "campaign：{}".format(campaign_dir.resolve()),
        "Gate C：{}".format(output_root.resolve()),
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--watch", type=int, default=0)
    args = parser.parse_args()
    while True:
        if args.watch:
            print("\033[2J\033[H", end="")
        print(build_dashboard(args.campaign_dir, args.output_root, args.max_epochs), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
