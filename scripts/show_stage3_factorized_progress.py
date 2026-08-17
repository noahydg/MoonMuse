#!/usr/bin/env python3
"""Readable live dashboard for the Stage 3 factorized-VA smoke gate."""

import argparse
import json
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path


def read_text(path):
    try:
        return path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return ''


def load_json(path):
    try:
        value = json.loads(read_text(path))
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def parse_training_step(log_text):
    values = [int(value) for value in re.findall(r'\bstep = (\d+)', log_text)]
    if not values:
        values = [
            int(value)
            for value in re.findall(r'^\s*\d+\s+(\d+)\s+', log_text, re.MULTILINE)
        ]
    return max(values, default=0)


def progress_bar(done, total, width=28):
    done = min(max(done, 0), total)
    filled = int(width * done / max(total, 1))
    return '█' * filled + '░' * (width - filled)


def gpu_status():
    try:
        result = subprocess.run(
            [
                'nvidia-smi',
                '--query-gpu=utilization.gpu,memory.used,memory.total',
                '--format=csv,noheader,nounits',
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return 'GPU 状态不可读'
    values = [part.strip() for part in result.stdout.splitlines()[0].split(',')] \
        if result.stdout.strip() else []
    if len(values) != 3:
        return 'GPU 状态不可读'
    return '利用率 {}%｜显存 {}/{} MiB'.format(*values)


def build_dashboard(run_root, expected_batches=100):
    run_root = Path(run_root)
    data_report = load_json(Path(
        'events/m2p/emopia_m2p_factorized_va_functional/factorized_data_report.json'
    ))
    migration_report = load_json(
        run_root / 'migrated_stage3_finetune_params.pt.migration.json'
    )
    migrated_integrity = load_json(
        run_root / 'migrated_checkpoint_integrity.json'
    )
    smoke_integrity = load_json(run_root / 'smoke_checkpoint_integrity.json')
    status = read_text(run_root / 'status.txt').strip()
    current_stage = read_text(run_root / 'current_stage.txt').strip() or '等待开始'
    training_log = read_text(run_root / 'logs/training.log')
    batches = min(parse_training_step(training_log), expected_batches)
    updates = (batches + 1) // 2

    if status.startswith('STAGE3_FACTORIZED_SMOKE_PASS'):
        overall = '已通过 smoke；下一步是固定源干预训练，不是直接写论文结果'
    elif smoke_integrity.get('passed') is False:
        overall = 'smoke 检查点门禁失败，请查看日志'
    elif batches:
        overall = '正在训练：这里只验证能否正确训练，不判断音乐好不好听'
    elif migration_report:
        overall = '检查点已迁移，等待/进入 100-batch smoke 训练'
    elif data_report:
        overall = '因子化数据已准备，正在迁移检查点'
    else:
        overall = '正在准备因子化数据'

    lines = [
        'MoonMuse｜Stage 3 因子化唤醒度 smoke 看板',
        '更新时间：{}'.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
        '',
        '现在在做什么：{}'.format(current_stage),
        '结论边界：{}'.format(overall),
        'GPU：{}'.format(gpu_status()),
        '',
        '1. 数据转换：{}（{} / 1071）'.format(
            '完成' if data_report.get('converted_files') == 1071 else '未完成',
            data_report.get('converted_files', 0),
        ),
        '2. 检查点迁移：{}；完整性：{}'.format(
            '完成' if migration_report else '未完成',
            '通过' if migrated_integrity.get('passed') else '未通过/未执行',
        ),
        '3. smoke 训练：{} {} / {} batches'.format(
            progress_bar(batches, expected_batches), batches, expected_batches
        ),
        '   梯度累计=2，因此当前约 {} / {} 次参数更新'.format(
            updates, (expected_batches + 1) // 2
        ),
        '4. 最终检查点：{}'.format(
            '通过' if smoke_integrity.get('passed') else '未通过/未执行'
        ),
        '',
        '运行目录：{}'.format(run_root.resolve()),
    ]
    if status:
        lines.append('状态标记：{}'.format(status))
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_root')
    parser.add_argument('--expected-batches', type=int, default=100)
    parser.add_argument('--watch', type=int, default=0,
                        help='refresh interval in seconds; 0 prints once')
    args = parser.parse_args()
    if args.expected_batches < 1:
        parser.error('--expected-batches must be positive')
    if args.watch < 0:
        parser.error('--watch cannot be negative')

    while True:
        if args.watch:
            print('\033[2J\033[H', end='')
        print(build_dashboard(args.run_root, args.expected_batches), flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == '__main__':
    main()
