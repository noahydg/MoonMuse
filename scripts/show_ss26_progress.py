#!/usr/bin/env python3
"""Read-only SS26 training progress, displayed inside authenticated Jupyter."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from html import escape
import json
from pathlib import Path
import re
from statistics import median
import time

CST = timezone(timedelta(hours=8))


def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        # The existing runner writes JSON directly; a read can overlap its write.
        return default


def tail(path, size=65536):
    try:
        with path.open('rb') as handle:
            handle.seek(max(0, path.stat().st_size - size))
            return handle.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def key(run):
    return (run.get('seed'), run.get('phase'), run.get('branch'))


def epochs(run_dir):
    values = re.findall(r'^ep(\d+)\s*\|', tail(run_dir / 'checkpoints/valloss.txt'), re.M)
    return max(map(int, values), default=0)


def snapshot(root, evidence, now=None):
    root, evidence = Path(root), Path(evidence)
    now = time.time() if now is None else now
    planned = read_json(root / 'planned_commands.json', [])
    completed = read_json(root / 'completed.json', [])
    completed = {key(r): r for r in completed if r.get('status') == 'PASS'}
    current = read_json(root / 'current.json', {})
    protocol = read_json(root / 'protocol.json', {})
    maximum = protocol.get('max_epochs_per_phase', 30)
    total = len(planned)
    done = len(completed)
    state = 'RUNNING' if current else 'WAITING'
    run_dir = Path(current['run_dir']) if current.get('run_dir') else None
    progress = float(epochs(run_dir)) if run_dir else 0.0
    epoch = int(progress)
    if run_dir:
        batches = re.findall(r'-- epoch\s+(\d+)\s*\|\s*batch\s+(\d+)/(\d+)', tail(run_dir / 'training.log'))
        if batches:
            ep, batch, count = map(int, batches[-1])
            epoch = max(epoch, ep)
            progress = max(progress, ep - 1 + min(batch / max(count, 1), .99))
    progress = min(maximum, progress)
    active = key(current) not in completed and run_dir is not None
    fraction = min(1, (done + (progress / maximum if active else 0)) / total) if total else 0
    rates = []
    for r in completed.values():
        n = epochs(Path(r['run_dir']))
        if n:
            rates.append((r['phase'], r['branch'], r['elapsed_seconds'] / n))
    elapsed = max(0, now - current.get('started_unix', now))
    if active and progress >= 2:
        rates.append((current.get('phase'), current.get('branch'), elapsed / progress))

    def rate(command):
        matched = [s for p, b, s in rates if p == command['phase'] and b == command['branch']]
        same_phase = [s for p, b, s in rates if p == command['phase']]
        return median(matched or same_phase or [s for p, b, s in rates])

    remaining = None
    if rates:
        remaining = sum(rate(r) * (maximum - progress if active and key(r) == key(current) else maximum)
                        for r in planned if key(r) not in completed)
    rc_text = tail(evidence / 'campaign.exitcode').strip()
    rc = int(rc_text) if re.fullmatch(r'-?\d+', rc_text) else None
    trained = read_json(root / 'status.json', {}).get('status') == 'PASS'
    backup = read_json(evidence / 'backup.json', {})
    if trained:
        state, fraction, remaining = 'BACKUP', 1.0, None
        if rc == 0 and backup.get('sha256'):
            state = 'COMPLETE'
    if rc is not None and rc != 0:
        state, remaining = 'FAILED', None
    elif run_dir and read_json(run_dir / 'status.json', {}).get('status') == 'FAILED':
        state, remaining = 'FAILED', None
    elif state == 'RUNNING' and run_dir:
        log = run_dir / 'training.log'
        last = log.stat().st_mtime if log.exists() else current.get('started_unix', now)
        if now - last > 180:
            state, remaining = 'STALE', None
    return dict(state=state, updated_unix=now, total=total, done=done,
                fraction=fraction, epoch=min(epoch, maximum), max_epochs=maximum,
                seed=current.get('seed'), phase=current.get('phase'), branch=current.get('branch'),
                remaining_seconds=remaining, exitcode=rc)


def clock(ts):
    return datetime.fromtimestamp(ts, CST).strftime('%m月%d日 %H:%M')


def duration(seconds):
    minutes = max(1, round(seconds / 60))
    hours, minutes = divmod(minutes, 60)
    return (f'{hours} 小时 ' if hours else '') + (f'{minutes} 分钟' if minutes else '')


def render(s):
    state = s['state']
    titles = {'RUNNING': '训练进行中', 'WAITING': '正在读取进度', 'STALE': '日志暂未更新',
              'BACKUP': '训练完成，正在备份', 'COMPLETE': '训练与备份已完成', 'FAILED': '训练已停止，请检查日志'}
    color = '#c45c30' if state in ('FAILED', 'STALE') else '#287d69'
    percentage = round(s['fraction'] * 100, 1)
    current = f"第 {min(s['done'] + 1, s['total'])}/{s['total']} 阶段 · 第 {s['epoch']}/{s['max_epochs']} 轮"
    branch = {'Aplus': '有 alignment', 'A0': '无 alignment'}.get(s['branch'], '')
    details = f"seed {s['seed']} · {branch} · 训练阶段 {s['phase']}"
    eta = '正在积累运行速度，稍后显示预计时间'
    if s['remaining_seconds'] is not None:
        eta = f"预计训练结束：{clock(s['updated_unix'] + s['remaining_seconds'])} 左右<br><span style='font-size:16px;color:#58645f'>约剩 {duration(s['remaining_seconds'])}</span>"
    if state == 'BACKUP':
        current, details, eta = f"{s['done']}/{s['total']} 阶段已完成", '', '备份完成后自动关机'
    elif state == 'COMPLETE':
        current, details, eta = f"{s['done']}/{s['total']} 阶段已完成", '', '结果已备份，随后自动关机'
    elif state == 'FAILED':
        eta = f"退出码：{s['exitcode']}；保留当前结果与日志，不视为完成"
    elif state == 'STALE':
        eta = '超过 3 分钟未见新日志，暂不估算结束时间'
    updated = datetime.fromtimestamp(s['updated_unix'], CST).strftime('%H:%M:%S')
    return f"""<div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;max-width:760px;margin:24px auto;padding:34px;border:1px solid #e0e7e1;border-radius:18px;background:#f9fcfa;color:#20352b;box-sizing:border-box">
    <div style="font-size:13px;letter-spacing:1px;color:#667d70">MOONMUSE · SS26</div>
    <div style="display:flex;justify-content:space-between;align-items:center;margin:16px 0">
    <strong style="font-size:25px">{titles[state]}</strong><strong style="font-size:30px;color:{color}">{percentage:g}%</strong></div>
    <div role="progressbar" aria-label="整体训练进度" aria-valuenow="{percentage}" aria-valuemin="0" aria-valuemax="100" style="height:14px;background:#e2ebe6;border-radius:20px;overflow:hidden"><div style="width:{percentage}%;height:100%;background:{color}"></div></div>
    <div style="font-size:19px;margin-top:20px">{escape(current)}</div>
    <div style="font-size:14px;color:#667d70;margin-top:6px">{escape(details)}</div>
    <div style="font-size:21px;margin-top:28px;line-height:1.7">{eta}</div>
    <div style="font-size:13px;color:#667d70;margin-top:20px;line-height:1.7">按各阶段最多 {s['max_epochs']} 轮估算；提前停止时会更早完成。<br>更新时间 {updated}（北京时间） · 每 5 秒刷新</div>
    </div>"""


def show_notebook(root, evidence, interval=5):
    from IPython.display import HTML, display
    handle = display(HTML('<p>正在读取训练日志…</p>'), display_id=True)
    while True:
        s = snapshot(root, evidence)
        handle.update(HTML(render(s)))
        # A separate progress snapshot; no training files or configurations changed.
        target = Path(evidence) / 'live-progress.json'
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(s, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(target)
        if s['state'] in ('COMPLETE', 'FAILED'):
            break
        time.sleep(interval)
