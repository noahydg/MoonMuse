import importlib.util
import json
import os
from pathlib import Path

spec = importlib.util.spec_from_file_location('progress', Path(__file__).parents[1] / 'scripts/show_ss26_progress.py')
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def setup(tmp_path):
    root, evidence = tmp_path / 'training', tmp_path / 'evidence'
    root.mkdir(); evidence.mkdir()
    runs = [dict(seed=42, phase=1, branch=b, run_dir=str(root / b)) for b in ('Aplus', 'A0')]
    for r in runs:
        d = Path(r['run_dir']); (d / 'checkpoints').mkdir(parents=True)
    (root / 'planned_commands.json').write_text(json.dumps(runs))
    (root / 'protocol.json').write_text(json.dumps({'max_epochs_per_phase': 30}))
    (root / 'current.json').write_text(json.dumps({**runs[0], 'status': 'RUNNING', 'started_unix': 1000}))
    return root, evidence, runs


def test_live_progress_and_maximum_epoch_eta(tmp_path):
    root, evidence, runs = setup(tmp_path)
    d = Path(runs[0]['run_dir'])
    (d / 'checkpoints/valloss.txt').write_text('ep001 | valloss: 0.5\nep002 | valloss: 0.4\n')
    (d / 'training.log').write_text(' -- epoch 003 | batch 123/246: len: tensor')
    os.utime(d / 'training.log', (1060, 1060))
    s = p.snapshot(root, evidence, now=1060)
    assert s['state'] == 'RUNNING'
    assert s['epoch'] == 3 and s['fraction'] == 2.5 / 60
    assert s['remaining_seconds'] == (60 - 2.5) * 24
    assert '预计训练结束' in p.render(s)


def test_completed_current_not_double_counted(tmp_path):
    root, evidence, runs = setup(tmp_path)
    (root / 'completed.json').write_text(json.dumps([{**runs[0], 'status': 'PASS', 'elapsed_seconds': 120}]))
    s = p.snapshot(root, evidence, now=1060)
    assert s['fraction'] == .5 and s['done'] == 1


def test_failure_overrides_running_and_complete_waits_for_backup(tmp_path):
    root, evidence, runs = setup(tmp_path)
    (evidence / 'campaign.exitcode').write_text('124\n')
    assert p.snapshot(root, evidence)['state'] == 'FAILED'
    (evidence / 'campaign.exitcode').write_text('0\n')
    (root / 'status.json').write_text('{"status":"PASS"}')
    assert p.snapshot(root, evidence)['state'] == 'BACKUP'
    (evidence / 'backup.json').write_text('{"sha256":"abc"}')
    assert p.snapshot(root, evidence)['state'] == 'COMPLETE'


def test_stale_and_partial_json_do_not_claim_live_eta(tmp_path):
    root, evidence, runs = setup(tmp_path)
    assert p.snapshot(root, evidence, now=1300)['state'] == 'STALE'
    (root / 'current.json').write_text('{"status":')
    s = p.snapshot(root, evidence, now=1300)
    assert s['state'] == 'WAITING' and s['remaining_seconds'] is None
