"""Bounded campaign supervision, final evidence backup, and unconditional shutdown."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback


class Supervisor:
    def __init__(self, out, backup, repo, total_seconds=12600, reserve_seconds=900,
                 shutdown_command=None, total_phases=36):
        if not 0 < reserve_seconds < total_seconds:
            raise ValueError('Backup reserve must fit inside the total time limit')
        self.out, self.backup, self.repo = map(Path, (out, backup, repo))
        self.deadline = time.monotonic() + total_seconds
        self.work_deadline = self.deadline - reserve_seconds
        self.shutdown_command = shutdown_command
        self.total_phases = total_phases
        self.child = None
        self.closed = False
        self.lock = threading.Lock()
        self.timer = threading.Timer(total_seconds, self.shutdown)
        self.timer.daemon = True

    def stop_child(self):
        child = self.child
        if child is None or child.poll() is not None: return
        try:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=3)
        except ProcessLookupError:
            pass

    def command(self, command, log, deadline, env=None):
        left = deadline - time.monotonic()
        if left <= 0: raise TimeoutError('Campaign deadline reached')
        with Path(log).open('wb') as stream:
            self.child = subprocess.Popen(command, cwd=self.repo, env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try: rc = self.child.wait(timeout=left)
            except subprocess.TimeoutExpired:
                self.stop_child(); raise TimeoutError('Command deadline reached')
            finally:
                if self.child.poll() is None: self.stop_child()
        if rc != 0: raise subprocess.CalledProcessError(rc, command)

    def shutdown(self):
        with self.lock:
            if self.closed: return
            self.closed = True
        self.stop_child()
        if self.shutdown_command:
            try: subprocess.run(self.shutdown_command, timeout=10, check=True)
            except Exception as error:
                print('Shutdown command failed; platform scheduled stop remains required:', error, flush=True)

    def digest(self, path):
        h = hashlib.sha256()
        with Path(path).open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                if time.monotonic() >= self.deadline-2: raise TimeoutError('Backup checksum deadline')
                h.update(block)
        return h.hexdigest()

    def backup_results(self):
        archive = self.out.with_suffix('.tar.gz')
        partial = Path(str(archive) + '.partial')
        self.command(['tar', '-czf', str(partial), '-C', str(self.out.parent), self.out.name],
                     self.out.with_suffix('.pack.log'), self.deadline - 2)
        partial.rename(archive)
        digest = self.digest(archive)
        checksum = Path(str(archive) + '.sha256')
        checksum.write_text(digest + '  ' + archive.name + '\n')
        self.backup.mkdir(parents=True, exist_ok=True)
        self.command(['cp', str(archive), str(checksum), str(self.backup)],
                     self.out.with_suffix('.copy.log'), self.deadline - 2)
        if self.digest(self.backup / archive.name) != digest:
            raise ValueError('Persistent backup checksum mismatch')
        receipt = dict(status='VERIFIED', archive=archive.name, sha256=digest,
                       campaign_exitcode=int((self.out / 'campaign.exitcode').read_text()))
        (self.backup / (self.out.name + '.receipt.json')).write_text(json.dumps(receipt, indent=2))
        return receipt

    def run(self, test_command, campaign_command, env):
        rc, backup_ok = 1, False
        self.timer.start()
        try:
            self.out.mkdir(parents=True, exist_ok=False)
            (self.out / 'progress.json').write_text(json.dumps(dict(status='RUNNING', phase='unit_tests',
                completed=0, total=self.total_phases, deadline_epoch=time.time() + self.deadline-time.monotonic())))
            try:
                self.command(test_command, self.out / 'pytest.log', self.work_deadline, env)
                self.command(campaign_command, self.out / 'campaign.log', self.work_deadline, env)
                rc = 0
            except Exception as error:
                rc = 124 if isinstance(error, TimeoutError) else (error.returncode if isinstance(error, subprocess.CalledProcessError) else 1)
                (self.out / 'failure.txt').write_text(traceback.format_exc())
            except KeyboardInterrupt:
                rc = 130
                (self.out / 'failure.txt').write_text('Interrupted before campaign completion\n')
            finally:
                self.stop_child()
                (self.out / 'campaign.exitcode').write_text(str(rc) + '\n')
                (self.out / ('COMPLETE' if rc == 0 else 'INCOMPLETE')).write_text(
                    'Campaign execution complete; backup receipt is stored separately.\n' if rc == 0 else 'No scientific conclusion from incomplete execution.\n')
                try: progress = json.loads((self.out / 'progress.json').read_text())
                except Exception: progress = {}
                progress.update(status='PACKING', campaign_exitcode=rc)
                (self.out / 'progress.json').write_text(json.dumps(progress))
                try:
                    self.backup_results(); backup_ok = True
                except Exception:
                    (self.out / 'backup_failure.txt').write_text(traceback.format_exc())
                progress.update(status='COMPLETE' if rc == 0 and backup_ok else 'NEEDS_ATTENTION',
                                backup_verified=backup_ok)
                (self.out / 'progress.json').write_text(json.dumps(progress))
        finally:
            self.shutdown()
            self.timer.cancel()
        return rc if rc != 0 else (0 if backup_ok else 74)


def main():
    disk = Path(os.environ.get('SS26_DISK', '/root/autodl-tmp'))
    repo = Path(os.environ.get('SS26_REPO', str(disk / 'MoonMuse-SS26/MoonMuse-repo')))
    out = Path(os.environ.get('OUT', str(disk / 'ss26-stage2-pad-20260928-v2')))
    backup = Path(os.environ.get('SS26_BACKUP', '/root/autodl-fs/moonmuse/ss26-stage2-pad-20260928-v2'))
    total = int(os.environ.get('TIME_LIMIT_SECONDS', '12600'))
    reserve = int(os.environ.get('BACKUP_RESERVE_SECONDS', '900'))
    shutdown = ['/usr/bin/shutdown'] if os.environ.get('AUTO_SHUTDOWN', '1') == '1' else None
    if shutdown and not str(repo).startswith('/root/autodl-tmp/'):
        raise RuntimeError('Real shutdown is only allowed on the specified AutoDL checkout')
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHON_BIN=sys.executable, SS26_MANAGED='1',
        OUT=str(out), PLAN_ROOT=str(disk / 'MoonMuse-SS26/ss26-stage1-plans-20260928'),
        ASSET_MANIFEST=str(disk / 'MoonMuse-SS26/ss26-stage2-pad-assets-v2.json'),
        SS26_TOTAL_DEADLINE=str(time.time()+total))
    supervisor = Supervisor(out, backup, repo, total, reserve, shutdown)
    def interrupted(signum, frame): raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupted)
    tests = [sys.executable, '-m', 'pytest', '-q', 'tests/test_ss26_stage2_pad_objective.py',
             'tests/test_ss26_stage2_integrity.py']
    return supervisor.run(tests, ['bash', 'scripts/run_ss26_stage2_pad_pair.sh'], env)


if __name__ == '__main__':
    sys.exit(main())
