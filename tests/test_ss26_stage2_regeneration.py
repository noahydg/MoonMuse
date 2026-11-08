from pathlib import Path

from scripts.run_ss26_stage2_regeneration import commands, smoke_commands, main, shutdown_command, RUN_NAME, TOTAL_PHASES
import subprocess
import pytest


def test_six_conditions_only_reuse_finetuned_checkpoints_with_identical_policy():
    rows = commands(Path('/prior'), Path('/plans'), Path('/out'))
    assert {(seed, mode) for seed, mode, _ in rows} == {
        (s, m) for s in (42, 43, 44) for m in ('ignore', 'count')}
    for seed, mode, command in rows:
        assert command[1] == 'stage2_melody/inference.py'
        assert command[command.index('-i')+1] == f'/prior/checkpoints/seed{seed}/{mode}/finetune/params/best_params.pt'
        assert command[command.index('--max_events')+1] == '1536'
        assert '--structured-generation' in command
        assert command[command.index('-od')+1] == f'/out/melodies/{mode}/seed{seed}'
    assert RUN_NAME != 'ss26-stage2-regeneration-20260930-v1'


def test_shell_shutdown_wrapper_handles_executable_without_shebang(tmp_path):
    script = tmp_path / 'shutdown'
    marker = tmp_path / 'called'
    script.write_text('printf stopped > "$1"\n')
    script.chmod(0o700)
    command = shutdown_command()
    assert command == ['/bin/bash', '-c', '/usr/bin/shutdown']
    # Execute only a harmless local stand-in, never the real shutdown command.
    subprocess.run(command[:2] + ['"$0" "$1"', str(script), str(marker)], check=True)
    assert marker.read_text() == 'stopped'


def test_smoke_is_separate_from_the_fixed_six_full_batches():
    rows = smoke_commands(Path('/prior'), Path('/plans'), Path('/out'))
    assert TOTAL_PHASES == 10 and {mode for mode, _ in rows} == {'ignore', 'count'}
    for mode, command in rows:
        assert command[command.index('--seed')+1] == '42'
        assert command[command.index('-id')+1] == '/out/smoke/plans'
        assert command[command.index('-od')+1] == f'/out/smoke/{mode}'


def test_no_unapproved_launch(monkeypatch):
    monkeypatch.setattr('sys.argv', ['run_ss26_stage2_regeneration.py'])
    monkeypatch.delenv('SS26_REGEN_APPROVED', raising=False)
    with pytest.raises(RuntimeError, match='Not launched'):
        main()
