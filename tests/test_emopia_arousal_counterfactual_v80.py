from pathlib import Path

import mido

from scripts.audit_emopia_arousal_counterfactual_v80 import signatures, transform_midi


def write_midi(path: Path) -> None:
    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
    track.append(mido.Message("program_change", program=0, time=0))
    track.append(mido.Message("note_on", note=60, velocity=64, time=0))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(path)


def test_transform_changes_only_declared_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.mid"
    output = tmp_path / "output.mid"
    write_midi(source)

    report = transform_midi(source, output, velocity_shift=16, tempo_multiplier=1.25)
    transformed = signatures(mido.MidiFile(output))

    assert transformed["velocities"] == [80]
    assert transformed["tempos"] == [400000]
    assert report["source_mean_velocity"] == 64.0
    assert report["variant_mean_velocity"] == 80.0
    assert report["source_median_bpm"] == 120.0
    assert report["variant_median_bpm"] == 150.0
