#!/usr/bin/env python3
"""Common SS26 conversion: fixed degree mapping and shared repeat-note release."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def cap_repeated_notes(midi):
    """End a note by the next strictly later onset of its pitch, in each track.

    Simultaneous duplicate notes remain intact. This changes releases only.
    Apply identically to learned and rule performances; retain the raw MIDI.
    """
    changed = 0
    for instrument in midi.instruments:
        starts = defaultdict(set)
        for note in instrument.notes:
            starts[note.pitch].add(note.start)
        successors = {}
        for pitch, values in starts.items():
            ordered = sorted(values)
            successors.update({(pitch, a): b for a, b in zip(ordered, ordered[1:])})
        for note in instrument.notes:
            next_start = successors.get((note.pitch, note.start))
            if next_start is not None and note.end > next_start:
                note.end = next_start
                changed += 1
    return changed


def convert(events, raw_dir, playback_dir, bars=8):
    # The inherited converter chooses two minor-key aliases at import time.
    # Fix its import seed in a fresh converter process for every condition.
    if 'txt2midi' in sys.modules:
        raise RuntimeError('Run SS26 conversion in a fresh process to fix the degree mapping')
    random.seed(0)
    import txt2midi
    import miditoolkit
    txt2midi.convert_txt_directory(str(events), str(raw_dir), bars=bars, condition_track='Melody')
    playback_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for src in sorted(raw_dir.glob('*.mid')):
        midi = miditoolkit.MidiFile(str(src))
        before = Counter((i, n.start, n.pitch, n.velocity) for i, ins in enumerate(midi.instruments) for n in ins.notes)
        changed = cap_repeated_notes(midi)
        dst = playback_dir / src.name
        midi.dump(str(dst))
        checked = miditoolkit.MidiFile(str(dst))
        after = Counter((i, n.start, n.pitch, n.velocity) for i, ins in enumerate(checked.instruments) for n in ins.notes)
        if before != after:
            raise ValueError(f'Playback conversion changed notes, onsets, pitches or velocities: {src}')
        rows.append(dict(file=src.name, notes=sum(before.values()), shortened_note_releases=changed,
                         raw_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
                         playback_sha256=hashlib.sha256(dst.read_bytes()).hexdigest()))
    report = dict(status='PASS', formalStudyData=False, import_random_seed=0,
                  minor_degree_aliases={x:txt2midi.roman2minorDegree[x] for x in ('II#','V#')},
                  common_processing='cap note release at the next strictly later same-pitch onset; no onset or note deletion',
                  required_scope='identical conversion and playback processing for EVERY comparison condition',
                  files=rows)
    (playback_dir.parent / 'conversion.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events', type=Path, required=True)
    p.add_argument('--raw-midi', type=Path, required=True)
    p.add_argument('--playback-midi', type=Path, required=True)
    a = p.parse_args()
    convert(a.events, a.raw_midi, a.playback_midi)
