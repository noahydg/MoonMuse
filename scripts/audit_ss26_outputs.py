#!/usr/bin/env python3
"""Audit physical MIDI pitch/onset preservation against the original score."""
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts.render_rule_performance_ss26 import note_records, track_bars

PC={'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
MAJOR={'I':0,'I#':1,'II':2,'II#':3,'III':4,'IV':5,'IV#':6,'V':7,'V#':8,'VI':9,'VI#':10,'VII':11}
MINOR={**MAJOR,'III':3,'VI':8,'VI#':9,'VII':10}


def score_counter(path):
    tokens=path.read_text().splitlines()
    key=next(t[4:] for t in tokens if t.startswith('Key_'))
    degrees=MINOR if key.islower() else MAJOR
    notes=note_records(track_bars(tokens,'Melody',8))
    return Counter((n['tick'],min(108,max(21,int(n['octave'].rsplit('_',1)[1])*12+PC[key.upper()]+degrees[n['degree'].removeprefix('Note_Degree_')]))) for n in notes)


def check_midi(source, midi_path):
    import miditoolkit
    midi=miditoolkit.MidiFile(str(midi_path))
    expected=score_counter(source)
    actual=Counter((n.start,n.pitch) for ins in midi.instruments for n in ins.notes)
    if midi.ticks_per_beat!=480 or not expected or expected!=actual:
        raise ValueError(f'MIDI score mismatch: {midi_path}; extra={sum((actual-expected).values())}, missing={sum((expected-actual).values())}')
    notes=[n for ins in midi.instruments for n in ins.notes]
    if any(n.end<=n.start for n in notes):raise ValueError('Non-positive duration')
    return dict(midi_skeleton_equal=True,notes=len(notes),tempo_values=sorted({t.tempo for t in midi.tempo_changes}),
                velocity_values=[n.velocity for n in notes],duration_ticks=[n.end-n.start for n in notes])
