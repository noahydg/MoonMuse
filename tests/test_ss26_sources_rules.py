from collections import Counter
import csv
import hashlib
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.select_ss26_sources import SourceIndex
from scripts.render_rule_performance_ss26 import intervals, note_records, quantize, render_tokens, track_bars
from scripts.convert_ss26_events import cap_repeated_notes
from scripts.prepare_ss26_comparison import canonical_condition
from scripts.audit_ss26_outputs import score_counter


def spec():
    return dict(parameters={'Q1':dict(tempo_median=125,velocity_median=80,duration_ioi_ratio_median=1,
                                     beat_offsets={str(i):0 for i in range(16)})},
                bounds={'Q1':{'Tempo':dict(min=110,max=149),'Note_Duration':dict(min=240,max=960),'Note_Velocity':dict(min=67,max=88)}},
                vocabulary={'Tempo_':[110,120,130,150],'Note_Duration_':[120,240,480,960],'Note_Velocity_':[60,70,80,90]})


SOURCE=['Emotion_Q1','Key_C','Track_Melody','Bar_None','Beat_0','Chord_Root_I','Chord_Quality_M',
        'Note_Octave_4','Note_Degree_I','Note_Octave_4','Note_Degree_V','Beat_1','Note_Octave_4','Note_Degree_I','EOS_None']


def test_polyphonic_ioi_and_final_group():
    notes=note_records(track_bars(SOURCE,'Melody'))
    gaps,last=intervals(notes)
    assert [n['tick'] for n in notes]==[0,0,120]
    assert gaps=={0:120,120:120} and last==120
    assert intervals([{'tick':100}])[0]=={100:480}


def test_deterministic_rule_preserves_notes_and_strict_raw_bounds():
    one,info=render_tokens(SOURCE,'Q1',spec())
    assert one==render_tokens(SOURCE,'Q1',spec())[0]
    before=note_records(track_bars(SOURCE,'Melody'))
    after=note_records(track_bars(one,'Performance'),True)
    assert [(n['tick'],n['octave'],n['degree']) for n in before]==[(n['tick'],n['octave'],n['degree']) for n in after]
    assert [n['duration'] for n in after]==[240,240,240]
    assert quantize(125,'Tempo_','Q1',spec())==120
    assert quantize(999,'Note_Velocity_','Q1',spec())==80


def test_incomplete_prefix_does_not_remove_complete_notes():
    source=SOURCE[:-1]+['Note_Octave_7','EOS_None']
    rendered,info=render_tokens(source,'Q1',spec())
    assert len(note_records(track_bars(rendered,'Performance')))==3
    assert info['discarded_incomplete_prefixes']==1


def test_bare_terminal_track_marker_does_not_add_an_empty_bar():
    source=SOURCE[:-1]+['Track_Melody']
    rendered,_=render_tokens(source,'Q1',spec())
    assert len(track_bars(rendered,'Performance'))==1
    assert len(note_records(track_bars(rendered,'Performance')))==3


def test_canonical_input_keeps_original_key_and_avoids_empty_track_tokens():
    tokens,dropped=canonical_condition([x.replace('Key_C','Key_E') for x in SOURCE[:-1]]+['Track_Melody'])
    assert tokens[0]=='Key_E' and not any(t.startswith('Track_') for t in tokens)
    assert tokens.count('Bar_None')==1 and dropped==0


def test_physical_pitch_audit_uses_source_key_and_minor_degree(tmp_path):
    source=tmp_path/'source.txt'
    source.write_text('\n'.join(['Emotion_Q3','Key_a','Track_Melody','Bar_None','Beat_0','Note_Octave_4','Note_Degree_VI#']))
    assert score_counter(source)==Counter({(0,66):1})


def test_common_release_cap_keeps_simultaneous_duplicates():
    notes=[SimpleNamespace(pitch=60,start=0,end=480,velocity=80),SimpleNamespace(pitch=60,start=0,end=360,velocity=70),
           SimpleNamespace(pitch=60,start=120,end=600,velocity=75),SimpleNamespace(pitch=67,start=0,end=480,velocity=65)]
    midi=SimpleNamespace(instruments=[SimpleNamespace(notes=notes)])
    before=Counter((n.start,n.pitch,n.velocity) for n in notes)
    assert cap_repeated_notes(midi)==2
    assert [n.end for n in notes]==[120,120,600,480]
    assert before==Counter((n.start,n.pitch,n.velocity) for n in notes)


def test_history_normalizes_newlines_and_avoids_comparator_collision(tmp_path):
    source=tmp_path/'score.txt';source.write_bytes(b'Emotion_Q1\r\nKey_C\r\nTrack_Melody\r\nBar_None\r\n')
    h=hashlib.sha256(source.read_bytes()).hexdigest()
    with (tmp_path/'source_manifest.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed','quadrant','sample_index','source_sha256','frozen_path'])
        w.writeheader();w.writerow(dict(seed=42,quadrant='Q1',sample_index=0,source_sha256=h,frozen_path='score.txt'))
    index=SourceIndex(tmp_path)
    normalized=hashlib.sha256(source.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    assert index.match({'source_sha256':normalized},Path('source_manifest.json'))[0]=={h}
    row=dict(system_private='moonmuse_c2p',seed_private='42',quadrant_private='Q1',sample_index_private='0')
    assert not index.match(row,Path('v84-full-study/master_manifest_private.csv'))[0]
    row['system_private']='moonmuse_full'
    assert index.match(row,Path('v84-full-study/master_manifest_private.csv'))[0]=={h}
