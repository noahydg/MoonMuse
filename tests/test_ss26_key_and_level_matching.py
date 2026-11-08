import ast
from pathlib import Path
import mido
import numpy as np
import pytest
from source_key import declared_source_key
from scripts.build_ss26_level_matched_rule import bounded_mean_shift, transform, features, skeleton


@pytest.mark.parametrize('prefix',[[],['Emotion_Q2'],['Valence_Negative','Arousal_High']])
@pytest.mark.parametrize('stage',['stage2_melody','stage3_performance'])
def test_actual_reader_retains_minor_key(tmp_path,prefix,stage):
    # Exercise the real reader without importing its GPU model dependencies.
    path=Path(__file__).parents[1]/stage/'inference.py'
    tree=ast.parse(path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='read_generated_events')
    ns={'np':np,'declared_source_key':declared_source_key}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),ns)
    inp=tmp_path/'score.txt';inp.write_text('\n'.join(prefix+['Key_f#','Track_Melody','Bar_None','Beat_0'])+'\n')
    key,bars=ns['read_generated_events'](inp,{'Bar_None':0,'Beat_0':1})
    assert key=='Key_f#' and bars==[[0,1]]


@pytest.mark.parametrize('events',[[],['Emotion_Q2'],['Key_C','Key_c'],['Key_E#']])
def test_key_reader_rejects_missing_ambiguous_or_invalid(events):
    with pytest.raises(ValueError):declared_source_key(events)


def test_velocity_shift_respects_bounds_and_target_with_clipped_accents():
    x=bounded_mean_shift([40,50,60,70],65.25,40,67)
    assert sum(x)==261 and min(x)>=40 and max(x)<=67
    assert x==sorted(x)


def test_midi_transform_preserves_notes_releases_and_timing(tmp_path):
    midi=mido.MidiFile();track=mido.MidiTrack();midi.tracks.append(track)
    track.extend([mido.MetaMessage('set_tempo',tempo=600000,time=0),
        mido.Message('note_on',note=60,velocity=50,time=240),
        mido.MetaMessage('set_tempo',tempo=400000,time=120),
        mido.Message('note_off',note=60,velocity=0,time=120),
        mido.Message('note_on',note=64,velocity=70,time=0),
        mido.Message('note_off',note=64,velocity=0,time=480)])
    src=tmp_path/'old.mid';dst=tmp_path/'new.mid';midi.save(src)
    transform(src,dst,110.5,73.5,40,91)
    assert skeleton(src)==skeleton(dst)
    f=features(dst)
    assert abs(f['effective_bpm']-110.5)<.001 and f['velocity_mean']==73.5
    assert sum(msg.type=='set_tempo' for t in mido.MidiFile(dst).tracks for msg in t)==1


def test_rounding_does_not_depend_on_simultaneous_note_storage_order(tmp_path):
    outputs=[]
    for index,order in enumerate(((60,64,67),(67,60,64))):
        midi=mido.MidiFile();track=mido.MidiTrack();midi.tracks.append(track)
        track.extend(mido.Message('note_on',note=p,velocity=60,time=0) for p in order)
        track.extend(mido.Message('note_off',note=p,time=480 if i==0 else 0) for i,p in enumerate(order))
        old=tmp_path/f'{index}.mid';new=tmp_path/f'{index}_out.mid';midi.save(old)
        transform(old,new,120,60+1/3,40,91);outputs.append(skeleton(new,True))
    assert outputs[0]==outputs[1]
