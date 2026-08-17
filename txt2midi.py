import os
import random
import argparse
import hashlib
import json
import miditoolkit
from miditoolkit.midi import containers
import numpy as np
from itertools import chain

##############################
# constants
##############################

max_bars = 128

DEFAULT_BEAT_RESOL = 480
DEFAULT_BAR_RESOL = 480 * 4
DEFAULT_FRACTION = 16


class NoValidNotesError(ValueError):
    pass

DEFAULT_SCALE = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'G#', 'A', 'Bb', 'B']
MAJOR_KEY = np.array(['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'])
MINOR_KEY = np.array(['c', 'c#', 'd', 'd#', 'e', 'f', 'f#', 'g', 'g#', 'a', 'a#', 'b'])

quality_conversion_table = {
    #           1     2     3     4  5     6     7
    'maj':     [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min':     [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'aug':     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    'dim':     [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
    'sus4':    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'sus4(b7)':[1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus4(b7,9)':[1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0],
    'sus2':    [1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    '7':       [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj7':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min7':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    'minmaj7': [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    'maj6':    [1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6':    [1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    '9':       [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj9':    [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    'min9':    [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
    '7(#9)':   [1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0],
    'maj6(9)': [1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
    'min6(9)': [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    'maj(9)':  [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    'min(9)':  [1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    'maj(11)': [1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min(11)': [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
    '11':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0],
    'maj9(11)':[1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1],
    'min11':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0],
    '13':      [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    'maj13':   [1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
    'min13':   [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 0],
    'dim7':    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
    'hdim7':   [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0],
    #'5':       [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    }
IDX_TO_KEY = {
    9: 'A',
    10: 'A#',
    11: 'B',
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#'
}
KEY_TO_IDX = {
    v: k for k, v in IDX_TO_KEY.items()
}
quality_name_table = {
    'M': 'maj',
    'm': 'min',
    '+': 'aug',
    'o': 'dim',
    'sus4': 'sus4',
    'sus2': 'sus2',
    '7': '7',
    'M7': 'maj7',
    'm7': 'min7',
    'o7': 'dim7',
    '/o7': 'hdim7',
    'None': 'None'
}
majorDegree2roman = {
    0: 'I',
    1: 'I#',
    2: 'II',
    3: 'II#',
    4: 'III',
    5: 'IV',
    6: 'IV#',
    7: 'V',
    8: 'V#',
    9: 'VI',
    10: 'VI#',
    11: 'VII',
}
roman2majorDegree = {v: k for k, v in majorDegree2roman.items()}
roman2minorDegree = {
    'I': 0,
    'I#': 1,
    'II': 2,
    'II#': random.choice([2, 3]),
    'III': 3,
    'IV': 5,
    'IV#': 6,
    'V': 7,
    'V#': random.choice([7, 8]),
    'VI': 8,
    'VI#': 9,
    'VII': 10
}

##############################
# containers for conversion
##############################
class ConversionEvent(object):
    def __init__(self, event, is_full_event=False):
        if not is_full_event:
            if 'Note' in event:
                self.name, self.value = '_'.join(event.split('_')[:-1]), event.split('_')[-1]
            elif 'Chord' in event:
                self.name, self.value = event.split('_')[0], '_'.join(event.split('_')[1:])
            else:
                self.name, self.value = event.split('_')
        else:
            self.name, self.value = event['name'], event['value']

    def __repr__(self):
        return 'Event(name: {} | value: {})'.format(self.name, self.value)

class NoteEvent(object):
    def __init__(self, pitch, bar, position, duration, velocity, microtiming=None):
        self.pitch = pitch
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)
        self.quantized_duration = duration
        self.velocity = velocity

        if microtiming is not None:
            self.start_tick += microtiming

    def set_microtiming(self, microtiming):
        self.start_tick += microtiming

    def set_velocity(self, velocity):
        self.velocity = velocity

    def __repr__(self):
        return 'Note(pitch = {}, duration = {}, start_tick = {})'.format(
            self.pitch, self.quantized_duration, self.start_tick
        )

class TempoEvent(object):
    def __init__(self, tempo, bar, position):
        self.tempo = tempo
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

    def set_tempo(self, tempo):
        self.tempo = tempo

    def __repr__(self):
        return 'Tempo(tempo = {}, start_tick = {})'.format(
            self.tempo, self.start_tick
        )

class ChordEvent(object):
    def __init__(self, chord_val, bar, position):
        self.chord_val = chord_val
        self.start_tick = bar * DEFAULT_BAR_RESOL + position * (DEFAULT_BAR_RESOL // DEFAULT_FRACTION)

##############################
# conversion functions
##############################
def event_to_midi(key, events, mode, output_midi_path=None, is_full_event=False,
                  return_tempos=False, enforce_tempo=False, enforce_tempo_evs=None, play_chords=False):
    events = [ConversionEvent(ev, is_full_event=is_full_event) for ev in events]
    # print (events[:20])

    keyname = key.split('_')[1].upper()
    print('[degug] keyname:', keyname)
    start = np.where(MAJOR_KEY == keyname)[0][0]
    scale_range = np.concatenate([MAJOR_KEY[start:], MAJOR_KEY[:start]], axis=0)

    # assert events[0].name == 'Bar'
    temp_notes = []
    temp_tempos = []
    temp_chords = []
    temp_root = None
    temp_quality = None

    cur_bar = -1
    cur_position = 0

    for i in range(len(events)):
        # print('[debug] event:', events[i])
        if events[i].name == 'Bar':
            cur_bar += 1
        elif events[i].name == 'Beat':
            cur_position = int(events[i].value)
            assert cur_position >= 0 and cur_position < DEFAULT_FRACTION
        #   print (cur_bar, cur_position)
        elif events[i].name == 'Tempo' and 'Conti' not in events[i].value:
            temp_tempos.append(TempoEvent(
                int(events[i].value), max(cur_bar, 0), cur_position
            ))
        elif 'Note_Pitch' in events[i].name:
            if mode == 'full' and \
                    (i + 1) < len(events) and 'Note_Duration' in events[i + 1].name and \
                    (i + 2) < len(events) and 'Note_Velocity' in events[i + 2].name:
                # check if the 3 events are of the same instrument
                temp_notes.append(
                    NoteEvent(
                        pitch=int(events[i].value),
                        bar=cur_bar, position=cur_position,
                        duration=int(events[i + 1].value), velocity=int(events[i + 2].value)
                    )
                )
            elif mode == 'skyline' and \
                    (i + 1) < len(events) and 'Note_Duration' in events[i + 1].name:
                temp_notes.append(
                    NoteEvent(
                        pitch=int(events[i].value),
                        bar=cur_bar, position=cur_position,
                        duration=int(events[i + 1].value), velocity=80
                    )
                )
        elif 'Chord' in events[i].name and 'Conti' not in events[i].value:
            # print('[debug] chord:', events[i].value)
            temp_chords.append(
                ChordEvent(events[i].value, cur_bar, cur_position)
            )
        elif events[i].name in ['EOS', 'PAD']:
            continue
    
    # print('[debug] temp_chords:', temp_chords)
    # print('[debug] temp_notes:', temp_notes)
    print('[debug] temp_tempos:', temp_tempos)
    print('# tempo changes:', len(temp_tempos), '| # notes:', len(temp_notes))
    midi_obj = miditoolkit.midi.parser.MidiFile()
    midi_obj.instruments = [
        miditoolkit.Instrument(program=0, is_drum=False, name='Piano')
    ]

    for n in temp_notes:
        midi_obj.instruments[0].notes.append(
            miditoolkit.Note(int(n.velocity), n.pitch, int(n.start_tick), int(n.start_tick + n.quantized_duration))
        )

    if enforce_tempo is False:
        for t in temp_tempos:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )
    else:
        if enforce_tempo_evs is None:
            enforce_tempo_evs = temp_tempos[1]
        for t in enforce_tempo_evs:
            midi_obj.tempo_changes.append(
                miditoolkit.TempoChange(t.tempo, int(t.start_tick))
            )

    for c in temp_chords:
        if 'None' in c.chord_val:
            midi_obj.markers.append(
                miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
            )
        else:
            chord_val = c.chord_val
            root = chord_val.split('_')[0]
            if keyname in MAJOR_KEY:
                root = roman2majorDegree[root]
            else:
                root = roman2minorDegree[root]
            quality = chord_val.split('_')[1]
            c.chord_val = scale_range[int(root)] + '_' + quality
            midi_obj.markers.append(
                miditoolkit.Marker('Chord-{}'.format(c.chord_val), int(c.start_tick))
            )
    for b in range(cur_bar):
        midi_obj.markers.append(
            miditoolkit.Marker('Bar-{}'.format(b + 1), int(DEFAULT_BAR_RESOL * b))
        )

    if not midi_obj.instruments[0].notes:
        raise NoValidNotesError('no complete note events were generated')
    midi_obj.max_tick = max(note.end for note in midi_obj.instruments[0].notes)

    if play_chords:
        add_chords(midi_obj)

    if output_midi_path is not None:
        midi_obj.dump(output_midi_path)

    if not return_tempos:
        return midi_obj
    else:
        return midi_obj, temp_tempos

def add_chords(midi_obj):
    default_velocity = 63

    markers = [marker for marker in midi_obj.markers if 'Chord' in marker.text]
    prev_chord = None
    dedup_chords = []
    for m in markers:
        if m.text == 'Chord-None_None':
            continue
        if m.text != prev_chord:
            prev_chord = m.text
            dedup_chords.append(m)
    markers = dedup_chords

    midi_maps = [chord_to_midi(marker.text.split('-')[1]) for marker in markers]
    midi_obj.instruments.append(containers.Instrument(program=0, is_drum=False, name='Piano'))

    if not len(midi_maps) == 0:
        for midi_map, prev_marker, next_marker in zip(midi_maps, markers[:-1], markers[1:]):
            for midi_pitch in midi_map:
                midi_note = containers.Note(start=prev_marker.time, end=next_marker.time, pitch=midi_pitch,
                                            velocity=default_velocity)
                midi_obj.instruments[1].notes.append(midi_note)
        for midi_pitch in midi_maps[-1]:
            midi_note = containers.Note(start=markers[-1].time, end=midi_obj.max_tick, pitch=midi_pitch,
                                        velocity=default_velocity)
            midi_obj.instruments[1].notes.append(midi_note)

    return midi_obj

def chord_to_midi(chord):
    root, quality = chord.split('_')
    bass = root

    root_c = 60
    bass_c = 36
    root_pc = KEY_TO_IDX[root]
    if quality in quality_name_table:
        quality = quality_name_table[quality]
    chord_map = list(np.where(np.array(quality_conversion_table[quality]) == 1)[0])
    bass_pc = KEY_TO_IDX[bass]

    return [bass_c + bass_pc] + [root_c + root_pc + i for i in chord_map]

def degree2pitch(key, octave, roman):
    # major key
    if key in MAJOR_KEY:
        tonic = KEY_TO_IDX[key]
        pitch = octave * 12 + tonic + roman2majorDegree[roman]
    # minor key
    elif key in MINOR_KEY:
        tonic = KEY_TO_IDX[key.upper()]
        pitch = octave * 12 + tonic + roman2minorDegree[roman]
    else:
        raise NameError('Wrong key name {}.'.format(key))

    return pitch

def extract_performance_bars(events, condition_track='Melody'):
    """Return each performance track up to its next condition-track marker."""
    condition_marker = 'Track_{}'.format(condition_track)
    condition_starts = [
        index for index, event in enumerate(events)
        if event == condition_marker
    ]
    perf_starts = [
        index for index, event in enumerate(events)
        if event == 'Track_Performance'
    ]

    midi_bars = []
    condition_index = 0
    for start in perf_starts:
        while condition_index < len(condition_starts) and condition_starts[condition_index] <= start:
            condition_index += 1
        end = (
            condition_starts[condition_index]
            if condition_index < len(condition_starts)
            else len(events)
        )
        midi_bars.append(events[start + 1:end])
    return midi_bars


def extract_midi_events_from_generation(
        key, events, relative_melody=False, condition_track='Melody'):
    if relative_melody:
        new_events = []
        keyname = key.split('_')[1]
        octave = None
        root = None
        quality = None
        for evs in events:
            if 'Note_Octave' in evs:
                octave = int(evs.split('_')[2])
            elif 'Note_Degree' in evs:
                if octave is None:
                    continue
                roman = evs.split('_')[2]
                pitch = degree2pitch(keyname, octave, roman)
                pitch = max(21, pitch)
                pitch = min(108, pitch)
                if pitch < 21 or pitch > 108:
                    raise ValueError('Pitch value must be in (21, 108), but gets {}'.format(pitch))
                new_events.append('Note_Pitch_{}'.format(pitch))
            elif 'Chord_Root' in evs:
                root = evs.split('_')[-1]
            elif 'Chord_Quality' in evs:
                quality = evs.split('_')[-1]
                if root is not None:
                    new_events.append('Chord_{}_{}'.format(root, quality))
                root = None
                quality = None
            # elif 'Chord_' in evs:
            #     if 'None' in evs or 'Conti' in evs:
            #         new_events.append(evs)
            #     else:
            #         root, quality = evs.split('_')[1], evs.split('_')[2]
            #         if keyname in MAJOR_KEY:
            #             root = roman2majorDegree[root]
            #         else:
            #             root = roman2minorDegree[root]
            #         new_events.append('Chord_{}_{}'.format(root, quality))
            else:
                new_events.append(evs)
        events = new_events

    return extract_performance_bars(events, condition_track=condition_track)


def flatten_midi_bars(generated_bars, bars=max_bars):
    """Flatten generated bars while preserving exactly one Bar marker per bar."""
    normalized = []
    for bar_events in generated_bars[:bars]:
        if bar_events and bar_events[0] == 'Bar_None':
            normalized.append(bar_events)
        else:
            normalized.append(['Bar_None'] + bar_events)
    return list(chain.from_iterable(normalized))


def collapse_exact_onset_pitch_duplicates(events):
    """Keep the first complete note for each bar/beat/pitch and audit removals.

    This transformation operates on absolute-pitch conversion events. It leaves
    every non-duplicate token unchanged and is disabled by default. Equal Beat
    tokens remain the same onset; a new Bar or different Beat resets the pitch
    set. The caller must preserve the raw generated event text separately.
    """
    output = []
    bar_index = -1
    beat = 0
    multiplicities = {}
    removed = []
    index = 0
    while index < len(events):
        event = events[index]
        if event == 'Bar_None':
            bar_index += 1
            beat = 0
            output.append(event)
            index += 1
            continue
        if event.startswith('Beat_'):
            beat = int(event.rsplit('_', 1)[1])
            output.append(event)
            index += 1
            continue
        complete_note = (
            event.startswith('Note_Pitch_')
            and index + 2 < len(events)
            and events[index + 1].startswith('Note_Duration_')
            and events[index + 2].startswith('Note_Velocity_')
        )
        if not complete_note:
            output.append(event)
            index += 1
            continue
        pitch = int(event.rsplit('_', 1)[1])
        onset_key = (bar_index, beat, pitch)
        multiplicities[onset_key] = multiplicities.get(onset_key, 0) + 1
        note_group = events[index:index + 3]
        if multiplicities[onset_key] == 1:
            output.extend(note_group)
        else:
            removed.append(
                {
                    'bar_number': bar_index + 1,
                    'beat_subdivision': beat,
                    'pitch': pitch,
                    'removed_tokens': note_group,
                }
            )
        index += 3
    duplicate_groups = [
        {
            'bar_number': key[0] + 1,
            'beat_subdivision': key[1],
            'pitch': key[2],
            'multiplicity': count,
        }
        for key, count in sorted(multiplicities.items())
        if count > 1
    ]
    report = {
        'input_token_count': len(events),
        'output_token_count': len(output),
        'removed_note_group_count': len(removed),
        'removed_token_count': len(events) - len(output),
        'duplicate_onset_pitch_group_count': len(duplicate_groups),
        'max_same_pitch_onset_count': max(
            (row['multiplicity'] for row in duplicate_groups), default=1
        ),
        'duplicate_groups': duplicate_groups,
        'removed_note_groups': removed,
        'non_duplicate_tokens_preserved': len(events) - len(output) == 3 * len(removed),
    }
    return output, report


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def txt_to_event(input_event_path):
    with open(input_event_path, 'r') as f:
        # 读取所有行并去除换行符
        events = [line.strip() for line in f.readlines()]
    return events


def merge_chord_events(data_list):
    new_list = []
    i = 0
    n = len(data_list)
    
    while i < n:
        current = data_list[i]
        
        # 检查当前元素是否是Chord_Root且下一个元素是Chord_Quality
        if (current.startswith('Chord_Root_') and 
            i+1 < n and 
            data_list[i+1].startswith('Chord_Quality_')):
            
            # 提取root和quality
            root = current.split('_')[-1]
            quality = data_list[i+1].split('_')[-1]
            
            # 创建合并后的chord
            merged_chord = f'Chord_{root}_{quality}'
            new_list.append(merged_chord)
            
            # 跳过下一个元素，因为已经处理了
            i += 2
        else:
            new_list.append(current)
            i += 1
            
    return new_list


def convert_txt_directory(
        input_dir, output_dir=None, relative_melody=True, bars=max_bars,
        condition_track='Melody', collapse_duplicate_onsets=False,
        transformation_report=None):
    if collapse_duplicate_onsets and transformation_report is None:
        raise ValueError(
            'an explicit transformation_report is required when duplicate '
            'collapse is enabled'
        )
    if output_dir is None:
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    files = sorted(file for file in os.listdir(input_dir) if file.endswith('.txt'))
    converted = 0
    skipped = 0
    file_reports = []
    for file in files:
        input_event_path = os.path.join(input_dir, file)
        output_midi_path = os.path.join(output_dir, file.replace('.txt', '.mid'))
        generated = txt_to_event(input_event_path)
        key = next((event for event in generated if event.startswith('Key_')), 'Key_C')

        generated = merge_chord_events(generated)
        generated = extract_midi_events_from_generation(
            key,
            generated,
            relative_melody=relative_melody,
            condition_track=condition_track,
        )
        midi_events = flatten_midi_bars(generated, bars=bars)
        if collapse_duplicate_onsets:
            midi_events, duplicate_report = collapse_exact_onset_pitch_duplicates(
                midi_events
            )
        else:
            duplicate_report = None

        try:
            event_to_midi(
                key,
                midi_events,
                mode='full',
                output_midi_path=output_midi_path
            )
            converted += 1
            if duplicate_report is not None:
                file_reports.append(
                    {
                        'file': file,
                        'input_event_sha256': sha256_file(input_event_path),
                        'output_midi_sha256': sha256_file(output_midi_path),
                        **duplicate_report,
                    }
                )
        except NoValidNotesError as exc:
            skipped += 1
            print('[warning] skipped {}: {}'.format(file, exc))

    print('MIDI_CONVERSION_SUMMARY converted={} skipped={}'.format(converted, skipped))
    if converted == 0:
        raise NoValidNotesError('no generated samples contained complete note events')
    report = {
        'schema': 'moonmuse-explicit-onset-dedup-conversion-v1',
        'formalStudyData': False,
        'enabled': bool(collapse_duplicate_onsets),
        'raw_event_text_modified': False,
        'representative_rule': 'keep_first_complete_note_group',
        'converted': converted,
        'skipped': skipped,
        'files': file_reports,
        'interpretation_boundary': (
            'This is an explicit audited conversion intervention, not evidence that '
            'the model learned to avoid duplicate notes.'
        ),
    }
    if transformation_report is not None:
        report_dir = os.path.dirname(os.path.abspath(transformation_report))
        os.makedirs(report_dir, exist_ok=True)
        with open(transformation_report, 'w') as handle:
            json.dump(report, handle, indent=2)
            handle.write('\n')
    return report


def main():
    parser = argparse.ArgumentParser(description='Convert generated event txt files to MIDI.')
    parser.add_argument('-i', '--input_dir', default='generation/stage3_performance',
                        help='directory containing generated event txt files')
    parser.add_argument('-o', '--output_dir', default=None,
                        help='directory to write MIDI files; defaults to input_dir')
    parser.add_argument('--absolute_melody', action='store_true',
                        help='input events already use absolute Note_Pitch tokens')
    parser.add_argument('--max_bars', type=int, default=max_bars,
                        help='maximum number of bars to convert per file')
    parser.add_argument('--condition-track', choices=['Melody', 'Chord'],
                        default='Melody',
                        help='track that conditions each generated performance bar')
    parser.add_argument('--collapse-exact-onset-pitch-duplicates', action='store_true',
                        help='explicitly keep only the first complete note per onset/pitch')
    parser.add_argument('--transformation-report', default=None,
                        help='required JSON audit path when duplicate collapse is enabled')
    args = parser.parse_args()

    convert_txt_directory(
        args.input_dir,
        output_dir=args.output_dir,
        relative_melody=not args.absolute_melody,
        bars=args.max_bars,
        condition_track=args.condition_track,
        collapse_duplicate_onsets=args.collapse_exact_onset_pitch_duplicates,
        transformation_report=args.transformation_report,
    )


if __name__ == '__main__':
    main()
