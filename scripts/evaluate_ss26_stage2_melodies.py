#!/usr/bin/env python3
"""Deterministic symbolic descriptors for Stage 2 pitch-onset scores (SS26 PAD diagnostic).

Reads functional-representation event text (Stage 2 outputs) or EMOPIA c2m event
pickles, keeps the first N bars of the melody track and computes technical and
harmony-fit descriptors. These are structural checks of the generated score,
not measures of musical quality or perceived emotion.
"""
import argparse
import ast
import csv
import hashlib
import json
import math
import pickle
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage2_event_validation import validate_events
MAJOR = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_OFF = {'I': 0, 'I#': 1, 'II': 2, 'II#': 3, 'III': 4, 'IV': 5, 'IV#': 6,
             'V': 7, 'V#': 8, 'VI': 9, 'VI#': 10, 'VII': 11}
# convert_key.py draws II#/V# in minor at import time (2|3, 7|8); fixed here to 3/8.
MINOR_OFF = {'I': 0, 'I#': 1, 'II': 2, 'II#': 3, 'III': 3, 'IV': 5, 'IV#': 6,
             'V': 7, 'V#': 8, 'VI': 8, 'VI#': 9, 'VII': 10}
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10, 11}  # natural minor plus leading tone
STRONG_BEATS = {0, 8}
METRICS = ['complete_bars', 'completion_known', 'reached_target_bar', 'notes_per_bar', 'empty_bar_rate', 'onsets_per_bar',
           'notes_per_onset', 'chord_tone_ratio', 'strong_chord_tone_ratio',
           'scale_tone_ratio', 'pitch_range', 'mean_abs_interval', 'large_leap_rate',
           'repeated_top_note_rate', 'invalid_tokens']
DISTANCE_METRICS = ['notes_per_bar', 'empty_bar_rate', 'onsets_per_bar', 'notes_per_onset',
                    'chord_tone_ratio', 'strong_chord_tone_ratio', 'scale_tone_ratio',
                    'pitch_range', 'mean_abs_interval', 'large_leap_rate', 'repeated_top_note_rate']


def load_quality_tables():
    tree = ast.parse((ROOT / 'stage3_performance' / 'convert2midi.py').read_text())
    tables = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ('quality_conversion_table', 'quality_name_table'):
            tables[node.targets[0].id] = ast.literal_eval(node.value)
    conv, names = tables['quality_conversion_table'], tables['quality_name_table']
    out = {}
    for short, long_name in names.items():
        if long_name in conv:
            out[short] = [i for i, v in enumerate(conv[long_name]) if v]
    return out


QUALITY = load_quality_tables()


def describe(tokens, bars=8, generation=None):
    validation = validate_events(tokens, bars, generation)
    if not validation['valid']:
        result = {m: float('nan') for m in METRICS}
        result.update(invalid_tokens=float(validation['invalid_tokens']),
            reached_target_bar=float(validation['reached_target_bar']),
            completion_known=float(validation['completion_known']),
            complete_bars=float(validation['complete_bars']) if validation['completion_known'] else float('nan'),
            minor=float('nan'))
        return result
    key = next((t for t in tokens if t.startswith('Key_')), None)
    if key is None:
        raise ValueError('no key')
    name = key.split('_', 1)[1]
    minor = name[0].islower()
    tonic = MAJOR.index(name.upper())
    off = MINOR_OFF if minor else MAJOR_OFF
    scale = MINOR_SCALE if minor else MAJOR_SCALE

    track, bar_idx = None, {'Chord': -1, 'Melody': -1}
    chords = defaultdict(list)          # bar -> [(beat, pcs or None)]
    notes = defaultdict(list)           # bar -> [(beat, midi)]
    beat = {'Chord': 0, 'Melody': 0}
    pending_root, pending_octave, invalid = None, None, 0
    for t in tokens:
        if t.startswith('Track_'):
            track = t.split('_', 1)[1]
            continue
        if track not in ('Chord', 'Melody'):
            continue
        if t == 'Bar_None':
            bar_idx[track] += 1
            beat[track] = 0
        elif t.startswith('Beat_'):
            beat[track] = int(t.split('_')[1])
        elif t.startswith('Chord_Root_'):
            pending_root = t[len('Chord_Root_'):]
        elif t.startswith('Chord_Quality_'):
            q = t[len('Chord_Quality_'):]
            if track == 'Chord' and pending_root is not None:
                if pending_root == 'Conti':
                    pcs = 'Conti'
                elif pending_root in off and q in QUALITY:
                    pcs = {(tonic + off[pending_root] + i) % 12 for i in QUALITY[q]}
                else:
                    pcs = None
                chords[bar_idx['Chord']].append((beat['Chord'], pcs))
            pending_root = None
        elif t.startswith('Note_Octave_') and track == 'Melody':
            pending_octave = int(t.split('_')[-1])
        elif t.startswith('Note_Degree_') and track == 'Melody':
            deg = t[len('Note_Degree_'):]
            if pending_octave is None or deg not in off:
                invalid += 1
            else:
                notes[bar_idx['Melody']].append((beat['Melody'], pending_octave * 12 + tonic + off[deg]))
            pending_octave = None
        elif t in ('EOS_None', 'PAD_None', 'Pad_None') or t.startswith('Emotion_') or t.startswith('Key_'):
            if t in ('PAD_None', 'Pad_None'):
                invalid += 1
        else:
            invalid += 1

    active, chord_at = None, {}
    for b in range(bars):
        events = sorted(chords.get(b, []), key=lambda x: x[0])
        timeline = []
        for bt, pcs in events:
            if pcs != 'Conti':
                active = pcs
            timeline.append((bt, active))
        chord_at[b] = (timeline, active)

    def chord_for(b, bt):
        timeline, _ = chord_at[b]
        cur = None
        prev_bar_end = chord_at[b - 1][1] if b > 0 else None
        cur = prev_bar_end
        for t_beat, pcs in timeline:
            if t_beat <= bt:
                cur = pcs
        return cur

    n_bars = min(bar_idx['Melody'] + 1, bars)
    all_notes, onsets_top, ct, ct_strong, n_strong, n_ct_eval, n_strong_eval = [], [], 0, 0, 0, 0, 0
    empty = 0
    onset_count = 0
    for b in range(bars):
        bar_notes = notes.get(b, [])
        if not bar_notes:
            empty += 1
            continue
        by_beat = defaultdict(list)
        for bt, m in bar_notes:
            by_beat[bt].append(m)
        onset_count += len(by_beat)
        for bt in sorted(by_beat):
            onsets_top.append(max(by_beat[bt]))
            for m in by_beat[bt]:
                all_notes.append(m)
                pcs = chord_for(b, bt)
                if pcs:
                    n_ct_eval += 1
                    ct += int(m % 12 in pcs)
                    if bt in STRONG_BEATS:
                        n_strong_eval += 1
                        ct_strong += int(m % 12 in pcs)
    intervals = [abs(a - b) for a, b in zip(onsets_top[1:], onsets_top[:-1])]
    rel = [(m - tonic) % 12 for m in all_notes]
    safe = lambda a, b: (a / b) if b else float('nan')
    return dict(
        reached_target_bar=float(n_bars >= bars),
        complete_bars=float(validation['complete_bars']) if validation['completion_known'] else float('nan'),
        completion_known=float(validation['completion_known']),
        notes_per_bar=len(all_notes) / bars,
        empty_bar_rate=empty / bars,
        onsets_per_bar=onset_count / bars,
        notes_per_onset=safe(len(all_notes), onset_count),
        chord_tone_ratio=safe(ct, n_ct_eval),
        strong_chord_tone_ratio=safe(ct_strong, n_strong_eval),
        scale_tone_ratio=safe(sum(r in scale for r in rel), len(rel)),
        pitch_range=(max(all_notes) - min(all_notes)) if all_notes else float('nan'),
        mean_abs_interval=statistics.mean(intervals) if intervals else float('nan'),
        large_leap_rate=safe(sum(i > 12 for i in intervals), len(intervals)),
        repeated_top_note_rate=safe(sum(i == 0 for i in intervals), len(intervals)),
        invalid_tokens=float(validation['invalid_tokens']),
        minor=float(minor),
    )


def summarize(rows):
    out = {}
    for m in METRICS:
        vals = [r[m] for r in rows if math.isfinite(r[m])]
        out[m] = dict(mean=statistics.mean(vals) if vals else float('nan'),
                      sd=statistics.stdev(vals) if len(vals) > 1 else float('nan'),
                      n=len(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', action='append', nargs=2, metavar=('LABEL', 'DIR'), default=[],
                    help='label and directory of Stage 2 *_melody.txt outputs (recursive)')
    ap.add_argument('--emopia-reference', type=Path,
                    help='events/c2m/emopia_c2m_functional (uses data_splits/valid.pkl)')
    ap.add_argument('--bars', type=int, default=8)
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f'refusing to overwrite {args.output_dir}')
    args.output_dir.mkdir(parents=True)

    per_file = []
    for label, d in args.group:
        for p in sorted(Path(d).rglob('*_melody.txt')):
            toks = [t for t in p.read_text().split('\n') if t]
            q = next((t.split('_')[1] for t in toks if t.startswith('Emotion_')), 'NA')
            row = dict(group=label, quadrant=q, file=str(p.relative_to(d)))
            sidecar = p.with_name(p.name + '.generation.json')
            generation = json.loads(sidecar.read_text()) if sidecar.exists() else None
            if generation is not None:
                assert generation['events_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest(), p
            row.update(describe(toks, args.bars, generation))
            per_file.append(row)
        if not any(r['group'] == label for r in per_file):
            raise ValueError(f'No generated files in group {label}: {d}')
    ref_rows = []
    if args.emopia_reference:
        split = pickle.load(open(args.emopia_reference / 'data_splits' / 'valid.pkl', 'rb'))
        for name in split:
            ev = pickle.load(open(args.emopia_reference / 'events' / name, 'rb'))[2]
            toks = [f"{e['name']}_{e['value']}" for e in ev]
            row = dict(group='emopia_valid_reference', quadrant=name.split('_')[0], file=name)
            row.update(describe(toks, args.bars))
            ref_rows.append(row)
    all_rows = per_file + ref_rows
    with open(args.output_dir / 'per_file.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0]))
        w.writeheader()
        w.writerows(all_rows)

    groups = defaultdict(list)
    for r in all_rows:
        groups[r['group']].append(r)
        groups[f"{r['group']}|{r['quadrant']}"].append(r)
    summary = {g: summarize(rs) for g, rs in groups.items()}
    if ref_rows:
        ref = summary['emopia_valid_reference']
        for g in [g for g in summary if '|' not in g and g != 'emopia_valid_reference']:
            dists = {m: abs(summary[g][m]['mean'] - ref[m]['mean']) / ref[m]['sd']
                     for m in DISTANCE_METRICS if ref[m]['sd'] and math.isfinite(ref[m]['sd'])
                     and math.isfinite(summary[g][m]['mean'])}
            summary[g]['standardized_distance_to_emopia'] = dists
            summary[g]['mean_standardized_distance'] = statistics.mean(dists.values()) if dists else float('nan')
    meta = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                bars=args.bars, groups=[g for g, _ in args.group],
                files={g: sum(1 for r in per_file if r['group'] == g) for g, _ in args.group},
                reference_files=len(ref_rows),
                completion='complete_bars needs actual generation stop evidence or retained final EOS; reached_target_bar alone is not completion',
                boundary='structural descriptors of symbolic scores; not musical quality or emotion')
    def finite_json(value):
        if isinstance(value, dict): return {k: finite_json(v) for k, v in value.items()}
        if isinstance(value, list): return [finite_json(v) for v in value]
        return None if isinstance(value, float) and not math.isfinite(value) else value
    (args.output_dir / 'summary.json').write_text(json.dumps(finite_json(dict(meta=meta, summary=summary)), indent=2, allow_nan=False))
    print(json.dumps(meta, indent=2))
    for g in [g for g in summary if '|' not in g]:
        s = summary[g]
        print(g, ' '.join(f"{m}={s[m]['mean']:.3f}" for m in METRICS),
              f"dist={s.get('mean_standardized_distance', float('nan')):.3f}")


if __name__ == '__main__':
    main()
