"""Prospective plan-level partitions, preserving prior melody exposure records."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path


def read(path):
    with Path(path).open() as f: return list(csv.DictReader(f))


def write(path, rows):
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def family(row): return row['seed'], row['valence'].lower(), row['sample_index']


def rank(key):
    return hashlib.sha256(('SS26-plan-v2|' + '|'.join(key)).encode()).hexdigest()


def partition(history, roles):
    all_families = defaultdict(list)
    for row in history: all_families[family(row)].append(row)
    calibration = {family(row) for row in roles if row['role'] == 'calibration'}
    eligible = defaultdict(list)
    for row in roles:
        if row['role'] == 'held_out_candidate' and family(row) not in calibration:
            eligible[family(row)].append(row)
    used = set(); selected = []
    for seed in ('42', '43', '44'):
        for valence, quadrants in [('positive', ('Q1', 'Q4')), ('negative', ('Q2', 'Q3'))]:
            keys = sorted([k for k in eligible if k[:2] == (seed, valence)], key=rank)
            for quadrant in quadrants:
                for _ in range(2):
                    key = next(k for k in keys if k not in used and any(r['quadrant'] == quadrant for r in eligible[k]))
                    row = next(r for r in eligible[key] if r['quadrant'] == quadrant)
                    selected.append(dict(row, pool='system', plan_id='|'.join(key)))
                    used.add(key)
            remaining = [k for k in keys if k not in used]
            assert len(remaining) >= 7
            native_counts = Counter()
            for key in remaining[:7]:
                row = min(eligible[key], key=lambda r:(native_counts[r['quadrant']],r['ranking_digest']))
                native_counts[row['quadrant']] += 1
                selected.append(dict(row, pool='within', plan_id='|'.join(key)))
                used.add(key)
    assert len(used) == len(selected) == 66
    assert not used & calibration
    for row in selected:
        members = all_families[family(row)]
        row['plan_has_prior_H1_H2_sibling'] = any(r['history_class'] != 'H0' for r in members)
        row['plan_rank'] = rank(family(row))
    strict = [r for key in eligible if all(x['history_class'] == 'H0' for x in all_families[key])
              for r in eligible[key]]
    summary = dict(status='CONDITIONAL_PLAN_GROUPED_CANDIDATES', formalStudyData=False,
        prior_calibration_plans=len(calibration), candidate_plans_before_selection=len(eligible),
        selected_plans=66, within_sources=42, system_slots=24, reserved_candidate_plans=len(eligible)-66,
        cross_calibration_plans=0, cross_block_plans=0, multiple_melodies_per_selected_plan=0,
        selected_with_prior_H1_H2_sibling=sum(r['plan_has_prior_H1_H2_sibling'] for r in selected),
        strict_whole_plan_history_clear_candidates=len({family(r) for r in strict}),
        boundary='Main option retains earlier H0 eligibility at melody level and discloses sibling history; strict whole-plan option is separate. Source policy remains prospective.')
    return selected, strict, summary


def main(root, out):
    out.mkdir(exist_ok=False)
    selected, strict, summary = partition(read(root/'results/ss26-sources-20260927/source_history.csv'),
        read(root/'results/ss26-level-matched-rule-20260928-r2/source_roles.csv'))
    write(out/'candidate_slots.csv', selected)
    write(out/'strict_whole_plan_candidates.csv', strict)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.root,a.out)
