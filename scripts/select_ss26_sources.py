#!/usr/bin/env python3
"""Classify preserved source histories and rank candidates without reading scores."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized_digest(path):
    return digest(('\n'.join(line.strip() for line in path.read_text().splitlines() if line.strip()) + '\n').encode())


def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


class SourceIndex:
    def __init__(self, root):
        self.rows = list(csv.DictReader((root / 'source_manifest.csv').open()))
        self.hashes = defaultdict(set)
        self.coords = defaultdict(set)
        self.by_hash = {}
        for r in self.rows:
            p = root / r['frozen_path']
            h = digest(p.read_bytes())
            if h != r['source_sha256']:
                raise ValueError(f'Frozen source changed: {p}')
            self.by_hash[h] = r
            self.hashes[h].add(h)
            self.hashes[normalized_digest(p)].add(h)
            self.coords[int(r['seed']), r['quadrant'], int(r['sample_index'])].add(h)

    def match(self, row, path):
        # Explicit hashes are preferred. Newline-normalized aliases are needed
        # because old Linux texts and preserved CRLF copies are the same score.
        matched = set()
        for field, value in row.items():
            if 'sha256' in field:
                matched.update(self.hashes.get(str(value), set()))
        if matched:
            return matched, 'content_hash_or_newline_alias'
        system = str(row.get('system_private') or row.get('system_id') or row.get('system') or '')
        if system and system not in {'full', 'moonmuse_full', 'moonmuse_v70', 'v1_full', 'v69', 'v70', 'moonmuse_v63'}:
            return set(), 'other_system'
        # Only identity fields: never inspect classifier scores or human ratings.
        identity = ' '.join(str(v) for k, v in row.items()
                            if any(x in k for x in ('source', 'file', 'path', 'locator')) and 'sha256' not in k)
        seed = row.get('seed_private') or row.get('seed')
        if not str(seed).isdigit():
            m = re.search(r'seed(42|43|44)(?:\D|$)', identity)
            seed = m.group(1) if m else None
        q = row.get('source_quadrant') or row.get('quadrant_private') or row.get('quadrant') or row.get('target_quadrant')
        idx = row.get('sample_index_private') or row.get('sample_index') or row.get('source_index')
        m = re.search(r'samp_(\d+)_(?:Positive|Negative)_(Q[1-4])_melody', identity)
        if m:
            idx, q = m.groups()
        # Hash prefixes occur in renamed fixed-source performances.
        prefixes = re.findall(r'source_Q[1-4]_([0-9a-f]{12})_melody', identity)
        for prefix in prefixes:
            for h, values in self.hashes.items():
                if h.startswith(prefix):
                    matched.update(values)
        if matched:
            return matched, 'unique_content_hash_prefix'
        old_family = any(t in str(path) for t in ('listening-study-v1', 'v82-human-', 'v84-full-', 'va-blind-', 'q4-ending-', 'stage2-motif-rerank-', 'v2-melodic-hook-'))
        if seed and q and str(idx).isdigit() and (old_family or m):
            return self.coords.get((int(seed), q, int(idx)), set()), 'original_population_coordinates'
        # This early listening pack omits seed. Exclude all possible seed matches
        # instead of calling any one of them unheard.
        if m and 'quality-inference-listening-precheck-' in str(path):
            for (s, cq, ci), hs in self.coords.items():
                if cq == q and ci == int(idx):
                    matched.update(hs)
            return matched, 'conservative_missing_seed'
        return set(), 'no_frozen_source_match'


def history_files(results):
    files = {}
    for pattern, level, reason in [
        ('**/master_manifest_private.csv', 'H2', 'included in a prepared listening/author-preview set'),
        ('**/private_mapping.csv', 'H2', 'included in a prepared listening/author-preview set'),
        ('**/source_manifest.json', 'H1', 'selected for development or diagnostic generation'),
        ('v2-melodic-hook-*/per_file_private.csv', 'H1', 'individual diagnostic material'),
        ('stage2-motif-rerank-*/ranked_candidates_private.csv', 'AUTO', 'population-level automatic ranking'),
        ('gems-idyom-extension-*/**/derived_manifest.csv', 'AUTO', 'population-level automatic analysis'),
        ('v81-v70-baseline-superiority-*/population_manifest.csv', 'AUTO', 'population-level automatic screen'),
    ]:
        for p in results.glob(pattern):
            if not any('ss26-' in part or 'synthetic' in part for part in p.parts):
                files[p] = (level, reason)
    return files


def select(project, output):
    results = project / 'results'
    freeze = results / 'v81-v70-source-freeze-20260809'
    index = SourceIndex(freeze)
    output.mkdir(parents=True, exist_ok=False)
    uses, inventory = [], []
    levels = {h: 'H0' for h in index.by_hash}
    auto = Counter()
    for path, (level, reason) in sorted(history_files(results).items()):
        data = json.loads(path.read_text()).get('sources', []) if path.suffix == '.json' else list(csv.DictReader(path.open()))
        mapped = 0
        for n, row in enumerate(data, 1):
            matches, method = index.match(row, path)
            mapped += bool(matches)
            for h in sorted(matches):
                uses.append(dict(source_sha256=h, history_class=level, manifest=str(path.relative_to(project)), row=n, match_method=method, reason=reason))
                if level == 'AUTO':
                    auto[h] += 1
                elif level > levels[h]:
                    levels[h] = level
        inventory.append(dict(path=str(path.relative_to(project)), sha256=digest(path.read_bytes()), history_class=level, rows=len(data), matched_rows=mapped, reason=reason))
    classified = []
    for r in index.rows:
        h = r['source_sha256']
        classified.append({**r, 'history_class': levels[h], 'automatic_history_records': auto[h],
                           'ranking_digest': digest(('SS26|' + h).encode())})
    eligible = sorted((r.copy() for r in classified if r['history_class'] == 'H0'),
                      key=lambda r: (int(r['seed']), r['quadrant'], r['ranking_digest']))
    ranks = Counter()
    for r in eligible:
        cell = r['seed'], r['quadrant']; ranks[cell] += 1; r['rank_within_cell'] = ranks[cell]
    fields = list(classified[0])
    write_csv(output / 'source_history.csv', classified, fields)
    write_csv(output / 'eligible_ranked.csv', eligible, fields + ['rank_within_cell'])
    write_csv(output / 'history_evidence.csv', uses, ['source_sha256','history_class','manifest','row','match_method','reason'])
    (output / 'history_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    excluded = {h for h, level in levels.items() if level != 'H0'}
    assert not excluded.intersection(r['source_sha256'] for r in eligible)
    report = dict(status='PASS', formalStudyData=False, source_count=len(classified),
                  counts=dict(Counter(levels.values())),
                  eligible_per_cell={f'seed{s}-{q}':ranks[s,q] for s in ('42','43','44') for q in ('Q1','Q2','Q3','Q4')},
                  eligible_excluded_intersection=0, history_manifests=len(inventory),
                  script_sha256=digest(Path(__file__).read_bytes()),
                  ranking='ascending sha256(SS26|original_source_sha256) within seed/quadrant; no score filtering',
                  scope='H0 means no matched development-selection or prepared-listening record in the inventoried local manifests; not independently new or proven unheard. Later population-level automatic analyses are retained as AUTO history.',
                  H2='conservative exclusion of materials prepared for listening; does not assert every file was actually heard',
                  source_root=str(freeze.resolve()))
    (output / 'source_classification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(select(a.project.resolve(), a.output.resolve()), indent=2))
