# MoonMuse

## Manuscript materials (November 2026)

This snapshot accompanies *MoonMuse: Content-Preserving Performance Interventions
for Attributable Arousal Control in Symbolic Piano Generation*. It adds the
key-preserving generation updates, matched alignment experiments, fixed-cohort
listening analyses (78 and 69 listeners), and evaluator-audit tools used in the
current manuscript. The experimental Stage 2 V2 directory remains separate from
the paper's three-stage system.

Aggregate listening-study results and processed figure tables are in
[`source_data/`](source_data/README.md). These are numerical research outputs;
raw questionnaire responses, participant identifiers, training datasets and
model checkpoints are not part of this release.

### Analysis entry points

Run scripts from the repository root with `python scripts/<name>.py --help`:

| Analysis | Script |
| --- | --- |
| Original 78-listener comparison | `analyze_v82_human_study.py`, `summarize_v82_manuscript_results.py` |
| Twelve fixed-source intervention pairs | `prepare_stage3_counterfactual_sources.py`, `analyze_stage3_arousal_counterfactual.py` |
| Matched alignment training and generated comparisons | `run_ss26_matched_alignment.py`, `analyze_ss26_comparison.py` |
| Final 69-listener comparison | `analyze_ss26_returnee_final.py` |
| Cumulative playback sensitivity | `analyze_ss26_playback_sensitivity.py` |
| Fixed-content evaluator perturbations | `audit_emopia_arousal_counterfactual_v80.py` |
| Classifier and listening-stimulus pairing | `evaluate_ss26_new_score_classifier.py`, `pair_ss26_listeners_with_classifier.py` |

Training/inference require the original licensed datasets, checkpoints and
experiment workspace. Historical campaign launchers retain their original
server-layout checks. Participant-level reanalysis additionally requires the
controlled-access study files; the aggregate tables alone cannot reproduce
participant-clustered confidence intervals. Requests for those materials follow
the manuscript's Data Availability statement.

For CPU analysis tests, install `requirements-analysis.txt`, then run:

```bash
python -m pytest -q tests/test_ss26_descriptive_analysis.py tests/test_ss26_key_and_level_matching.py tests/test_ss26_evaluator_pairing.py
```


MoonMuse is a three-stage symbolic piano generation pipeline for affect-conditioned chord, melody, and performance generation.

```text
emotion + key
    -> Stage 1: chord generation
    -> Stage 2: melody generation
    -> Stage 3: performance generation
    -> MIDI
```

## Project structure

- `stage1_chord/`: functional chord generation.
- `stage2_melody/`: chord-conditioned melody generation.
- `stage3_performance/`: melody-conditioned performance generation.
- `stage2_melody_v2/`: experimental structured-melody components.
- `representations/`: MIDI/event conversion, task extraction, vocabularies, and data splits.
- `scripts/`: data preparation, training, generation, validation, evaluation, and listening-study utilities.
- `tests/`: core unit and pipeline tests.

The root modules implement shared conditioning, generation budgets, score locking, and Stage 3 affect-control objectives.

## Environment

```bash
conda create -n moonmuse python=3.9
conda activate moonmuse
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data

Datasets and model checkpoints are not included. Place local artifacts in the following ignored directories:

```text
midi_data/
events/
new_events/
ckpt/
generation/
```

See `DATA_LICENSES.md` before downloading, processing, or redistributing dataset-derived files.

Prepare task-specific event data from a compatible processed archive:

```bash
python scripts/prepare_data_from_official.py \
  --official-events-zip /path/to/events.zip

python scripts/rebuild_content_isolated_split.py \
  --dataset-root events/chord/hooktheory_chord_functional

python scripts/audit_data.py \
  --events-root events \
  --split-dir-name data_splits_content_isolated \
  --fallback-split-dir-name data_splits \
  --output-dir generation/data-audit \
  --strict
```

## Training

Run a bounded six-phase pilot:

```bash
PRETRAIN_EPOCHS=3 FINETUNE_EPOCHS=5 SEED=42 WORKERS=4 \
  bash scripts/run_formal_pilot.sh
```

Run one full training seed:

```bash
SEED=42 bash scripts/run_formal_seed.sh
```

Run the remaining seeds after the first seed passes validation:

```bash
SEEDS="43 44" bash scripts/run_formal_campaign.sh
```

## Generation

Generate and validate a complete three-stage run:

```bash
SEED=42 bash scripts/run_formal_generation.sh \
  generation/formal-full-seed42-YYYYMMDD-HHMMSS
```

The generation runner invokes Stage 1, Stage 2, the Stage 2-to-Stage 3 connector, Stage 3, MIDI conversion, MIDI validation, and descriptor evaluation.

## Controlled performance generation

Stage 3 supports factorized valence/arousal conditioning and skeleton-locked generation. In skeleton-locked mode, score-level events are preserved while performance attributes such as tempo, duration, and velocity remain controllable.

The relevant implementation is in:

- `emotion_conditioning.py`
- `factorized_conditioning.py`
- `performance_skeleton.py`
- `stage3_arousal_alignment.py`
- `stage3_attribute_objective.py`
- `stage3_performance/inference.py`

## Listening-study utilities

The V82 utilities prepare blinded study materials, render globally normalized audio, build standalone questionnaires, validate a launch package, and analyze exported responses:

```text
scripts/prepare_v82_human_study.py
scripts/render_v82_global_gain_audio.py
scripts/build_v82_questionnaires.py
scripts/validate_v82_prelaunch.py
scripts/analyze_v82_human_study.py
```

Study launch mode requires a complete ethics record. Private mappings and participant response files must not be committed.

## Tests

```bash
python -m pytest -q
```

## License

The software is released under the MIT License. Dataset terms are separate and are described in `DATA_LICENSES.md`.
