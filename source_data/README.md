# Manuscript source data

These tables contain reported aggregate results and stimulus-level numerical
measurements. No file contains participant identifiers or individual responses.

| Manuscript result | Files |
| --- | --- |
| 78-listener system comparison; Fig. 3 | `listening_n78_summary.json`, `listening_tradeoff_system_summary.csv`, `listening_tradeoff_effect_intervals.csv` |
| Twelve fixed-source pairs; Fig. 4 | `stage3_directional_effects_pairs.csv`, `stage3_directional_effects_invariants.csv` |
| Matched alignment comparison | `matched_alignment_arousal_gaps.csv`, `matched_alignment_gap_contrasts.csv` |
| 69-listener same-score and system comparisons; Fig. 5 | `listening_n69_summary.json`, `fig5_same_score_listening.json`, `listening_n69_source_estimates.csv` |
| Playback-duration sensitivity | `listening_n69_playback_sensitivity.json` |
| Fixed-content evaluator audit; Fig. 6 | `automatic_arousal_confounds_panel_c_counterfactuals.csv`, `automatic_arousal_confounds_panel_c_summary.csv`, `fig6_factorial_predictions.csv`, `fig6_fingerprint_medians.csv` |
| Exploratory classifier/listener comparison | `evaluator_listening_source_contrasts.csv` |

The 69-listener analyses use equal quadrant/source-cell weighting. The main
intervals are `cluster_sum_HC1`; `two_way_HC1` is the sensitivity estimate.
In the same-score summaries, `model` means MoonMuse and `other` means the
source-matched rule. In the system block, `other` means EMO-Disentangler.
`moonmuse_v70` is the historical full-model label for the original 78-listener
study. Source hashes identify musical stimuli, not people.

Tempo is measured in BPM, velocity in MIDI units, and arousal ratings on the
seven-point scale. The playback threshold measures cumulative time including
repeats, not unique temporal coverage. Missing/unestimable results are retained
as such. See the paper and analysis scripts for estimands and inclusion rules.

Third-party dataset access and licensing remain separate; see
[`DATA_LICENSES.md`](../DATA_LICENSES.md). Cite the associated MoonMuse paper when
using these study results.
