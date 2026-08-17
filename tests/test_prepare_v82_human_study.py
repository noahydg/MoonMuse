from __future__ import annotations

import unittest

from scripts.prepare_v82_human_study import (
    FORMS,
    QUADRANTS,
    SEEDS,
    SYSTEMS,
    assigned_form,
    parse_reason_list,
    select_slots,
    stable_digest,
    validate_design,
)


class V82HumanStudyPreparationTest(unittest.TestCase):
    def test_reason_parser_accepts_frozen_semicolon_csv_and_legacy_lists(self):
        self.assertEqual(parse_reason_list(""), [])
        self.assertEqual(parse_reason_list("large_gap;duplicate"), ["large_gap", "duplicate"])
        self.assertEqual(parse_reason_list("['large_gap']"), ["large_gap"])

    def joined_rows(self):
        rows = []
        for seed in SEEDS:
            for quadrant in QUADRANTS:
                for sample_index in range(25):
                    digest = stable_digest("test", seed, quadrant, sample_index)
                    rows.append(
                        {
                            "seed": seed,
                            "quadrant": quadrant,
                            "sample_index": sample_index,
                            "selection_digest": digest,
                            "candidate_midi_sha256": f"c-{seed}-{quadrant}-{sample_index}",
                            "baseline_midi_sha256": f"b-{seed}-{quadrant}-{sample_index}",
                        }
                    )
        return rows

    def test_selection_is_two_lowest_digests_per_cell_and_order_invariant(self):
        rows = self.joined_rows()
        selected_a, audit_a = select_slots(rows)
        selected_b, audit_b = select_slots(list(reversed(rows)))
        keys_a = [
            (row["seed"], row["quadrant"], row["sample_index"])
            for row in selected_a
        ]
        keys_b = [
            (row["seed"], row["quadrant"], row["sample_index"])
            for row in selected_b
        ]
        self.assertEqual(keys_a, keys_b)
        self.assertEqual(len(selected_a), 24)
        self.assertEqual(sum(bool(row["selected"]) for row in audit_a), 24)
        self.assertEqual(audit_a, audit_b)

    def test_form_assignment_reverses_each_cell_pair(self):
        for rank in (1, 2):
            forms = {assigned_form(rank, system) for system in SYSTEMS}
            self.assertEqual(forms, set(FORMS))
        self.assertNotEqual(
            assigned_form(1, "moonmuse_v70"),
            assigned_form(2, "moonmuse_v70"),
        )

    def test_design_validation_accepts_exact_counterbalance(self):
        rows = []
        index = 0
        for seed in SEEDS:
            for quadrant in QUADRANTS:
                for rank in (1, 2):
                    index += 1
                    slot = f"SL{index:02d}"
                    for system in SYSTEMS:
                        rows.append(
                            {
                                "stimulus_id": f"X{len(rows):03d}",
                                "slot_id": slot,
                                "form": assigned_form(rank, system),
                                "system_private": system,
                                "seed": seed,
                                "quadrant": quadrant,
                            }
                        )
        report = validate_design(rows)
        self.assertEqual(report["stimuli"], 48)
        self.assertTrue(report["slot_system_and_form_reversal"])


if __name__ == "__main__":
    unittest.main()
