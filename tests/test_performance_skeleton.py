import unittest

from performance_skeleton import (
    FORCED,
    SAMPLED,
    allowed_indices,
    bounded_allowed_indices,
    bounds_for_prefix,
    build_performance_bar_plan,
    discard_incomplete_identity_prefixes,
    validate_plan_vocabulary,
)


class PerformanceSkeletonTest(unittest.TestCase):
    def test_plan_copies_structure_and_inserts_only_attribute_slots(self):
        condition = [
            "Bar_None",
            "Beat_0",
            "Chord_Root_I",
            "Chord_Quality_M",
            "Note_Octave_4",
            "Note_Degree_V",
            "Beat_2",
            "Note_Octave_5",
            "Note_Degree_I",
        ]
        self.assertEqual(
            build_performance_bar_plan(condition),
            [
                (FORCED, "Bar_None"),
                (FORCED, "Beat_0"),
                (SAMPLED, "Tempo_"),
                (FORCED, "Chord_Root_I"),
                (FORCED, "Chord_Quality_M"),
                (FORCED, "Note_Octave_4"),
                (FORCED, "Note_Degree_V"),
                (SAMPLED, "Note_Duration_"),
                (SAMPLED, "Note_Velocity_"),
                (FORCED, "Beat_2"),
                (FORCED, "Note_Octave_5"),
                (FORCED, "Note_Degree_I"),
                (SAMPLED, "Note_Duration_"),
                (SAMPLED, "Note_Velocity_"),
            ],
        )

    def test_invalid_condition_grammar_is_rejected(self):
        with self.assertRaises(ValueError):
            build_performance_bar_plan(["Bar_None", "Note_Octave_4", "Beat_0"])
        with self.assertRaises(ValueError):
            build_performance_bar_plan(["Bar_None", "Beat_4", "Beat_2"])

    def test_incomplete_identity_prefixes_are_discarded_before_strict_planning(self):
        condition = [
            "Bar_None",
            "Beat_0",
            "Note_Octave_4",
            "Note_Octave_5",
            "Note_Degree_I",
            "Chord_Root_III",
            "Chord_Root_V",
            "Chord_Quality_M",
            "Note_Octave_7",
        ]
        cleaned, discarded = discard_incomplete_identity_prefixes(condition)
        self.assertEqual(
            cleaned,
            [
                "Bar_None",
                "Beat_0",
                "Note_Octave_5",
                "Note_Degree_I",
                "Chord_Root_V",
                "Chord_Quality_M",
            ],
        )
        self.assertEqual(
            discarded,
            [
                (2, "Note_Octave_4", "Note_Octave_5"),
                (5, "Chord_Root_III", "Chord_Root_V"),
                (8, "Note_Octave_7", "<EOF>"),
            ],
        )
        build_performance_bar_plan(cleaned)

    def test_vocabulary_validation_checks_forced_and_sampled_events(self):
        plan = [(FORCED, "Bar_None"), (SAMPLED, "Tempo_")]
        vocabulary = {"Bar_None": 0, "Tempo_Conti": 1, "Tempo_100": 2}
        validate_plan_vocabulary(plan, vocabulary)
        self.assertEqual(allowed_indices(vocabulary, "Tempo_"), [1, 2])
        with self.assertRaises(KeyError):
            validate_plan_vocabulary([(FORCED, "Beat_0")], vocabulary)
        with self.assertRaises(ValueError):
            allowed_indices(vocabulary, "Note_Duration_")

    def test_bounded_indices_keep_continuation_and_in_range_values(self):
        vocabulary = {
            "Tempo_Conti": 1,
            "Tempo_60": 2,
            "Tempo_100": 3,
            "Tempo_180": 4,
            "Beat_0": 5,
        }
        self.assertEqual(
            bounded_allowed_indices(vocabulary, "Tempo_", {"min": 80, "max": 160}),
            [1, 3],
        )
        self.assertEqual(
            bounds_for_prefix({"Tempo": {"min": 80, "max": 160}}, "Tempo_"),
            {"min": 80, "max": 160},
        )
        with self.assertRaisesRegex(ValueError, "no numeric vocabulary"):
            bounded_allowed_indices(vocabulary, "Tempo_", {"min": 300, "max": 400})

    def test_bounds_require_every_requested_attribute(self):
        with self.assertRaisesRegex(ValueError, "missing bounds"):
            bounds_for_prefix({}, "Note_Velocity_")


if __name__ == "__main__":
    unittest.main()
