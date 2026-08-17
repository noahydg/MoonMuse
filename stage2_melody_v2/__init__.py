"""Architecture-level Stage 2 components for the isolated MoonMuse V2 branch."""

from .motif_memory import (
    CellMotif,
    MelodyOnset,
    MotifComparison,
    cadence_target,
    compare_motifs,
    extract_cell_motifs,
    parse_melody_onsets,
)
from .planner import build_eight_bar_plan
from .factorized_control import (
    AffectTarget,
    ArousalTarget,
    DevelopmentDegree,
    EncodedFactorizedControl,
    FactorizedControlPlan,
    FactorizedControlSlot,
    MotifOperation,
    ValenceTarget,
    build_factorized_eight_bar_control,
    encode_factorized_control,
    factorized_control_from_phrase_plan,
)
from .rhythm_identity_diagnostic import (
    RhythmIdentityDiagnosticInterface,
    RhythmIdentityPairDiagnostic,
    RhythmIdentityScores,
    build_rhythm_identity_diagnostic_interface,
    normalize_rhythm_signature,
)
from .schema import (
    CadenceGoal,
    MotifRelation,
    PhrasePlan,
    PhraseRole,
    PhraseSlot,
)

__all__ = [
    "CadenceGoal",
    "AffectTarget",
    "ArousalTarget",
    "CellMotif",
    "DevelopmentDegree",
    "EncodedFactorizedControl",
    "FactorizedControlPlan",
    "FactorizedControlSlot",
    "MelodyOnset",
    "MotifOperation",
    "MotifComparison",
    "MotifRelation",
    "PhrasePlan",
    "PhraseRole",
    "PhraseSlot",
    "RhythmIdentityDiagnosticInterface",
    "RhythmIdentityPairDiagnostic",
    "RhythmIdentityScores",
    "build_eight_bar_plan",
    "build_factorized_eight_bar_control",
    "build_rhythm_identity_diagnostic_interface",
    "cadence_target",
    "compare_motifs",
    "extract_cell_motifs",
    "encode_factorized_control",
    "factorized_control_from_phrase_plan",
    "parse_melody_onsets",
    "normalize_rhythm_signature",
    "ValenceTarget",
]
