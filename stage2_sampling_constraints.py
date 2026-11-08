"""Optional functional-event pair constraints; no edits to accepted output."""
from stage2_event_validation import DEGREES, QUALITIES
from generation_budget import OnsetPitchTracker


class FunctionalMelodyState:
    """Optional structured-v2 state for sampled melody-track events.

    Key/emotion and forced harmony belong to the conditioning prefix. This
    state sees sampled, accepted events only, not inserted chord-track tokens.
    It permits empty bars and repeated pitches at different onsets; it never
    forces a musical ending when the sampling budget runs out.
    """

    def __init__(self):
        self.bar_started = False
        self.beat_started = False
        self.pitches = OnsetPitchTracker()

    def inadmissibles(self, event2idx, final_bar, base=None):
        blocked = set(self.pitches.inadmissibles(event2idx, base=base) or [])
        for event, token in event2idx.items():
            if event == 'Bar_None':
                allowed = not self.bar_started
            elif event == 'Track_Chord':
                allowed = self.bar_started
            elif event == 'EOS_None':
                allowed = self.bar_started and final_bar
            elif event.startswith('Beat_'):
                try: position = int(event[5:])
                except ValueError: position = -1
                allowed = self.bar_started and 0 <= position < 16
            elif event.startswith('Note_Octave_'):
                # Do not enter a pair whose every degree is already blocked.
                available = [i for name, i in event2idx.items()
                             if name.startswith('Note_Degree_') and name[12:] in DEGREES
                             and i not in blocked
                             and (event, name) not in self.pitches.functional_pitches]
                allowed = self.beat_started and bool(available)
            elif event.startswith(('Note_Degree_', 'Chord_Root_', 'Chord_Quality_')):
                allowed = self.beat_started
            else:
                # In particular Key_*, Emotion_*, PAD and Track_Melody cannot
                # be resampled inside the already-open melody track.
                allowed = False
            if not allowed:
                blocked.add(token)
        return sorted(blocked) or None

    def observe(self, event):
        if event == 'Track_Chord':
            self.bar_started = self.beat_started = False
            self.pitches.reset()
            return
        if event == 'Bar_None': self.bar_started = True
        if event.startswith('Beat_'): self.beat_started = True
        self.pitches.observe(event)


def pair_inadmissibles(event2idx, previous_event, remaining_events,
                      remaining_attempts=None, base=None):
    """Keep octave/degree and root/quality adjacent, within both budgets.

    This is a pair constraint, not a complete musical grammar. The independent
    output validator still decides whether a candidate is valid and complete.
    ``remaining_attempts`` includes the sampling attempt about to be made.
    """
    blocked = set(base or [])
    note_pending = previous_event.startswith('Note_Octave_')
    root = previous_event[11:] if previous_event.startswith('Chord_Root_') else None
    pair_fits = remaining_events >= 2 and (
        remaining_attempts is None or remaining_attempts >= 2)
    for event, token in event2idx.items():
        if note_pending:
            allowed = event.startswith('Note_Degree_') and event[12:] in DEGREES
        elif root is not None:
            quality = event[14:] if event.startswith('Chord_Quality_') else None
            allowed = quality in QUALITIES and (
                quality == root if root in {'Conti', 'None'}
                else root in DEGREES and quality not in {'Conti', 'None'})
        elif event.startswith(('Note_Degree_', 'Chord_Quality_')):
            allowed = False
        elif event.startswith('Note_Octave_'):
            allowed = pair_fits and event[12:] in {str(i) for i in range(1, 10)}
        elif event.startswith('Chord_Root_'):
            allowed = pair_fits and event[11:] in DEGREES | {'Conti', 'None'}
        else:
            allowed = True
        if not allowed:
            blocked.add(token)
    return sorted(blocked) or None
