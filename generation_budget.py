class RejectionBudget:
    """Bound cumulative rejected sampling attempts for one generated candidate."""

    def __init__(self, maximum=None):
        if maximum is not None and maximum <= 0:
            raise ValueError("maximum must be positive or None")
        self.maximum = maximum
        self.rejections = 0

    def record(self):
        self.rejections += 1
        return self.maximum is not None and self.rejections >= self.maximum


class SegmentSamplingBudget:
    """Bound all model sampling attempts within one generated track segment."""

    def __init__(self, maximum=None):
        if maximum is not None and maximum <= 0:
            raise ValueError("maximum must be positive or None")
        self.maximum = maximum
        self.attempts = 0

    @property
    def exhausted(self):
        return self.maximum is not None and self.attempts >= self.maximum

    def record(self):
        if self.exhausted:
            raise RuntimeError("sampling budget is already exhausted")
        self.attempts += 1

    def reset(self):
        self.attempts = 0


def monotonic_beat_inadmissibles(event2idx, current_position, base=None):
    """Return token ids that would move a generated bar backwards in time."""
    inadmissibles = set(base or [])
    for event, token_id in event2idx.items():
        if not event.startswith("Beat_"):
            continue
        try:
            position = int(event.rsplit("_", 1)[1])
        except ValueError:
            continue
        if position < current_position:
            inadmissibles.add(token_id)
    return sorted(inadmissibles) or None


def strip_terminal_eos(generated, idx2event):
    """Remove a sampled EOS without discarding a real final bar boundary."""
    if generated and idx2event[generated[-1]] == "EOS_None":
        return generated[:-1]
    return generated


class OnsetPitchTracker:
    """Track pitches already emitted at the current onset.

    REMI events encode absolute notes as ``Note_Pitch_*`` and functional notes
    as an adjacent ``Note_Octave_*`` / ``Note_Degree_*`` pair.  The tracker
    exposes token ids that would repeat an exact pitch without changing the
    onset.  It does not edit an accepted event sequence or converted MIDI.
    """

    def __init__(self):
        self.current_beat = None
        self.absolute_pitches = set()
        self.functional_pitches = set()
        self.pending_octave = None

    def reset(self):
        self.current_beat = None
        self._reset_onset()

    def _reset_onset(self):
        self.absolute_pitches.clear()
        self.functional_pitches.clear()
        self.pending_octave = None

    def inadmissibles(self, event2idx, base=None):
        """Return base masks plus exact pitches already used at this onset."""
        inadmissibles = set(base or [])
        for event in self.absolute_pitches:
            token_id = event2idx.get(event)
            if token_id is not None:
                inadmissibles.add(token_id)
        if self.pending_octave is not None:
            for octave, degree in self.functional_pitches:
                if octave != self.pending_octave:
                    continue
                token_id = event2idx.get(degree)
                if token_id is not None:
                    inadmissibles.add(token_id)
        return sorted(inadmissibles) or None

    def observe(self, event):
        """Update state after one sampled event has passed syntax checks."""
        if event.startswith("Beat_"):
            try:
                position = int(event.rsplit("_", 1)[1])
            except ValueError:
                self.pending_octave = None
                return
            if self.current_beat is None or position > self.current_beat:
                self._reset_onset()
            else:
                self.pending_octave = None
            self.current_beat = position
            return

        if event == "Bar_None" or event.startswith("Track_"):
            self.reset()
            return

        if event.startswith("Note_Pitch_"):
            self.pending_octave = None
            self.absolute_pitches.add(event)
            return

        if event.startswith("Note_Octave_"):
            self.pending_octave = event
            return

        if event.startswith("Note_Degree_"):
            if self.pending_octave is not None:
                self.functional_pitches.add((self.pending_octave, event))
            self.pending_octave = None
            return

        # A functional pitch is valid only for an adjacent octave/degree pair.
        self.pending_octave = None
