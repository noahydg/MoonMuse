"""Pure helpers for emotion-conditioned counterfactual generation."""

import hashlib


QUADRANTS = ('Q1', 'Q2', 'Q3', 'Q4')


def emotion_candidates_from_filename(filename, overrides=None):
    """Resolve Stage 3 emotion conditions without changing legacy defaults.

    ``overrides`` is intended for fixed-source counterfactual generation: the
    same melody can be rendered under two explicitly selected quadrants.  When
    no override is supplied, the historical filename-based behaviour is kept.
    """
    if overrides:
        invalid = [condition for condition in overrides if condition not in QUADRANTS]
        if invalid:
            raise ValueError('invalid emotion override(s): {}'.format(', '.join(invalid)))
        return list(dict.fromkeys(overrides))

    for quadrant in QUADRANTS:
        if quadrant in filename:
            return [quadrant]
    if 'Positive' in filename:
        return ['Q1', 'Q4']
    if 'Negative' in filename:
        return ['Q2', 'Q3']
    if 'None' in filename:
        return ['None']
    raise ValueError('wrong emotion label')


def paired_sampling_seed(base_seed, source_name):
    """Derive one stable seed per source, deliberately excluding condition.

    Reusing this seed for each condition aligns the random-number stream for a
    paired counterfactual while retaining a different stream for each source.
    """
    payload = '{}\0{}'.format(base_seed, source_name).encode('utf-8')
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:4], byteorder='big', signed=False)
