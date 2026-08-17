"""Pure helpers for factorized valence/arousal conditioning in Stage 3.

The legacy quadrant labels remain in the vocabulary for checkpoint compatibility.
Factorized datasets replace the leading ``Emotion_*`` event with one valence and
one arousal event so that high/low arousal is represented explicitly.
"""

CONDITIONING_MODES = ('legacy_quadrant', 'factorized_va')

FACTORIZED_TOKENS = (
    'Valence_Positive',
    'Valence_Negative',
    'Valence_None',
    'Arousal_High',
    'Arousal_Low',
    'Arousal_None',
)

QUADRANT_TO_FACTORS = {
    'Q1': ('Valence_Positive', 'Arousal_High'),
    'Q2': ('Valence_Negative', 'Arousal_High'),
    'Q3': ('Valence_Negative', 'Arousal_Low'),
    'Q4': ('Valence_Positive', 'Arousal_Low'),
    'None': ('Valence_None', 'Arousal_None'),
}


def factor_tokens(quadrant):
    """Return the valence/arousal token pair for a quadrant label."""
    try:
        return QUADRANT_TO_FACTORS[quadrant]
    except KeyError as exc:
        raise ValueError('unsupported quadrant: {}'.format(quadrant)) from exc


def primer_event_names(quadrant, key=None, mode='legacy_quadrant'):
    """Build Stage 3 primer names without importing Torch-dependent modules."""
    if mode not in CONDITIONING_MODES:
        raise ValueError('unsupported conditioning mode: {}'.format(mode))

    if mode == 'legacy_quadrant':
        names = ['Emotion_{}'.format(quadrant)]
    else:
        names = list(factor_tokens(quadrant))

    if key is not None:
        names.append(key)
    return names


def control_token_indices(event2idx):
    """Return control-only token indices that must not be sampled as music."""
    prefixes = ('Emotion_', 'Valence_', 'Arousal_')
    return tuple(sorted(
        index for token, index in event2idx.items()
        if token.startswith(prefixes)
    ))


def factorize_leading_emotion(events):
    """Replace the first event's ``Emotion`` label with two factor events.

    A fresh list and fresh factor dictionaries are returned. The remaining event
    dictionaries are copied to prevent accidental mutation of source samples.
    """
    if not events:
        raise ValueError('event sequence is empty')
    first = events[0]
    if first.get('name') != 'Emotion':
        raise ValueError('first event must be Emotion, got {!r}'.format(first))

    valence, arousal = factor_tokens(str(first.get('value')))
    valence_value = valence.split('_', 1)[1]
    arousal_value = arousal.split('_', 1)[1]
    return [
        {'name': 'Valence', 'value': valence_value},
        {'name': 'Arousal', 'value': arousal_value},
    ] + [dict(event) for event in events[1:]]


def append_factorized_vocabulary(event2idx, idx2event):
    """Append factor tokens while proving all existing token indices are stable."""
    new_event2idx = dict(event2idx)
    new_idx2event = dict(idx2event)
    if set(new_event2idx).intersection(FACTORIZED_TOKENS):
        raise ValueError('dictionary already contains factorized tokens')

    expected_indices = set(range(len(new_event2idx)))
    if set(new_event2idx.values()) != expected_indices:
        raise ValueError('legacy dictionary indices must be contiguous from zero')
    for token, index in new_event2idx.items():
        if new_idx2event.get(index) != token:
            raise ValueError('event2idx/idx2event disagree at {}'.format(token))

    for token in FACTORIZED_TOKENS:
        index = len(new_event2idx)
        new_event2idx[token] = index
        new_idx2event[index] = token
    return new_event2idx, new_idx2event
