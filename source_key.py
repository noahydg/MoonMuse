"""Read the declared source key without depending on affect-token order."""

VALID_KEYS = {n for tonic in ('C','C#','D','D#','E','F','F#','G','G#','A','A#','B') for n in (tonic,tonic.lower())}


def declared_source_key(events):
    keys = {event.strip() for event in events if event.strip().startswith('Key_')}
    if len(keys) != 1:
        raise ValueError(f'Expected one unambiguous source key, found {sorted(keys)}')
    key = next(iter(keys))
    if key[4:] not in VALID_KEYS:
        raise ValueError(f'Unsupported source key: {key}')
    return key
