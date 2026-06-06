'''Semantic sequence compression adapter inspired by SmartGen SSC/SPPC.

The first offline version uses bag-of-control vectors and cosine similarity. It
keeps the interface simple so a later implementation can swap in SmartGen's
Transformer/SPPC embedding without changing downstream code.
'''

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List

from causal_gen_guard.data.schemas import BehaviorSequence


def encode_sequence_for_compression(sequence: BehaviorSequence) -> Dict[str, object]:
    '''Encode a sequence as a normalized bag-of-control representation.'''
    counts = Counter(str(event.control_id) for event in sequence.events)
    norm = math.sqrt(sum(value * value for value in counts.values())) or 1.0
    vector = {key: value / norm for key, value in sorted(counts.items())}
    return {
        'sequence_id': sequence.sequence_id,
        'length': len(sequence.events),
        'counts': dict(counts),
        'vector': vector,
        'control_order': [str(event.control_id) for event in sequence.events],
    }


def _cosine_from_vectors(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(key, 0.0) for key, value in left.items())


def compress_sequences_by_similarity(sequences: List[BehaviorSequence], threshold: float) -> List[BehaviorSequence]:
    '''Greedily keep sequences whose bag-of-control similarity is below threshold.'''
    if threshold < 0.0 or threshold > 1.0:
        raise ValueError('threshold must be in [0, 1]')
    kept: List[BehaviorSequence] = []
    kept_vectors: List[Dict[str, float]] = []
    for sequence in sequences:
        encoded = encode_sequence_for_compression(sequence)
        vector = encoded['vector']
        assert isinstance(vector, dict)
        if all(_cosine_from_vectors(vector, previous) < threshold for previous in kept_vectors):
            context = dict(sequence.context)
            context['compression_adapter'] = 'smartgen_ssc_bag_cosine'
            context['compression_threshold'] = threshold
            kept.append(
                BehaviorSequence(
                    sequence_id=sequence.sequence_id,
                    events=list(sequence.events),
                    context=context,
                    label=sequence.label,
                    anomaly_type=sequence.anomaly_type,
                )
            )
            kept_vectors.append(vector)
    return kept
