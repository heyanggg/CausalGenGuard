'''Causal-TOF filtering for offline SmartGen synthetic normal sequences.

Causal-TOF chains lightweight legality checks, SmartGuard reconstruction
filtering, causal graph deviation filtering, and a placeholder utility-selection
stage. It is offline by design: no LLM API calls, no key requirements, and no
model training is launched by default.
'''

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence


RejectedRecord = Dict[str, Any]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return _json_safe(value.tolist())
        except Exception:
            pass
    return value


def _control_key(control_id: Any) -> Any:
    if hasattr(control_id, 'item'):
        try:
            control_id = control_id.item()
        except Exception:
            pass
    try:
        hash(control_id)
    except TypeError:
        return repr(control_id)
    return control_id


def _device_key(event: BehaviorEvent) -> Any:
    if event.device_id is not None:
        return event.device_id
    for key in ('device', 'device_id', 'persistence_or_device'):
        if key in event.raw_fields:
            return event.raw_fields[key]
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _sequence_payload(sequence: BehaviorSequence) -> dict[str, Any]:
    try:
        return sequence.to_dict()
    except Exception:
        return {
            'sequence_id': sequence.sequence_id,
            'events': [event.to_dict() for event in sequence.events if getattr(event, 'control_id', None) is not None],
            'context': dict(sequence.context),
            'label': _json_safe(sequence.label),
            'anomaly_type': sequence.anomaly_type,
        }


def _reject(sequence: BehaviorSequence, stage: str, reason: str, **extra: Any) -> RejectedRecord:
    record = {
        'sequence_id': sequence.sequence_id,
        'reject_stage': stage,
        'reject_reason': reason,
        'sequence': _sequence_payload(sequence),
    }
    record.update({key: _json_safe(value) for key, value in extra.items()})
    return record


def _allowed_controls_for_event(event: BehaviorEvent, device_action_map: Any) -> Optional[set[Any]]:
    if device_action_map is None:
        return None
    if isinstance(device_action_map, (list, tuple, set)):
        return {_control_key(item) for item in device_action_map}
    if not isinstance(device_action_map, dict):
        return None

    device = _device_key(event)
    if device in device_action_map:
        allowed = device_action_map[device]
    elif str(device) in device_action_map:
        allowed = device_action_map[str(device)]
    elif '__all__' in device_action_map:
        allowed = device_action_map['__all__']
    else:
        values = []
        for item in device_action_map.values():
            if isinstance(item, (list, tuple, set)):
                values.extend(item)
            else:
                values.append(item)
        allowed = values if values else list(device_action_map.keys())
    if isinstance(allowed, (list, tuple, set)):
        return {_control_key(item) for item in allowed}
    return {_control_key(allowed)}


def legality_filter(
    sequences: List[BehaviorSequence],
    device_action_map: Any = None,
    min_length: int = 2,
    max_dominant_ratio: float = 0.85,
) -> Tuple[List[BehaviorSequence], List[RejectedRecord]]:
    '''Filter empty, too-short, repetitive, or illegal-control sequences.'''
    kept: List[BehaviorSequence] = []
    rejected: List[RejectedRecord] = []
    for sequence in sequences:
        if not sequence.events:
            rejected.append(_reject(sequence, 'legality', 'empty_sequence'))
            continue
        if len(sequence.events) < min_length:
            rejected.append(_reject(sequence, 'legality', 'too_short_sequence', length=len(sequence.events)))
            continue

        illegal_control = None
        for event in sequence.events:
            allowed = _allowed_controls_for_event(event, device_action_map)
            control = _control_key(event.control_id)
            if allowed is not None and control not in allowed:
                illegal_control = {'control_id': control, 'device_id': _device_key(event)}
                break
        if illegal_control is not None:
            rejected.append(_reject(sequence, 'legality', 'illegal_control', **illegal_control))
            continue

        controls = [_control_key(event.control_id) for event in sequence.events]
        dominant_count = Counter(controls).most_common(1)[0][1]
        dominant_ratio = dominant_count / max(len(controls), 1)
        if len(controls) >= 4 and dominant_ratio > max_dominant_ratio:
            rejected.append(
                _reject(
                    sequence,
                    'legality',
                    'repeated_control_dominates_sequence',
                    dominant_ratio=dominant_ratio,
                    max_dominant_ratio=max_dominant_ratio,
                )
            )
            continue
        kept.append(sequence)
    return kept, rejected


def _control_mapping_for_backbone(sequences: List[BehaviorSequence], vocab_size: int, backbone: Any) -> Tuple[Dict[Any, int], Optional[str]]:
    for attr in ('control_vocab', 'vocab'):
        mapping = getattr(backbone, attr, None)
        if isinstance(mapping, dict):
            return {_control_key(key): int(value) for key, value in mapping.items()}, None

    controls: List[Any] = []
    direct_ok = True
    for sequence in sequences:
        for event in sequence.events:
            control = _control_key(event.control_id)
            controls.append(control)
            if not isinstance(control, int) or control < 0 or control >= vocab_size:
                direct_ok = False
    if direct_ok:
        return {control: int(control) for control in sorted(set(controls))}, None

    mapping: Dict[Any, int] = {}
    for control in controls:
        if control not in mapping:
            mapping[control] = len(mapping)
    if len(mapping) > vocab_size:
        return mapping, f'candidate control vocabulary size {len(mapping)} exceeds backbone vocab_size {vocab_size}'
    return mapping, 'controls_remapped_locally_for_backbone'


def _sequence_to_backbone_tensor(sequence: BehaviorSequence, control_map: Dict[Any, int], vocab_size: int) -> Tuple[Any, Any]:
    import torch

    fields = torch.zeros(1, len(sequence.events), 4, dtype=torch.float32)
    mask = torch.ones(1, len(sequence.events), dtype=torch.float32)
    for index, event in enumerate(sequence.events):
        control = _control_key(event.control_id)
        if control not in control_map:
            raise KeyError(f'control_id {control!r} is missing from backbone control mapping')
        control_index = int(control_map[control])
        if control_index < 0 or control_index >= vocab_size:
            raise ValueError(f'control index {control_index} is outside backbone vocab_size {vocab_size}')
        fields[0, index, 0] = _as_float(event.day)
        fields[0, index, 1] = _as_float(event.hour)
        fields[0, index, 2] = _as_float(event.duration)
        fields[0, index, 3] = float(control_index)
    return fields, mask


def reconstruction_filter(
    sequences: List[BehaviorSequence],
    backbone: Any,
    threshold: float,
) -> Tuple[List[BehaviorSequence], List[RejectedRecord]]:
    '''Filter sequences by SmartGuardBackbone mean reconstruction loss.'''
    if backbone is None:
        raise ValueError('backbone is required for reconstruction_filter')
    import torch

    vocab_size = int(getattr(backbone, 'vocab_size'))
    control_map, warning = _control_mapping_for_backbone(sequences, vocab_size, backbone)
    if warning and warning.startswith('candidate control vocabulary size'):
        return [], [_reject(sequence, 'reconstruction', warning) for sequence in sequences]

    was_training = backbone.training
    backbone.eval()
    kept: List[BehaviorSequence] = []
    rejected: List[RejectedRecord] = []
    with torch.no_grad():
        for sequence in sequences:
            try:
                fields, mask = _sequence_to_backbone_tensor(sequence, control_map, vocab_size)
                outputs = backbone(fields, attention_mask=mask)
                score = float((outputs['token_losses'].sum() / mask.sum().clamp_min(1.0)).item())
            except Exception as exc:
                rejected.append(_reject(sequence, 'reconstruction', f'reconstruction_failed: {exc}'))
                continue
            sequence.context['reconstruction_score'] = score
            if warning:
                sequence.context['reconstruction_warning'] = warning
            if score <= threshold:
                kept.append(sequence)
            else:
                rejected.append(_reject(sequence, 'reconstruction', 'reconstruction_score_above_threshold', score=score, threshold=threshold))
    if was_training:
        backbone.train()
    return kept, rejected


def causal_filter(
    sequences: List[BehaviorSequence],
    causal_model: Any,
    A_norm: Any,
    vocab: Dict[Any, int],
    threshold: float,
    window_size: Optional[int] = None,
) -> Tuple[List[BehaviorSequence], List[RejectedRecord]]:
    '''Filter sequences by gradient-causality deviation from A_norm.'''
    if causal_model is None:
        raise ValueError('causal_model is required for causal_filter')
    if A_norm is None:
        raise ValueError('A_norm is required for causal_filter')
    if vocab is None:
        raise ValueError('vocab is required for causal_filter')

    import torch
    from causal_gen_guard.data.behavior_event_tensor import sequence_to_event_tensor, sliding_windows_from_tensor
    from causal_gen_guard.models.causal_graph import causal_deviation_score, compute_gradient_causality, normal_causal_pattern, sparsify_causality

    kept: List[BehaviorSequence] = []
    rejected: List[RejectedRecord] = []
    was_training = causal_model.training
    causal_model.eval()
    for sequence in sequences:
        try:
            X, _, mask = sequence_to_event_tensor(sequence, vocab, max_len=None, include_time_features=False)
            valid_len = int(mask.sum())
            if valid_len < 2:
                rejected.append(_reject(sequence, 'causal', 'too_short_for_causal_window', length=valid_len))
                continue
            seq_window_size = window_size or min(4, valid_len - 1)
            windows, targets = sliding_windows_from_tensor(X[:valid_len], window_size=seq_window_size, pred_horizon=1)
            if windows.shape[0] == 0:
                rejected.append(_reject(sequence, 'causal', 'no_sliding_windows', length=valid_len, window_size=seq_window_size))
                continue
            graph_batch = compute_gradient_causality(
                causal_model,
                torch.from_numpy(windows).float(),
                torch.from_numpy(targets).float(),
            )
            graph = normal_causal_pattern(sparsify_causality(graph_batch, threshold=0.0))
            score_tensor = causal_deviation_score(graph, A_norm)
            score = float(score_tensor.detach().cpu().item() if hasattr(score_tensor, 'detach') else score_tensor)
        except Exception as exc:
            rejected.append(_reject(sequence, 'causal', f'causal_filter_failed: {exc}'))
            continue
        sequence.context['causal_deviation_score'] = score
        if score <= threshold:
            kept.append(sequence)
        else:
            rejected.append(_reject(sequence, 'causal', 'causal_deviation_above_threshold', score=score, threshold=threshold))
    if was_training:
        causal_model.train()
    return kept, rejected


def utility_selection(
    candidate_outliers: List[BehaviorSequence],
    train_sequences: Optional[List[BehaviorSequence]],
    val_sequences: Optional[List[BehaviorSequence]],
    config: Optional[dict[str, Any]],
) -> Tuple[List[BehaviorSequence], List[RejectedRecord], dict[str, Any]]:
    '''Placeholder utility selection stage.

    The full version would retrain or fine-tune with each candidate and reject it
    if validation reconstruction loss rises. That is intentionally skipped here
    to avoid hidden training cost. Candidates are kept and the report records the
    no-op decision unless config['enabled'] is explicitly implemented later.
    '''
    config = dict(config or {})
    log = {
        'enabled': bool(config.get('enabled', False)),
        'mode': 'no_op_interface',
        'reason': 'full utility retraining is intentionally not run in offline Causal-TOF stage',
        'train_sequence_count': len(train_sequences or []),
        'val_sequence_count': len(val_sequences or []),
        'candidate_count': len(candidate_outliers),
    }
    return list(candidate_outliers), [], log


def _write_jsonl(path: Path, rows: Iterable[Any]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            if isinstance(row, BehaviorSequence):
                payload = row.to_dict()
            else:
                payload = row
            handle.write(json.dumps(_json_safe(payload), ensure_ascii=False) + '\n')
            count += 1
    return count


def run_causal_tof(
    sequences: List[BehaviorSequence],
    output_dir: str | Path,
    device_action_map: Any = None,
    backbone: Any = None,
    reconstruction_threshold: Optional[float] = None,
    causal_model: Any = None,
    A_norm: Any = None,
    vocab: Optional[Dict[Any, int]] = None,
    causal_threshold: Optional[float] = None,
    train_sequences: Optional[List[BehaviorSequence]] = None,
    val_sequences: Optional[List[BehaviorSequence]] = None,
    utility_config: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    '''Run legality -> reconstruction -> causal -> utility and write artifacts.'''
    output_dir = Path(output_dir)
    all_rejected: List[RejectedRecord] = []
    report: dict[str, Any] = {
        'initial_count': len(sequences),
        'steps': {},
        'outputs': {
            'kept_jsonl': str(output_dir / 'kept.jsonl'),
            'rejected_jsonl': str(output_dir / 'rejected.jsonl'),
            'report_json': str(output_dir / 'report.json'),
        },
    }

    current, rejected = legality_filter(sequences, device_action_map=device_action_map)
    all_rejected.extend(rejected)
    report['steps']['legality'] = {'input': len(sequences), 'kept': len(current), 'rejected': len(rejected)}

    if backbone is not None and reconstruction_threshold is not None:
        before = len(current)
        current, rejected = reconstruction_filter(current, backbone, threshold=reconstruction_threshold)
        all_rejected.extend(rejected)
        report['steps']['reconstruction'] = {
            'input': before,
            'kept': len(current),
            'rejected': len(rejected),
            'threshold': reconstruction_threshold,
        }
    else:
        report['steps']['reconstruction'] = {'skipped': True, 'reason': 'missing backbone or threshold'}

    if causal_model is not None and A_norm is not None and vocab is not None and causal_threshold is not None:
        before = len(current)
        current, rejected = causal_filter(current, causal_model, A_norm, vocab, threshold=causal_threshold)
        all_rejected.extend(rejected)
        report['steps']['causal'] = {
            'input': before,
            'kept': len(current),
            'rejected': len(rejected),
            'threshold': causal_threshold,
        }
    else:
        report['steps']['causal'] = {'skipped': True, 'reason': 'missing causal_model, A_norm, vocab, or threshold'}

    before = len(current)
    current, rejected, utility_log = utility_selection(current, train_sequences, val_sequences, utility_config)
    all_rejected.extend(rejected)
    report['steps']['utility'] = {'input': before, 'kept': len(current), 'rejected': len(rejected), **utility_log}
    report['final_kept_count'] = len(current)
    report['final_rejected_count'] = len(all_rejected)

    output_dir.mkdir(parents=True, exist_ok=True)
    kept_count = _write_jsonl(output_dir / 'kept.jsonl', current)
    rejected_count = _write_jsonl(output_dir / 'rejected.jsonl', all_rejected)
    report['outputs']['kept_count_written'] = kept_count
    report['outputs']['rejected_count_written'] = rejected_count
    with (output_dir / 'report.json').open('w', encoding='utf-8') as handle:
        json.dump(_json_safe(report), handle, ensure_ascii=False, indent=2)
    return report
