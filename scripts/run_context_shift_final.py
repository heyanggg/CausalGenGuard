#!/usr/bin/env python3
'''Final context-shift adaptation runner for CausalGenGuard.

The final route treats SmartGuard standard experiments as sanity checks and
focuses on whether target-context synthetic normal data reduces target-normal
false positives. Causal information is used only for synthetic filtering
(Causal-TOF), not as the main detection branch.
'''

from __future__ import annotations

import argparse
import copy
import csv
import json
import pickle
import random
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import yaml
except Exception:  # pragma: no cover - runtime fallback.
    yaml = None

from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.attack_injector import inject_smartguard_style
from causal_gen_guard.data.schemas import BehaviorEvent, BehaviorSequence
from causal_gen_guard.evaluation.metrics import compute_binary_metrics
from causal_gen_guard.models.causal_graph import compute_gradient_causality, normal_causal_pattern, sparsify_causality
from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor
from causal_gen_guard.models.fusion_detector import FusionDetector
from causal_gen_guard.models.smartguard_backbone import SmartGuardBackbone, compute_reconstruction_loss
from causal_gen_guard.training.train_backbone import SequenceTensorDataset
from causal_gen_guard.training.train_causal import WindowDataset, build_window_arrays


METHOD_ORDER = [
    'source_only',
    'source_plus_raw_synthetic',
    'source_plus_tof_synthetic',
    'source_plus_causal_tof_synthetic',
    'oracle_target',
]

TOF_FILTER_STRATEGIES = [
    'no_filter',
    'iqr_1.5',
    'iqr_3.0',
    'keep_top_80_percent_by_low_loss',
    'keep_top_90_percent_by_low_loss',
]

CAUSAL_TOF_FILTER_STRATEGIES = [
    'tof_only',
    'relaxed_causal_keep_90_percent',
    'relaxed_causal_keep_95_percent',
    'causal_filter_disabled',
]

TARGET_CONTEXT_ANOMALY_TYPES = [
    'SD_light_flickering',
    'SD_camera_flickering',
    'SD_tv_flickering',
    'MD_camera_off_while_lock',
    'MD_window_open_while_lock',
    'DM_window_open_midnight',
    'DM_watervalve_open_midnight',
    'DD_microwave_long_time',
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'detach'):
        try:
            return json_safe(value.detach().cpu().numpy())
        except Exception:
            pass
    if hasattr(value, 'tolist'):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def resolve_path(path_value: Any) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError('PyYAML is required to read context_shift_fr.yaml')
    with path.open('r', encoding='utf-8') as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError('config must be a YAML mapping')
    return payload


def read_mapping(config: Dict[str, Any]) -> Tuple[Dict[str, int], Dict[int, str]]:
    paths = dict(config.get('paths', {}))
    control_to_id = json.loads(resolve_path(paths['control_to_id']).read_text(encoding='utf-8'))
    id_to_control_raw = json.loads(resolve_path(paths['id_to_control']).read_text(encoding='utf-8'))
    id_to_control = {int(key): value for key, value in id_to_control_raw.items()}
    return {str(key): int(value) for key, value in control_to_id.items()}, id_to_control


def _as_list(value: Any) -> List[Any]:
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, list):
        return value
    return [value]


def _int_or_none(value: Any) -> Optional[int]:
    if hasattr(value, 'item'):
        try:
            value = value.item()
        except Exception:
            pass
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _split_control(control: str) -> Tuple[str, str]:
    if ':' in control:
        return control.split(':', 1)
    return 'unknown', control


def canonical_event(
    day: Any,
    hour: Any,
    raw_device: Any,
    raw_control: Any,
    control_to_id: Dict[str, int],
    id_to_control: Dict[int, str],
    position: int,
    source_format: str,
) -> Optional[BehaviorEvent]:
    raw_control_id = _int_or_none(raw_control)
    if isinstance(raw_control, str) and raw_control in control_to_id:
        canonical_control = raw_control
        raw_control_id = control_to_id[raw_control]
    elif raw_control_id is not None and raw_control_id in id_to_control:
        canonical_control = id_to_control[raw_control_id]
    else:
        return None
    raw_device_id = _int_or_none(raw_device)
    device, action = _split_control(canonical_control)
    return BehaviorEvent(
        day=day,
        hour=hour,
        device_id=device,
        control_id=canonical_control,
        duration=raw_device_id if raw_device_id is not None else 1.0,
        raw_fields={
            'position': position,
            'day': day,
            'hour': hour,
            'raw_device_id': raw_device,
            'raw_control_id': raw_control_id,
            'device': device,
            'canonical_control': canonical_control,
            'action': action,
            'source_format': source_format,
        },
    )


def parse_record(
    record: Any,
    sequence_id: str,
    context: Dict[str, Any],
    control_to_id: Dict[str, int],
    id_to_control: Dict[int, str],
) -> Optional[BehaviorSequence]:
    if isinstance(record, dict) and 'events' in record:
        sequence = BehaviorSequence.from_dict(record)
        for event in sequence.events:
            canonical = (event.raw_fields or {}).get('canonical_control')
            if canonical in control_to_id:
                event.control_id = canonical
        sequence.context.update(context)
        sequence.label = context.get('label', sequence.label)
        sequence.anomaly_type = context.get('anomaly_type', sequence.anomaly_type)
        return sequence

    flat = _as_list(record)
    events: List[BehaviorEvent] = []
    if len(flat) >= 4 and len(flat) % 4 == 0:
        for offset in range(0, len(flat), 4):
            day, hour, raw_device, raw_control = flat[offset : offset + 4]
            event = canonical_event(day, hour, raw_device, raw_control, control_to_id, id_to_control, offset // 4, 'flat_4')
            if event is not None:
                events.append(event)
    else:
        for position, raw_control in enumerate(flat):
            event = canonical_event(0, position, None, raw_control, control_to_id, id_to_control, position, 'control_sequence')
            if event is not None:
                events.append(event)
    if not events:
        return None
    return BehaviorSequence(sequence_id=sequence_id, events=events, context=dict(context), label=context.get('label'), anomaly_type=context.get('anomaly_type'))


def load_pickle_sequences(
    path: Path,
    context: Dict[str, Any],
    control_to_id: Dict[str, int],
    id_to_control: Dict[int, str],
    limit: Optional[int] = None,
) -> List[BehaviorSequence]:
    with path.open('rb') as handle:
        obj = pickle.load(handle)
    records = obj.values() if isinstance(obj, dict) else obj
    sequences: List[BehaviorSequence] = []
    for index, record in enumerate(_as_list(records)):
        sequence = parse_record(record, f'{path.stem}_{index:06d}', context, control_to_id, id_to_control)
        if sequence is not None:
            sequences.append(sequence)
        if limit is not None and len(sequences) >= limit:
            break
    return sequences


def context_dir(config: Dict[str, Any], context: str) -> Path:
    return resolve_path(config['paths']['smartgen_root']) / context


def context_has_normal_data(config: Dict[str, Any], context: str) -> bool:
    directory = context_dir(config, context)
    return any((directory / name).exists() for name in ('trn.pkl', 'vld.pkl', 'test.pkl', 'split_test.pkl'))


def choose_context(config: Dict[str, Any], candidates: Sequence[str], role: str) -> str:
    for candidate in candidates:
        if context_has_normal_data(config, candidate):
            return candidate
    raise FileNotFoundError(f'No {role} context data found for candidates: {list(candidates)}')


def load_context_normals(
    config: Dict[str, Any],
    context: str,
    split_names: Sequence[str],
    control_to_id: Dict[str, int],
    id_to_control: Dict[int, str],
    limit: Optional[int],
) -> List[BehaviorSequence]:
    directory = context_dir(config, context)
    sequences: List[BehaviorSequence] = []
    for split in split_names:
        path = directory / f'{split}.pkl'
        if not path.exists():
            continue
        loaded = load_pickle_sequences(
            path,
            {
                'dataset': config.get('dataset', 'fr'),
                'context': context,
                'split': split,
                'source_project': 'SmartGen',
                'source_path': str(path),
                'label': 0,
                'anomaly_type': 'normal',
            },
            control_to_id,
            id_to_control,
            None if limit is None else max(limit - len(sequences), 0),
        )
        sequences.extend(loaded)
        if limit is not None and len(sequences) >= limit:
            break
    return sequences[:limit] if limit is not None else sequences


def find_generation_files(config: Dict[str, Any], target_context: str) -> List[Path]:
    directory = context_dir(config, target_context)
    keywords = tuple(config.get('smartgen', {}).get('generated_keywords', ['generation']))
    suffix_prefs = list(config.get('smartgen', {}).get('generated_suffixes', ['_seq.pkl', '.pkl']))
    files = [path for path in directory.rglob('*.pkl') if any(keyword in path.name for keyword in keywords)]

    def rank(path: Path) -> Tuple[int, int, str]:
        name = path.name
        suffix_rank = next((idx for idx, suffix in enumerate(suffix_prefs) if name.endswith(suffix)), len(suffix_prefs))
        textual_rank = 0 if '_seq' in name or 'text' in name.lower() else 1
        return suffix_rank, textual_rank, name

    return sorted(files, key=rank)


def load_target_synthetic(
    config: Dict[str, Any],
    target_context: str,
    control_to_id: Dict[str, int],
    id_to_control: Dict[int, str],
    limit: Optional[int],
) -> Tuple[List[BehaviorSequence], List[str]]:
    sequences: List[BehaviorSequence] = []
    loaded_files: List[str] = []
    for path in find_generation_files(config, target_context):
        loaded = load_pickle_sequences(
            path,
            {
                'dataset': config.get('dataset', 'fr'),
                'context': target_context,
                'source_project': 'SmartGen',
                'source_path': str(path),
                'synthetic_target_normal': True,
                'label': 0,
                'anomaly_type': 'normal',
            },
            control_to_id,
            id_to_control,
            None if limit is None else max(limit - len(sequences), 0),
        )
        if loaded:
            loaded_files.append(str(path))
            sequences.extend(loaded)
        if limit is not None and len(sequences) >= limit:
            break
    return (sequences[:limit] if limit is not None else sequences), loaded_files


def find_target_anomaly_files(config: Dict[str, Any], target_context: str) -> List[Path]:
    directory = context_dir(config, target_context)
    if not directory.exists():
        return []
    return sorted(path for path in directory.rglob('*.pkl') if any(token in path.name.lower() for token in ('attack', 'anomaly', 'abnormal')))


def load_target_anomalies(
    config: Dict[str, Any],
    target_context: str,
    control_to_id: Dict[str, int],
    id_to_control: Dict[int, str],
    limit: Optional[int],
) -> Tuple[List[BehaviorSequence], List[str]]:
    sequences: List[BehaviorSequence] = []
    files: List[str] = []
    for path in find_target_anomaly_files(config, target_context):
        loaded = load_pickle_sequences(
            path,
            {
                'dataset': config.get('dataset', 'fr'),
                'context': target_context,
                'source_project': 'SmartGen',
                'source_path': str(path),
                'label': 1,
                'anomaly_type': 'target_context_anomaly',
            },
            control_to_id,
            id_to_control,
            None if limit is None else max(limit - len(sequences), 0),
        )
        if loaded:
            files.append(str(path))
            sequences.extend(loaded)
        if limit is not None and len(sequences) >= limit:
            break
    return (sequences[:limit] if limit is not None else sequences), files


def target_labeled_paths(config: Dict[str, Any]) -> Tuple[Path, Path]:
    dataset = str(config.get('dataset', 'fr'))
    paths = dict(config.get('paths', {}))
    jsonl_value = paths.get('target_labeled_jsonl') or f'outputs/labels/{dataset}_target_context_labeled.jsonl'
    report_value = paths.get('target_labeled_report') or f'outputs/labels/{dataset}_target_context_labeled_report.json'
    return resolve_path(jsonl_value), resolve_path(report_value)


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with path.open('r', encoding='utf-8') as handle:
        for line in handle:
            if line.strip():
                count += 1
    return count


def write_sequence_jsonl(sequences: Sequence[BehaviorSequence], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence.to_dict(), ensure_ascii=False) + '\n')


def write_json_report(report: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def read_labeled_jsonl(path: Path, max_normal: int, max_anomaly: int) -> Tuple[List[BehaviorSequence], List[BehaviorSequence]]:
    normals: List[BehaviorSequence] = []
    anomalies: List[BehaviorSequence] = []
    if not path.exists():
        return normals, anomalies
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sequence = BehaviorSequence.from_dict(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid target labeled JSONL line {line_number}: {path}') from exc
            if label_to_int(sequence.label) == 1:
                if len(anomalies) < max_anomaly:
                    anomalies.append(sequence)
            elif len(normals) < max_normal:
                normals.append(sequence)
            if len(normals) >= max_normal and len(anomalies) >= max_anomaly:
                break
    return normals, anomalies


def _target_normal_copy(sequence: BehaviorSequence, index: int) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = f'target_normal::{index:06d}::{copied.sequence_id}'
    copied.label = 0
    copied.anomaly_type = 'normal'
    copied.context = dict(copied.context)
    copied.context['target_context_labeled'] = True
    copied.context['target_labeled_role'] = 'normal'
    return copied


def _target_anomaly_copy(sequence: BehaviorSequence, index: int, anomaly_type: str) -> BehaviorSequence:
    copied = copy.deepcopy(sequence)
    copied.sequence_id = f'target_anomaly::{index:06d}::{copied.sequence_id}'
    copied.label = 1
    copied.anomaly_type = anomaly_type
    copied.context = dict(copied.context)
    copied.context['target_context_labeled'] = True
    copied.context['target_labeled_role'] = 'anomaly'
    copied.context['target_labeled_index'] = index
    return copied


def _dedupe_sequences(sequences: Sequence[BehaviorSequence]) -> List[BehaviorSequence]:
    seen: set[str] = set()
    deduped: List[BehaviorSequence] = []
    for sequence in sequences:
        if sequence.sequence_id in seen:
            continue
        seen.add(sequence.sequence_id)
        deduped.append(sequence)
    return deduped


def build_target_context_labeled_set(
    config: Dict[str, Any],
    target_context: str,
    target_normals: List[BehaviorSequence],
    synthetic: List[BehaviorSequence],
    bounds: Dict[str, Any],
    seed: int,
) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], Dict[str, Any]]:
    max_normal = min(int(bounds.get('max_target_labeled_normal', 500)), 500)
    max_anomaly = min(int(bounds.get('max_target_labeled_anomaly', 500)), 500)
    source_pool = _dedupe_sequences(target_normals + synthetic)[:max_normal]
    labeled_normals = [_target_normal_copy(sequence, index) for index, sequence in enumerate(source_pool)]

    success_by_type: Dict[str, List[int]] = {anomaly_type: [] for anomaly_type in TARGET_CONTEXT_ANOMALY_TYPES}
    skipped_reason_counts: Dict[str, Counter[str]] = {anomaly_type: Counter() for anomaly_type in TARGET_CONTEXT_ANOMALY_TYPES}
    for anomaly_type in TARGET_CONTEXT_ANOMALY_TYPES:
        for index, sequence in enumerate(source_pool):
            injected, report = inject_smartguard_style(sequence, anomaly_type)
            if injected is None:
                skipped_reason_counts[anomaly_type][str(report.get('skipped_reason', 'unknown'))] += 1
            else:
                success_by_type[anomaly_type].append(index)

    candidate_count = {anomaly_type: len(indices) for anomaly_type, indices in success_by_type.items()}
    injectable_types = [anomaly_type for anomaly_type, count in candidate_count.items() if count > 0]
    candidate_pairs = [(anomaly_type, source_index) for anomaly_type in injectable_types for source_index in success_by_type[anomaly_type]]
    rng = random.Random(seed)
    rng.shuffle(candidate_pairs)

    labeled_anomalies: List[BehaviorSequence] = []
    injected_reports: List[Dict[str, Any]] = []
    generation_failures: List[Dict[str, Any]] = []
    per_anomaly_type_success = {anomaly_type: 0 for anomaly_type in TARGET_CONTEXT_ANOMALY_TYPES}
    for anomaly_type, source_index in candidate_pairs[:max_anomaly]:
        injected, report = inject_smartguard_style(source_pool[source_index], anomaly_type)
        if injected is None:
            generation_failures.append(report)
            continue
        labeled_anomalies.append(_target_anomaly_copy(injected, len(labeled_anomalies), anomaly_type))
        injected_reports.append(report)
        per_anomaly_type_success[anomaly_type] += 1

    canonical_reference = resolve_path(dict(config.get('paths', {})).get('canonical_normal_jsonl', 'outputs/processed/fr_sequences_canonical.jsonl'))
    report = {
        'dataset': config.get('dataset', 'fr'),
        'target_context': target_context,
        'canonical_reference_path': str(canonical_reference),
        'canonical_reference_count': count_jsonl(canonical_reference),
        'source_pool_count': len(source_pool),
        'real_target_normal_source_count': len(target_normals),
        'synthetic_target_normal_source_count': len(synthetic),
        'normal_count': len(labeled_normals),
        'anomaly_count': len(labeled_anomalies),
        'max_normal_bound': max_normal,
        'max_anomaly_bound': max_anomaly,
        'candidate_count': candidate_count,
        'injectable_anomaly_types': injectable_types,
        'per_anomaly_type_success': per_anomaly_type_success,
        'skipped_type_count': {anomaly_type: count for anomaly_type, count in candidate_count.items() if count == 0},
        'scan_skipped_reasons': {anomaly_type: dict(counter) for anomaly_type, counter in skipped_reason_counts.items()},
        'generation_failure_count': len(generation_failures),
        'generation_failures': generation_failures[:30],
        'injected': injected_reports[:50],
        'seed': seed,
        'notes': [
            'Only anomaly types with candidate_count > 0 are injected.',
            'The target labeled set is bounded to at most 500 normal and 500 anomaly sequences for smoke-test scale.',
        ],
    }
    return labeled_normals, labeled_anomalies, report


def write_target_context_labeled_set(
    config: Dict[str, Any],
    target_context: str,
    target_normals: List[BehaviorSequence],
    synthetic: List[BehaviorSequence],
    bounds: Dict[str, Any],
    seed: int,
) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], Dict[str, Any], List[str]]:
    jsonl_path, report_path = target_labeled_paths(config)
    labeled_normals, labeled_anomalies, report = build_target_context_labeled_set(
        config,
        target_context,
        target_normals,
        synthetic,
        bounds,
        seed,
    )
    write_sequence_jsonl(labeled_normals + labeled_anomalies, jsonl_path)
    write_json_report(report, report_path)
    return labeled_normals, labeled_anomalies, report, [str(jsonl_path), str(report_path)]


def max_event_len(sequences: List[BehaviorSequence]) -> int:
    return max((len(sequence.events) for sequence in sequences), default=0)


def safe_nhead(hidden_dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


def train_backbone(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    seed: int,
) -> Tuple[SmartGuardBackbone, Dict[str, Any]]:
    if not sequences:
        raise ValueError('cannot train source-only detector without sequences')
    set_seed(seed)
    max_len = max_event_len(sequences)
    dataset = SequenceTensorDataset(sequences, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SmartGuardBackbone(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        nhead=safe_nhead(hidden_dim),
        num_layers=1,
        dropout=0.1,
        use_ttpe=True,
    )
    model.control_vocab = dict(vocab)
    model.vocab = dict(vocab)
    model.max_len = max_len
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0.0
        for fields, mask in loader:
            optimizer.zero_grad()
            outputs = model(fields, attention_mask=mask)
            targets = fields[..., 3].long()
            loss = compute_reconstruction_loss(outputs['logits'], targets, mask=mask, reduction='mean')
            loss.backward()
            optimizer.step()
            token_count = float(mask.sum().item())
            total_loss += float(loss.detach().item()) * token_count
            total_tokens += token_count
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / max(total_tokens, 1.0)})
    model.eval()
    return model, {'history': history, 'max_len': max_len, 'train_count': len(sequences)}


def train_causal_for_filter(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    seed: int,
) -> Tuple[Optional[BehaviorCausalPredictor], Optional[np.ndarray], Dict[str, Any]]:
    if len(sequences) < 2 or max_event_len(sequences) < 2:
        return None, None, {'skipped': True, 'reason': 'too_few_sequences'}
    set_seed(seed)
    window_size = max(1, min(4, max_event_len(sequences) - 1))
    windows, time_windows, targets = build_window_arrays(sequences, vocab, window_size=window_size, pred_horizon=1)
    dataset = WindowDataset(windows, time_windows, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BehaviorCausalPredictor(len(vocab), hidden_dim=hidden_dim, time_feature_dim=time_windows.shape[-1], num_layers=1, dropout=0.1)
    model.window_size = window_size
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history: List[Dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_count = 0
        for batch_windows, batch_time, batch_targets in loader:
            optimizer.zero_grad()
            pred = model(batch_windows, batch_time)
            loss = model.compute_loss(pred, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * int(batch_windows.size(0))
            total_count += int(batch_windows.size(0))
        history.append({'epoch': epoch + 1, 'train_loss': total_loss / max(total_count, 1)})
    model.eval()
    sample_count = min(64, len(dataset))
    graphs = compute_gradient_causality(model, dataset.windows[:sample_count], dataset.targets[:sample_count])
    A_norm = normal_causal_pattern(sparsify_causality(graphs, threshold=0.0)).detach().cpu().numpy().astype(np.float32)
    return model, A_norm, {'history': history, 'window_size': window_size, 'window_count': len(dataset)}


def detector_for(backbone: SmartGuardBackbone, vocab: Dict[Any, int]) -> FusionDetector:
    return FusionDetector(backbone=backbone, causal_model=None, vocab=vocab, A_norm_bank=None, alpha=1.0, beta=0.0)


def calibrate_threshold(detector: FusionDetector, val_sequences: List[BehaviorSequence], quantile: float) -> float:
    return detector.calibrate_threshold(val_sequences, quantile=quantile)


def score_values(detector: FusionDetector, sequences: List[BehaviorSequence]) -> List[float]:
    return [float(item['score']) for item in detector.score_batch(sequences)]


def label_to_int(label: Any) -> int:
    if label is None:
        return 0
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label != 0)
    text = str(label).strip().lower()
    return 0 if text in ('0', 'normal', 'benign', 'clean', 'false', 'none') else 1


def evaluate(
    detector: FusionDetector,
    threshold: float,
    target_normals: List[BehaviorSequence],
    target_anomalies: List[BehaviorSequence],
) -> Dict[str, Any]:
    normal_scores = score_values(detector, target_normals) if target_normals else []
    target_normal_fpr = sum(score > threshold for score in normal_scores) / max(len(normal_scores), 1) if normal_scores else None
    result = {
        'target_normal_count': len(target_normals),
        'target_normal_fpr': target_normal_fpr,
        'target_normal_score_mean': float(np.mean(normal_scores)) if normal_scores else None,
        'anomaly_count': len(target_anomalies),
        'precision': None,
        'recall': None,
        'f1': None,
        'anomaly_f1': None,
        'auroc': None,
        'auprc': None,
    }
    if target_anomalies:
        eval_sequences = target_normals + target_anomalies
        y_true = [label_to_int(sequence.label) for sequence in eval_sequences]
        y_score = score_values(detector, eval_sequences)
        metrics = compute_binary_metrics(y_true, y_score, threshold)
        result.update(
            {
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1'],
                'anomaly_f1': metrics['f1'],
                'auroc': metrics['auroc'],
                'auprc': metrics['auprc'],
            }
        )
    return result


def legality_filter(sequences: List[BehaviorSequence], allowed_controls: set[Any], min_len: int = 2) -> Tuple[List[BehaviorSequence], List[Dict[str, Any]]]:
    kept: List[BehaviorSequence] = []
    rejected: List[Dict[str, Any]] = []
    for sequence in sequences:
        if len(sequence.events) < min_len:
            rejected.append({'sequence_id': sequence.sequence_id, 'stage': 'legality', 'reason': 'too_short'})
            continue
        illegal = next((event.control_id for event in sequence.events if event.control_id not in allowed_controls), None)
        if illegal is not None:
            rejected.append({'sequence_id': sequence.sequence_id, 'stage': 'legality', 'reason': 'unknown_control', 'control': illegal})
            continue
        kept.append(sequence)
    return kept, rejected


def reconstruction_filter(
    sequences: List[BehaviorSequence],
    detector: FusionDetector,
    threshold: float,
) -> Tuple[List[BehaviorSequence], List[Dict[str, Any]]]:
    kept: List[BehaviorSequence] = []
    rejected: List[Dict[str, Any]] = []
    for sequence, score in zip(sequences, score_values(detector, sequences)):
        sequence.context['tof_reconstruction_score'] = score
        if score <= threshold:
            kept.append(sequence)
        else:
            rejected.append({'sequence_id': sequence.sequence_id, 'stage': 'reconstruction', 'reason': 'score_above_threshold', 'score': score})
    return kept, rejected


def causal_scores(
    sequences: List[BehaviorSequence],
    causal_model: Optional[BehaviorCausalPredictor],
    A_norm: Optional[np.ndarray],
    vocab: Dict[Any, int],
) -> List[float]:
    if causal_model is None or A_norm is None:
        return [0.0 for _ in sequences]
    detector = FusionDetector(backbone=None, causal_model=causal_model, vocab=vocab, A_norm_bank=A_norm, alpha=0.0, beta=1.0)
    return [float(item['causal_score']) for item in detector.score_batch(sequences)]


def causal_filter(
    sequences: List[BehaviorSequence],
    causal_model: Optional[BehaviorCausalPredictor],
    A_norm: Optional[np.ndarray],
    vocab: Dict[Any, int],
    threshold: float,
) -> Tuple[List[BehaviorSequence], List[Dict[str, Any]]]:
    kept: List[BehaviorSequence] = []
    rejected: List[Dict[str, Any]] = []
    scores = causal_scores(sequences, causal_model, A_norm, vocab)
    for sequence, score in zip(sequences, scores):
        sequence.context['causal_tof_score'] = score
        if score <= threshold:
            kept.append(sequence)
        else:
            rejected.append({'sequence_id': sequence.sequence_id, 'stage': 'causal', 'reason': 'score_above_threshold', 'score': score})
    return kept, rejected


def score_summary(scores: Sequence[float], threshold: Optional[float] = None) -> Dict[str, Any]:
    if not scores:
        return {'count': 0, 'threshold': threshold}
    arr = np.asarray(scores, dtype=float)
    return {
        'count': int(arr.size),
        'mean': float(np.mean(arr)),
        'min': float(np.min(arr)),
        'p25': float(np.quantile(arr, 0.25)),
        'median': float(np.quantile(arr, 0.50)),
        'p75': float(np.quantile(arr, 0.75)),
        'max': float(np.max(arr)),
        'threshold': threshold,
    }


def _low_score_keep_indices(scores: Sequence[float], keep_fraction: float) -> set[int]:
    if not scores:
        return set()
    keep_count = max(1, int(np.ceil(len(scores) * keep_fraction)))
    ranked = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    return set(ranked[:keep_count])


def _selection_payload(
    sequences: List[BehaviorSequence],
    scores: Sequence[float],
    keep_indices: set[int],
    base_rejected: List[Dict[str, Any]],
    stage: str,
    score_key: str,
    reject_reason: str,
    filter_strategy: str,
    threshold: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    kept: List[BehaviorSequence] = []
    rejected = copy.deepcopy(base_rejected)
    for index, sequence in enumerate(sequences):
        sequence_copy = copy.deepcopy(sequence)
        score = float(scores[index]) if index < len(scores) else None
        if score is not None:
            sequence_copy.context[score_key] = score
        if index in keep_indices:
            kept.append(sequence_copy)
        else:
            rejected.append(
                {
                    'sequence_id': sequence.sequence_id,
                    'stage': stage,
                    'reason': reject_reason,
                    'score': score,
                    'filter_strategy': filter_strategy,
                }
            )
    payload = {
        'sequences': kept,
        'kept_count': len(kept),
        'rejected': rejected,
        'filter_strategy': filter_strategy,
        'score_summary': score_summary(scores, threshold),
    }
    if extra:
        payload.update(extra)
    return payload


def reconstruction_strategy_payload(
    synthetic: List[BehaviorSequence],
    legal_kept: List[BehaviorSequence],
    legal_rejected: List[Dict[str, Any]],
    scores: Sequence[float],
    strategy: str,
) -> Dict[str, Any]:
    if strategy == 'no_filter':
        return {
            'sequences': copy.deepcopy(synthetic),
            'kept_count': len(synthetic),
            'rejected': [],
            'filter_strategy': strategy,
            'score_summary': {'count': len(synthetic), 'threshold': None},
            'causal_info': None,
        }
    if not legal_kept:
        return {
            'sequences': [],
            'kept_count': 0,
            'rejected': copy.deepcopy(legal_rejected),
            'filter_strategy': strategy,
            'score_summary': score_summary([], None),
            'causal_info': None,
        }
    if strategy.startswith('iqr_'):
        multiplier = float(strategy.split('_', 1)[1])
        arr = np.asarray(scores, dtype=float)
        q1 = float(np.quantile(arr, 0.25))
        q3 = float(np.quantile(arr, 0.75))
        threshold = q3 + multiplier * (q3 - q1)
        keep_indices = {index for index, score in enumerate(scores) if score <= threshold}
        return _selection_payload(
            legal_kept,
            scores,
            keep_indices,
            legal_rejected,
            'reconstruction',
            'tof_reconstruction_score',
            'score_above_iqr_threshold',
            strategy,
            threshold,
            {'causal_info': None},
        )
    if strategy == 'keep_top_80_percent_by_low_loss':
        keep_indices = _low_score_keep_indices(scores, 0.80)
        threshold = max((scores[index] for index in keep_indices), default=None)
        return _selection_payload(
            legal_kept,
            scores,
            keep_indices,
            legal_rejected,
            'reconstruction',
            'tof_reconstruction_score',
            'outside_low_loss_percentile',
            strategy,
            float(threshold) if threshold is not None else None,
            {'causal_info': None},
        )
    if strategy == 'keep_top_90_percent_by_low_loss':
        keep_indices = _low_score_keep_indices(scores, 0.90)
        threshold = max((scores[index] for index in keep_indices), default=None)
        return _selection_payload(
            legal_kept,
            scores,
            keep_indices,
            legal_rejected,
            'reconstruction',
            'tof_reconstruction_score',
            'outside_low_loss_percentile',
            strategy,
            float(threshold) if threshold is not None else None,
            {'causal_info': None},
        )
    raise ValueError(f'unknown TOF filter strategy: {strategy}')


def causal_strategy_payload(
    base_payload: Dict[str, Any],
    causal_model: Optional[BehaviorCausalPredictor],
    A_norm: Optional[np.ndarray],
    vocab: Dict[Any, int],
    strategy: str,
    causal_info: Dict[str, Any],
) -> Dict[str, Any]:
    base_sequences = base_payload['sequences']
    base_rejected = base_payload['rejected']
    if strategy in ('tof_only', 'causal_filter_disabled') or not base_sequences or causal_model is None or A_norm is None:
        payload = {
            'sequences': copy.deepcopy(base_sequences),
            'kept_count': len(base_sequences),
            'rejected': copy.deepcopy(base_rejected),
            'filter_strategy': strategy,
            'base_tof_strategy': base_payload.get('filter_strategy'),
            'score_summary': {'count': len(base_sequences), 'threshold': None},
            'causal_info': causal_info,
        }
        if causal_model is None or A_norm is None:
            payload['causal_filter_note'] = 'causal_filter_unavailable'
        return payload

    scores = causal_scores(base_sequences, causal_model, A_norm, vocab)
    if strategy == 'relaxed_causal_keep_90_percent':
        keep_indices = _low_score_keep_indices(scores, 0.90)
    elif strategy == 'relaxed_causal_keep_95_percent':
        keep_indices = _low_score_keep_indices(scores, 0.95)
    else:
        raise ValueError(f'unknown Causal-TOF filter strategy: {strategy}')
    threshold = max((scores[index] for index in keep_indices), default=None)
    return _selection_payload(
        base_sequences,
        scores,
        keep_indices,
        base_rejected,
        'causal',
        'causal_tof_score',
        'outside_relaxed_causal_percentile',
        strategy,
        float(threshold) if threshold is not None else None,
        {'base_tof_strategy': base_payload.get('filter_strategy'), 'causal_info': causal_info},
    )


def build_synthetic_sets(
    synthetic: List[BehaviorSequence],
    source_train: List[BehaviorSequence],
    source_val: List[BehaviorSequence],
    source_detector: FusionDetector,
    source_threshold: float,
    vocab: Dict[Any, int],
    control_to_id: Dict[str, int],
    bounds: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    allowed = set(control_to_id.keys())
    legal_kept, legal_rejected = legality_filter(copy.deepcopy(synthetic), allowed)
    reconstruction_scores = score_values(source_detector, legal_kept) if legal_kept else []
    tof_candidates = {
        strategy: reconstruction_strategy_payload(synthetic, legal_kept, legal_rejected, reconstruction_scores, strategy)
        for strategy in TOF_FILTER_STRATEGIES
    }
    causal_model, A_norm, causal_info = train_causal_for_filter(
        source_train,
        vocab,
        int(bounds.get('epochs', 3)),
        int(bounds.get('batch_size', 32)),
        int(bounds.get('hidden_dim', 64)),
        seed,
    )
    return {
        'raw': {
            'sequences': copy.deepcopy(synthetic),
            'kept_count': len(synthetic),
            'rejected': [],
            'filter_strategy': 'no_filter',
            'score_summary': {'count': len(synthetic), 'threshold': None},
            'causal_info': None,
        },
        'tof_candidates': tof_candidates,
        'causal_model': causal_model,
        'A_norm': A_norm,
        'causal_info': causal_info,
    }


def train_and_eval_method(
    method_id: str,
    train_sequences: List[BehaviorSequence],
    source_val: List[BehaviorSequence],
    target_normals: List[BehaviorSequence],
    target_anomalies: List[BehaviorSequence],
    vocab: Dict[Any, int],
    bounds: Dict[str, Any],
    threshold_quantile: float,
    seed: int,
    synthetic_count: int = 0,
    kept_count: int = 0,
    rejected_count: int = 0,
    filter_strategy: str = '',
    selected: bool = False,
) -> Tuple[Dict[str, Any], FusionDetector, float]:
    backbone, train_info = train_backbone(
        train_sequences,
        vocab,
        int(bounds.get('epochs', 3)),
        int(bounds.get('batch_size', 32)),
        int(bounds.get('hidden_dim', 64)),
        seed,
    )
    detector = detector_for(backbone, vocab)
    threshold = calibrate_threshold(detector, source_val, threshold_quantile)
    metrics = evaluate(detector, threshold, target_normals, target_anomalies)
    row = {
        'method': method_id,
        'filter_strategy': filter_strategy,
        'selected': selected,
        'train_count': len(train_sequences),
        'source_count': len([sequence for sequence in train_sequences if not sequence.context.get('synthetic_target_normal')]),
        'synthetic_count': synthetic_count,
        'kept_count': kept_count,
        'rejected_count': rejected_count,
        'threshold': threshold,
        **metrics,
        'train_info': train_info,
    }
    return row, detector, threshold


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        'method',
        'filter_strategy',
        'selected',
        'train_count',
        'source_count',
        'synthetic_count',
        'kept_count',
        'rejected_count',
        'target_normal_count',
        'target_normal_fpr',
        'anomaly_count',
        'precision',
        'recall',
        'f1',
        'anomaly_f1',
        'auroc',
        'auprc',
        'adaptation_gain',
    ]
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json_safe(row.get(field)) for field in fields})


def clone_row_for_strategy(
    row: Dict[str, Any],
    method_id: str,
    filter_strategy: str,
    kept_count: Optional[int] = None,
    rejected_count: Optional[int] = None,
    selected: bool = False,
) -> Dict[str, Any]:
    cloned = copy.deepcopy(row)
    cloned['method'] = method_id
    cloned['filter_strategy'] = filter_strategy
    cloned['selected'] = selected
    if kept_count is not None:
        cloned['kept_count'] = kept_count
    if rejected_count is not None:
        cloned['rejected_count'] = rejected_count
    cloned['adaptation_gain'] = None
    return cloned


def best_by_target_fpr(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        raise ValueError('cannot choose best row from an empty strategy sweep')

    def key(row: Dict[str, Any]) -> Tuple[float, int, int, str]:
        fpr = row.get('target_normal_fpr')
        fpr_value = float(fpr) if fpr is not None else float('inf')
        return (fpr_value, -int(row.get('kept_count') or 0), int(row.get('rejected_count') or 0), str(row.get('filter_strategy') or ''))

    return min(rows, key=key)


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'method': row.get('method'),
        'filter_strategy': row.get('filter_strategy', ''),
        'selected': bool(row.get('selected')),
        'kept_count': row.get('kept_count'),
        'rejected_count': row.get('rejected_count'),
        'target_normal_fpr': row.get('target_normal_fpr'),
        'precision': row.get('precision'),
        'recall': row.get('recall'),
        'f1': row.get('f1'),
        'anomaly_f1': row.get('anomaly_f1'),
        'auroc': row.get('auroc'),
        'auprc': row.get('auprc'),
        'adaptation_gain': row.get('adaptation_gain'),
    }


def format_metric(value: Any) -> str:
    if value is None:
        return 'skipped'
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (float, np.floating)):
        return f'{float(value):.6f}'
    return str(value)


def write_tof_filter_analysis(path: Path, payload: Dict[str, Any]) -> None:
    rows = payload['rows']
    raw_row = next(row for row in rows if row['method'] == 'source_plus_raw_synthetic')
    tof_rows = [row for row in rows if row['method'] == 'source_plus_tof_synthetic']
    causal_rows = [row for row in rows if row['method'] == 'source_plus_causal_tof_synthetic']
    best_tof = next(row for row in tof_rows if row.get('selected'))
    best_causal = next(row for row in causal_rows if row.get('selected'))
    raw_fpr = raw_row.get('target_normal_fpr')

    over_rejecting = []
    for row in tof_rows + causal_rows:
        fpr = row.get('target_normal_fpr')
        if row.get('rejected_count', 0) > 0 and raw_fpr is not None and fpr is not None and fpr > raw_fpr:
            over_rejecting.append(row)

    lines = [
        '# TOF Filter Analysis',
        '',
        '## Summary',
        '',
        f"- Source context: `{payload['source_context']}`",
        f"- Target context: `{payload['target_context']}`",
        f"- Synthetic count: `{payload['synthetic_count']}`",
        f"- Raw synthetic target_normal_fpr: `{format_metric(raw_fpr)}`",
        f"- Selected TOF strategy: `{best_tof.get('filter_strategy')}` with target_normal_fpr `{format_metric(best_tof.get('target_normal_fpr'))}`",
        f"- Selected Causal-TOF strategy: `{best_causal.get('filter_strategy')}` with target_normal_fpr `{format_metric(best_causal.get('target_normal_fpr'))}`",
        '',
        '## Why Raw Synthetic Is Currently Strong',
        '',
        'Raw target-context synthetic normal keeps the full generated target-context diversity. The stricter filters rank generated sequences by source-trained reconstruction or causal scores, so they can reject normal target-context patterns precisely because those patterns differ from the source context.',
        '',
        '## Filter Strategy Sweep',
        '',
        '| method | filter_strategy | selected | kept_count | rejected_count | target_normal_fpr |',
        '| --- | --- | --- | ---: | ---: | ---: |',
    ]
    for row in tof_rows + causal_rows:
        lines.append(
            '| {method} | {strategy} | {selected} | {kept} | {rejected} | {fpr} |'.format(
                method=row.get('method'),
                strategy=row.get('filter_strategy') or '',
                selected='yes' if row.get('selected') else 'no',
                kept=row.get('kept_count', 0),
                rejected=row.get('rejected_count', 0),
                fpr=format_metric(row.get('target_normal_fpr')),
            )
        )

    lines.extend(['', '## Over-Rejection Notes', ''])
    if over_rejecting:
        lines.append('The following strategies rejected synthetic data and performed worse than raw synthetic:')
        lines.append('')
        for row in over_rejecting:
            lines.append(
                f"- `{row.get('method')}` / `{row.get('filter_strategy')}` rejected `{row.get('rejected_count')}` and produced FPR `{format_metric(row.get('target_normal_fpr'))}`."
            )
    else:
        lines.append('No strategy both rejected synthetic data and performed worse than raw synthetic in this run.')

    lines.extend(
        [
            '',
            '## Final Choice',
            '',
            f"- `source_plus_tof_synthetic` uses `{best_tof.get('filter_strategy')}`.",
            f"- `source_plus_causal_tof_synthetic` uses `{best_causal.get('filter_strategy')}`.",
            '- Causal information remains part of Causal-TOF filtering/explanation only; it is not promoted to a detector branch in the final route.',
            '',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def write_final_run_report(path: Path, payload: Dict[str, Any]) -> None:
    selected_rows = [row for row in payload['rows'] if row.get('selected')]
    sweep_rows = [
        row
        for row in payload['rows']
        if row.get('method') in ('source_plus_tof_synthetic', 'source_plus_causal_tof_synthetic')
    ]
    source_row = next((row for row in payload['rows'] if row.get('method') == 'source_only'), None)
    non_oracle_selected = [row for row in selected_rows if row.get('method') not in ('source_only', 'oracle_target')]
    best_non_oracle = best_by_target_fpr(non_oracle_selected) if non_oracle_selected else None
    target_labeled_report = dict(payload.get('target_labeled_report') or {})

    lines = [
        '# Final Context Shift Run Report',
        '',
        f'Date: {date.today().isoformat()}',
        f"Project path: `{PROJECT_ROOT}`",
        '',
        '## Run Status',
        '',
        '- Ran through: `True`',
        '- Command:',
        '',
        '```bash',
        'PYTHONPATH=src /home/heyang/miniconda3/envs/smartguard_env/bin/python scripts/run_context_shift_final.py --config configs/context_shift_fr.yaml',
        '```',
        '',
        '## Contexts',
        '',
        f"- Source context: `{payload['source_context']}`",
        f"- Target context: `{payload['target_context']}`",
        '',
        '## Synthetic Data',
        '',
        f"- synthetic_count: `{payload['synthetic_count']}`",
        '',
        '## Target Labeled Anomaly Set',
        '',
        f"- target_labeled_jsonl: `{payload.get('target_labeled_jsonl')}`",
        f"- target_labeled_report: `{payload.get('target_labeled_report_path')}`",
        f"- labeled_normal_count: `{target_labeled_report.get('normal_count', 0)}`",
        f"- labeled_anomaly_count: `{target_labeled_report.get('anomaly_count', 0)}`",
        f"- target_anomaly_source: `{payload.get('target_anomaly_source')}`",
        f"- injectable_anomaly_types: `{target_labeled_report.get('injectable_anomaly_types', [])}`",
        '',
        '| anomaly_type | candidate_count | injected_count |',
        '| --- | ---: | ---: |',
    ]
    candidate_count = dict(target_labeled_report.get('candidate_count') or {})
    per_success = dict(target_labeled_report.get('per_anomaly_type_success') or {})
    for anomaly_type in TARGET_CONTEXT_ANOMALY_TYPES:
        lines.append(f"| {anomaly_type} | {candidate_count.get(anomaly_type, 0)} | {per_success.get(anomaly_type, 0)} |")

    lines.extend(
        [
            '',
        '## Selected Method Results',
        '',
        '| method | filter_strategy | kept_count | rejected_count | target_normal_fpr | precision | recall | f1 | auroc | auprc | adaptation_gain |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    )
    for row in selected_rows:
        lines.append(
            '| {method} | {strategy} | {kept} | {rejected} | {fpr} | {precision} | {recall} | {f1} | {auroc} | {auprc} | {gain} |'.format(
                method=row.get('method'),
                strategy=row.get('filter_strategy') or '',
                kept=row.get('kept_count', 0),
                rejected=row.get('rejected_count', 0),
                fpr=format_metric(row.get('target_normal_fpr')),
                precision=format_metric(row.get('precision')),
                recall=format_metric(row.get('recall')),
                f1=format_metric(row.get('f1')),
                auroc=format_metric(row.get('auroc')),
                auprc=format_metric(row.get('auprc')),
                gain=format_metric(row.get('adaptation_gain')),
            )
        )

    lines.extend(
        [
            '',
            '## Filter Strategy Sweep',
            '',
            '| method | filter_strategy | selected | kept_count | rejected_count | target_normal_fpr | precision | recall | f1 | auroc | auprc | adaptation_gain |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        ]
    )
    for row in sweep_rows:
        lines.append(
            '| {method} | {strategy} | {selected} | {kept} | {rejected} | {fpr} | {precision} | {recall} | {f1} | {auroc} | {auprc} | {gain} |'.format(
                method=row.get('method'),
                strategy=row.get('filter_strategy') or '',
                selected='yes' if row.get('selected') else 'no',
                kept=row.get('kept_count', 0),
                rejected=row.get('rejected_count', 0),
                fpr=format_metric(row.get('target_normal_fpr')),
                precision=format_metric(row.get('precision')),
                recall=format_metric(row.get('recall')),
                f1=format_metric(row.get('f1')),
                auroc=format_metric(row.get('auroc')),
                auprc=format_metric(row.get('auprc')),
                gain=format_metric(row.get('adaptation_gain')),
            )
        )

    lines.extend(
        [
            '',
            '## Metric Availability',
            '',
            '- `target_normal_fpr`: available for all methods.',
            f"- `precision`: {'available' if payload['target_anomaly_available'] else 'skipped'}.",
            f"- `recall`: {'available' if payload['target_anomaly_available'] else 'skipped'}.",
            f"- `f1`: {'available' if payload['target_anomaly_available'] else 'skipped'}.",
            f"- `auroc`: {'available' if payload['target_anomaly_available'] else 'skipped'}.",
            f"- `auprc`: {'available' if payload['target_anomaly_available'] else 'skipped'}.",
        ]
    )
    if not payload['target_anomaly_available']:
        lines.append('- Skip reason: no target-context anomaly data was found for the selected target context, so anomaly-label metrics were not fabricated.')
    lines.extend(['', '## Best Method By Target-Normal FPR', ''])
    if best_non_oracle is not None:
        lines.extend(
            [
                f"- Best non-oracle method: `{best_non_oracle.get('method')}`",
                f"- Best non-oracle filter_strategy: `{best_non_oracle.get('filter_strategy')}`",
                f"- Best non-oracle target_normal_fpr: `{format_metric(best_non_oracle.get('target_normal_fpr'))}`",
            ]
        )
    if source_row is not None:
        lines.append(f"- Source-only target_normal_fpr: `{format_metric(source_row.get('target_normal_fpr'))}`")
    lines.extend(
        [
            '',
            '## Output Files',
            '',
            '- `outputs/labels/fr_target_context_labeled.jsonl`',
            '- `outputs/labels/fr_target_context_labeled_report.json`',
            '- `outputs/results/context_shift_final_fr.csv`',
            '- `outputs/results/context_shift_final_fr.json`',
            '- `outputs/logs/TOF_FILTER_ANALYSIS.md`',
            '- `outputs/logs/FINAL_CONTEXT_SHIFT_RUN_REPORT.md`',
            '',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    set_seed(int(config.get('run_bounds', {}).get('seed', 42)))
    control_to_id, id_to_control = read_mapping(config)
    contexts = dict(config.get('contexts', {}))
    source_context = choose_context(config, contexts.get('source_candidates', ['winter', 'single']), 'source')
    target_context = choose_context(config, contexts.get('target_candidates', ['spring', 'multiple', 'night']), 'target')
    bounds = dict(config.get('run_bounds', {}))
    threshold_quantile = float(config.get('evaluation', {}).get('threshold_quantile', 0.95))
    seed = int(bounds.get('seed', 42))

    source_train = load_context_normals(config, source_context, ['trn'], control_to_id, id_to_control, int(bounds.get('max_source_train', 500)))
    source_val = load_context_normals(config, source_context, ['vld', 'trn'], control_to_id, id_to_control, int(bounds.get('max_source_val', 200)))
    target_normals = load_context_normals(config, target_context, ['test', 'split_test', 'vld', 'trn'], control_to_id, id_to_control, int(bounds.get('max_target_normal', 300)))
    target_anomalies, target_anomaly_files = load_target_anomalies(config, target_context, control_to_id, id_to_control, int(bounds.get('max_target_anomaly', 300)))
    synthetic, synthetic_files = load_target_synthetic(config, target_context, control_to_id, id_to_control, int(bounds.get('max_synthetic', 500)))
    target_labeled_normals, target_labeled_anomalies, target_labeled_report, target_labeled_files = write_target_context_labeled_set(
        config,
        target_context,
        target_normals,
        synthetic,
        bounds,
        seed + 1000,
    )
    if target_labeled_anomalies:
        target_anomaly_limit = min(int(bounds.get('max_target_anomaly', 300)), 500)
        target_anomalies = target_labeled_anomalies[:target_anomaly_limit]
        target_anomaly_files = target_labeled_files
        target_anomaly_source = 'target_context_labeled_jsonl'
    else:
        target_anomaly_source = 'smartgen_existing_anomaly_files' if target_anomalies else 'none'

    if not source_train or not source_val or not target_normals:
        raise ValueError('source_train, source_val, and target_normals must all be non-empty')

    all_vocab_sequences = source_train + source_val + target_normals + target_labeled_normals + target_anomalies + synthetic
    vocab = build_vocab(all_vocab_sequences)
    rows: List[Dict[str, Any]] = []

    source_row, source_detector, source_threshold = train_and_eval_method(
        'source_only',
        source_train,
        source_val,
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed,
        selected=True,
    )
    rows.append(source_row)
    synthetic_sets = build_synthetic_sets(synthetic, source_train, source_val, source_detector, source_threshold, vocab, control_to_id, bounds, seed + 1)

    raw_payload = synthetic_sets['raw']
    raw_train = raw_payload['sequences']
    raw_row, _, _ = train_and_eval_method(
        'source_plus_raw_synthetic',
        source_train + raw_train,
        source_val + raw_train[: max(1, min(len(raw_train), len(source_val)))],
        target_normals,
        target_anomalies,
        vocab,
        bounds,
        threshold_quantile,
        seed + 1,
        synthetic_count=len(synthetic),
        kept_count=raw_payload['kept_count'],
        rejected_count=len(raw_payload['rejected']),
        filter_strategy='no_filter',
        selected=True,
    )
    rows.append(raw_row)

    tof_payloads = synthetic_sets['tof_candidates']
    tof_rows: List[Dict[str, Any]] = []
    for index, strategy in enumerate(TOF_FILTER_STRATEGIES):
        payload = tof_payloads[strategy]
        if strategy == 'no_filter':
            row = clone_row_for_strategy(
                raw_row,
                'source_plus_tof_synthetic',
                strategy,
                kept_count=payload['kept_count'],
                rejected_count=len(payload['rejected']),
                selected=False,
            )
        else:
            synthetic_train = payload['sequences']
            row, _, _ = train_and_eval_method(
                'source_plus_tof_synthetic',
                source_train + synthetic_train,
                source_val + synthetic_train[: max(1, min(len(synthetic_train), len(source_val)))],
                target_normals,
                target_anomalies,
                vocab,
                bounds,
                threshold_quantile,
                seed + 10 + index,
                synthetic_count=len(synthetic),
                kept_count=payload['kept_count'],
                rejected_count=len(payload['rejected']),
                filter_strategy=strategy,
            )
        tof_rows.append(row)
    best_tof_row = best_by_target_fpr(tof_rows)
    best_tof_row['selected'] = True
    rows.extend(tof_rows)

    best_tof_strategy = str(best_tof_row.get('filter_strategy') or 'no_filter')
    best_tof_payload = tof_payloads[best_tof_strategy]
    causal_payloads: Dict[str, Dict[str, Any]] = {}
    causal_rows: List[Dict[str, Any]] = []
    for index, strategy in enumerate(CAUSAL_TOF_FILTER_STRATEGIES):
        payload = causal_strategy_payload(
            best_tof_payload,
            synthetic_sets['causal_model'],
            synthetic_sets['A_norm'],
            vocab,
            strategy,
            synthetic_sets['causal_info'],
        )
        causal_payloads[strategy] = payload
        if strategy in ('tof_only', 'causal_filter_disabled'):
            row = clone_row_for_strategy(
                best_tof_row,
                'source_plus_causal_tof_synthetic',
                strategy,
                kept_count=payload['kept_count'],
                rejected_count=len(payload['rejected']),
                selected=False,
            )
        else:
            synthetic_train = payload['sequences']
            row, _, _ = train_and_eval_method(
                'source_plus_causal_tof_synthetic',
                source_train + synthetic_train,
                source_val + synthetic_train[: max(1, min(len(synthetic_train), len(source_val)))],
                target_normals,
                target_anomalies,
                vocab,
                bounds,
                threshold_quantile,
                seed + 30 + index,
                synthetic_count=len(synthetic),
                kept_count=payload['kept_count'],
                rejected_count=len(payload['rejected']),
                filter_strategy=strategy,
            )
        causal_rows.append(row)
    best_causal_row = best_by_target_fpr(causal_rows)
    best_causal_row['selected'] = True
    rows.extend(causal_rows)

    if target_normals:
        oracle_row, _, _ = train_and_eval_method(
            'oracle_target',
            source_train + target_normals,
            source_val + target_normals[: max(1, min(len(target_normals), len(source_val)))],
            target_normals,
            target_anomalies,
            vocab,
            bounds,
            threshold_quantile,
            seed + 99,
            synthetic_count=0,
            kept_count=0,
            rejected_count=0,
            selected=True,
        )
        rows.append(oracle_row)

    source_fpr = source_row.get('target_normal_fpr')
    for row in rows:
        if source_fpr is not None and row.get('target_normal_fpr') is not None and row['method'] != 'source_only':
            row['adaptation_gain'] = source_fpr - row['target_normal_fpr']
        else:
            row['adaptation_gain'] = None

    output_csv = resolve_path(config['paths']['output_csv'])
    output_json = resolve_path(config['paths']['output_json'])
    write_csv(output_csv, rows)

    def filter_report(value: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'filter_strategy': value.get('filter_strategy'),
            'base_tof_strategy': value.get('base_tof_strategy'),
            'synthetic_count': len(synthetic),
            'kept_count': value.get('kept_count'),
            'rejected_count': len(value.get('rejected', [])),
            'rejected_reasons': value.get('rejected', [])[:30],
            'score_summary': value.get('score_summary'),
            'causal_info': value.get('causal_info'),
            'causal_filter_note': value.get('causal_filter_note'),
        }

    payload = {
        'dataset': config.get('dataset', 'fr'),
        'source_context': source_context,
        'target_context': target_context,
        'synthetic_count': len(synthetic),
        'rows': rows,
        'selected_methods': {
            'source_only': compact_row(source_row),
            'source_plus_raw_synthetic': compact_row(raw_row),
            'source_plus_tof_synthetic': compact_row(best_tof_row),
            'source_plus_causal_tof_synthetic': compact_row(best_causal_row),
        },
        'filter_strategy_sweep': {
            'tof': [compact_row(row) for row in tof_rows],
            'causal_tof': [compact_row(row) for row in causal_rows],
        },
        'synthetic_files': synthetic_files,
        'target_anomaly_files': target_anomaly_files,
        'target_anomaly_source': target_anomaly_source,
        'target_labeled_jsonl': str(target_labeled_files[0]),
        'target_labeled_report_path': str(target_labeled_files[1]),
        'target_labeled_report': target_labeled_report,
        'target_anomaly_available': bool(target_anomalies),
        'skipped_metrics': [] if target_anomalies else ['precision', 'recall', 'f1', 'anomaly_f1', 'auroc', 'auprc'],
        'synthetic_filter_reports': {
            'raw': filter_report(raw_payload),
            'tof_selected': filter_report(best_tof_payload),
            'causal_tof_selected': filter_report(causal_payloads[str(best_causal_row.get('filter_strategy'))]),
            'tof_candidates': {strategy: filter_report(value) for strategy, value in tof_payloads.items()},
            'causal_tof_candidates': {strategy: filter_report(value) for strategy, value in causal_payloads.items()},
        },
        'notes': {
            'causal_branch_role': 'Causal branch is used only by Causal-TOF filtering/explanation, not as detection main branch.',
            'smartguard_standard_role': 'SmartGuard standard experiment remains a sanity check.',
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_tof_filter_analysis(PROJECT_ROOT / 'outputs/logs/TOF_FILTER_ANALYSIS.md', payload)
    write_final_run_report(PROJECT_ROOT / 'outputs/logs/FINAL_CONTEXT_SHIFT_RUN_REPORT.md', payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run final FR context-shift adaptation experiment.')
    parser.add_argument('--config', default='configs/context_shift_fr.yaml')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(resolve_path(args.config))
    payload = run(config)
    print('Wrote {}'.format(resolve_path(config['paths']['output_csv'])))
    print('Wrote {}'.format(resolve_path(config['paths']['output_json'])))
    print('source_context={} target_context={}'.format(payload['source_context'], payload['target_context']))
    if not payload['target_anomaly_available']:
        print('Target anomaly data not found; anomaly_f1/AUROC/AUPRC skipped.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
