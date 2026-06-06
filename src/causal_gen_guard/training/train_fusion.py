'''Context-shift training pipeline for the CausalGenGuard fusion detector.

The pipeline is intentionally conservative: it is CPU-only, uses short default
training loops, and provides a --smoke-test mode for validating source-to-target
context experiments without launching a full benchmark run.
'''

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import yaml
except Exception:  # pragma: no cover - handled at runtime with a clear message.
    yaml = None

from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorSequence
from causal_gen_guard.evaluation.metrics import compute_binary_metrics
from causal_gen_guard.generation.causal_tof import run_causal_tof
from causal_gen_guard.models.causal_graph import graph_centrality, normal_causal_pattern, sparsify_causality, compute_gradient_causality
from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor
from causal_gen_guard.models.fusion_detector import FusionDetector
from causal_gen_guard.models.losses import causal_aware_noise_weight
from causal_gen_guard.models.smartguard_backbone import SmartGuardBackbone, compute_reconstruction_loss
from causal_gen_guard.training.train_backbone import SequenceTensorDataset
from causal_gen_guard.training.train_causal import WindowDataset, build_window_arrays


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> Dict[str, Any]:
    if yaml is None:
        raise RuntimeError('PyYAML is required to read context-shift YAML configs')
    with Path(path).open('r', encoding='utf-8') as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError('config must be a YAML mapping')
    return data


def deep_update(base: Dict[str, Any], updates: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in dict(updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def resolve_path(path_value: Any, root: Optional[Path] = None) -> Optional[Path]:
    if path_value in (None, ''):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return path
    return (root or project_root()) / path


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
    return value


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        json.dump(json_safe(payload), handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(json_safe(row), ensure_ascii=False) + '\n')
            count += 1
    return count


def load_jsonl_sequences(path: Path, max_sequences: Optional[int] = None, required: bool = True) -> List[BehaviorSequence]:
    if path is None or not path.exists():
        if required:
            raise FileNotFoundError('sequence JSONL not found: {}'.format(path))
        return []
    sequences: List[BehaviorSequence] = []
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sequences.append(BehaviorSequence.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError('failed to parse {}:{}: {}'.format(path, line_number, exc)) from exc
            if max_sequences is not None and len(sequences) >= max_sequences:
                break
    if required and not sequences:
        raise ValueError('no BehaviorSequence records loaded from {}'.format(path))
    return sequences


def label_to_int(label: Any) -> int:
    if label is None:
        return 0
    if isinstance(label, bool):
        return int(label)
    if isinstance(label, (int, float)):
        return int(label != 0)
    text = str(label).strip().lower()
    if text in ('0', 'normal', 'benign', 'clean', 'false', 'none'):
        return 0
    return 1


def limit_for_smoke(sequences: List[BehaviorSequence], smoke_test: bool, count: int = 8) -> List[BehaviorSequence]:
    if not smoke_test:
        return sequences
    return list(sequences[:count])


def safe_nhead(hidden_dim: int) -> int:
    for candidate in (8, 4, 2, 1):
        if hidden_dim % candidate == 0:
            return candidate
    return 1


def max_event_len(sequences: List[BehaviorSequence]) -> int:
    return max((len(sequence.events) for sequence in sequences), default=0)


def train_backbone_stage(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    config: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
    smoke_test: bool = False,
    causal_centrality: Optional[np.ndarray] = None,
) -> Tuple[SmartGuardBackbone, Dict[str, Any]]:
    if not sequences:
        raise ValueError('cannot train SmartGuardBackbone without sequences')
    model_config = dict(config.get('model', {}).get('backbone', {}))
    hidden_dim = int(model_config.get('embedding_dim', model_config.get('hidden_dim', 64)))
    layers = int(model_config.get('layers', model_config.get('num_layers', 1)))
    epochs = int(model_config.get('epochs', 1))
    if smoke_test:
        epochs = min(epochs, 1)
        hidden_dim = min(hidden_dim, 32)
    nhead = int(model_config.get('nhead', safe_nhead(hidden_dim)))
    if hidden_dim % nhead != 0:
        nhead = safe_nhead(hidden_dim)
    batch_size = int(model_config.get('batch_size', 16 if smoke_test else 32))
    learning_rate = float(model_config.get('learning_rate', 1e-3))
    dropout = float(model_config.get('dropout', 0.1))
    nwrl_mu = float(model_config.get('nwrl_mu', 1.0))
    nwrl_gamma = float(model_config.get('nwrl_gamma', 1.0))
    nwrl_min_w = float(model_config.get('nwrl_min_w', 0.1))
    nwrl_max_w = float(model_config.get('nwrl_max_w', 2.0))
    centrality_tensor = None
    if causal_centrality is not None:
        candidate = torch.as_tensor(causal_centrality, dtype=torch.float32)
        if candidate.numel() == len(vocab):
            centrality_tensor = candidate.reshape(-1)
    max_len = int(model_config.get('max_len') or max_event_len(sequences))
    if max_len <= 0:
        raise ValueError('all sequences are empty; cannot train backbone')

    dataset = SequenceTensorDataset(sequences, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = SmartGuardBackbone(
        vocab_size=len(vocab),
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=layers,
        dropout=dropout,
        use_ttpe=True,
    )
    model.control_vocab = dict(vocab)
    model.max_len = max_len
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_tokens = 0.0
        for fields, mask in loader:
            optimizer.zero_grad()
            outputs = model(fields, attention_mask=mask)
            targets = fields[..., 3].long()
            token_losses = compute_reconstruction_loss(outputs['logits'], targets, mask=mask, reduction='none')
            if centrality_tensor is not None:
                with torch.no_grad():
                    loss_vector = torch.zeros(len(vocab), dtype=torch.float32, device=token_losses.device)
                    loss_counts = torch.zeros(len(vocab), dtype=torch.float32, device=token_losses.device)
                    valid = mask > 0
                    if valid.any():
                        valid_targets = targets[valid]
                        valid_losses = token_losses.detach()[valid]
                        loss_vector.scatter_add_(0, valid_targets, valid_losses)
                        loss_counts.scatter_add_(0, valid_targets, torch.ones_like(valid_losses))
                    loss_vector = loss_vector / loss_counts.clamp_min(1.0)
                    channel_weights = causal_aware_noise_weight(
                        loss_vector,
                        centrality_tensor.to(token_losses.device),
                        mu=nwrl_mu,
                        gamma=nwrl_gamma,
                        min_w=nwrl_min_w,
                        max_w=nwrl_max_w,
                    )
                token_weights = channel_weights[targets] * mask
                loss = (token_losses * token_weights).sum() / token_weights.sum().clamp_min(1.0)
            else:
                loss = compute_reconstruction_loss(outputs['logits'], targets, mask=mask, reduction='mean')
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                token_count = float(mask.sum().item())
                total_loss += float(loss.detach().item()) * token_count
                total_tokens += token_count
        mean_loss = total_loss / max(total_tokens, 1.0)
        history.append({'epoch': float(epoch + 1), 'train_loss': float(mean_loss)})
        print('{} backbone epoch {}/{} loss={:.6f}'.format(stage_name, epoch + 1, epochs, mean_loss))

    checkpoint_path = output_dir / 'checkpoints' / '{}_smartguard_backbone.pt'.format(stage_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab': dict(vocab),
            'inverse_vocab': inverse_vocab_list(vocab),
            'max_len': max_len,
            'model_config': {
                'vocab_size': len(vocab),
                'hidden_dim': hidden_dim,
                'nhead': nhead,
                'num_layers': layers,
                'dropout': dropout,
                'use_ttpe': True,
                'causal_aware_nwrl': centrality_tensor is not None,
            },
            'history': history,
        },
        checkpoint_path,
    )
    return model, {'checkpoint': str(checkpoint_path), 'history': history, 'max_len': max_len, 'causal_aware_nwrl': centrality_tensor is not None}


def train_causal_stage(
    sequences: List[BehaviorSequence],
    vocab: Dict[Any, int],
    config: Dict[str, Any],
    output_dir: Path,
    stage_name: str,
    smoke_test: bool = False,
) -> Tuple[Optional[BehaviorCausalPredictor], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    if not sequences:
        return None, None, None, {'skipped': True, 'reason': 'no training sequences'}
    model_config = dict(config.get('model', {}).get('causal', {}))
    tau = int(model_config.get('tau', model_config.get('window_size', 4)))
    if max_event_len(sequences) < 2:
        return None, None, None, {'skipped': True, 'reason': 'sequences are too short for causal windows'}
    window_size = max(1, min(tau, max_event_len(sequences) - 1))
    hidden_dim = int(model_config.get('hidden_dim', 64))
    if smoke_test:
        hidden_dim = min(hidden_dim, 32)
    layers = int(model_config.get('layers', model_config.get('num_layers', 1)))
    epochs = int(model_config.get('epochs', 1))
    if smoke_test:
        epochs = min(epochs, 1)
    batch_size = int(model_config.get('batch_size', 16 if smoke_test else 32))
    learning_rate = float(model_config.get('learning_rate', 1e-3))
    dropout = float(model_config.get('dropout', 0.1))
    threshold_h = float(model_config.get('threshold_h', model_config.get('sparse_threshold', 0.0)))

    try:
        windows, time_windows, targets = build_window_arrays(sequences, vocab, window_size, pred_horizon=1)
    except ValueError as exc:
        return None, None, None, {'skipped': True, 'reason': str(exc)}
    dataset = WindowDataset(windows, time_windows, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = BehaviorCausalPredictor(
        input_channels=len(vocab),
        hidden_dim=hidden_dim,
        time_feature_dim=time_windows.shape[-1],
        num_layers=layers,
        dropout=dropout,
    )
    model.window_size = window_size
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
        mean_loss = total_loss / max(total_count, 1)
        history.append({'epoch': float(epoch + 1), 'train_loss': float(mean_loss)})
        print('{} causal epoch {}/{} loss={:.6f}'.format(stage_name, epoch + 1, epochs, mean_loss))

    sample_count = min(int(model_config.get('causality_samples', 16 if smoke_test else 64)), len(dataset))
    graph_batch = compute_gradient_causality(model, dataset.windows[:sample_count], dataset.targets[:sample_count])
    sparse_graphs = sparsify_causality(graph_batch, threshold=threshold_h)
    A_norm_tensor = normal_causal_pattern(sparse_graphs)
    centrality_tensor = graph_centrality(A_norm_tensor)
    A_norm = A_norm_tensor.detach().cpu().numpy().astype(np.float32)
    centrality = centrality_tensor.detach().cpu().numpy().astype(np.float32)

    checkpoint_path = output_dir / 'checkpoints' / '{}_causal_predictor.pt'.format(stage_name)
    graph_path = output_dir / 'results' / '{}_A_norm.npy'.format(stage_name)
    centrality_path = output_dir / 'results' / '{}_causal_centrality.npy'.format(stage_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state_dict': model.state_dict(),
            'vocab': dict(vocab),
            'inverse_vocab': inverse_vocab_list(vocab),
            'model_config': {
                'input_channels': len(vocab),
                'hidden_dim': hidden_dim,
                'time_feature_dim': time_windows.shape[-1],
                'num_layers': layers,
                'dropout': dropout,
            },
            'window_size': window_size,
            'pred_horizon': 1,
            'history': history,
        },
        checkpoint_path,
    )
    np.save(graph_path, A_norm)
    np.save(centrality_path, centrality)
    info = {
        'checkpoint': str(checkpoint_path),
        'A_norm': str(graph_path),
        'causal_centrality': str(centrality_path),
        'history': history,
        'window_size': window_size,
        'window_count': int(len(dataset)),
    }
    return model, A_norm, centrality, info


def inverse_vocab_list(vocab: Dict[Any, int]) -> List[Any]:
    inverse: List[Any] = [None] * len(vocab)
    for key, value in dict(vocab).items():
        if 0 <= int(value) < len(inverse):
            inverse[int(value)] = key
    return inverse


def calibrate_component_thresholds(detector: FusionDetector, validation_sequences: List[BehaviorSequence], quantile: float) -> Dict[str, float]:
    raw_scores = [detector._raw_score_sequence(sequence) for sequence in validation_sequences]
    if not raw_scores:
        return {'reconstruction_threshold': 0.0, 'causal_threshold': 0.0}
    rec_values = np.asarray([item['rec_score'] for item in raw_scores], dtype=np.float32)
    causal_values = np.asarray([item['causal_score'] for item in raw_scores], dtype=np.float32)
    return {
        'reconstruction_threshold': float(np.quantile(rec_values, quantile)),
        'causal_threshold': float(np.quantile(causal_values, quantile)),
    }


def build_detector(
    backbone: SmartGuardBackbone,
    causal_model: Optional[BehaviorCausalPredictor],
    vocab: Dict[Any, int],
    A_norm: Optional[np.ndarray],
    config: Dict[str, Any],
) -> FusionDetector:
    fusion_config = dict(config.get('fusion', {}))
    alpha = float(fusion_config.get('alpha', 0.6))
    beta = float(fusion_config.get('beta', 0.4)) if causal_model is not None else 0.0
    delta = float(fusion_config.get('delta', 0.0))
    if A_norm is None:
        A_norm = np.zeros((len(vocab), len(vocab)), dtype=np.float32)
    return FusionDetector(backbone, causal_model, vocab, A_norm_bank=A_norm, alpha=alpha, beta=beta, delta=delta)


def compact_score_record(item: Dict[str, Any]) -> Dict[str, Any]:
    return {
        'sequence_id': item.get('sequence_id'),
        'label': item.get('label'),
        'anomaly_type': item.get('anomaly_type'),
        'score': item.get('score'),
        'rec_score': item.get('rec_score'),
        'causal_score': item.get('causal_score'),
        'pred': item.get('pred'),
        'context_graph': item.get('context_graph'),
    }


def evaluate_detector(detector: FusionDetector, test_sequences: List[BehaviorSequence], threshold: float) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    if not test_sequences:
        empty_metrics = {
            'precision': float('nan'),
            'recall': float('nan'),
            'f1': float('nan'),
            'fpr': float('nan'),
            'fnr': float('nan'),
            'auroc': float('nan'),
            'auprc': float('nan'),
        }
        return empty_metrics, []
    detector.threshold = float(threshold)
    score_items = detector.score_batch(test_sequences)
    labels = [label_to_int(sequence.label) for sequence in test_sequences]
    scores = [float(item['score']) for item in score_items]
    metrics = compute_binary_metrics(labels, scores, threshold)
    return metrics, [compact_score_record(item) for item in score_items]


def load_context_sequences(config: Dict[str, Any], smoke_test: bool) -> Tuple[List[BehaviorSequence], List[BehaviorSequence], List[BehaviorSequence], List[BehaviorSequence], List[str]]:
    root = project_root()
    paths = dict(config.get('paths', {}))
    notes: List[str] = []
    smoke_count = int(config.get('smoke_max_sequences', 8))

    source_train_path = resolve_path(paths.get('source_train_jsonl'), root)
    source_val_path = resolve_path(paths.get('source_val_jsonl'), root)
    target_test_path = resolve_path(paths.get('target_normal_test_jsonl'), root)
    synthetic_path = resolve_path(paths.get('synthetic_target_jsonl'), root)

    source_train = load_jsonl_sequences(source_train_path, max_sequences=smoke_count if smoke_test else None, required=True)
    source_val = load_jsonl_sequences(source_val_path, max_sequences=smoke_count if smoke_test else None, required=False)
    if not source_val:
        source_val = limit_for_smoke(source_train, True, max(1, min(len(source_train), smoke_count)))
        notes.append('source_val_jsonl missing or empty; reused source_train subset for calibration')

    target_test = load_jsonl_sequences(target_test_path, max_sequences=smoke_count if smoke_test else None, required=False)
    if not target_test:
        notes.append('target_normal_test_jsonl missing or empty; metrics will be NaN')

    synthetic = load_jsonl_sequences(synthetic_path, max_sequences=smoke_count if smoke_test else None, required=False)
    if not synthetic:
        notes.append('synthetic_target_jsonl missing or empty; SmartGen augmentation skipped')
    return source_train, source_val, target_test, synthetic, notes


def run_causal_tof_prefilter(
    synthetic_sequences: List[BehaviorSequence],
    source_train: List[BehaviorSequence],
    source_val: List[BehaviorSequence],
    config: Dict[str, Any],
    output_dir: Path,
    smoke_test: bool,
) -> Tuple[List[BehaviorSequence], Dict[str, Any]]:
    if not synthetic_sequences:
        return [], {'skipped': True, 'reason': 'no synthetic sequences'}
    pre_dir = output_dir / 'causal_tof_prefilter'
    source_vocab = build_vocab(source_train)
    pre_backbone, pre_backbone_info = train_backbone_stage(source_train, source_vocab, config, pre_dir, 'pre_tof', smoke_test)
    use_causal = bool(config.get('experiments', {}).get('use_causal', True))
    pre_causal_model = None
    pre_A_norm = None
    pre_causal_info: Dict[str, Any] = {'skipped': True, 'reason': 'causal disabled'}
    if use_causal:
        pre_causal_model, pre_A_norm, _, pre_causal_info = train_causal_stage(source_train, source_vocab, config, pre_dir, 'pre_tof', smoke_test)
    detector = build_detector(pre_backbone, pre_causal_model, source_vocab, pre_A_norm, config)
    quantile = float(config.get('fusion', {}).get('threshold_quantile', 0.95))
    detector.calibrate_threshold(source_val, quantile=quantile)
    component_thresholds = calibrate_component_thresholds(detector, source_val, quantile)
    report = run_causal_tof(
        synthetic_sequences,
        output_dir=pre_dir / 'filtered_synthetic',
        backbone=pre_backbone,
        reconstruction_threshold=component_thresholds['reconstruction_threshold'],
        causal_model=pre_causal_model,
        A_norm=pre_A_norm,
        vocab=source_vocab,
        causal_threshold=component_thresholds['causal_threshold'] if pre_causal_model is not None else None,
        train_sequences=source_train,
        val_sequences=source_val,
        utility_config={'enabled': False},
    )
    kept_path = Path(report['outputs']['kept_jsonl'])
    kept_sequences = load_jsonl_sequences(kept_path, required=False)
    report['prefilter_backbone'] = pre_backbone_info
    report['prefilter_causal'] = pre_causal_info
    report['component_thresholds'] = component_thresholds
    return kept_sequences, report


def run_pipeline(config: Dict[str, Any], smoke_test: bool = False, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    config = deep_update(config, overrides)
    seed = int(config.get('seed', 42))
    set_seed(seed)
    dataset = str(config.get('dataset', 'unknown'))
    transition = '{}->{}'.format(config.get('source_context', 'source'), config.get('target_context', 'target'))
    output_dir = resolve_path(config.get('paths', {}).get('output_dir') or 'outputs/results/context_shift_{}'.format(dataset), project_root())
    run_name = str(config.get('run_name', 'fusion_smoke' if smoke_test else 'fusion'))
    output_dir = output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_train, source_val, target_test, synthetic, notes = load_context_sequences(config, smoke_test=smoke_test)
    experiments = dict(config.get('experiments', {}))
    use_smartgen = bool(experiments.get('use_smartgen', True))
    use_causal = bool(experiments.get('use_causal', True))
    use_causal_tof = bool(experiments.get('use_causal_tof', True))

    synthetic_used: List[BehaviorSequence] = []
    causal_tof_report: Dict[str, Any] = {'skipped': True}
    if use_smartgen and synthetic:
        synthetic_used = synthetic
        if use_causal_tof:
            synthetic_used, causal_tof_report = run_causal_tof_prefilter(synthetic, source_train, source_val, config, output_dir, smoke_test)
            notes.append('Causal-TOF filtered synthetic sequences from {} to {}'.format(len(synthetic), len(synthetic_used)))
    elif use_smartgen:
        notes.append('use_smartgen is true but no synthetic sequences were available')
    else:
        notes.append('SmartGen augmentation disabled by experiment config')

    train_sequences = list(source_train) + list(synthetic_used)
    if not train_sequences:
        raise ValueError('no training sequences available after loading and filtering')
    vocab = build_vocab(train_sequences)
    final_backbone, backbone_info = train_backbone_stage(train_sequences, vocab, config, output_dir, 'final', smoke_test)
    causal_model = None
    A_norm = None
    centrality = None
    if use_causal:
        causal_model, A_norm, centrality, causal_info = train_causal_stage(train_sequences, vocab, config, output_dir, 'final', smoke_test)
    else:
        causal_info = {'skipped': True, 'reason': 'causal branch disabled'}

    use_causal_aware_nwrl = bool(experiments.get('use_causal_aware_nwrl', False))
    if use_causal_aware_nwrl and centrality is not None:
        final_backbone, nwrl_info = train_backbone_stage(
            train_sequences,
            vocab,
            config,
            output_dir,
            'final_nwrl',
            smoke_test,
            causal_centrality=centrality,
        )
        backbone_info['causal_aware_nwrl_finetune'] = nwrl_info
        notes.append('causal-aware NWRL fine-tune applied after causal centrality estimation')
    elif use_causal_aware_nwrl:
        notes.append('causal-aware NWRL requested but causal centrality was unavailable')

    detector = build_detector(final_backbone, causal_model, vocab, A_norm, config)
    quantile = float(config.get('fusion', {}).get('threshold_quantile', 0.95))
    threshold = detector.calibrate_threshold(source_val, quantile=quantile)
    metrics, score_records = evaluate_detector(detector, target_test, threshold)

    result_paths = {
        'summary': str(output_dir / 'summary.json'),
        'fusion_config': str(output_dir / 'fusion_detector_config.json'),
        'scores': str(output_dir / 'fusion_scores.jsonl'),
    }
    write_jsonl(output_dir / 'fusion_scores.jsonl', score_records)
    fusion_payload = {
        'dataset': dataset,
        'transition': transition,
        'threshold': threshold,
        'alpha': detector.alpha,
        'beta': detector.beta,
        'delta': detector.delta,
        'rec_mean': detector.rec_mean,
        'rec_std': detector.rec_std,
        'causal_mean': detector.causal_mean,
        'causal_std': detector.causal_std,
        'vocab': dict(vocab),
        'inverse_vocab': inverse_vocab_list(vocab),
        'A_norm_path': causal_info.get('A_norm') if isinstance(causal_info, dict) else None,
        'centrality_path': causal_info.get('causal_centrality') if isinstance(causal_info, dict) else None,
    }
    write_json(output_dir / 'fusion_detector_config.json', fusion_payload)

    summary = {
        'dataset': dataset,
        'transition': transition,
        'smoke_test': smoke_test,
        'counts': {
            'source_train': len(source_train),
            'source_val': len(source_val),
            'target_test': len(target_test),
            'synthetic_loaded': len(synthetic),
            'synthetic_used': len(synthetic_used),
            'train_total': len(train_sequences),
            'vocab_size': len(vocab),
        },
        'experiments': experiments,
        'metrics': metrics,
        'threshold': threshold,
        'backbone': backbone_info,
        'causal': causal_info,
        'causal_tof': causal_tof_report,
        'notes': notes,
        'outputs': result_paths,
    }
    write_json(output_dir / 'summary.json', summary)
    print('Saved context-shift fusion summary to {}'.format(output_dir / 'summary.json'))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run a context-shift CausalGenGuard fusion training pipeline.')
    parser.add_argument('--config', required=True, help='YAML config such as configs/context_shift_fr.yaml')
    parser.add_argument('--smoke-test', action='store_true', help='Use a tiny subset and one epoch per stage')
    parser.add_argument('--run-name', default=None, help='Optional subdirectory name under paths.output_dir')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    overrides: Dict[str, Any] = {}
    if args.run_name:
        overrides['run_name'] = args.run_name
    run_pipeline(config, smoke_test=args.smoke_test, overrides=overrides)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
