'''Command-line evaluation runner for CausalGenGuard FusionDetector.'''

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from causal_gen_guard.data.behavior_event_tensor import build_vocab
from causal_gen_guard.data.schemas import BehaviorSequence
from causal_gen_guard.evaluation.explain import (
    explain_one_sequence,
    plot_causal_graph_diff,
    plot_score_distribution,
    save_explanation_json,
)
from causal_gen_guard.evaluation.metrics import compute_binary_metrics
from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor
from causal_gen_guard.models.fusion_detector import FusionDetector
from causal_gen_guard.models.smartguard_backbone import SmartGuardBackbone


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
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
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float32)
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if hasattr(value, 'detach'):
        try:
            return value.detach().cpu().numpy().astype(np.float32)
        except Exception:
            pass
    return np.asarray(value, dtype=np.float32)


def load_jsonl_sequences(path: str | Path) -> List[BehaviorSequence]:
    sequences: List[BehaviorSequence] = []
    path = Path(path)
    with path.open('r', encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                sequences.append(BehaviorSequence.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError('Failed to parse {}:{}: {}'.format(path, line_number, exc)) from exc
    if not sequences:
        raise ValueError('No sequences loaded from {}'.format(path))
    return sequences


def load_backbone_checkpoint(path: Optional[str]) -> Tuple[Any, Optional[Dict[Any, int]]]:
    if not path:
        return None, None
    checkpoint = torch.load(path, map_location='cpu')
    config = dict(checkpoint.get('model_config', {}))
    if 'vocab_size' not in config:
        config['vocab_size'] = len(checkpoint.get('vocab', {})) or len(checkpoint.get('inverse_vocab', []))
    config.pop('causal_aware_nwrl', None)
    model = SmartGuardBackbone(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    vocab = checkpoint.get('vocab')
    if isinstance(vocab, dict):
        model.vocab = vocab
        model.control_vocab = vocab
    model.eval()
    return model, vocab if isinstance(vocab, dict) else None


def load_causal_checkpoint(path: Optional[str]) -> Tuple[Any, Optional[Dict[Any, int]]]:
    if not path:
        return None, None
    checkpoint = torch.load(path, map_location='cpu')
    config = dict(checkpoint.get('model_config', {}))
    if 'input_channels' not in config:
        config['input_channels'] = len(checkpoint.get('vocab', {})) or len(checkpoint.get('inverse_vocab', []))
    model = BehaviorCausalPredictor(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.window_size = int(checkpoint.get('window_size', 4))
    model.eval()
    vocab = checkpoint.get('vocab')
    return model, vocab if isinstance(vocab, dict) else None


def load_a_norm(path: Optional[str]) -> Any:
    if not path:
        return None
    path_obj = Path(path)
    if path_obj.suffix.lower() == '.json':
        with path_obj.open('r', encoding='utf-8') as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            return {key: np.asarray(value, dtype=np.float32) for key, value in payload.items()}
        return np.asarray(payload, dtype=np.float32)
    return np.load(path_obj)


def choose_vocab(sequences: List[BehaviorSequence], *vocabs: Optional[Dict[Any, int]]) -> Dict[Any, int]:
    for vocab in vocabs:
        if isinstance(vocab, dict) and vocab:
            return vocab
    return dict(build_vocab(sequences))


def inverse_vocab(vocab: Dict[Any, int]) -> List[Any]:
    inverse: List[Any] = [None] * len(vocab)
    for key, value in dict(vocab).items():
        index = int(value)
        if 0 <= index < len(inverse):
            inverse[index] = key
    return inverse


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


def norm_graph_for_score(score_item: Dict[str, Any], A_norm: Any, vocab_size: int) -> np.ndarray:
    if A_norm is None:
        return np.zeros((vocab_size, vocab_size), dtype=np.float32)
    if isinstance(A_norm, dict):
        context_graph = score_item.get('context_graph')
        if context_graph in A_norm:
            return _to_numpy(A_norm[context_graph])
        if 'global' in A_norm:
            return _to_numpy(A_norm['global'])
        if A_norm:
            return _to_numpy(next(iter(A_norm.values())))
        return np.zeros((vocab_size, vocab_size), dtype=np.float32)
    return _to_numpy(A_norm)


def safe_plot(warnings_out: List[str], description: str, func: Any, *args: Any, **kwargs: Any) -> bool:
    try:
        func(*args, **kwargs)
        return True
    except Exception as exc:
        warnings_out.append('{} failed: {}'.format(description, exc))
        return False


def save_explanations(
    detector: FusionDetector,
    sequences: List[BehaviorSequence],
    scores: List[Dict[str, Any]],
    A_norm: Any,
    vocab: Dict[Any, int],
    explanation_dir: Path,
    figure_dir: Path,
    num_explanations: int,
    warnings_out: List[str],
) -> List[Dict[str, Any]]:
    explanation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(range(len(scores)), key=lambda index: float(scores[index].get('score', 0.0)), reverse=True)[: max(0, num_explanations)]
    inverse = inverse_vocab(vocab)
    manifest: List[Dict[str, Any]] = []
    for rank, index in enumerate(selected, start=1):
        sequence = sequences[index]
        score_item = scores[index]
        safe_id = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in sequence.sequence_id) or 'sequence_{}'.format(index)
        explanation_path = explanation_dir / '{}_{:03d}.json'.format(safe_id, rank)
        graph_path = figure_dir / '{}_{:03d}_causal_diff.png'.format(safe_id, rank)
        explanation = explain_one_sequence(detector, sequence)
        explanation['case_study_rank'] = rank
        explanation['source_score_record'] = {
            'score': score_item.get('score'),
            'rec_score': score_item.get('rec_score'),
            'causal_score': score_item.get('causal_score'),
            'pred': score_item.get('pred'),
            'label': sequence.label,
        }
        explanation['figures'] = {}
        graph = _to_numpy(score_item.get('graph'))
        if graph.size:
            norm_graph = norm_graph_for_score(score_item, A_norm, len(vocab))
            if safe_plot(warnings_out, 'causal graph diff plot for {}'.format(sequence.sequence_id), plot_causal_graph_diff, graph, norm_graph, inverse, graph_path):
                explanation['figures']['causal_graph_diff'] = str(graph_path)
        else:
            warnings_out.append('No causal graph available for explanation {}'.format(sequence.sequence_id))
        explanation['warnings'] = [warning for warning in warnings_out if sequence.sequence_id in warning]
        save_explanation_json(explanation, explanation_path)
        manifest.append({'sequence_id': sequence.sequence_id, 'rank': rank, 'path': str(explanation_path), 'figure': str(graph_path) if graph_path.exists() else None})
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Evaluate CausalGenGuard FusionDetector on JSONL sequences.')
    parser.add_argument('--sequences', required=True, help='Test JSONL containing BehaviorSequence records.')
    parser.add_argument('--checkpoint-backbone', default=None, help='SmartGuard backbone checkpoint path.')
    parser.add_argument('--checkpoint-causal', default=None, help='Behavior causal predictor checkpoint path.')
    parser.add_argument('--a-norm', default=None, help='A_norm .npy or JSON graph bank path.')
    parser.add_argument('--threshold', type=float, default=None, help='Anomaly threshold. If omitted, calibrates on provided sequences.')
    parser.add_argument('--output', default='outputs/results/eval.json')
    parser.add_argument('--alpha', type=float, default=0.6)
    parser.add_argument('--beta', type=float, default=0.4)
    parser.add_argument('--delta', type=float, default=0.0)
    parser.add_argument('--calibration-quantile', type=float, default=0.95)
    parser.add_argument('--save-explanations', action='store_true', help='Save per-sequence explanation JSON and optional figures.')
    parser.add_argument('--num-explanations', type=int, default=5, help='Number of highest-score sequences to explain.')
    parser.add_argument('--explanation-dir', default='outputs/results/explanations')
    parser.add_argument('--figure-dir', default='outputs/figures')
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_explanations < 0:
        raise ValueError('--num-explanations must be non-negative')

    warnings_out: List[str] = []
    sequences = load_jsonl_sequences(args.sequences)
    backbone, backbone_vocab = load_backbone_checkpoint(args.checkpoint_backbone)
    causal_model, causal_vocab = load_causal_checkpoint(args.checkpoint_causal)
    vocab = choose_vocab(sequences, causal_vocab, backbone_vocab)
    A_norm = load_a_norm(args.a_norm)

    detector = FusionDetector(
        backbone=backbone,
        causal_model=causal_model,
        vocab=vocab,
        A_norm_bank=A_norm,
        alpha=args.alpha,
        beta=args.beta,
        delta=args.delta,
    )
    if args.threshold is None:
        threshold = detector.calibrate_threshold(sequences, quantile=args.calibration_quantile)
        threshold_source = 'calibrated_from_sequences'
    else:
        threshold = float(args.threshold)
        detector.threshold = threshold
        threshold_source = 'cli'

    scores = detector.score_batch(sequences)
    predictions = [int(item['score'] > threshold) for item in scores]
    labels = [sequence.label for sequence in sequences]
    has_labels = all(label is not None for label in labels)
    metrics = None
    if has_labels:
        metrics = compute_binary_metrics([label_to_int(label) for label in labels], [item['score'] for item in scores], threshold=threshold)

    explanations_manifest: List[Dict[str, Any]] = []
    figures: Dict[str, Any] = {}
    if args.save_explanations:
        explanation_dir = Path(args.explanation_dir)
        figure_dir = Path(args.figure_dir)
        if has_labels:
            normal_scores = [float(item['score']) for item, label in zip(scores, labels) if label_to_int(label) == 0]
            anomaly_scores = [float(item['score']) for item, label in zip(scores, labels) if label_to_int(label) == 1]
            score_dist_path = figure_dir / 'score_distribution.png'
            if safe_plot(warnings_out, 'score distribution plot', plot_score_distribution, normal_scores, anomaly_scores, score_dist_path):
                figures['score_distribution'] = str(score_dist_path)
        else:
            warnings_out.append('Labels missing; score distribution normal/anomaly plot skipped')
        explanations_manifest = save_explanations(
            detector,
            sequences,
            scores,
            A_norm,
            vocab,
            explanation_dir,
            figure_dir,
            args.num_explanations,
            warnings_out,
        )

    result = {
        'sequence_count': len(sequences),
        'threshold': threshold,
        'threshold_source': threshold_source,
        'weights': {'alpha': args.alpha, 'beta': args.beta, 'delta': args.delta},
        'normalization': {
            'rec_mean': detector.rec_mean,
            'rec_std': detector.rec_std,
            'causal_mean': detector.causal_mean,
            'causal_std': detector.causal_std,
            'context_mean': detector.context_mean,
            'context_std': detector.context_std,
        },
        'metrics': metrics,
        'warnings': warnings_out,
        'figures': figures,
        'explanations': explanations_manifest,
        'scores': [
            {
                'sequence_id': item['sequence_id'],
                'score': item['score'],
                'rec_score': item['rec_score'],
                'causal_score': item['causal_score'],
                'z_rec': item['z_rec'],
                'z_causal': item['z_causal'],
                'pred': predictions[index],
                'label': labels[index],
                'anomaly_type': item.get('anomaly_type'),
                'context_graph': item.get('context_graph'),
            }
            for index, item in enumerate(scores)
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        json.dump(_json_safe(result), handle, ensure_ascii=False, indent=2)
    print('Wrote evaluation results to {}'.format(output))
    if warnings_out:
        print('Evaluation warnings: {}'.format('; '.join(warnings_out)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
