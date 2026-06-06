'''Fusion detector for reconstruction and causal-deviation anomaly scores.'''

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from causal_gen_guard.data.behavior_event_tensor import sequence_to_event_tensor, sliding_windows_from_tensor
from causal_gen_guard.data.schemas import BehaviorSequence
from causal_gen_guard.models.causal_graph import causal_deviation_score, compute_gradient_causality, normal_causal_pattern, sparsify_causality


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


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float32)
    return np.asarray(value, dtype=np.float32)


class FusionDetector:
    '''Fuse SmartGuard reconstruction and GCAD-style causal deviation scores.'''

    def __init__(
        self,
        backbone: Any,
        causal_model: Any,
        vocab: Dict[Any, int],
        A_norm_bank: Any = None,
        alpha: float = 0.6,
        beta: float = 0.4,
        delta: float = 0.0,
    ) -> None:
        self.backbone = backbone
        self.causal_model = causal_model
        self.vocab = {_control_key(key): int(value) for key, value in dict(vocab).items()}
        self.A_norm_bank = A_norm_bank
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.delta = float(delta)
        self.rec_mean = 0.0
        self.rec_std = 1.0
        self.causal_mean = 0.0
        self.causal_std = 1.0
        self.context_mean = 0.0
        self.context_std = 1.0
        self.threshold: Optional[float] = None
        self._last_raw_scores: List[dict[str, Any]] = []

    def _sequence_to_backbone_input(self, sequence: BehaviorSequence) -> Tuple[torch.Tensor, torch.Tensor]:
        fields = torch.zeros(1, len(sequence.events), 4, dtype=torch.float32)
        mask = torch.ones(1, len(sequence.events), dtype=torch.float32)
        for index, event in enumerate(sequence.events):
            control = _control_key(event.control_id)
            if control not in self.vocab:
                raise KeyError('control_id {!r} is missing from fusion vocab'.format(control))
            fields[0, index, 0] = _as_float(event.day)
            fields[0, index, 1] = _as_float(event.hour)
            fields[0, index, 2] = _as_float(event.duration)
            fields[0, index, 3] = float(self.vocab[control])
        return fields, mask

    def _reconstruction_detail(self, sequence: BehaviorSequence) -> Tuple[float, List[dict[str, Any]]]:
        if self.backbone is None or not sequence.events:
            return 0.0, []
        was_training = self.backbone.training
        self.backbone.eval()
        with torch.no_grad():
            fields, mask = self._sequence_to_backbone_input(sequence)
            outputs = self.backbone(fields, attention_mask=mask)
            token_losses_tensor = outputs['token_losses'][0].detach().cpu()
            score = float((outputs['token_losses'].sum() / mask.sum().clamp_min(1.0)).item())
        if was_training:
            self.backbone.train()
        token_details: List[dict[str, Any]] = []
        for index, event in enumerate(sequence.events):
            token_details.append(
                {
                    'position': index,
                    'control_id': event.control_id,
                    'rec_loss': float(token_losses_tensor[index].item()),
                    'day': event.day,
                    'hour': event.hour,
                }
            )
        return score, token_details

    def _candidate_norm_graphs(self, sequence: BehaviorSequence) -> List[Tuple[str, np.ndarray]]:
        if self.A_norm_bank is None:
            return [('global', np.zeros((len(self.vocab), len(self.vocab)), dtype=np.float32))]
        if isinstance(self.A_norm_bank, dict):
            context_id = sequence.context.get('context_id') or sequence.context.get('context') or sequence.context.get('season')
            if context_id is not None and context_id in self.A_norm_bank:
                return [(str(context_id), _to_numpy(self.A_norm_bank[context_id]))]
            return [(str(key), _to_numpy(value)) for key, value in self.A_norm_bank.items()]
        return [('global', _to_numpy(self.A_norm_bank))]

    def _causal_detail(self, sequence: BehaviorSequence) -> Tuple[float, str, np.ndarray, List[dict[str, Any]]]:
        channel_count = len(self.vocab)
        empty_graph = np.zeros((channel_count, channel_count), dtype=np.float32)
        if self.causal_model is None or not sequence.events or channel_count == 0:
            return 0.0, 'none', empty_graph, []

        X, _, mask = sequence_to_event_tensor(sequence, self.vocab, max_len=None, include_time_features=False)
        valid_len = int(mask.sum())
        if valid_len < 2:
            return 0.0, 'too_short', empty_graph, []
        window_size = int(getattr(self.causal_model, 'window_size', min(4, valid_len - 1)))
        window_size = max(1, min(window_size, valid_len - 1))
        windows, targets = sliding_windows_from_tensor(X[:valid_len], window_size=window_size, pred_horizon=1)
        if windows.shape[0] == 0:
            return 0.0, 'no_windows', empty_graph, []

        was_training = self.causal_model.training
        self.causal_model.eval()
        graph_batch = compute_gradient_causality(
            self.causal_model,
            torch.from_numpy(windows).float(),
            torch.from_numpy(targets).float(),
        )
        graph = normal_causal_pattern(sparsify_causality(graph_batch, threshold=0.0))
        graph_np = _to_numpy(graph)
        if was_training:
            self.causal_model.train()

        best_score: Optional[float] = None
        best_name = 'global'
        best_norm = None
        for name, norm_graph in self._candidate_norm_graphs(sequence):
            score_value = causal_deviation_score(graph_np, norm_graph)
            score = float(np.asarray(score_value).reshape(-1)[0])
            if best_score is None or score < best_score:
                best_score = score
                best_name = name
                best_norm = norm_graph
        if best_score is None:
            best_score = 0.0
            best_norm = empty_graph

        edge_scores = np.maximum(graph_np - np.asarray(best_norm, dtype=np.float32), 0.0)
        inverse_vocab = self._inverse_vocab()
        edge_details: List[dict[str, Any]] = []
        for src in range(edge_scores.shape[0]):
            for dst in range(edge_scores.shape[1]):
                weight = float(edge_scores[src, dst])
                if weight > 0.0:
                    edge_details.append({'from': inverse_vocab[src], 'to': inverse_vocab[dst], 'weight': weight})
        edge_details.sort(key=lambda item: item['weight'], reverse=True)
        return float(best_score), best_name, graph_np, edge_details

    def _inverse_vocab(self) -> List[Any]:
        inverse = [None] * len(self.vocab)
        for key, value in self.vocab.items():
            if 0 <= value < len(inverse):
                inverse[value] = key
        return inverse

    def _z(self, value: float, mean: float, std: float) -> float:
        return (float(value) - float(mean)) / max(float(std), 1e-8)

    def _raw_score_sequence(self, sequence: BehaviorSequence) -> dict[str, Any]:
        rec_score, token_details = self._reconstruction_detail(sequence)
        causal_score, context_graph, graph_np, edge_details = self._causal_detail(sequence)
        context_score = _as_float(sequence.context.get('context_score'), 0.0)
        return {
            'sequence_id': sequence.sequence_id,
            'rec_score': rec_score,
            'causal_score': causal_score,
            'context_score': context_score,
            'context_graph': context_graph,
            'token_details': token_details,
            'edge_details': edge_details,
            'graph': graph_np,
            'label': sequence.label,
            'anomaly_type': sequence.anomaly_type,
        }

    def _finalize_score(self, raw: dict[str, Any]) -> dict[str, Any]:
        z_rec = self._z(raw['rec_score'], self.rec_mean, self.rec_std)
        z_causal = self._z(raw['causal_score'], self.causal_mean, self.causal_std)
        z_context = self._z(raw['context_score'], self.context_mean, self.context_std) if self.delta else 0.0
        score = self.alpha * z_rec + self.beta * z_causal + self.delta * z_context
        result = dict(raw)
        result.update({'z_rec': z_rec, 'z_causal': z_causal, 'z_context': z_context, 'score': float(score)})
        if self.threshold is not None:
            result['pred'] = int(score > self.threshold)
        return result

    def score_sequence(self, sequence: BehaviorSequence) -> dict[str, Any]:
        '''Return fused score details for one sequence.'''
        return self._finalize_score(self._raw_score_sequence(sequence))

    def score_batch(self, sequences: List[BehaviorSequence]) -> List[dict[str, Any]]:
        '''Return fused score details for a batch of sequences.'''
        return [self.score_sequence(sequence) for sequence in sequences]

    def predict(self, sequences: List[BehaviorSequence], threshold: float) -> List[int]:
        '''Predict anomalies using score > threshold.'''
        self.threshold = float(threshold)
        return [int(item['score'] > self.threshold) for item in self.score_batch(sequences)]

    def calibrate_threshold(self, validation_sequences: List[BehaviorSequence], quantile: float = 0.95) -> float:
        '''Calibrate normalization statistics and threshold from validation sequences.'''
        if not validation_sequences:
            raise ValueError('validation_sequences must not be empty')
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError('quantile must be in [0, 1]')
        raw_scores = [self._raw_score_sequence(sequence) for sequence in validation_sequences]
        rec_values = np.asarray([item['rec_score'] for item in raw_scores], dtype=np.float32)
        causal_values = np.asarray([item['causal_score'] for item in raw_scores], dtype=np.float32)
        context_values = np.asarray([item['context_score'] for item in raw_scores], dtype=np.float32)
        self.rec_mean = float(rec_values.mean())
        self.rec_std = float(rec_values.std() or 1.0)
        self.causal_mean = float(causal_values.mean())
        self.causal_std = float(causal_values.std() or 1.0)
        self.context_mean = float(context_values.mean())
        self.context_std = float(context_values.std() or 1.0)
        finalized = [self._finalize_score(item) for item in raw_scores]
        self.threshold = float(np.quantile([item['score'] for item in finalized], quantile))
        self._last_raw_scores = raw_scores
        return self.threshold

    def explain(self, sequence: BehaviorSequence, top_k_tokens: int = 5, top_k_edges: int = 5) -> dict[str, Any]:
        '''Return compact token and edge explanations for one sequence.'''
        detail = self.score_sequence(sequence)
        tokens = sorted(detail.get('token_details', []), key=lambda item: item.get('rec_loss', 0.0), reverse=True)[:top_k_tokens]
        edges = detail.get('edge_details', [])[:top_k_edges]
        pred = detail.get('pred')
        if pred is None and self.threshold is not None:
            pred = int(detail['score'] > self.threshold)
        return {
            'score': detail['score'],
            'rec_score': detail['rec_score'],
            'causal_score': detail['causal_score'],
            'pred': pred,
            'top_tokens': tokens,
            'top_edges': edges,
            'context_graph': detail.get('context_graph'),
        }
