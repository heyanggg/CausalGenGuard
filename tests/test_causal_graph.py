'''Tests for behavior causal graph utilities.'''

from __future__ import annotations

import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError('torch is required for causal graph tests')


def test_compute_gradient_causality_shape() -> None:
    _require_torch()
    from causal_gen_guard.models.causal_graph import compute_gradient_causality
    from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor

    torch.manual_seed(0)
    model = BehaviorCausalPredictor(input_channels=3, hidden_dim=8, num_layers=1, dropout=0.0)
    windows = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)

    A = compute_gradient_causality(model, windows, targets)

    assert A.shape == (2, 3, 3)
    assert torch.all(A >= 0)


def test_sparsify_causality_is_nonnegative() -> None:
    _require_torch()
    from causal_gen_guard.models.causal_graph import sparsify_causality

    A = torch.tensor(
        [[0.0, 0.8, 0.1], [0.2, 0.0, 0.7], [0.4, 0.1, 0.0]],
        dtype=torch.float32,
    )
    sparse = sparsify_causality(A, threshold=0.25)

    assert sparse.shape == (3, 3)
    assert torch.all(sparse >= 0)
    assert sparse[0, 1] > 0
    assert sparse[1, 0] == 0


def test_causal_deviation_score_same_graph_near_zero_and_different_larger() -> None:
    _require_torch()
    from causal_gen_guard.models.causal_graph import causal_deviation_score, normal_causal_pattern

    graphs = torch.tensor(
        [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
            [[0.0, 1.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    A_norm = normal_causal_pattern(graphs)
    same_score = causal_deviation_score(A_norm, A_norm)
    different = A_norm.clone()
    different[2, 0] = 2.0
    different_score = causal_deviation_score(different, A_norm)

    assert float(same_score) < 1e-6
    assert float(different_score) > float(same_score)


def main() -> None:
    if torch is None:
        print('test_causal_graph.py: skipped because torch is not installed')
        return
    test_compute_gradient_causality_shape()
    test_sparsify_causality_is_nonnegative()
    test_causal_deviation_score_same_graph_near_zero_and_different_larger()
    print('test_causal_graph.py: all checks passed')


if __name__ == '__main__':
    main()
