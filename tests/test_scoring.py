'''Tests for reconstruction and causal-aware loss weighting.'''

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
        raise RuntimeError('torch is required for scoring tests')


def test_noise_aware_weight_downweights_high_loss() -> None:
    _require_torch()
    from causal_gen_guard.models.losses import noise_aware_weight_from_loss_vector

    losses = torch.tensor([0.1, 2.0, 8.0], dtype=torch.float32)
    weights = noise_aware_weight_from_loss_vector(losses, mu=2.0)

    assert weights[0] > weights[1]
    assert weights[1] > weights[2]


def test_causal_weight_keeps_high_loss_low_centrality_low() -> None:
    _require_torch()
    from causal_gen_guard.models.losses import causal_aware_noise_weight

    losses = torch.tensor([0.1, 8.0, 8.0], dtype=torch.float32)
    centrality = torch.tensor([0.1, 0.1, 1.0], dtype=torch.float32)
    weights = causal_aware_noise_weight(losses, centrality, mu=2.0, gamma=4.0, min_w=0.05, max_w=2.0)

    assert weights[1] < weights[0]


def test_causal_weight_boosts_high_loss_high_centrality() -> None:
    _require_torch()
    from causal_gen_guard.models.losses import causal_aware_noise_weight

    losses = torch.tensor([0.1, 8.0, 8.0], dtype=torch.float32)
    centrality = torch.tensor([0.1, 0.1, 1.0], dtype=torch.float32)
    weights = causal_aware_noise_weight(losses, centrality, mu=2.0, gamma=4.0, min_w=0.05, max_w=2.0)

    assert weights[2] > weights[1]


def main() -> None:
    if torch is None:
        print('test_scoring.py: skipped because torch is not installed')
        return
    test_noise_aware_weight_downweights_high_loss()
    test_causal_weight_keeps_high_loss_low_centrality_low()
    test_causal_weight_boosts_high_loss_high_centrality()
    print('test_scoring.py: all checks passed')


if __name__ == '__main__':
    main()
