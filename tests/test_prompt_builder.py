'''Tests for offline SmartGen prompt builder.'''

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.generation.prompt_builder import build_smartgen_prompt


def test_prompt_contains_context_and_causal_hints() -> None:
    prompt = build_smartgen_prompt(
        original_context={'home': 'fr', 'season': 'winter'},
        new_context={'home': 'fr', 'season': 'spring'},
        device_info={'devices': ['light'], 'controls': ['on', 'off']},
        compressed_sequences=[['light_on', 'light_off']],
        transition_hints=[{'from': 'light_on', 'to': 'light_off', 'count': 3}],
        causal_hints=[{'from': 'light_on', 'to': 'light_off', 'weight': 0.9}],
    )

    assert 'IoT expert' in prompt
    assert 'original_context' in prompt
    assert 'new_context' in prompt
    assert 'causal_hints' in prompt
    assert 'spring' in prompt
    assert 'light_off' in prompt
    assert 'JSON' in prompt or '<seq' in prompt


def main() -> None:
    test_prompt_contains_context_and_causal_hints()
    print('test_prompt_builder.py: all checks passed')


if __name__ == '__main__':
    main()
