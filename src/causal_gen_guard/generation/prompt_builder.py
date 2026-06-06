'''Prompt builder for offline SmartGen-compatible context generation.

This module only builds prompt strings. It never calls online LLM APIs and never
requires API keys.
'''

from __future__ import annotations

import json
from typing import Any


def _json_block(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def build_smartgen_prompt(
    original_context: dict[str, Any],
    new_context: dict[str, Any],
    device_info: dict[str, Any],
    compressed_sequences: Any,
    transition_hints: Any,
    causal_hints: Any,
) -> str:
    '''Build an offline SmartGen-style prompt for future generation calls.'''
    return '\n'.join(
        [
            'Role: You are an IoT expert for behavior sequence generation.',
            '',
            'Task: Adapt historical IoT behavior sequences from the original context to the target context without violating device semantics or causal hints.',
            '',
            'original_context:',
            _json_block(original_context),
            '',
            'new_context:',
            _json_block(new_context),
            '',
            'device_info and control action set:',
            _json_block(device_info),
            '',
            'historical compressed sequences:',
            _json_block(compressed_sequences),
            '',
            'high-frequency transitions:',
            _json_block(transition_hints),
            '',
            'causal_hints / key causal edges:',
            _json_block(causal_hints),
            '',
            'Output format: return either JSON { sequences: [[...], ...], notes: ...} or one sequence per line using <seq ... seq>.',
            'Do not include API credentials or online tool calls.',
        ]
    )
