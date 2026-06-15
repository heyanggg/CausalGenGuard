'''Prepare offline SmartGen synthetic/filter data for CausalGenGuard.

This script never calls online LLM APIs. It scans an existing SmartGen checkout
for local generated/filter/result files and converts them to BehaviorSequence
JSONL records. Optional Causal-TOF filtering is offline and only runs when
--apply-causal-tof is passed.
'''

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / 'src'
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from causal_gen_guard.generation.offline_synth_loader import load_smartgen_offline_sequences


def _resolve_path(value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json_file(path: Path | None) -> Any:
    if path is None:
        return None
    with path.open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _load_backbone(path: Path | None) -> Any:
    if path is None:
        return None
    import torch
    from causal_gen_guard.models.smartguard_backbone import SmartGuardBackbone

    checkpoint = torch.load(path, map_location='cpu')
    config = dict(checkpoint.get('model_config', {}))
    if 'vocab_size' not in config:
        config['vocab_size'] = len(checkpoint.get('vocab', {})) or len(checkpoint.get('inverse_vocab', []))
    model = SmartGuardBackbone(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'vocab' in checkpoint:
        model.vocab = checkpoint['vocab']
    return model


def _load_causal_model(path: Path | None) -> tuple[Any, dict[Any, int] | None]:
    if path is None:
        return None, None
    import torch
    from causal_gen_guard.models.causal_predictor import BehaviorCausalPredictor

    checkpoint = torch.load(path, map_location='cpu')
    config = dict(checkpoint.get('model_config', {}))
    if 'input_channels' not in config:
        config['input_channels'] = len(checkpoint.get('vocab', {})) or len(checkpoint.get('inverse_vocab', []))
    model = BehaviorCausalPredictor(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('vocab')


def _load_array(path: Path | None) -> Any:
    if path is None:
        return None
    import numpy as np

    return np.load(path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert offline SmartGen synthetic/filter data into JSONL.')
    parser.add_argument('--smartgen-root', default='../SmartGen', help='Path to the SmartGen project root.')
    parser.add_argument('--dataset', default='fr', choices=['fr', 'sp', 'us', 'an'], help='SmartGen dataset name.')
    parser.add_argument('--transition', default='ST', choices=['ST', 'TT', 'NT'], help='Context transition label.')
    parser.add_argument('--output', default='outputs/synthetic/fr_ST.jsonl', help='Output JSONL path.')
    parser.add_argument('--limit', type=int, default=None, help='Optional maximum number of sequences to export.')
    parser.add_argument('--offline', action='store_true', default=True, help='Offline mode is always enabled.')
    parser.add_argument('--apply-causal-tof', action='store_true', help='Apply offline legality/reconstruction/causal TOF filters.')
    parser.add_argument('--tof-output-dir', default=None, help='Directory for Causal-TOF kept/rejected/report outputs.')
    parser.add_argument('--device-action-map', default=None, help='Optional JSON map of device -> allowed controls, or __all__.')
    parser.add_argument('--backbone-checkpoint', default=None, help='Optional SmartGuardBackbone checkpoint for reconstruction filtering.')
    parser.add_argument('--reconstruction-threshold', type=float, default=None, help='Keep sequences with reconstruction score <= threshold.')
    parser.add_argument('--causal-checkpoint', default=None, help='Optional BehaviorCausalPredictor checkpoint for causal filtering.')
    parser.add_argument('--a-norm', default=None, help='Optional A_norm.npy for causal deviation filtering.')
    parser.add_argument('--causal-threshold', type=float, default=None, help='Keep sequences with causal deviation <= threshold.')
    return parser


def _write_jsonl(path: Path, sequences: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence.to_dict(), ensure_ascii=False) + '\n')


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root_arg = Path(args.smartgen_root)
    output_arg = Path(args.output)
    smartgen_root = root_arg if root_arg.is_absolute() else (PROJECT_ROOT / root_arg).resolve()
    output = output_arg if output_arg.is_absolute() else (PROJECT_ROOT / output_arg).resolve()

    try:
        sequences = load_smartgen_offline_sequences(
            smartgen_root,
            dataset=args.dataset,
            transition=args.transition,
            limit=args.limit,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f'SmartGen offline preparation failed: {exc}', file=sys.stderr)
        return 2

    if args.apply_causal_tof:
        try:
            from causal_gen_guard.generation.causal_tof import run_causal_tof

            device_action_map = _load_json_file(_resolve_path(args.device_action_map))
            backbone = _load_backbone(_resolve_path(args.backbone_checkpoint))
            causal_model, checkpoint_vocab = _load_causal_model(_resolve_path(args.causal_checkpoint))
            A_norm = _load_array(_resolve_path(args.a_norm))
            tof_output_dir = _resolve_path(args.tof_output_dir) or output.with_suffix('').parent / f'{output.stem}_causal_tof'
            report = run_causal_tof(
                sequences,
                output_dir=tof_output_dir,
                device_action_map=device_action_map,
                backbone=backbone,
                reconstruction_threshold=args.reconstruction_threshold,
                causal_model=causal_model,
                A_norm=A_norm,
                vocab=checkpoint_vocab,
                causal_threshold=args.causal_threshold,
                utility_config={'enabled': False},
            )
            kept_path = Path(report['outputs']['kept_jsonl'])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(kept_path.read_text(encoding='utf-8'), encoding='utf-8')
            report_path = report['outputs']['report_json']
            final_kept_count = report['final_kept_count']
            print(f'Applied Causal-TOF. Report: {report_path}')
            print(f'Wrote {final_kept_count} kept SmartGen sequences to {output}')
            return 0
        except Exception as exc:
            print(f'Causal-TOF preparation failed: {exc}', file=sys.stderr)
            return 3

    _write_jsonl(output, sequences)
    print(f'Wrote {len(sequences)} offline SmartGen sequences to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
