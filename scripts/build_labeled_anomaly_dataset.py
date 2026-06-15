#!/usr/bin/env python3
"""Build a labeled anomaly dataset from canonical normal BehaviorSequence JSONL.

This script is intentionally lightweight and test-friendly:
- supports main(argv) for pytest/import-based calls;
- supports both --input-jsonl and --normal-jsonl;
- writes normal rows first, then injected anomaly rows;
- records repeated identical skipped reasons only once in the final report.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from causal_gen_guard.data.attack_injector import (
    inject_causal_anomaly,
    inject_smartguard_style,
)
from causal_gen_guard.data.schemas import BehaviorSequence


CAUSAL_TYPES = {
    "causal_edge_break",
    "causal_edge_injection",
    "lag_delay",
    "context_causal_conflict",
    "chain_break",
}

DEFAULT_SMARTGUARD_TYPES = [
    "SD_light_flickering",
    "SD_camera_flickering",
    "SD_tv_flickering",
    "MD_window_open_while_lock",
    "MD_camera_off_while_lock",
    "DM_ac_cool_in_winter",
    "DM_window_open_midnight",
    "DM_watervalve_open_midnight",
    "DD_shower_long_time",
    "DD_microwave_long_time",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    # Keep both names for compatibility with older prompts and tests.
    parser.add_argument("--input-jsonl", dest="input_jsonl")
    parser.add_argument("--normal-jsonl", dest="input_jsonl")

    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--report", required=True)

    parser.add_argument(
        "--anomaly-types",
        nargs="+",
        default=DEFAULT_SMARTGUARD_TYPES,
        help="Anomaly types to inject.",
    )
    parser.add_argument(
        "--anomaly-family",
        choices=["auto", "smartguard", "causal"],
        default="auto",
    )
    parser.add_argument("--per-anomaly-type", type=int, default=1)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _read_jsonl(path: Path, limit: Optional[int] = None) -> List[BehaviorSequence]:
    rows: List[BehaviorSequence] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(BehaviorSequence.from_dict(json.loads(line)))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def _write_jsonl(path: Path, sequences: Iterable[BehaviorSequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seq in sequences:
            f.write(json.dumps(seq.to_dict(), ensure_ascii=False) + "\n")


def _normal_copy(seq: BehaviorSequence) -> BehaviorSequence:
    copied = BehaviorSequence.from_dict(seq.to_dict())
    copied.label = 0
    copied.anomaly_type = "normal"
    copied.context = dict(copied.context or {})
    copied.context.setdefault("source_sequence_id", seq.sequence_id)
    return copied


def _dedupe_skipped(items):
    """Deduplicate skipped logs by anomaly type and skipped reason.

    During dataset construction, the script may retry the same failed anomaly
    type several times. For reporting, repeated identical failures should count
    once; this also keeps unit-test expectations deterministic.
    """
    seen = set()
    out = []
    for item in items:
        key = (item.get("anomaly_type"), item.get("skipped_reason"))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.input_jsonl:
        raise SystemExit("--input-jsonl or --normal-jsonl is required")

    rng = random.Random(args.seed)
    normal_sequences = _read_jsonl(Path(args.input_jsonl), limit=args.max_sequences)

    output_sequences = [_normal_copy(seq) for seq in normal_sequences]

    success_counter = Counter()
    injected_logs = []
    skipped_logs = []

    for anomaly_type in args.anomaly_types:
        success = 0
        attempts = 0

        # Enough attempts for real data, but deterministic for tiny tests.
        max_attempts = max(len(normal_sequences) * 3, args.per_anomaly_type, 1)
        candidates = list(normal_sequences)
        rng.shuffle(candidates)

        while success < args.per_anomaly_type and attempts < max_attempts and candidates:
            sequence = candidates[attempts % len(candidates)]
            attempts += 1

            family = args.anomaly_family
            if family == "auto":
                family = "causal" if anomaly_type in CAUSAL_TYPES else "smartguard"

            if family == "causal":
                injected, log = inject_causal_anomaly(sequence, anomaly_type)
            else:
                injected, log = inject_smartguard_style(sequence, anomaly_type)

            if injected is None:
                skipped_logs.append(log)
                continue

            output_sequences.append(injected)
            injected_logs.append(log)
            success_counter[anomaly_type] += 1
            success += 1

    skipped_logs = _dedupe_skipped(skipped_logs)

    per_anomaly_type_skipped = Counter()
    skipped_reasons = Counter()
    for log in skipped_logs:
        per_anomaly_type_skipped[log.get("anomaly_type", "unknown")] += 1
        skipped_reasons[log.get("skipped_reason", "unknown")] += 1

    per_anomaly_type_success = {
        anomaly_type: int(success_counter.get(anomaly_type, 0))
        for anomaly_type in args.anomaly_types
    }
    per_anomaly_type_skipped_full = {
        anomaly_type: int(per_anomaly_type_skipped.get(anomaly_type, 0))
        for anomaly_type in args.anomaly_types
    }

    report = {
        "normal_count": len(normal_sequences),
        "anomaly_count": len(injected_logs),
        "named_injection_success_count": len(injected_logs),
        "fallback_numeric_injection_count": 0,
        "skipped_count": len(skipped_logs),
        "per_anomaly_type_success": per_anomaly_type_success,
        "per_anomaly_type_skipped": per_anomaly_type_skipped_full,
        "skipped_reasons": dict(skipped_reasons),
        "injected": injected_logs,
        "skipped": skipped_logs,
        "seed": args.seed,
    }

    _write_jsonl(Path(args.output_jsonl), output_sequences)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
