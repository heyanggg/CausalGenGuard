#!/usr/bin/env python3
"""Build labeled anomalies from target-context normal sequences.

Use this script after preparing target-context real normal data.  It creates a
labeled test set so context-shift experiments can report F1/AUROC/AUPRC instead
of only target-normal false-positive rate.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from causal_gen_guard.data.attack_injector import CAUSAL_ANOMALIES, SMARTGUARD_ANOMALIES, inject_causal_anomaly, inject_smartguard_style
from causal_gen_guard.data.schemas import BehaviorSequence


def load_jsonl(path: Path, limit: int | None = None) -> list[BehaviorSequence]:
    sequences: list[BehaviorSequence] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if limit is not None and len(sequences) >= limit:
                break
            line = line.strip()
            if line:
                sequences.append(BehaviorSequence.from_dict(json.loads(line), validate=False))
    if not sequences:
        raise ValueError(f"No sequences found in {path}")
    return sequences


def write_jsonl(path: Path, sequences: list[BehaviorSequence]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sequence in sequences:
            handle.write(json.dumps(sequence.to_dict(validate=False), ensure_ascii=False) + "\n")


def load_control_pool(mapping_dir: Path | None) -> dict[str, int]:
    if mapping_dir is None:
        return {}
    control_to_id = mapping_dir / "control_to_id.json"
    id_to_control = mapping_dir / "id_to_control.json"
    if control_to_id.exists():
        raw = json.loads(control_to_id.read_text(encoding="utf-8"))
        return {str(key): int(value) for key, value in raw.items()}
    if id_to_control.exists():
        raw = json.loads(id_to_control.read_text(encoding="utf-8"))
        return {str(value): int(key) for key, value in raw.items()}
    return {}


def normal_copy(sequence: BehaviorSequence) -> BehaviorSequence:
    copied = BehaviorSequence.from_dict(sequence.to_dict(validate=False), validate=False)
    copied.label = 0
    copied.anomaly_type = "normal"
    copied.context = dict(copied.context)
    copied.context.setdefault("target_context_label", copied.context.get("context_id", "target"))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-normal-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mapping-dir", type=Path, default=None)
    parser.add_argument("--anomaly-family", choices=("smartguard", "causal", "mixed"), default="mixed")
    parser.add_argument("--per-anomaly-type", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--max-attempts-per-type", type=int, default=5000)
    args = parser.parse_args()

    normals = load_jsonl(args.target_normal_jsonl, args.max_sequences)
    if args.anomaly_family == "smartguard":
        anomaly_types = list(SMARTGUARD_ANOMALIES)
    elif args.anomaly_family == "causal":
        anomaly_types = list(CAUSAL_ANOMALIES)
    else:
        anomaly_types = list(SMARTGUARD_ANOMALIES) + list(CAUSAL_ANOMALIES)
    control_pool = load_control_pool(args.mapping_dir)
    rng = random.Random(args.seed)

    labeled = [normal_copy(sequence) for sequence in normals]
    success: Counter[str] = Counter()
    skipped: Counter[str] = Counter()
    skipped_reasons: Counter[str] = Counter()
    logs: list[dict[str, Any]] = []

    for anomaly_type in anomaly_types:
        attempts = 0
        while success[anomaly_type] < args.per_anomaly_type and attempts < args.max_attempts_per_type:
            attempts += 1
            source = rng.choice(normals)
            if anomaly_type in CAUSAL_ANOMALIES:
                injected, log = inject_causal_anomaly(source, anomaly_type, control_pool=control_pool)
            else:
                injected, log = inject_smartguard_style(source, anomaly_type, control_pool=control_pool)
            if injected is None:
                skipped[anomaly_type] += 1
                skipped_reasons[log.get("skipped_reason", "unknown")] += 1
                continue
            injected.sequence_id = f"target__{source.sequence_id}__{anomaly_type}__{success[anomaly_type]:04d}"
            injected.label = 1
            injected.context = dict(injected.context)
            injected.context["source_sequence_id"] = source.sequence_id
            injected.context["target_context_label"] = source.context.get("context_id", "target")
            injected.context["injection_log"] = log
            labeled.append(injected)
            success[anomaly_type] += 1
            logs.append(log)

    rng.shuffle(labeled)
    write_jsonl(args.output_jsonl, labeled)
    report = {
        "target_normal_count": len(normals),
        "anomaly_count": int(sum(success.values())),
        "labeled_count": len(labeled),
        "per_anomaly_type_success": dict(success),
        "per_anomaly_type_skipped": dict(skipped),
        "skipped_reasons": dict(skipped_reasons),
        "control_pool_size": len(control_pool),
        "seed": args.seed,
        "output_jsonl": str(args.output_jsonl),
        "log_examples": logs[:20],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
