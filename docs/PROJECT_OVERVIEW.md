# CausalGenGuard Project Overview

CausalGenGuard is a behavior-based anomaly detection project for smart home and
IoT logs.  A behavior means one device action, for example:

```text
Camera:switch off
Light:switch on
SmartLock:lock unlock
WaterValve:valve open
```

A sequence is an ordered list of these behaviors.  The project learns what
normal sequences look like, then marks unusual sequences as anomalies.

## Why the project exists

SmartGuard is already strong on the original static benchmark, so only trying to
increase its F1 score on the same injected anomalies is not a strong research
story.  The more realistic problem is context shift: a detector trained on one
context, such as winter or daytime, may treat normal behavior in another context,
such as spring or night, as suspicious.

CausalGenGuard therefore focuses on this question:

> Can synthetic target-context normal data reduce false alarms when an IoT
> anomaly detector is trained mainly on source-context normal data?

## Main components

1. **Behavior preprocessing** converts raw SmartGuard/SmartGen/SmartSense/ARGUS
   data into a shared `BehaviorSequence` schema.
2. **SmartGuard-style reconstruction backbone** reconstructs device-control
   tokens and produces a reconstruction anomaly score.
3. **SmartGen-style context generation layer** supplies synthetic target-context
   normal sequences.
4. **Causal graph layer** converts sequences into event-channel tensors and
   estimates behavior dependency graphs.
5. **Causal-TOF filter** keeps synthetic sequences that are legal, not extreme
   under reconstruction scoring, and not too far from normal causal structure.
6. **Evaluation and explanation** report target-context false-positive rate,
   anomaly metrics when labels exist, and token/edge-level explanations.

## Final experimental route

The current recommended main experiment compares five methods:

```text
source_only
source_plus_raw_synthetic
source_plus_tof_synthetic
source_plus_causal_tof_synthetic
oracle_target
```

The causal branch is useful for filtering and explanation.  It should not be
claimed as the primary detector unless later experiments support that claim.
