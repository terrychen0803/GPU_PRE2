# I4: Isolated pretraining + context-aware Semi-Markov structured training

## What is actually implemented

Experimental single shared Semi-Markov recognizer using **only RTX4090 numerical P-event features**:
1. Pretrain on **S1–S3 isolated** examples. They teach single-operator type, local shape variation and *not splitting inside* one operator.
2. Continue with **S1–S3 + S4 Dependent / Independent** graph examples, group-balanced, to teach *not merging neighboring operators*.
3. Four comparable configurations: \`isolated_only\`, \`context_only\`, \`pretrain_context\`, \`pretrain_mixed\`. Same architecture, model initialization/seed, validation groups, and typed-boundary metrics.
4. N0 ratio-residual 43→64→32→1 is available separately; operator predictor consumes source P-only features and independent target E2 hardware descriptor, not target test trace or reference labels.

**Do not confuse this new implementation with the frozen historical I4 P0–P2 model.** Its 29→64→32→11 architecture is mirrored, but the precise E1 27-dimensional segment feature aggregator and normalizer from historical \`data/predictor_input_v2\` are now available in `gpu_pre2.models.historical` with their original frozen checkpoints outside Git. The new candidate scorer currently uses **mean-pooled** event features. Consequently historical checkpoints MUST NOT be loaded as if compatible, and the historical module has independently reproduced the three v2 repeats and evaluation outputs; the mean-pooled mixed trainer remains a separate experiment.

## Data contract

Labeled JSONL example:
\`\`\`json
{"sample_id":"example-01","group_id":"signature-or-connected-graph-group","source_gpu":"RTX4090","split":"train","kind":"context_d","source_event_features":[[0.0,0.0]],"source_hw":[0.1,0.2],"gold_segments":[{"start":0,"end":1,"class_id":0}],"quality":{"usable_for_segmentation":true}}
\`\`\`
**Illustrative shape only:** \`source_event_features\` in a real record must contain exactly **27 finite numbers per GPU event**. \`gold_segments\` must fully partition the event index range and contain verified integer class IDs within the frozen taxonomy. Unknown types, unfixed Memset geometry, missing activity and non-contiguous/non-interleaving-invalid ownership must **not** be invented or silently dropped to make the dataset fit.

- \`isolated\`: one gold segment (S1–S3)
- \`context_d\`, \`context_i\`: multiple gold segments, real graph execution (S4)
- \`group_id\`: graph/signature connected component; **same configuration on all devices, D/I/S/repeats/views stays in the same group**
- \`split\`: train/validation/test/confirmation; the training CLI accepts only train+validation and rejects group collisions
- \`gold_segments\`: R/evaluator-only supervision. \`predictor_view\` strips all gold, identifiers and target data before model inference.
- Target P CTX labels for N0 must be independently audited, matched P/P by target direction and separately permitted for train supervision. GI is reserved for I5 and is not a substitute for P CTX.

### Example CLI (after real validated rows are exported)

\`\`\`bash
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/train_i4.py --train /data/i4/train.jsonl --validation /data/i4/val.jsonl \
    --mode pretrain_mixed --seed 17 --pretrain-updates 720 --second-updates 720 \
    --out /data/i4/fits/mixed_seed17
\`\`\`
The CLI **does not** derive R labels, fetch Nsight traces, construct real model features, or verify reference completeness; dataset conversion from existing local I4 P0–P2 exports is still a migration gate. Do not use the 40 iterations or three process repeats as independent train/validation groups.

## Critical scientific warnings

- Current S4 scale0: **one connected context group**. It can inform development, not held-out context generalization or OOF by splitting repeats.
- Four missing classes: Hardswish/Hardsigmoid/Mul/Softmax. A full gold partition is required; do not relabel them as known classes or truncate them and claim complete trace coverage.
- Five-event Conv including 17,280-byte Memset is unsupported in old E1 geometry; update event features with explicit event type/missingness mask before claiming entire-graph coverage.
- Report blind typed F1 and boundary F1 on independent context groups, isolated F1, coverage and CTX accuracy separately.
- Maintain frozen historical isolated-only model as baseline; never pretend this new code was the exact original O3 checkpoint.

## Local historical replay

See LOCAL_DATA_LAYOUT.md. Use `gpu_pre2.models.historical.predict` for the frozen E1 candidate-segment baseline. Its checkpoint/normalizer/data contract is distinct from this trainer's per-event mean-pooling contract. Migration did not fit a new scientific mixed model or construct a validated real-data converter for it.
