# Migration status (2026-11-07)

## Scope and source

Main research objective: RTX4090 P-only trace -> RTX3090/GB10 operator CTX and single-forward GI, with the six specified inference models / seven workload groups.

| Component | Intended transfer | Actual status of this clean repo |
|---|---|---|
| E2 | Executable `e2_calibration_v3` | Source copied byte-for-byte from old fixed commit; **existing measured profiles remain off-repo** |
| O3 / I4 | Semi-Markov and N0 | Clean I4 experimental implementation and mixed-training API; old frozen model weights and original training code **not yet imported** |
| I2 | Seven-A scoped forward GI/P/R and R inventory | Inventory/collector extraction from old code requires compatibility regression; source identities documented |
| I3 S1-S3 | 72 operator configurations | Catalog and capture metadata; raw/asset files still on hosts |
| I3 S4 | Six scale0 D/I/S graphs | Context plans and metadata; raw/derived D assets still on hosts |
| I5 | Single forward GI prediction | Contract and baseline plan only; no ITER model fitted |

**Never claim new formal reference verification from migration.** All known historical `strict_reference_verified=false` flags retain their meaning. No new GPU runs or dataset training have occurred in this repository.

## Data migration prerequisite

Before claiming end-to-end reproduction, locate and hash each device's raw capture archive, weights/inputs, calibration profile, I4 checkpoint and train/test split manifest. Store only SHA256/index metadata in Git; stage raw data on controlled external storage. Block unsupported/missing records rather than fabricate P/R/GI or download substitute assets.

## Mandatory gates

1. Unit tests for source-only predictor and grouped train/validation separation.
2. Byte-stable E2 run source provenance and calibration compatibility check.
3. I2/I3 capture replay comparison on a fixed subset with original source commit, data schema and R ownership.
4. I4 isolated baseline replay and new context-mixed training against independent graph groups; do not use S4 scale0 repeats as separate held-out groups.
5. I5 only after I4 coverage/OOF and matched single-forward GI labels are available.
