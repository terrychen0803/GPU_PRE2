# Migration status — 2026-11-07

Local E2/I2/I3/I4 integration and historical numerical replay completed. Old local stage trees were removed only after byte/hash and offline regression gates. Remote predecessor repositories and branches were not deleted.

| Component | Status |
| --- | --- |
| E2 | Three historical profiles/samples relocated and SHA-verified; compatibility checks retain missing UUID/software mismatch |
| I2 | 189 cells, 7,560 measured; executable scoped workers/coordinator and shared parser ported; 126 saved traces replayed byte-exact |
| I3 | 3,024 cells, 120,960 measured; shared isolated and A/BC context implementations; 672 repeat2 P/R traces replayed |
| I4 frozen | Exact E1 candidate features, normalizers, 3 recognizer + 3 N0 checkpoints; all three v2 repeats and 12 evaluation outputs replay byte-exact |
| I4 mixed | Existing experimental trainer retained; separate feature contract, no new scientific fit or data conversion claimed |
| Coverage | 7,200 rows from actual source R/P; incomplete semantic arguments prevent exact signature claims |
| I5 | Not implemented; GI contract only |

The new package has no imports of predecessor O1/O2/O4 packages or dataset path injection. Dataset IDs, stage allowlists and explicit protocols replace discovery/fallback behavior. Repeated source snapshots are inert content-addressed provenance, not runnable dependencies.

Remaining gates: live collector smoke on each device; missing 3090 isolated operand evidence and 4090 smoke delivery; strict reference qualification; non-kernel feature treatment and missing taxonomy; independent context validation; scientific I4 mixed fits and I5. The 3090 no-cuDNN and 4090 conditional Groups4 captures remain distinct variants; their data replay does not qualify new GPU collection.

See LOCAL_DATA_LAYOUT.md and MIGRATION_REPORT_ZH_TW.md. The full external hash index and raw/checkpoint assets remain outside Git; their pointers and SHA are tracked under data/manifests.
