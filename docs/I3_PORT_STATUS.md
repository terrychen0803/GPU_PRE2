# I3 port status

`gpu_pre2.i3.isolated` provides the selected S1–S3 asset/worker/coordinator/report system. `gpu_pre2.i3.context` handles both frozen A and BC S4 plans through explicit plan hashes. Shared dispatch, taxonomy, runtime and profiling are imported once from this package.

The 174-entry catalog remains a design pool. Actual retained data is 72 isolated configs + 40 context work items per device: 3,024 cells, 120,960 measured across three devices. S4 Scale0 remains one connected context group.

Offline replay covered all repeat2 P/R configurations on the three devices (672 traces). Of 36,992 artifacts, 36,902 were byte-exact. Ninety Groups4 predictor files preserve four distinct capture-local stream aliases instead of the historical flattened s0; all other fields agree. The exception still does not prove all-stream GI completion or strict reference.

New collection uses `gpu_pre2:` protocol IDs and frozen source manifests, with no per-run full source copy or legacy bootstrap. No new GPU smoke was performed. Historical 3090 no-cuDNN and 4090 Groups4 variants retain their own evidence and limits; data replay does not qualify a new run of those variants.

See LOCAL_DATA_LAYOUT.md for explicit dataset IDs, manifest/hash verification and collector entry points.
