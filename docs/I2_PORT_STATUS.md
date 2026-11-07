# I2 port status

Seven scoped A adapters, fixed-asset validation, GI/P/R worker and formal coordinator are in `gpu_pre2.i2`. Shared SQLite parser/quality/alignment lives in `gpu_pre2.profiling`; no old O1/O2/O4 import is required.

All three devices' seven A groups are registered: 189 cells, 7,560 measured, 126 P/R captures. All 126 captures and 6,990 parser outputs replay byte-exact. GI remains a single no-profiler scoped forward, with 40 measurements × 3 fresh processes.

New collection protocol IDs carry `gpu_pre2:` to distinguish the port. Code release hashes are frozen before collection, and assets must match the explicitly supplied manifest. No new GPU smoke was run; historical replay is not proof of live collector numerical equivalence. Reference warnings, unassigned Qwen copy events, stack differences and stability reviews remain.

Run `./run.sh -m gpu_pre2.i2.run_formal --help`; local assets/data/replay commands are documented in LOCAL_DATA_LAYOUT.md.
