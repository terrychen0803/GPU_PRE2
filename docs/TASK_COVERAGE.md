# I2 → I3 → S4 → I4 → I5 coverage

The measured source mapping is in `results/coverage/TASK_COVERAGE.csv`, with a summary JSON beside it. It uses RTX4090 repeat2, measured iteration 7, and independently replays each workload's saved exact R/P alignment before transferring R event indices onto P.

There are 7,200 rows: 7,198 recorded operations plus two unassigned Qwen copy events. Full event-index lists, workload/operator IDs, ATen names, input shapes, matching measured isolated/context configurations, I4 limitations and target GI join IDs are retained. This table is evaluator-only and never predictor input.

I2 reference records lack full bound semantic arguments. Matches are explicitly `shape_only`, not complete runtime-signature exact matches. Geometry/length/class checks cannot establish correct blind recognition or time prediction. Zero recorded work, missing classes, non-kernel geometry, noncontiguous ownership and over-L8 cases retain their reasons and denominators.

Different GPU/software stacks can expose different Swin/Qwen decompositions; cross-device operator indices are not forcibly paired. S4 scale0 remains one group; its D/I/S and repeats cannot become independent held-out samples. No task-level prediction accuracy or formal exact coverage is inferred from this migration table.
