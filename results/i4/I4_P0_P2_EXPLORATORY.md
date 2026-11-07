# I4 P0–P2 historical exploratory findings, not GPU_PRE2 retraining

Source: REPORT_ZH_TW2222.md supplied by researcher on 2026-11-07.

- Historical 3-seed Semi-Markov L8 recognizer 29→64→32→11, and 3-seed shared N0 43→64→32→1.
- Historical 11-group isolated test: mean blind typed F1 approximately 97%; RTX4090→GB10 N0 oracle CTX MdAPE 8.05%; RTX4090→RTX3090 oracle CTX MdAPE 3.20%.
- S4 dependent six graphs / 28 positions: boundary F1 = 0%,15%,0%; typed F1 = 0%,5%,0% across seeds.
- S4 missing Hardswish/Hardsigmoid/Mul/Softmax. Resize-Concat-Conv includes 17,280-byte Memset with unsupported old E1 geometry.
- Old frozen canonical flow: data/predictor_input_v2 -> predictions_v2 -> evaluation_v2.
- Strict verification remains false. The historical tests were already inspected, not new confirmatory tests.
- IMPORTANT: original six weight files, normalizers, exact original candidate aggregate features and local input rows are NOT yet migrated. The new experimental mean-pool Semi-Markov code is NOT claimed checkpoint-compatible.
