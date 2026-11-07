# I4 synthetic integration smoke (not research data)

These JSONL rows are invented, CPU-only fixtures to check that isolated pretraining plus D/I context mixed learning can run from the command line.

    python -m pip install -e .
    python scripts/train_i4.py --train examples/i4/train.synthetic.jsonl --validation examples/i4/validation.synthetic.jsonl --mode pretrain_mixed --seed 17 --pretrain-updates 2 --second-updates 2 --log-every 1 --out outputs/synthetic_mixed

Outputs: recognizer.pt, normalizer.json, report.json. They **do not** carry any meaningful scientific accuracy claim and do not replace I3 P/R captures or the historical frozen I4 dataset. This is only a source-only API smoke test. All groups across train and validation are disjoint.
