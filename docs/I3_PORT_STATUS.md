# I3 catalog migration status

Existing design definitions imported from exact historical SHA 3db5871d43abd87fefbdadc21371beee59ac6b88:
- 174 I3 S1–S3 *planned catalog entries*, not 174 measured entries; only the selected 72 configurations have engineering results on the three nodes.
- S4 Context A and BC: six graphs, 28 positions, 40 D/I/S work item definitions, one connected Scale0 group.
- All three nodes have reported S1–S4 40×3 engineering collections, but original file bytes are stored on experiment hosts; results summaries alone are not a replacement for P/R/GI source raw captures.

Run CPU-only structure audit:
    python -m pip install -e .
    python scripts/audit_i3_catalog.py

**Not yet ported / ready:** exact old I3 collection code and frozen input assets, producer-to-dataset converter, full Nsight/CUPTI parser and R-to-P alignment/quality replay. The catalog file itself has its original historical status field (including planned_only), which describes the design generation version, not current completion. Reported capture status lives in separate manifests.

When porting the collector, avoid the original repeated source bundles i3_pilot/i3_config_pilot/i3_context_pilot and o4_pilot/vendor/legacy. Extract a common runtime and versioned protocol handlers, compare a saved trace's exact event/label/quality outputs against old code, then cut over. Do not claim bit-identical capture behavior before the regression.
