# mergeforge — delivery log (SOP §11 write-back)

Every world-class system delivered by the random-ai-system-delivery pipeline is
appended here so future runs avoid re-delivering a covered domain.

| date       | system    | repo                 | domain            | grade | note |
|------------|-----------|----------------------|-------------------|-------|------|
| 2026-10-13 | mergeforge | CJX0712/mergeforge | Model Merging / Task Arithmetic | S | Flagship MergeFuse (TIES sign election + per-tensor λ calibration + non-inferiority guard); pretext-base + frozen-trunk block-loss regime (A11); gates G1-G6 all pass, recovery vs oracle 0.99 / vs joint 0.93; bitwise deterministic; 109 tests; CI py3.12/3.13; full A1→A11 negative-result log in docs/model_card.md |

## Coverage notes
- Domain chosen: **Model Merging / Task Arithmetic** — verified uncovered
  (255 prior forges, none in merging).
- Author attribution: 晨星.
- Quality grade: **S** (world-class): preregistered gates, paired-seed
  significance, oracle-reference recovery, ablation + ≥3 failure cases with
  attribution, bitwise determinism, CI matrix, honest negative results
  (A1-A10 structural failures documented).
- Key engineering lessons preserved: flat-parameter ordering must be verified
  by roundtrip tests; every gradient branch (incl. class_block) needs its own
  numerical check; full-ft Adam on undertrained basins yields orthogonal task
  vectors — merging requires a converged shared basin + decomposable increments.
