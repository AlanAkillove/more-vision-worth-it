"""Report subpackage: M7 figures (plan §16 / §10.1) and the Phase-0 audit report (§10.3).

`figures.py` renders the nine required PNGs strictly from the underlying tables already
written by M2-M6 (never from in-memory recompute), re-deriving only the few helper tables a
figure needs (accuracy_by_resolution, transition matrix, scalar-vs-oracle). `report.py`
assembles docs/phase0_report.md from metrics.json and derives the OVERALL GO / NO-GO /
NEEDS-ONE-FOLLOW-UP verdict.
"""
