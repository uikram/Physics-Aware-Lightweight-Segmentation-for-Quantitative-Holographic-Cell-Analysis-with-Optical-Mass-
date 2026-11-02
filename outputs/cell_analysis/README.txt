cell_analysis: per-cell analyses of every requested run, computed from the saved predictions (no training).
Class names are in the MANUSCRIPT order (D-25): mask value 2 = spherocyte, mask value 3 = echinocyte. Nothing here is to be
swapped again (manuscript/results.py reads cell_analysis.json without the swap).
Statistics over seeds: mean and sample SD (n - 1) over the seeds present (3). Cells flagged as phase-wrapped (D-21) are
excluded from the *_nowrap quantities; the *_all quantities keep them.
