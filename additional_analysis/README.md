# additional_analysis — statistical analyses and the published tables

These scripts run on the CPU (except `w9 --infer`, which needs a GPU for inference). None of them trains a network
or changes a result. They read `results/`, `configs/config.yaml` and each other's outputs, and they write only to
`additional_analysis/results/` (and `results/tables_for_manuscript/` for the table generator). Every JSON output
starts with a metadata block that lists the script, the package versions and every input file.

`common.py` reads the selected (representative) LoRA rank of each architecture from
`results/consolidated/selection/rank_selection.csv`. It stops if that file is missing, so every script below
needs the selection to have run first (see the main README, step 6).

| Script | What it computes | Used in |
|---|---|---|
| `w1_cell_paired_stats.py` | Cell-level paired comparisons of two objectives on the same reference cells at the selected configurations: hierarchical bootstrap (storage days → images → cells, 10,000 resamples), exact day-block permutation test (2¹¹ arrangements), Holm correction | Table S8, Fig. 4, Fig. S5 |
| `w2_per_architecture_tests.py` | Run-level comparisons of the proposed objective with the six other objectives, per architecture: 12 LoRA pairs, exact Wilcoxon signed-rank test, Holm over the six comparisons | Table 2, Table S6 |
| `w4_cell_vs_global_ipp.py` | One-factor comparison of the full objective with cIPP against the full objective with global IPP (36 pooled pairs and per architecture) | Table S7 |
| `w5_reference_cells_checks.py` | Reference cells per storage day, crowding, and the optical-axis superposition screen | Table S1, Fig. S1, Fig. S2 |
| `w6_superposition_and_fov.py` | Phase profiles of the superposition candidates, and the annotated field of view per split (needs `data/prepared/`) | Supplementary Section S1, Fig. S1, Fig. S2 |
| `w7_followup_checks.py` | Matched cells with dry-mass errors above 10 pg, with their merged, touching and day-15 shares; class confusion of matched cells; cancellation of signed cell errors within an image | Tables S14, S15; Supplementary Section S9 |
| `w8_signed_errors.py` | Signed and absolute integrated-phase errors of matched cells, touching and isolated | Table S16 |
| `w9_validation_selection.py` | Evaluates every run on the 41 validation images (`--infer`, GPU) and applies the rank-selection rule to those metrics (`--select`) | Section 3.5; read by `scripts/evaluation/objective_and_rank_selection.py` |
| `w10_claim_checks.py` | 90th percentile of the matched-cell error, and the image-level MdARE of the per-image median dry mass | Sections 4.3 and 4.8 |
| `w11_validation_sensitivity.py` | The rank the rule chooses at each tolerance from 0 to 0.05 score points, on the validation metrics | Supplementary Section S6 |
| `tables_for_manuscript.py` | Builds Tables 1–5 and S1–S16 from the outputs above and from `results/consolidated/`, as CSV, JSON and XLSX, with `provenance.json` (every input file and its SHA-256) and `flags.md` (values that could not be traced; none for the published build) | All tables |
| `make_tables.sh` | Reruns W-2, W-4, W-5, W-8, W-7 and W-1 at the selected ranks, then `tables_for_manuscript.py` | — |

## Running

From the repository root, after the selection step:

```bash
# validation-based rank selection (step 5 of the main README)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python additional_analysis/w9_validation_selection.py --infer
python additional_analysis/w9_validation_selection.py --select

# the published tables (about 10 minutes; the W-1 bootstrap dominates)
bash additional_analysis/make_tables.sh
BOOT=2000 bash additional_analysis/make_tables.sh        # quicker W-1 bootstrap, not for the final tables
SKIP_ANALYSES=1 bash additional_analysis/make_tables.sh  # tables only, from existing outputs

# the remaining analyses
python additional_analysis/w6_superposition_and_fov.py   # needs data/prepared/
python additional_analysis/w10_claim_checks.py
python additional_analysis/w11_validation_sensitivity.py
```

The per-run files must carry `metadata.rescored`, which `scripts/run_rescore.sh` writes. The table generator
refuses older files unless it is given `--allow-stale`, which is for testing only.
