# T1_dataset: Dataset after the corrected preprocessing (float32 phase, background referenced)

| Item | Value | Source |
|---|---|---|
| Training images (base, with float32 phase map) | 182 | metrics.json n_train_views / 4 |
| Training views (4 lossless 90° rotations per base image) | 728 | metrics.json n_train_views |
| Validation (= evaluation) images | 66 | metrics.json n_val_images |
| Storage days in the validation set | 11 (days 0, 5, 8, 12, 15, 19, 23, 27, 30, 37, 47) | cells_gt.csv |
| Validation images per storage day | 6/6/6/6/6/6/6/6/6/6/6 | cells_gt.csv |
| Annotated cells (area 15-200 µm², not cut by the annotated field of view) | 2049 | compiled cell_analysis annotation |
|   of which phase-wrapped (excluded from phase-derived results, D-21) | 61 (3.0%) in 29 images | compiled cell_analysis annotation |
| Annotated cells used for phase-derived results | 1988 | compiled cell_analysis annotation |
|   discocyte (1): cells (wrapped) | 1450 (8, 0.6%) | compiled cell_analysis annotation |
|   spherocyte (2): cells (wrapped) | 259 (14, 5.4%) | compiled cell_analysis annotation |
|   echinocyte (3): cells (wrapped) | 230 (28, 12.2%) | compiled cell_analysis annotation |
|   stomatocyte (4): cells (wrapped) | 110 (11, 10.0%) | compiled cell_analysis annotation |

Notes:

* The 74 base training images without a float32 map (DECISIONS.md D-05, Q-1 closed) are not used; the count 256 of the 8-bit set comes from DECISIONS.md, not from outputs/ (data/processed is not in the repository).
* Wrapped cells: raw phase within 0.3 rad of 2π or a neighbour jump > π within 2 px of the cell (D-21).
