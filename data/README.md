# Data

The quantitative phase maps and annotation masks are **not redistributed** in this repository. As stated in the
article, the data analysed in this study are available from the corresponding author
(inkyu.moon@dgist.ac.kr) upon reasonable request.

## Expected layout (`configs/config.yaml`, `paths`)

```
data/raw/
├── dataset/
│   ├── X_train/   8-bit exports of the 182 training-split images (used only to register the masks)
│   ├── X_val/     8-bit exports of the 66 validation-split images (used as the test set)
│   ├── Y_train/   annotation masks drawn on the exports: 0 background, 1 discocyte, 2 spherocyte,
│   └── Y_val/                                          3 echinocyte, 4 stomatocyte
├── raw_data/      32-bit float phase maps in radians, 768 × 768, matched to the exports by file stem
└── X_train_raw/   (further float phase maps; every directory in paths.raw_phase_dirs is searched)
```

Image stems of the form `YYYYMMDD_n` carry the acquisition date; the storage day is counted from `data.day0`
(2011-05-11). The phase maps must be float32 radians: `qpicell/physics.py` refuses 8-bit, integer or rescaled data.

## Preparation

```bash
python scripts/utilities/prepare_data.py --workers 16
```

This builds `data/prepared/` once, on the CPU:

- `data/prepared/<split>/phase/` holds each float phase map minus its annotation-free second-order background.
- `data/prepared/<split>/mask/` holds the masks registered onto the 768 × 768 phase grid (ECC affine, nearest
  neighbour); 255 marks pixels outside the annotated field.
- `data/prepared/<split>/cells/` holds the reference cells (watershed, h = 3 px).
- `data/prepared/<split>/wrapped/` holds the pixels flagged as possibly phase-wrapped.
- `manifest.csv` records, for every image, the registration result, the background coefficients, the storage day
  and SHA-256 checksums.

Every later script verifies those checksums before it reads a prepared map. For the published dataset,
`report.json` records 248 accepted images (182 training split, 66 validation split, none rejected), 3,397 and
3,084 reference cells, and one borderline registration accepted under the documented rule.

The 182 training-split images are divided deterministically by `sha1(stem) % 5` into 141 training and 41
validation images. The 66 validation-split images form the test set.
