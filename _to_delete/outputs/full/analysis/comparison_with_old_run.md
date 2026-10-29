# Current runs against the superseded half-scale runs (old_run/)

Positive difference = current minus old. Dice-only runs are the control (segmentation input unchanged).

## Dice-only runs (control)

3 runs. Median absolute difference: Dice 0.0069, AJI 0.0096, BF1 0.0080, IPE 0.103 pp; largest Dice difference 0.2011; runs whose number of lost classes changed: 1.

## Runs with PMC / BGA / IPP

24 runs. Median absolute difference: Dice 0.0040, AJI 0.0156, BF1 0.0072, IPE 0.149 pp; largest Dice difference 0.1944; runs whose number of lost classes changed: 1.

## Per run

| series | run | Dice old | Dice new | AJI old | AJI new | BF1 old | BF1 new | IPE old (%) | IPE new (%) | classes lost old/new |
|---|---|---|---|---|---|---|---|---|---|---|
| full | edge_sam_abl_dice_only | 0.737 | 0.730 | 0.383 | 0.348 | 0.635 | 0.643 | 4.70 | 4.60 | 0/0 |
| full | edge_sam_abl_pmc | 0.748 | 0.739 | 0.323 | 0.350 | 0.649 | 0.656 | 3.05 | 3.28 | 0/0 |
| full | edge_sam_abl_pmc_bga | 0.731 | 0.747 | 0.346 | 0.341 | 0.638 | 0.642 | 3.02 | 3.01 | 0/0 |
| full | edge_sam_full_finetune | 0.754 | 0.559 | 0.283 | 0.278 | 0.692 | 0.635 | 2.08 | 8.31 | 0/1 |
| full | edge_sam_lora_r16 | 0.726 | 0.722 | 0.316 | 0.274 | 0.642 | 0.636 | 2.95 | 2.90 | 0/0 |
| full | edge_sam_lora_r2 | 0.675 | 0.685 | 0.266 | 0.257 | 0.615 | 0.621 | 3.27 | 2.86 | 0/0 |
| full | edge_sam_lora_r32 | 0.736 | 0.731 | 0.318 | 0.329 | 0.639 | 0.647 | 3.09 | 2.66 | 0/0 |
| full | edge_sam_lora_r4 | 0.714 | 0.710 | 0.259 | 0.233 | 0.625 | 0.627 | 3.47 | 2.99 | 0/0 |
| full | edge_sam_lora_r8 | 0.715 | 0.706 | 0.233 | 0.224 | 0.628 | 0.615 | 2.70 | 3.13 | 0/0 |
| full | mobile_sam_abl_dice_only | 0.604 | 0.805 | 0.231 | 0.222 | 0.743 | 0.799 | 13.10 | 1.81 | 1/0 |
| full | mobile_sam_abl_pmc | 0.806 | 0.813 | 0.217 | 0.202 | 0.794 | 0.806 | 1.75 | 1.86 | 0/0 |
| full | mobile_sam_abl_pmc_bga | 0.815 | 0.811 | 0.235 | 0.224 | 0.801 | 0.806 | 1.66 | 1.58 | 0/0 |
| full | mobile_sam_full_finetune | 0.217 | 0.217 | 0.375 | 0.448 | 0.806 | 0.808 | 2.01 | 2.16 | 3/3 |
| full | mobile_sam_lora_r16 | 0.810 | 0.811 | 0.227 | 0.221 | 0.790 | 0.803 | 1.40 | 1.41 | 0/0 |
| full | mobile_sam_lora_r2 | 0.807 | 0.806 | 0.256 | 0.261 | 0.796 | 0.801 | 1.23 | 1.38 | 0/0 |
| full | mobile_sam_lora_r32 | 0.812 | 0.818 | 0.236 | 0.266 | 0.803 | 0.799 | 1.42 | 1.51 | 0/0 |
| full | mobile_sam_lora_r4 | 0.811 | 0.807 | 0.250 | 0.227 | 0.792 | 0.803 | 1.46 | 1.42 | 0/0 |
| full | mobile_sam_lora_r8 | 0.811 | 0.810 | 0.247 | 0.243 | 0.805 | 0.792 | 1.45 | 1.42 | 0/0 |
| full | mobilenet_unet_abl_dice_only | 0.232 | 0.232 | 0.534 | 0.534 | 0.775 | 0.775 | 26.39 | 26.39 | 3/3 |
| full | mobilenet_unet_abl_pmc | 0.232 | 0.232 | 0.459 | 0.503 | 0.761 | 0.758 | 25.30 | 27.08 | 3/3 |
| full | mobilenet_unet_abl_pmc_bga | 0.232 | 0.232 | 0.464 | 0.477 | 0.762 | 0.756 | 25.79 | 26.79 | 3/3 |
| full | mobilenet_unet_full_finetune | 0.413 | 0.418 | 0.671 | 0.642 | 0.832 | 0.825 | 2.19 | 2.17 | 2/2 |
| full | mobilenet_unet_lora_r16 | 0.414 | 0.413 | 0.524 | 0.511 | 0.812 | 0.815 | 2.45 | 2.37 | 2/2 |
| full | mobilenet_unet_lora_r2 | 0.416 | 0.406 | 0.571 | 0.538 | 0.832 | 0.809 | 2.66 | 2.73 | 2/2 |
| full | mobilenet_unet_lora_r32 | 0.415 | 0.414 | 0.483 | 0.500 | 0.805 | 0.832 | 2.18 | 2.54 | 2/2 |
| full | mobilenet_unet_lora_r4 | 0.414 | 0.408 | 0.585 | 0.569 | 0.826 | 0.801 | 2.02 | 2.39 | 2/2 |
| full | mobilenet_unet_lora_r8 | 0.413 | 0.412 | 0.509 | 0.603 | 0.804 | 0.816 | 2.17 | 2.52 | 2/2 |
