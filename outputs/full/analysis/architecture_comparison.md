# Architectures at r8 (checkpoint last, wrapped cells excluded)

| metric | edge_sam | mobile_sam | mobilenet_unet |
|---|---|---|---|
| cells_predicted | 1269.0 | 1328.0 | 1734.0 |
| cells_annotated | 1988.0 | 1988.0 | 1988.0 |
| pearson_r | 0.343 | 0.212 | 0.631 |
| ccc | 0.216 | 0.188 | 0.619 |
| bias_pg | 1.105 | 0.471 | -0.247 |
| loa_low_pg | -6.346 | -4.557 | -3.071 |
| loa_high_pg | 8.557 | 5.498 | 2.578 |
| mdare_pct | 3.621 | 3.111 | 2.134 |
| day0_dry_mass_pg | 28.199 | 28.229 | 27.969 |
| composition_mean_abs_pp | 2.871 | 1.668 | 8.561 |
| area_rho_pred | -0.747 | -0.593 | -0.647 |
| area_rho_gt | -0.532 | -0.532 | -0.532 |
| drymass_rho_pred | -0.377 | -0.137 | -0.475 |
| drymass_rho_gt | -0.115 | -0.115 | -0.115 |
| drymass_late_vs_early_pct_pred | -7.68 | -3.053 | -8.686 |
| drymass_late_vs_early_pct_gt | -0.788 | -0.788 | -0.788 |
| count_ratio_discocyte | 0.585 | 0.596 | 1.068 |
| count_ratio_echinocyte | 0.914 | 0.927 | 0.0 |
| count_ratio_spherocyte | 0.673 | 0.871 | 0.96 |
| count_ratio_stomatocyte | 0.657 | 0.657 | 0.0 |
