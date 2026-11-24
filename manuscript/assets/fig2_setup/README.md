# Fig. 2 — optical setup (drawing, not data)

The schematic of the off-axis transmission Mach–Zehnder DHM setup is a drawing. It is not generated from results.
`fig2_setup.png` and `fig2_setup.pdf` are the figure printed in the manuscript, rendered from the SVG embedded in
the manuscript file. `manuscript/make_all.py` copies them to `manuscript/figures/main/` so that the figure set is
complete.

The optical parameters are given in the manuscript text (Section 3.1): a 666 nm laser diode, a 40×/0.75 NA
objective, and a CCD of 1920 × 1200 pixels of 5.86 µm, of which 1024 × 1024 were used. The wavelength in
`configs/config.yaml` (`physics.wavelength_um: 0.666`) is checked by `manuscript/verify.py`.
