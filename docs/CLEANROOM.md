# Provenance policy

This project is an independent implementation of *direct sparse odometry* as
described in the scientific literature. It must stay free of code derived from
GPL-licensed implementations so it can be used commercially.

## Allowed sources
- The paper: J. Engel, V. Koltun, D. Cremers, "Direct Sparse Odometry", IEEE TPAMI 2017 (arXiv:1607.02565).
- Follow-up papers (e.g. Stereo DSO, Wang et al. ICCV 2017; photometric calibration, Engel et al. 2016).
- Textbook material: Lie groups (Barfoot, "State Estimation for Robotics"), Gauss-Newton /
  Levenberg-Marquardt, Schur complement marginalization, First-Estimate Jacobians (Huang et al.).
- Permissively licensed libraries: Eigen (MPL2), Sophus (MIT), OpenCV (Apache-2), GoogleTest (BSD-3).

## Not allowed
- Reading, copying, translating or paraphrasing the source code of the original DSO repository
  (or its GPL forks: LDSO, Stereo-DSO ports, ...).
- Copying constant tables, file layouts or class structure from such code.

## When stuck
Reading the original code "only for ideas" is also not allowed: it establishes access, which
together with any similarity weakens the independent-creation defence. Instead:
1. J. Engel's PhD thesis (TUM, 2017) and follow-up papers (Stereo DSO, LDSO, DM-VIO) — text only.
2. General literature on direct alignment, photometric BA, marginalisation.
3. If code knowledge is truly needed: a *different* person reads it and writes a plain-language
   behavioural spec (no code, no identifiers); we implement from the spec and archive it.

## Practice
- Every non-obvious constant is documented with its source (paper section / own tuning).
- Design and naming are our own. Parameters that the paper states are cited as `[DSO §x]`.
- Before commercial release: legal review + patent freedom-to-operate check.
