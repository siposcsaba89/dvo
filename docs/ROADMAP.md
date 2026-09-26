# Roadmap

Goal: fast direct sparse VO for driving data, producing poses + a semi-dense point cloud
to initialise neural surface reconstruction / Gaussian Splatting instead of COLMAP.

| Step | Content | Validation |
|------|---------|------------|
| 1 | Skeleton, CMake/vcpkg, KITTI reader, image pyramid, trajectory evaluation (ATE, Sim3/SE3 align) | unit tests, loads seq 00 |
| 2 | Photometric residual model: residual pattern, affine brightness (a,b), Huber weighting, analytic Jacobians | numeric-vs-analytic Jacobian tests |
| 3 | Coarse frame tracking: pyramid direct alignment against keyframe depth, motion hypotheses | track with GT-free stereo depth on KITTI |
| 4 | Candidate point selection (region-adaptive gradient threshold) + immature points (epipolar search, depth interval) | visual checks, depth vs stereo |
| 5 | Monocular initialisation | first N frames |
| 6 | Sliding-window photometric bundle adjustment, Schur complement, marginalisation with FEJ | synthetic tests, then KITTI |
| 7 | Keyframe + point management (creation, activation, outlier removal, marginalisation strategy) | full pipeline on KITTI 00 |
| 8 | Export: poses + points in COLMAP text format (cameras/images/points3D) for GS / NeuS | load in a GS trainer |
| 9 | Stereo extension (metric scale, better for driving) | KITTI ATE in metres |
| 10 | Performance: multithreading, SIMD | real-time on KITTI |
| 11 | Tuning pass once the full pipeline runs (items below) | KITTI ATE / drift, point accuracy |

Deferred to step 11:
- Tracker speed (~72 ms/frame at step 3) and threaded image loading (~28 ms/frame).
- Immature points: high outlier / ambiguous rates, error bound too optimistic (stereo inside interval only 29 %).
- Fisheye validity mask and YAML camera config for non-KITTI data.
- Window BA on KITTI turns (frames 100-160): monocular scale shrinks ~1.5 % per keyframe relative to stereo
  (Sim3 scale 0.93 over 100 frames); not reproduced synthetically, not caused by marginalisation, FEJ, window
  size, point count or image periphery. Revisit after point management (step 7) and stereo residuals (step 9).
- Window BA speed (~300 ms per keyframe with ~5000 points).
- Full mono pipeline (step 7): ~110 ms/frame; scale drifts ~20 % over 250 frames of KITTI 00 (local scale
  1.13 -> 0.91), 4.8 % / 1.06 deg per 100 m. Candidates: photometric calibration (vignetting), keyframe and
  activation thresholds, candidate tracing quality.
- Stereo (step 9), 300 frames of KITTI 00: ATE 0.24 m SE3, 1.37 % translation drift, scale 1.00; rotation drift
  still ~1.1 deg/100m (same as mono) and ~160 ms/frame.

## Step 11 log

Benchmark: `tools/bench.sh <tag> stereo|mono 500` runs all nine 500-frame segments of KITTI 00 in parallel and
reports the mean ATE (Sim3, and SE3 for stereo) and the KITTI segment drift (100-800 m). The 300-frame numbers
above are dominated by 100 m segments and overstate rotation drift.

| Change | Stereo ATE SE3 / t % / r deg/100m | Mono ATE Sim3 / t % / r deg/100m |
|--------|-----------------------------------|----------------------------------|
| baseline | 0.474 / 0.85 / 0.435 | 17.9 / 15.7 / 4.16 (segment 4000 diverges) |
| drop points only without any good residual | 0.494 / 0.85 / 0.465 | 4.08 / 4.52 / 0.466 |
