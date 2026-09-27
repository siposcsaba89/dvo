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

Parameter sweep on top of that (not adopted; ms/frame measured with two benchmarks sharing the CPU):

| Variant | Stereo ATE SE3 / t % / r | Mono ATE Sim3 / t % / r |
|---------|--------------------------|-------------------------|
| 3000 active points | 0.471 / 0.84 / 0.439 (+50 % time) | 4.22 / 4.66 / 0.450 |
| 9 keyframes | 0.518 / 0.86 / 0.452 | 6.15 / 6.32 / 1.21 |
| stereo weight 2 / 0.5 | 0.492 / 0.88 / 0.471 ; 0.686 / 0.92 / 0.438 | - |
| fewer keyframes (kf-flow 180, tflow 75) | - | 7.73 / 7.94 / 1.88 |
| more keyframes (kf-flow 80, tflow 35) | - | 3.99 / 4.46 / 0.456 (+75 % time) |
| candidate activation also after Skipped/Ambiguous traces | 0.488 / 0.84 / 0.446 | 4.15 / 4.60 / 0.465 |
| 12 / 20 window BA iterations | 0.493 / 0.86 / 0.474 ; 0.494 / 0.86 / 0.450 | 4.12 / 4.56 / 0.461 ; 4.15 / 4.59 / 0.456 |

Full sequence (4541 frames, 3.7 km, segments 100-800 m as in the KITTI benchmark), before the camera correction:
stereo 0.78 % / 0.268 deg/100m, ATE SE3 3.5 m (Stereo DSO paper: 0.84 % / 0.26). Mono 39 % / 1.93: the scale
grows monotonically ~7x over 3000 frames and then collapses. So stereo was already on par; the short-segment
rotation drift was an evaluation artefact.

Mono scale drift root cause: every window BA moved existing points ~0.7 % farther away, in stereo too. Two parts:
1. The window BA did not converge: in every LM step about half of the points overshoot in depth (photometric
   error is far from linear in depth) and single points reject whole steps. Fixed with a per-point depth
   safeguard (keep the step, half of it, or the old depth, whichever has the lowest point energy).
2. The KITTI rectified camera model is slightly off. Per temporal residual, the best inverse depth is 1-2 %
   below the stereo optimum, growing with image radius and with the same sign for older and newer keyframes:
   residual pincushion distortion. EUCM alpha = -0.03 (both cameras) removes most of it; focal scale (+-1 %) and
   principal point (+-4 px) do not. Consistent with Cvisic et al., "Recalibrating the KITTI Dataset Camera Setup
   for Improved Odometry Accuracy", ECMR 2021 (weakly constrained intrinsics, 30 % / 50 % odometry gains from
   recalibration). `run_vo --cam-alpha` applies it (also to the COLMAP export), `--check-calibration` logs the
   temporal vs stereo depth bias per image radius for any stereo rig.

| Change | Stereo ATE SE3 / t % / r deg/100m | Mono ATE Sim3 / t % / r deg/100m |
|--------|-----------------------------------|----------------------------------|
| per-point depth safeguard in the window BA | 0.496 / 0.85 / 0.443 | 4.10 / 4.53 / 0.459 |
| + `--cam-alpha -0.03` | 0.501 / 0.79 / 0.428 | 0.995 / 1.28 / 0.430 |

Full-sequence mono with the safeguard: alpha 0: 39 % / 1.93, alpha -0.03: 6.4 % / 0.281, alpha -0.035: 1.15 % /
0.288 with ATE Sim3 6.9 m and local scale within +-5 % over 3.7 km; alpha -0.04 reverses the drift.
