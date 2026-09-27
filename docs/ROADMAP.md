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
| 10 | Performance: multithreading, SIMD (done: multithreading, see log) | real-time on KITTI |
| 11 | Tuning pass once the full pipeline runs (items below) | KITTI ATE / drift, point accuracy |

Deferred to step 11:
- Tracker speed (~72 ms/frame at step 3) and threaded image loading (~28 ms/frame).
- Immature points: high outlier / ambiguous rates, error bound too optimistic (stereo inside interval only 29 %).
- Fisheye validity mask and YAML camera config for non-KITTI data (done: ValidityMask, --camera, --video).
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

Frame-to-frame accuracy (relative pose error, 1 frame): RMS 2.9 cm / 0.072 deg on four 300-frame segments, but the
RMS is dominated by the ground truth: KITTI 00 frames 2275-2290 are linearly interpolated (constant 0.274 deg/frame
yaw, constant 0.77 m steps) while the estimate follows the real motion. Median rotation error per frame is
0.03-0.04 deg for keyframe pairs and pairs with a tracked frame alike (mono and stereo), i.e. tracked frames are
not worse than keyframes and the error is near the ground-truth noise. A tracked-frame refinement was therefore not
implemented; revisit only with better ground truth (synthetic or survey-grade).

## Step 10 log

Profile first (`tools/profile.sh <tag>`: single process, KITTI 00 frames 1000-1400 mono and stereo, optional
fisheye video; `run_vo` prints the time per stage). Before: the window BA was 70 % of the time (244 ms per
keyframe, KITTI makes a keyframe every ~2 frames), and within it the per-point depth safeguard 59 % and the
linearisation 28 %; the dense Schur complement and solve only 2 %. The cost was residual evaluation, not algebra.

Changes, all deterministic (fixed chunk count for parallel reductions, results identical run to run):
- Window BA: host/target pair table per pass (rotation matrix, adjoint, brightness scale) instead of per residual;
  energy-only evaluation (no Jacobians, intensity-only interpolation) for the safeguard, classification and energy;
  the FEJ second projection only for pairs whose linearisation point differs from the current state; linearisation,
  Schur complement, safeguard and classification in parallel over points. Marginalisation linearises all hosted
  points in one pass.
- Tracker: parallel linearisation, the coarse-level motion hypotheses in parallel. Immature point tracing, mono
  initialisation and pyramid construction in parallel.
- Input decoding in a background thread; `--start` / `--stride` skip frames without decoding.

| ms/frame (16 threads) | mono KITTI | stereo KITTI | fisheye 960x608 |
|-----------------------|------------|--------------|-----------------|
| before (+ input)      | 154 (+15)  | 222 (+82)    | 64 (+4)         |
| after (input hidden)  | 23         | 31           | 18              |

Window BA 244 -> 20 ms per keyframe. Benchmark (nine 500-frame segments, alpha -0.03) unchanged: stereo ATE SE3
0.500 m / 0.78 % / 0.425 deg/100m (before 0.501 / 0.79 / 0.428), mono 0.766 m / 1.11 % / 0.424 (0.995 / 1.28 /
0.430, within the mono run-to-run spread). aiMotive garage fisheye, all 1886 frames: 90 -> 18 ms/frame, same map.

Remaining per keyframe: BA ~20 ms (linearise 7, Schur 5, safeguard 5); per frame: pyramid 3-5 ms, tracking 3,
tracing 2-3. Not done: SIMD / float residuals (at most ~2x on the BA, ~4 ms/frame), a separate mapping thread
(tracking latency only; throughput is already bound by the parallel BA).

Own data (step 11 inputs): `run_vo --video file.h264 --camera cam.yaml --scale 0.5` with an EUCM YAML and an
obstruction mask (config/camera_example.yaml). aiMotive front fisheye, underground garage, 1886 frames at 960x608:
1826 posed (the first 60 are before monocular initialisation), 248 keyframes, 72.6k points, ~90 ms/frame, no weak
tracking. No ground truth; open: calibration pixel-centre convention, metric scale from the camera height.
