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
| 12 | Multi-camera rig: body poses, per-camera brightness, cross-camera residuals, rig YAML, multi-camera export | KITTI stereo as a 2-camera rig; synthetic surround rig metric; own rig |
| 13 | Denser point clouds: converged candidates as map points; semi-dense mapping pass with the final poses | synthetic depth accuracy; Prodigy garage |
| 14 | Place recognition: ORB features with map depth per keyframe camera, keyframe records, FBoW vocabulary, inverted-file database | retrieval on KITTI 00 (GT) and the garage laps |
| 15 | Loop detection: cross-camera candidates, rig P3P RANSAC, odometry-consistency and temporal checks | no false loops on KITTI 00 / garage / multi-floor |
| 16 | Pose graph (Ceres, SE3) over keyframes with loop edges; frames and points follow | KITTI 00 full ATE, garage lap consistency |
| 17 | Global bundle adjustment over all keyframes (optional) | cross-lap geometry |

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

## Step 12 log

A keyframe is a rig snapshot: one body pose (vehicle frame x forward, y left, z up) and affine brightness per
camera (6 + 2C parameters). A point is hosted in one camera of one keyframe (bearing + inverse distance) and has
residuals in every camera of every other keyframe and in the other cameras of its own keyframe (static residuals,
the former stereo residual, which only depend on depth and brightness). Frame Jacobians go through the extrinsic
adjoint (numeric test). Tracking estimates the body motion and per-camera brightness from all cameras. New keyframes
match their candidates into the other cameras (metric depth); activation checks occupancy in all cameras of the new
keyframe, so a surface seen by two cameras is hosted once (no ghost copies from double hosting).

- KITTI 00 stereo through the rig path, nine 500-frame segments: ATE SE3 0.490 m, 0.80 % / 0.435 deg/100m (stereo
  code before: 0.500 m, 0.78 % / 0.425); with 1000 points per camera 0.545 m / 0.81 % / 0.486. Both images are now
  processed fully (pyramids, tracking, residuals into right images of all keyframes): ~2x time per frame.
- Synthetic surround room: front/left/right fisheye rig tracks metrically without alignment; a front/rear rig
  without common view recovers metric scale only in turns (0.1 rad per keyframe: 0.1 % scale error; 0.04 rad: 3 %
  bias from rendering noise), as expected from Clipp et al. 2008. Straight driving needs overlapping cameras.
- Input: `run_vo --rig rig.yaml [--rig-cameras front left ...]` (config/rig_example.yaml), COLMAP export with one
  camera model and image folder per camera.

Open: cross-camera tracing of immature points (candidates are traced only in their own camera over time), real rig
data (extrinsics, synchronisation), dynamic-object masks, SIMD for the larger residual counts.

aiMotive Prodigy recording (e:/records/aimrec), four surround fisheyes (F/B_FISHEYE_C, M_FISHEYE_L/R, 1936x1220
EUCM, run at half resolution): `rig_from_sensorconfig` writes the rig from sensorconfig.yaml (streams by device
id, obstruction masks); `estimate_sync` found M_FISHEYE_L one frame late (fast body-frame angular velocity of
single-camera runs, cross-correlated). 1885 frames (75 s, 155 m, underground garage): all frames posed from the
first one, no weak tracking, 77 ms/frame, 195k coloured points; height stays within 0.2 m. Static vs temporal depth
bias +-0.06 % across the image radius: the sensorconfig calibration is consistent.

## Step 13 log

- Converged candidates (`--map-candidates`): candidates that were never activated become map points when their
  keyframe leaves the window, if they have 2+ good traces and a relative inverse-depth half interval below 5 %.
- Semi-dense pass (`--densify`, `SemiDenseMapper`): after odometry, with the final poses and per-frame brightness,
  every keyframe image hosts ~20k high-gradient pixels (region-adaptive selector). Each is traced into the other
  cameras of its frame and into all cameras of the following 30 frames (other cameras only where the current depth
  projects inside). Accepted with 3+ good traces, outliers <= 20 % of them, and a relative half interval <= 2 %.
  Tracing only searches inside the current interval, so a wrong first match can confirm itself; the multi-view check
  (`--densify-voxel 0.05 --densify-voxel-hosts 2`: a point needs points of another host image in its 5 cm voxel)
  removes most of those. Edges along the epipolar line stay unconstrained in one pair but not in another (cameras
  and frames give different epipolar directions).
- Synthetic surround room: median depth error < 0.5 %, 95 % < 3 %.
- Prodigy garage, lap 1 (850 frames, F/L/B fisheyes, half resolution): 73k active + 105k candidate + 374k semi-dense
  points, 493k after filtering (27k before). Densify pass ~200 ms/frame. Remaining: some haze at the ceiling,
  textureless floor stays empty, no dynamic-object masks yet. `render_cloud` renders a PLY view to PNG.

## Step 14 log

Loop closure works offline on keyframe records (`run_vo --keyframes-out file.kfr`): at marginalisation every keyframe
camera gets ~1000 ORB keypoints (grid-spread, masked) whose depth comes from the window points and converged
candidates projected into it (only where neighbours within 3 px agree to 5 %), plus the final body pose and the rig.
The interfaces are incremental (add a keyframe, query the earlier ones), so an online version can reuse them.
FBoW (MIT) provides the vocabulary; the inverted-file database with the L1 score and the normalisation by the
previous keyframe follow Nister & Stewenius 2006 and Galvez-Lopez & Tardos 2012. `train_vocabulary` trains on
records; `place_recall` measures top-1 retrieval against ground truth or the odometry poses (5 m, 150+ frames older,
every camera against every camera).

| Data | Keyframes | Revisit queries | Top-1 correct | Normalised score >= 1.0: accepted / correct |
|------|-----------|-----------------|---------------|---------------------------------------------|
| KITTI 00 stereo | 2653 | 479 | 93 % | 388 / 381 |
| Garage, F/L/B fisheyes, both laps | 180 | 84 | 80 % | 28 / 22 |
| Garage, KITTI-only vocabulary | 180 | 84 | 69 % | 36 / 26 |

Vocabulary k=10, 5 levels, trained on 3000 images of both sequences. Garage revisits run in the opposite direction:
56 of 67 correct matches are front against rear camera, which a single camera cannot find. A domain vocabulary helps
(69 -> 80 %); it should be trained on other aiMotive recordings than the test one. Score thresholds alone are not
precise enough (garage: 6 wrong of 28 at 1.0), so step 15 verifies geometrically. KITTI 00 full-sequence baseline
before loop closure: ATE SE3 3.95 m.
