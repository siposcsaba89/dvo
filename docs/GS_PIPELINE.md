# Gaussian-splatting reconstruction pipeline (current best)

How to turn an aiMotive recording into a Gaussian-splatting model with our pipeline: sdv odometry and loop
closure for poses and points, multi-view monocular depth to fill textureless surfaces, a TSDF fusion of that depth,
and gsplat trained with a dense depth loss. Every command and parameter of the current best setup is listed here, so
a run can be reproduced exactly. **Keep this file up to date with every change of the pipeline** (last section).

State: 2026-09-30. Reference result: garage record `voxelnet 20250414T141729Z_nm` (Zion), see [Results](#results).

## One command

```bash
# 1. rig (once per recording, see step 0)
# 2. everything else, resumable (each stage leaves RESULT_DIR/.done_<stage>; delete it to rerun a stage)
FORWARD=1.55 HEIGHT=1.50 tools/gs_pipeline.sh results/<name>      # FORWARD/HEIGHT: see step 10
```

[tools/gs_pipeline.sh](../tools/gs_pipeline.sh) runs steps 1-10 with exactly the parameters below. Its settings
(environment variables) are `VO_CAMS`, `GS_CAMS`, `DA3_GROUPS`, `VO_ARGS`, `DENSIFY_ARGS`, `REFINE_RIG`, `FORWARD`, `HEIGHT`,
and `STOP_AFTER=vo` or
`STOP_AFTER=loops` to stop after that stage and look at the trajectory plots first (run it again to continue). Log: redirect it, e.g.
`> results/<name>/pipeline.log`; start it with `setsid nohup` so it survives the terminal.

## Requirements

| What | Where / version |
|---|---|
| sdv (this repo) | `cmake --preset linux-clang && cmake --build --preset linux-release` → `build/linux/Release/` (clang, vcpkg `x64-linux-dynamic`) |
| CUDA (optional) | CUDA 13.4 at `/usr/local/cuda` (nvcc, host compiler clang or gcc); found automatically, `-DSDV_CUDA=OFF` builds without. Enables NVDEC decoding (`--decode`, below); FFmpeg from vcpkg with `nvcodec` |
| gsplat + our scripts | `/home/csaba/projects/gsplat` (gsplat 1.6.0, base commit `512d366b`) with our changes in `examples/` (below) |
| Python env for gsplat | `/home/csaba/mamba/envs/occnet` (torch 2.12 cu132; `tensorly` only for the bilateral grid) |
| Python env for DA3 | `/home/csaba/mamba/envs/depth-anything-3` (torch 2.13 cu130), model `depth-anything/DA3-BASE` (Apache-2.0) in the HF cache; run with `HF_HUB_OFFLINE=1` |
| Loop vocabulary | `results/voc_k10l5.fbow` |
| GPU | 16 GB (RTX 5060 Ti, WSL). Never run two GPU jobs at once: WSL reports spurious out-of-memory errors |
| RAM | 48 GB WSL. One heavy job at a time: `close_loops` of a long run (~19 GB) next to DA3 took the whole WSL VM down (2026-09-30). The host has 63 GB and WSL's page cache counts against Windows and is not given back while busy: with a 48 GB limit fuse filled it (6 GB used + 28 GB cache, vmmemWSL 45 GB) and Windows ran out of memory. Now `memory=40GB` in `.wslconfig`, and `tools/drop_cache.py` (started by the pipeline) drops the run's cached files above `MAX_CACHE_GB` (8) |
| Disk | A per-image depth folder is ~10 GB on Chili (10165 images, 960x540 float16), a 3.4 M checkpoint ~2.4 GB with optimizer state. The disk filled overnight 2026-10-01 (run B died at 49 k). The pipeline now deletes the raw DA3 and aligned depth after the TSDF (`KEEP_INTERMEDIATE=1` keeps them) and the numbered checkpoints of a finished run (the final one and the previews stay). Deleting inside WSL does not shrink the Windows `ext4.vhdx`: `wsl --shutdown`, then `Optimize-VHD -Mode Full` / diskpart `compact vdisk`, or `wsl --manage <distro> --set-sparse true` once |

Our gsplat files, **not committed yet** (gsplat `main` + working tree): `examples/mono_depth_mv.py`,
`align_depth.py`, `fuse_depth.py`, `tsdf_fuse.py`, `depth_vis.py`, `eval_cameras.py`, `render_flyby.py`,
`render_depth.py`, `mono_depth.py`, `datasets/colmap_text.py`, and changes to `simple_trainer.py` (dense depth loss,
fused Adam and per-image TV for the bilateral grid) and `datasets/colmap.py` (dense depth loading, symlinked folders).

## Inputs

- **Recording**: the directory with `record/` (`Camera_*_meta.dat`, `_data.h264`). No IMU/INS data is used.
- **Calibration**: a calibreye result, `final_sensorconfig.zip` (or `autocalib_sensorconfig.zip`), run on this
  recording or a calibration drive of the vehicle close in time, **with per-camera intrinsics**
  (`extrin.optimize_intrinsics=1`, or `--im_sc` with a properly calibrated config). The configs that come with
  recordings can hold nominal intrinsics (Chili 2026: five of six cameras with identical `fy cx cy alpha beta`):
  check the camera YAMLs of step 0 and stop if they are identical. Zips written on Windows (backslash entry names)
  are read as well.

## Steps

`R=results/<name>`, `B=build/linux/Release`, `V=results/voc_k10l5.fbow`, `G=/home/csaba/projects/gsplat/examples`.
Cameras: `VO="F_CTCAM_L F_CTCAM_R M_NEIGHBORLANECAM_L M_NEIGHBORLANECAM_R"`, `GS="$VO B_MIDRANGECAM_C"`.

### 0. Rig (manual, once)

```bash
$B/aimrec_extract --record REC/record --calib CALIB.zip --rig-out $R/rig      # prints cameras, frames, fps
```

Then edit `$R/rig/rig.yaml`: add `mask_inflate: 12` (rig-wide, a margin around the ego vehicle) and, per camera,
`image_width:` half the sensor width (`1920` for 3840 px, `1936` for 2896 px cameras): all later steps work at this
resolution, and the odometry at half of it again (`--scale 0.5`, ~960 px wide). Check the camera YAMLs (intrinsics,
see Inputs) and the masks once (`aimrec_extract ... -o DIR --first 3000 --count 1 --color` writes a frame and the
mask per camera).

### 0b. Rig rotations (optional, `REFINE_RIG=1`)

```bash
$B/run_vo --rig $R/rig/rig.yaml --rig-cameras $VO --scale 0.5 -n 3000 $VO_ARGS -o $R/rig_refine/poses.txt \
    --keyframes-out $R/rig_refine/keyframes.kfr
$B/close_loops --keyframes $R/rig_refine/keyframes.kfr --poses $R/rig_refine/poses.txt --vocabulary $V --rig $R/rig/rig.yaml \
    --rig-cameras $VO --scale 0.5 --photometric --pba-points 120 --pba-extrinsics --pba-extrinsic-fix-scale \
    --pba-extrinsic-sigma-t 0.05 \
    --rig-out $R/rig/rig_refined.yaml --out $R/rig_refine/poses_loop.txt
```

A joint photometric BA over the first 3000 frames (`RIG_FRAMES`, ~7 GB per 300 keyframes) refines the camera
extrinsics; the sum of the distances between the cameras is held (`--pba-extrinsic-fix-scale`), so the metric scale
stays while the cameras can move (5 cm prior). The log lists every baseline's change. All later steps then use
`rig/rig_refined.yaml` (close_loops takes the extrinsics from the keyframe records, so the odometry has to run with
the refined rig). Needed when a camera is off: with F_MIDRANGECAM_C in the odometry (Zion autocalib) it is pitched
-0.16 deg against the others; refined, its points' floater rate halves and the loop drift halves (ROADMAP step 24).
`--pba-intrinsics [CAMERAS]` refines the intrinsics as well (a calibration check: the Zion autocalib intrinsics
moved < 0.3 % / 0.6 px, F_MIDRANGECAM_C cy 2 px, i.e. the same pitch).

### 1. Odometry (CPU, ~40 ms/frame)

```bash
VO_ARGS="--trace-min-quality 3 --static-min-quality 3 --point-min-good-fraction 0.5"
$B/run_vo --rig $R/rig/rig.yaml --rig-cameras $VO --scale 0.5 $VO_ARGS -o $R/poses.txt --keyframes-out $R/keyframes.kfr \
    --png $R/odometry.png --trajectory-ply $R/odometry_trajectory.ply
python tools/plot_trajectory.py $R/odometry_trajectory.png $R/poses.txt      # top view by height + height over time
```

Odometry on the four CT + neighbour-lane cameras (the fisheyes were worse, see Rejected). `VO_ARGS`: a good trace
needs the second-best match along the epipolar line at 3x the best energy (default 2), between the cameras of a
keyframe too, and a point leaves the window when fewer than half of its residuals in the image are good. Zion
garage: odometry points 12.9 -> 9.4 % floaters, maximum loop drift 0.29 -> 0.06 m. Stricter settings lose the metric
scale (ambiguity 5: the run 42 % short): compare the run length with an earlier run. Look at `odometry.png`
(top view with the map points) and `odometry_trajectory.png` (levels of a multi-storey garage are height plateaus;
drift shows as a sloping floor). `--ply` would also write the coloured map points, at the cost of decoding every
keyframe image again; the pipeline does not need them.

### 2. Loops, photometric BA, densify (CPU, ~15 min per 3000 frames)

```bash
$B/detect_loops --keyframes $R/keyframes.kfr --vocabulary $V --out $R/loops.txt --verbose
$B/close_loops --keyframes $R/keyframes.kfr --poses $R/poses.txt --vocabulary $V --rig $R/rig/rig.yaml \
    --rig-cameras $VO --scale 0.5 --photometric --pba-points 120 [--pba-block-keyframes 200] \
    --densify --densify-cameras $GS --merge --densify-min-quality 3 --free-space --densify-drop-frames 5 \
    --densify-coarse-step 2 --out $R/diag/poses_loop.txt --ply $R/diag/cloud.ply --points-out $R/diag/points.ply
python tools/level_run.py $R/diag/poses_loop.txt $R/diag/cloud.ply $R/diag/points.ply
python tools/plot_trajectory.py $R/diag/trajectory.png $R/poses.txt $R/diag/poses_loop.txt
```

**Levelling**: without IMU the world "up" is the body up of the first frame, so a start on a slope tilts the whole map
(Chili: 2.1 deg, the lower level rose 1.2 m across the garage although it is flat). `level_run.py` takes the mean body
up axis over the drive (frames more than 3 deg off it, i.e. ramps, left out) as vertical and rotates poses and clouds
in place (original poses kept as `.unlevelled`). Check: in `trajectory.png` every level is a flat plateau (Chili
after levelling: lower level within 12 cm); revisits at the same place must agree in height (loop consistency), a
constant slope of a level means a remaining tilt, not a bad loop.

**Memory of the photometric BA**: ~3-4 KB per pattern residual in Ceres, plus the keyframe images. The joint problem
fits up to ~400 keyframes (garage: 270 keyframes, 0.9 M residuals, 5.9 GB). Longer runs use
`--pba-block-keyframes 200` (the pipeline does so above 400 keyframes): first a joint solve over all keyframes with
every m-th point (`--pba-block-coarse-residuals`, 3 M), which fixes the large-scale shape of the trajectory, then
blocks of 200 consecutive keyframes with all points, the rest fixed, 2 sweeps per round with the borders shifted by
half a block (`--pba-block-sweeps`). Garage check: same final rmse (6.52) and residuals as the joint solve, poses
within 8 mm median (22 mm max, 0.04 deg) of it, 4.0 GB. Without the coarse solve the blocks drift (a unit test covers
both). Chili (1357 keyframes, 7.5 M candidate residuals): ~19 GB instead of ~45 GB.

`--densify-min-quality 3`: the same ambiguity threshold for the densify traces. `--free-space`: points that two or
more other densify host images saw through (they measured a surface clearly behind it) are removed. Zion garage, held
out views: floaters 3.5 -> 1.2 %, 6 % fewer points, 9 s. `--densify-verify` (refines each point's inverse depth
over all buffered views and needs photometric agreement in several of them) gets ~0.8 % and a tighter depth, but
removes ~40 % of the points; not in the default `DENSIFY_ARGS` until a GS comparison. `--densify-drop-frames 5`: a candidate
without a good trace 5 frames after its host is not traced further (garage: traces -23 %, 1.5 % fewer points,
floaters unchanged). `--densify-coarse-step 2`: epipolar searches longer than 16 steps sample every 2nd pixel, then
pixel by pixel around the best match and the best one beyond the ambiguity exclusion (densify -15 %, same points and
precision).

`poses_loop.txt`: body poses of all frames. `cloud.ply`: densified, merged cloud (neighbour filter 3 within 0.2 m).
`points.ply`: every point with its attributes (observations, depth sigma, kept); the **trusted points** of the later
steps come from it (kept, >= 20 observations, sigma <= 0.004, an exported point within 3 cm). Check
`detect_loops.log`: in multi-storey garages, loops between levels would show as large corrections.

Egomotion for other tools (aiMotive `egomotion2.json`, keyed by record frame id, `RT_ECEF_body` = T_world_body in
the run's local levelled frame, not geodetic; `time`/`time_host` = earliest camera exposure start, host clock, s):

```bash
$B/export_egomotion --rig $R/rig/rig.yaml --rig-cameras $VO --poses $R/diag/poses_loop.txt -o $R/egomotion2.json
```

### 3. COLMAP export

```bash
$B/export_colmap --rig $R/rig/rig.yaml --rig-cameras $VO --cameras $GS --poses $R/diag/poses_loop.txt \
    --cloud $R/diag/cloud.ply --scale 0.5 --out $R/colmap
```

Defaults: a frame every 0.25 m or 5 deg (`--min-travel`, `--min-rotation`), virtual pinholes cropped to 100 deg
(`--max-fov`), points3D with 10+ neighbours within 0.2 m (`--min-neighbours`, against floaters), 3 cm voxels.
Garage: 456 frames x 5 cameras, 968x627 px, 1.96 M points.

### 4. Multi-view depth, DA3-BASE with our poses (GPU, the long step)

```bash
cd $G
DA3=/home/csaba/mamba/envs/depth-anything-3/bin/python
$DA3 mono_depth_mv.py --data_dir $R/colmap --window 3 --step 2 --keep 1 --process_res 756 --out mono_mv \
    --cameras $VO --save $VO
$DA3 mono_depth_mv.py --data_dir $R/colmap --window 3 --step 2 --keep 1 --process_res 756 --out mono_mv \
    --cameras B_MIDRANGECAM_C M_NEIGHBORLANECAM_L M_NEIGHBORLANECAM_R --save B_MIDRANGECAM_C
```

DA3 sees 7 frames (centre +-3 steps of 2) of every window camera with our intrinsics and world-to-camera poses, so
its depth comes out in our (metric) scale; the centre frame and its neighbours are kept. 16 GB fit 4 cameras x 7
frames (28 views) at 756 px; do not lower the resolution, use fewer cameras per window instead (30+ views run out of
memory). Resumable: finished images are skipped.

### 5. Alignment to our points

```bash
python align_depth.py --data_dir $R/colmap --reference $R/diag/points.ply --mono mono_mv --depth depth_mv --vis_every 20
```

Per image one scale, robust in log depth, fitted to the visible trusted points within 15 m (no inverse-depth
offset: it bent the near range). Trusted points more than 1.5x behind the raw DA3 depth are treated as occluded
(`--max_behind`: seen through textureless walls that none of our points occlude; they had scaled whole image blocks
by 2.5x). A scale more than 25 % off its camera's median is replaced by its neighbours' (`--max_jump`,
`--prior_frames`). Writes `depth_mv/`, `depth_mv_fits.csv`, and the error by distance.

### 6. Per-image fusion: correction, confidence, floaters

```bash
python fuse_depth.py --data_dir $R/colmap --reference $R/diag/points.ply --depth depth_mv --out $R/colmap_fused \
    --fill_voxel 0.06
```

Per image a smooth (planar, log-scale) correction and a confidence map: how well the depth agrees with nearby trusted
points (edge-aware, `--conf_px 40`, `--tol 0.1`), plus the floor: pixels whose depth agrees with the floor plane
under the camera (`--floor_tol 0.08`; floor height = lowest strong peak of the trusted points within 0.3 m of
*camera height - mounting height*, the mounting height estimated per camera from the data, so it works on every
level and ramp). Writes `depth_mv_fused/` and `depth_mv_conf/` (inputs of step 7); its own fill and floater filter are
superseded by step 7.

### 7. TSDF fusion (GPU, ~5 min)

```bash
python tsdf_fuse.py --data_dir $R/colmap --reference $R/diag/points.ply --depth depth_mv_fused --conf depth_mv_conf \
    --out $R/colmap_tsdf --name tsdf --min_conf 0.3 --min_weight 0.02 --max_up 0.0 --iterations 2 --refine quadratic
```

All depth maps and the trusted points (weight 4, `--point_weight`) go into one sparse TSDF (4 cm voxels, sorted
64-bit keys + binary search on the GPU; truncation max(8 cm, 3 % of the depth); weight confidence / depth^2). Second
iteration: every image's depth is corrected towards the fused surface (quadratic log-scale field) and integrated again.
Outputs:
- `colmap_tsdf/sparse/0/points3D.txt`: our points minus floaters + single-layer surface fill (6 cm voxels);
- `colmap/tsdf_fused/`, `colmap/tsdf_conf/`: depth targets raycast from the surface (first zero crossing, trilinear;
  searched around the corrected image depth, a wide search where that fails);
- floaters: our points in front of the surface by max(10 cm, 5 %) in >= 2 views on rays looking down (`--max_up 0`;
  nothing above camera height is removed, pipes and cable trays survive), unless trusted.
`--holdout 5` keeps every 5th trusted point out, for an unbiased check (see Evaluation); do not use it for the final run.

### 8. Training (GPU, ~25 min at 15-22 it/s)

```bash
python simple_trainer.py default --data_dir $R/colmap_tsdf --data_factor 1 --result_dir $R/gsplat_tsdf \
    --disable_viewer --disable_video --dense_depth tsdf_fused --dense_depth_conf tsdf_conf
```

30k steps, default strategy. Dense depth loss: L1 on disparity weighted by the confidence (>= 0.6, depth <= 15 m,
inside the mask), `--dense_depth_lambda 0.1` decaying to 10 % (`--dense_depth_final 0.1`).

### 9. Evaluation

```bash
python eval_cameras.py --data_dir $R/colmap_tsdf --result_dir $R/gsplat_tsdf --step 29999      # PSNR, floor ratio per camera
```

Validation: every 8th image. Floor depth ratio: rendered / true depth in the lower quarter of the images, the true
depth from the floor plane under the camera; ~1.0 means no mirror world or fog on the floor.

### 10. Renders

```bash
python render_flyby.py --data_dir $R/colmap_tsdf --ckpt $R/gsplat_tsdf/ckpts/ckpt_29999_rank0.pt \
    --poses $R/diag/poses_loop.txt --out $R/gsplat_tsdf/flyby_look_front.mp4 --forward 1.55 --height 1.50 \
    --yaw_amp 35 --sway_amp 0.4 --bob_amp 0.25            # novel views; --follow CAMERA renders a training camera
python render_depth.py --data_dir $R/colmap_tsdf --ckpt $R/gsplat_tsdf/ckpts/ckpt_29999_rank0.pt --out $R/gsplat_tsdf/depth \
    --split val --indices 20 45 80 100 140 160 200 230 260 280
```

The fly-by camera moves along the body trajectory; the body origin is at the rear axle, so `--forward` / `--height`
put it at the CT cameras (Zion 1.75 / 1.67, Chili 1.55 / 1.50: the CT camera translation in the rig).

## Results

Garage `voxelnet 20250414T141729Z_nm` (Zion, 3286 frames, 130 m, autocalib calibration), 5 cameras, validation on
285 images:

| Setup | PSNR | SSIM | LPIPS | Gaussians | Floor depth ratio 10-90 % |
|---|---|---|---|---|---|
| 4 cameras, no mono depth (baseline) | 31.02 (4 cams) | | | | ~1.08 (mirror world) |
| 5 cameras + dense depth, per-image fill | 29.89 | 0.953 | 0.145 | 1.59 M | 0.95-1.04 |
| **5 cameras + dense depth, TSDF fusion (current)** | **29.79** | **0.953** | **0.144** | **1.47 M** | **0.96-1.00** |

Depth quality against held-out trusted points (4-6 m): per-image depth median error 2.2 %, TSDF 0.6 %. Floor
thickness of the fill (spread per 1 m tile, median): per-image 13 cm, TSDF (2 iterations) 2 cm. The baseline has the
highest PSNR on the training views but a mirror world under the glossy floor; the depth runs fix the geometry.

## Rejected (do not retry without a new idea)

- **Fisheye cameras** for densify/GS: DA3 compresses their depth range (floor 1.32 near .. 0.57 far).
- **F_MIDRANGECAM_C** (behind the windshield): -2 dB and it degrades the other cameras (B_MID -3.6 dB); a
  per-camera colour fit recovers only 2 of 7 dB, so its error is geometry/sharpness (calibration ~0.1 deg pitch).
- **Bilateral grid** colour correction: -2 dB (colour is already consistent between cameras; +/-1 %). If used,
  `--bilateral_grid_fused` (pip `fused-bilagrid`) and the per-image TV now make it ~as fast as without.
- **`--opacity_reg 0.01`** (with `dense_depth_final 0.5`): pruned 80 % of the Gaussians, blurry, fog stayed.
- **SuiteSparse/CHOLMOD** for the BA's Cholesky: its Supernodal module and SPQR are GPL-2.0+ (not usable for a
  commercial codebase), and after the fixed-brightness coarse solve the linear solver is only ~25-30 % of a block
  solve.
- Dense per-image correction fields interpolated from sparse points (cloudy ghosting); an inverse-depth offset in
  the alignment (degenerate scales); single-image DA3 metric (twice the error of multi-view).

## Open issues

- Fog in novel views (glossy floor reflections, the moving car's ghost, the uncovered straight-ahead corridor):
  next steps are a fog measure in novel views, masking moving objects, a free-space loss against the TSDF.
- Multi-storey garages: the floor search follows the levels (steps 6, 9); loops between levels are not rejected
  specially yet.
- Trusted points behind textureless walls are only handled by the 1.5x rule; a visibility test against the TSDF
  would be cleaner.
- BA speed: the linear solver (Eigen sparse Cholesky, single-threaded; Ceres here has no SuiteSparse, LAPACK or CUDA)
  dominates large reduced systems. The coarse solve holds the brightness fixed (0.7 instead of 6.7 s of linear
  solver on the garage, same final rmse); `--pba-solver nesdis|iterative` were slower on the garage. Next, if still
  needed: measure the breakdown on a long run; Ceres' CUDA dense Schur for the coarse solve (licence-clean).
- Densify speed: hosts are keyframes, traced into all following frames (30) of all densify cameras, so every frame is
  decoded. Trace frames chosen by motion (>= 5 cm or 0.5 deg since the last one) would save ~22 % on Chili
  (11 % standstill), ~37 % on the Zion garage (35 % standstill); 10 cm ~ half. Needs a quality check (points, depth
  sigma, trusted points) on the garage.

## Changes

- 2026-09-30: first version. Chili calibration (Windows zip paths), floor plane relative to the camera mounting
  height (multi-storey garages), `tools/gs_pipeline.sh`. Blocked photometric BA for long runs
  (`--pba-block-keyframes`), `run_vo --trajectory-ply`, `tools/plot_trajectory.py`, `STOP_AFTER`; the `--ply`
  colour pass of `run_vo` decodes only the keyframe images. Levelling after the loop closure (`tools/level_run.py`); `close_loops --out` written right after the BA. Coarse BA solve with fixed brightness; `--pba-solver`; Ceres time breakdown in the log.
  `mono_depth_mv.py` resume redoes empty files (crash mid-write); `align_depth.py` keeps at most 2000 held-out
  samples per image for the error report (all points of Chili ran out of RAM, now flat at ~5 GB). An image with
  fewer than 200 of our points (standing in front of a plain wall) gets its scale from the neighbours instead of
  no depth file (fuse needs one per image).
  `fuse_depth.py` pass 2 reads the saved depth/confidence back instead of caching ~1.5 MB per image (15 GB on
  Chili, took WSL down); disk write queues of fuse and TSDF bounded to 16 images.
  `tools/drop_cache.py` in the pipeline (page cache of the run's files, posix_fadvise, no root needed).
  `simple_trainer.py --init_voxel` (mean point per voxel, m): clouds over 5 M points start from 8 cm voxels
  (Chili 11.3 M -> 3.4 M; 11.3 M ran out of GPU memory at the first densification, 'device not ready').
  Training resumable: `simple_trainer.py --resume CKPT` (full checkpoints: optimizers, densification state,
  schedulers; splats-only ones with fresh optimizers and fast-forwarded schedules), `--ckpt_every` (numbered full
  checkpoints, written atomically, with previews of `--preview_count` 15 validation views spread over the scene by
  farthest point sampling), `--balance_radius` (weighted sampling, 1 / same-camera images within r m). Pipeline:
  `STEPS`, `CKPT_EVERY` (10000), `BALANCE_RADIUS`, `GS_NAME`, `RESUME_FROM`, `TRAIN_ARGS`; a rerun resumes from the
  newest checkpoint. Chili 30 k (uniform): PSNR 29.10 on 1271 test images. Running: 60 k, balanced 3 m (levels
  77/13/10 % of the steps -> 60/22/17 %), densification until 30 k. `tsdf_fuse.py --readout prior` (crossing
  closest to the image's depth; default `first`) added, not yet evaluated. The fly-by needs ~26 GB RAM on Chili.
  Balanced 60 k result: the repeated opacity resets of densification until 30 k pruned to 507 k Gaussians (uniform
  30 k: 908 k); preview PSNR 30.26 vs 29.74 of the 30 k run on the same 15 views, still too few Gaussians for three
  levels. `TRAIN_CONFIG=mcmc` (fixed budget, `--strategy.cap_max`): Chili run 3 M, 60 k steps, balanced 3 m,
  `INIT_VOXEL=0.1` (2.25 M start, under the cap), 5 GB GPU, ~21 it/s. Diverged (previews ~19 dB and gray at
  10 k, loss rising after ~5 k), stopped; would need its own tuning with the dense-depth loss.
  `strategy.reset_stop_iter` (gsplat DefaultStrategy): no opacity resets from this step on, so densification after
  it is not pruned away again. Chili `gsplat_tsdf_bal60k_more`: 60 k, balanced 3 m, `TRAIN_ARGS="--strategy.
  refine_stop_iter 30000 --strategy.reset_stop_iter 15000 --strategy.grow_grad2d 0.0001"`: 3.43 M Gaussians
  (1.1 M at the last reset, grows until 30 k), ~6 GB GPU; preview PSNR 21.6 / 25.2 / 25.3 / 30.4 / 31.1 / 32.1 at
  10..60 k (bal60k 30.26), test PSNR 29.94 (uniform 30 k 29.10). Still +1 dB per 10 k at the end: continued as
  `gsplat_tsdf_bal120k` (`RESUME_FROM` its 60 k checkpoint, `STEPS=120000`, same arguments; the restored schedules
  keep decaying the means' learning rate, 3.5e-6 at 60 k, so the continuation mostly refines colour, opacity, shape).
  Nothing is pruned after `refine_stop_iter`: at 60 k 35 % of the 3.43 M had opacity < 0.005 and floaters were
  visible. Stopped at the 70 k checkpoint and continued with a pruning phase: `TRAIN_ARGS="--strategy.
  refine_stop_iter 82000 --strategy.reset_stop_iter 80000 --strategy.grow_grad2d 1e9"` (opacity resets at 72, 75,
  78 k, no split/clone, opacity and scale pruning until 82 k). Stopping a trainer: the dataloader's forkserver
  workers survive a kill of the main process (~0.9 GB RAM each); kill them too.
  Result: worse. Each reset pruned about a third (3.43 -> 2.02 -> 1.38 -> 0.91 -> 0.63 M): after a reset to 0.01 the
  Gaussians have 3 k steps to regain opacity, with the means' learning rate near zero and no growth to replace
  them. Preview PSNR 32.15 (60 k) -> 28.5 (80 k) -> 30.97 (120 k), test 29.62 (60 k: 29.94). Late opacity resets
  are the wrong tool for a large model; the best Chili model stays `gsplat_tsdf_bal60k_more` at 60 k.
  `carve_gaussians.py` (gsplat examples): removes Gaussians whose centre lies clearly in front of the TSDF depth
  (z < 0.9 D - 0.2 m) in >= 3 training views and >= half of the views that see it, and those with opacity < 0.005;
  counts cached in `<out>/carve_counts.npz`. `eval_cameras.py --render` renders the checkpoint itself (matches the
  trainer's renders: 29.97 vs 29.94). On the 60 k model: opacity < 0.005 is 1.2 M of 3.43 M and removing it changes
  nothing (29.97); free-space carving (233 k, 38 k visible) costs 0.42 dB (29.53, > 1 dB on 47 of 424 test images,
  mostly M_NEIGHBORLANECAM_L). The worst images lose real content: the view out of the open gate, floor near the
  camera, walls and ceiling; where the TSDF depth is off these are "in front" of it. Not usable as is: the depth
  maps are not accurate enough to carve against.
  Continuing a finished run does not help: 60 -> 70 k with the same settings, preview PSNR 32.15 -> 31.73 while the
  training loss kept falling (means frozen, no growth). Running: `gsplat_tsdf_bal120k_scratch`, the
  `bal60k_more` settings from scratch with `STEPS=120000` (the means' schedule stretched: at the end of
  densification, 30 k, 3x the learning rate of the 60 k run). Stopped after ~5 min for a schedule scaled to the
  image count: gsplat's defaults count steps, but what matters is passes over the images. Garage 2280 images,
  30 k steps: densification 7.5 passes, a reset every 1.5 passes; Chili 10165 images: a reset every 3000 steps is
  every 0.34 passes (most places not seen again before the next reset, the reason the resets pruned so hard).
  Running: `gsplat_tsdf_sched120k`, `STEPS=120000`, `TRAIN_ARGS="--strategy.reset_every 12000 --strategy.
  reset_stop_iter 36001 --strategy.refine_stop_iter 60000 --strategy.grow_grad2d 0.0002"` (resets 12/24/36 k,
  densification 6.7 passes). The garage schedule itself was never tuned either.
  Also scaled: gsplat's split and prune sizes are fractions of the scene extent (Chili 51 m, garage 31 m), so on
  Chili Gaussians up to 0.51 m were cloned instead of split (garage 0.31 m). Fixed to the garage's metres:
  `--strategy.grow_scale3d 0.006 --strategy.prune_scale3d 0.06`; that run is `gsplat_tsdf_sched120k_m` (the
  schedule-only `gsplat_tsdf_sched120k` stopped at 6.6 k). Result: test PSNR 30.84 (best before 29.94; every camera
  +0.6..1.2 dB) with 1.97 M Gaussians (3.43 M). Each reset still pruned ~40 % (12/24/36 k: 3.28 -> 1.90, 2.36 -> 1.36,
  1.81 -> 1.24 M), growth ~55 / step with grow_grad2d 2e-4 (1e-4: ~115). Previews 23.4 / 25.1 / 26.0 / 26.8 / 28.5 /
  29.1 at 10..60 k (behind while densifying), then 31.0 / 31.7 / 32.9 / 32.6 / 32.5 / 32.6 at 70..120 k: flat from
  90 k. Best Chili model now. Capacity target ~4.5 M: the garage (1 level, 2280
  images) ended at 1.47 M, Chili has 3 levels, while its upper levels have only ~1300 / ~1000 images each.
  TSDF depth targets are faceted everywhere (not only columns): normal maps of `tsdf_fused` show flat patches of
  about one 4 cm voxel on cars, columns, walls and a staircase on the floor, the per-image mono is smooth (both
  float16, so not the storage). Cause: the zero crossing of the voxel TSDF (each cell its own tilted plane, the
  nearest-voxel fallback where not all 8 neighbours are valid). Prototype (scratchpad `detail.py`, not in the
  pipeline): mono shape with TSDF scale, log(TSDF / mono) smoothed by a confidence-weighted guided filter on the log
  mono depth (r 15 px, local scale may follow depth), pixels disagreeing > 5 % keep the TSDF. Smooth where they
  agree; front cameras agree (median 0.6 %, 1 % of pixels kept), the side cameras do not (median 9-12 %, 10-23 %
  kept, still faceted there): the mono shape itself is off there.
  `detail_depth.py` (gsplat examples, from the prototype): full Chili run `colmap/tsdf_detail` (+ `_conf`, `_vis`),
  roughness (median angle between normals 2 px apart) mono 3.8-5.8, TSDF 15-20.5, detail 4.4-6.6 deg. Accuracy on
  held-out trusted points (`tsdf_fuse.py --holdout 10`, `colmap/tsdf_ho_*`, points <= 10 m, median |depth / point
  - 1|): TSDF 0.7-1.7 %, detail 0.8-2.3 % (+0.3..0.7 pp), corrected mono 1.7-12.9 % (+12 % too far below 2.5 m on
  B_MIDRANGECAM_C and M_NEIGHBORLANECAM_L: the near-range compression; the TSDF removes it). Beyond 10 m (the TSDF's
  `max_depth`) the TSDF depth reads a nearer surface, ~70 % short, on ~25 % of the far points; training uses
  <= 15 m (`dense_depth_max`), so those pixels pull far Gaussians forward: to look at.
  Sweep (every 50th image, held-out points <= 10 m, median |err| / roughness): TSDF 0.99 % / 18.0 deg; guided filter
  r 4 eps 1e-3 tol 0.02 1.10 % / 7.1 (smallest error), r 15 eps 1e-2 tol 0.05 1.58 % / 5.7; a joint bilateral of the
  TSDF itself 1.28-1.46 % / 2.7-4.7. The 90 % quantile is ~26 % for all, TSDF included: points on depth edges (a
  background point next to a foreground silhouette), not wrong surfaces. The column of
  M_NEIGHBORLANECAM_L/006091 shows the real defect: TSDF patches +-4 cm apart (row 300: 1.15 1.18 1.11 1.13 1.07),
  r 15 smooth (1.14 1.10 1.13), r 4 keeps the patches (it follows the TSDF within 2 %). Both built: `tsdf_detail`
  (r 15), `tsdf_detail4` (r 4); `gs_pipeline.sh` `DENSE_DEPTH` / `DENSE_CONF` choose the training targets.
  Overnight: run A `gsplat_tsdf_sched120k_m_detail` = the sched120k_m settings with `tsdf_detail` (the depth alone);
  run B `gsplat_tsdf_sched120k_m_g15` = the better depth of A / tsdf, grow_grad2d 1.5e-4 (more Gaussians).
  Run A: test PSNR 30.86 (tsdf targets 30.84), per camera within +-0.07 dB, 1.98 M Gaussians; previews 28.7 / 32.7 /
  33.1 at 60 / 90 / 120 k (tsdf 29.1 / 32.9 / 32.6). The depth targets do not move the test PSNR (test views lie
  between training views); whether the geometry is better needs the rendered depth / novel views.
  Run B died at 49 k when the disk filled; its only kept checkpoint was truncated (8 MB) although written to a
  `.tmp` and renamed: `simple_trainer.py` now syncs and checks the archive before the rename, and the pipeline resumes
  from the newest checkpoint that opens. Restarted from 0 (2026-10-01 07:48).
  Visible quality (user): the outside start and the upper level look bad (near bushes, pavement, columns streaked
  along the driving direction), the bottom level better; the test PSNR does not show it (per section 30.1-31.4,
  outside 30.7). Not speed: ~5 cm / frame outside, 15 on the ramp / upper level (the best PSNR). Subset test
  `sub_upper` (outside + upper level, frames 0-1600, 2326 images, points within 15 m and above the floor under the
  nearest camera, 4.95 M): trained alone and scored with the full models on the same 290 test images
  (`val_images.txt` in a data dir fixes the test images, `eval_names.py`). Full sched120k_m there: 30.88.
  `render_names.py`: ground truth | each model on the same views (`sub_upper/renders`). Outside: bushes streaked,
  far street, cars, sky mush; upper level M_NEIGHBORLANECAM_L/001247: the near column shifted and widened while the
  far car and wall line up (parallax: that camera's position or the near depth is off; check per camera = rig
  calibration vs along the drive = pose drift).
  `detail_depth.py --disagree drop`: no target where TSDF and mono disagree (instead of the TSDF depth), `--names`
  for a subset; `tsdf_agree` for the sub_upper images (targets on 55-66 % of the pixels vs 63-69 %), subset run 2
  `sub_upper/gsplat_agree`.
  Results on the 290 test images: subset alone 31.67 (full models 30.88 / 30.98): upper level frames 1000-1500 32.30
  vs 30.44 (+1.8 dB, "the best first floor so far" by eye), outside frames 0-500 29.38 vs 30.26 (no gain): the upper
  level is starved in the shared model (steps, Gaussians: 824 k for 2326 images alone vs 2 M for 10165), the
  outside has another problem (light, far geometry without depth, vegetation). Agree-only 31.58: no difference.
  Subset previews still +2 dB from 20 to 30 k (densification ends at 15 k): next a fresh 60 k subset run on the
  better of combined / weighted-mono depth (`detail_depth.py --disagree mono`: scaled mono everywhere, confidence
  1 where the TSDF agrees, 0.1 elsewhere and beyond the TSDF, trained with `--dense_depth_min_conf 0.05`; the depth
  loss is in disparity, so per cm a pixel at 1 m pulls ~100x a pixel at 10 m). A plain resume would continue the
  restored 30 k schedule (means at 1 %, frozen), which did not help on the full model (60 -> 70 k).
  Weighted mono 30 k: 31.72 (combined 31.67, agree-only 31.58): the depth variant is within noise; outside 29.3 in
  all three, so extra depth beyond 10 m does not fix the outside part.
  Image spacing: `export_colmap --min-travel 0.25` gives 0.29 m median on Chili (2033 of 8512 frames). A side
  camera (f 479 px) sees an object at 1 m jump ~140 px between training views (2 m: 70, 10 m: 14): the near field
  (bushes, kerbs, columns, parked cars) is sparsely sampled, and streaks along the driving direction are the typical
  result. Dense test `sub_d10`: export at 0.1 m, the same outside + upper-level cut, DA3 `--step 6` (the same
  +-1.8 m window), align, fuse, TSDF, weighted mono, schedule scaled by passes, the same 290 test images (their
  depth from the 0.25 m maps). Run B (full, grow 1.5e-4) dropped.
  The fused cloud of the dense subset (and the 0.25 m one) has stacked cars and walls on the ramp into the garage.
  Not the rig or the poses: the cameras' per-image depth agrees within 2-4 cm where the depth is right, and two
  passes over the same place agree within 2-5 cm. It is the per-image depth scale: per-image depth / trusted point
  (points hosted within 60 frames, z-buffered: the level below would bias it) is 0.995-1.000 outside, but 0.73-1.44
  per camera on the ramp (frames 1000-1040) and 1.06-1.24 on the upper level (1100-1240), the same in the 0.25 m
  export. One scale per image cannot fit DA3's depth-range compression where most points are near.
  Next test (`depthtest.txt`, 143 frames x 5 cameras on 4 stretches): multi-view DA3-BASE vs single-image DA3-Large
  (`mono_depth_mv.py --window 0 --no_poses`, 952 px; multi-view needs BASE for 28 views on 16 GB), each aligned with
  one scale and with `align_depth.py --fit power` (z = e^a z_mono^b per image, b pulled to 1, clipped 0.5-2).
  Result: the fit model was not the problem (power exponent ~0.99), the points were: `align_depth.py` fitted every
  image to all visible trusted points, and under the ramp and upper level the level below shows through the floor
  slab (few points on it). `--ref_frames 60` (only points made within 60 frames of the image): upper level
  0.991-0.996 per camera (was 1.15-1.26), cameras within 3-5 cm. Single-image DA3-Large (`--no_poses`,
  `align_depth.py --prescale`): per camera 0.71-1.31 even outside, 322 of 715 scales replaced, cameras 5-25 cm
  apart: multi-view DA3 stays (its joint inference with our poses is what makes the views consistent; the metric
  scale is only a Umeyama fit of its poses to ours, overridden per image by the alignment anyway). The scale
  prior replaced solid fits (F_CTCAM_L on the ramp 1.33 from 1800 points, 92 % inliers -> 1.0): now only fits
  with < `--trust_points` 500 points or < `--trust_inliers` 0.6 are replaced. F_CTCAM_L on the ramp (frames
  1000-1040) still reads 0.77 in the check although its fit says 1.29-1.34 (point/mono median 1.325 at frame 1004:
  the fit is right there; the check's point selection differs).
  Correction of the single-image verdict: its first alignment was unfair, the scale prior replaced 322 of 715 good
  fits (single-image scales are arbitrary per image, so jumps are normal). Fair (`--prescale --ref_frames 60`,
  solid fits kept): about as good as multi-view where there are points (upper level 0.99-1.00 per camera, power
  fit 1.000; cameras 3.5-4.5 cm), worse on the ramp (16-24 cm): there the right cameras see a bare wall at 2.3 m
  with 6-10 trusted points per image (B_MIDRANGECAM_C, M_NEIGHBORLANECAM_L ~400-540), the trusted filter
  (>= 20 observations) drops 95-99 % of the ramp's points (median 7-12 observations). `align_depth.py` fits now
  start at the consensus value (exact 1-D RANSAC: the densest +-5 % window) instead of the median, and the power fit's
  prior on b is fixed (it scaled with the depth spread, so narrow spreads ran b to its 0.5 limit). Single-image
  power fit with `--min_obs 8 --max_sigma 0.006`: upper level cameras 2.7-2.8 cm (multi-view 3.0-3.1), the worst
  upper window 6.7 cm (was 27-35), outside 4.6 cm; the ramp stays open (F_CTCAM_L 0.74, M_NEIGHBORLANECAM_R 1.36:
  no points to fit the right cameras to). Single-image DA3-Large is ~8x faster than multi-view BASE.
  Dense subset 0.1 m (first build, old alignment): test 33.93 (60 k subset 32.52, full 30.88), outside 0-500 31.69,
  upper 1000-1500 34.39; "the best model so far" by eye. Rebuild with the fixed alignment: `sub_d10b`.
  Balanced sampling (3 m) at 120 k steps: no image is skipped; uniform would draw each training image 13.5 times,
  balanced 7.1 (10 %) / 9.8 (median) / 28.4 (90 %) times; the weight is capped at the 10 % density, so the
  sparsest images get at most ~2x the uniform rate.
- 2026-10-01: point quality (ROADMAP step 24). `VO_ARGS` (`--trace-min-quality 3 --static-min-quality 3
  --point-min-good-fraction 0.5`), `DENSIFY_ARGS` (`--densify-min-quality 3 --free-space`), optional `REFINE_RIG=1`
  (stage rig, rotations from the first `RIG_FRAMES` frames, translations fixed). `tools/cloud_consistency.py`:
  floaters and depth agreement of a points.ply without ground truth (free-space test against the densify host images,
  hold-out mode for filters). 6 cameras with `REFINE_RIG=1 VO_CAMS="$VO B_MIDRANGECAM_C F_MIDRANGECAM_C"`
  (`results/zion_6cam`, up to the loops): loop drift max 0.07 m (4 cameras, old settings 0.29 m), 5.18 M points,
  0.5 % floaters (old 3.5 %); the defaults keep 4 odometry cameras until a GS comparison. Not yet trained with GS: the
  garage reference result above is from the old settings.
- 2026-10-01: single-image DA3-Large on the whole dense subset (`sub_s1`, 4730 images in 23 min, one camera per call,
  `--window 0 --no_poses --process_res 952`). The depth-fused cloud was clean on the ramp, the TSDF cloud showed a
  doubled ramp wall. Cause: F_CTCAM_L frames 1026-1040 were 1.37-1.69x too far. Scored against the points
  F_CTCAM_L itself tracked (host camera, +-15 frames, so surely visible) instead of all trusted points: 80 % of the
  trusted points made within 60 frames lie behind the bare ramp wall (the next turn of the ramp), the sparse z-buffer
  cannot hide them (no points on the wall), the unscaled single-image depth's median prescale lands on them and the
  consensus keeps them (12 000 points, "solid"). Multi-view DA3 is metric, so `--max_behind` removes them there (it was
  right on the ramp all along; the earlier "0.75 on the ramp" was the same check artefact, not glass). The ghost
  check (`ghost_check.py`) marks the real wall here (the far views see through it): it names the images involved,
  not which side is wrong. `align_depth.py`:
  - `--host_frames 15`: the points the image's own camera tracked within 15 frames set the starting scale (median),
    the fit uses the trusted points (`--ref_frames`) within `--anchor_tol` 30 % of it; fewer than `--host_min` 10 own
    points: neighbour pass. The host camera of the dump's camera index is found by the points' recorded distance.
  - neighbour pass: weak images are fitted to the nearest `--neighbour_same` 4 solid frames of their own camera
    (the same wall 0.1 m away, which covers the image and hides what lies behind), nearest first, and fitted images
    become solid (the scale is passed along the camera); other cameras' depth leaves gaps on a bare wall through
    which surfaces behind it win the fit (M_NEIGHBORLANECAM_L 1.7x). Neighbour fits start from the front-most
    group with >= `--neighbour_front` 30 % of the biggest (points can hide behind the surface, not lie in front).
  - `--drop_weak`: images the scale prior would rescale get no depth (single-image scales are per image, a prior
    from other frames is a guess).
  Ramp 990-1080 against own points: F_CTCAM_L 0.99-1.01 everywhere (was 1.37-1.69 on 1026-1040), B_MIDRANGECAM_C
  0.97-1.04, M_NEIGHBORLANECAM_L 0.88-1.04 up to 1038. Open: M_NEIGHBORLANECAM_R sees only a bare wall (0-3 own
  points), nothing verifies it (multi-view 1.6 m, single-image variants 1.1-3.6 m); the chained neighbour pass fits
  it anyway. Full run with these settings: `sub_s1b` (749 of 755 weak images fitted by the chained neighbour pass,
  7 dropped; `fuse_depth.py` skips images without depth). Its TSDF depth on the ramp reads 0.98-1.00 against the
  cameras' own points (M_NEIGHBORLANECAM_L 1040-1045 too, its per-image depth 1.36 there). Ghost check (views within
  6 m): 10.5 % ghost points (first single-image run 11.0 %, multi-view `sub_d10b` 7.1 %), frames 1050-1099 246 k
  (was 422 k). Test (same 290 images, same training): 33.86, multi-view `sub_d10b` 34.06; per camera -0.03 to
  -0.43 dB (B_MIDRANGECAM_C worst), upper level 1000-1500 34.49 vs 34.69. Single-image DA3-Large with the fixed
  alignment is close but not better: multi-view DA3 stays the default. Renders: `sub_upper/renders_s1b`.
- 2026-10-01: rig stage with the scale constraint (`--pba-extrinsic-fix-scale`, translations free otherwise):
  `results/zion_6cam_scale`, 6 cameras, loop drift max 0.03 m, 5.10 M points.
- 2026-10-01: `RIG_POINTS` (points per image of the rig BA, default 120; Chili 6 cameras needs 60: 2.6 M instead of
  4.6 M residuals, ~9 instead of ~16 GB); the rig stage reuses `rig_refine/keyframes.kfr` if present. Chili, 6 cameras
  (`results/chili_6cam`, up to the odometry + detect_loops): loop drift median 0.03 m, max 0.15 m (4 cameras, old
  settings: 0.45 / 1.55 m).
- 2026-10-01: running: full Chili trajectory with 6 GS cameras (F_MIDRANGECAM_C included, its rig now refined),
  `results/chili_6cam/run_gs_dense.sh` (after the loops stage; `.done_dense_<stage>` markers, resumable): export
  every 0.1 m with `rig/rig_refined.yaml` (~4400 frames x 6), single-image DA3-Large and the `sub_s1b` alignment
  (single-image is ~8x faster than multi-view and only 0.2 dB behind on the subset), fuse, TSDF, weighted mono,
  the `sub_s1b` training with the schedule scaled by passes (60 k steps per 2036 images, ~730 k steps), balanced
  3 m; test on every 16th exported frame (all cameras). On a GPU out-of-memory the run resumes from the newest
  checkpoint with 1.5x `grow_grad2d`. Outputs in `gs_dense/` (`eval.txt`, `renders`, `depth`, fly-by).
  Export: 26502 images, 6.16 M points; DA3 15 min per camera; alignment: all 6 host cameras found, 996 of 1005 weak
  images fitted by the neighbour pass, 14 dropped; TSDF 5.98 M own + 6.58 M fill points. The first attempt died at
  02:34 in the weighted mono: the WSL disk (`E:\vms\wsl\tumbleweed\ext4.vhdx`, 468 GB) had filled E: although `df /`
  showed ~500 GB free (the vhdx does not shrink when files are deleted). Every per-image folder of this run is
  ~26 GB: the script now writes no visualisations, deletes the raw / aligned / fused / TSDF depth once consumed, keeps
  the 2 newest checkpoints and stops training below 15 GB free on E: or /. The files written before the crash were
  all checked (load, PNG/JPEG end markers): none truncated.
  Stopped before training: the TSDF cloud had walls in the wrong place (the KIJÁRAT wall turned across the ramp, a
  wall over the floor on a lower level); the depth-fused cloud `colmap_fused/fused.ply` did not (its fill passes the
  multi-view check; the TSDF integrates every depth map). Not the poses: the chained neighbour pass (fitted images
  become references) drifts along bare walls. Ramp test (`ramp_test`, frames 930-1160, aligned depth / fused cloud,
  z-buffered): chained F_CTCAM_R 964-986 0.61 -> 0.13 (a wall at 1/8 of its distance), 988-998 1.65-2.11,
  M_NEIGHBORLANECAM_R 1052-1058 0.16-0.26; without chaining 0.60-1.17; own camera only (`--neighbour_same_only`):
  those images get no depth. `align_depth.py`: `--neighbour_chain` (the old behaviour, now off by default),
  `--neighbour_same_only`. New chain `run_gs_fused.sh` (after `run_da3.sh`, the raw DA3 depth was deleted): no
  chaining, own camera only, unverified images without depth, fuse, then training on the fused cloud with the
  per-image fused depth (confidence >= 0.6), no TSDF. Kept for comparison: `fused_v1_chained.ply` (clean),
  `tsdf_v1_ghosts.ply`. `mono_depth_mv.py --vis_every N` (previews of every n-th frame, 0: none; the run writes
  none) and `--no_conf`; the DA3 confidence (`colmap/mono_s1_conf`) is kept for sky masks.
  TensorRT for single-image DA3: `da3_trt_export.py` (ONNX of image -> depth, conf; ONNX has no `cartesian_prod`, the
  RoPE grid is built with meshgrid for the export), `trtexec --fp16` (TensorRT 10.14 in
  `/home/csaba/tools/TensorRT-10.14.1.48`, its Python wheel installed in the depth-anything-3 env), `mono_depth_mv.py
  --trt ENGINE` (needs `LD_LIBRARY_PATH=<TensorRT>/lib`; the engine has a fixed input size, 952x532 for 960x540 images
  at `--process_res 952`; `results/engines/da3l_fp16_952x532.engine`). DA3-Large on the RTX 5060 Ti: 19 images/s
  instead of 4.8 (PyTorch bf16; batching 4 images gains only 12 %); depth vs PyTorch: per-image median 0.28 %, worst
  image 2.4 % (a scale the alignment absorbs), confidence 0.8 %. The Chili 6-camera redo runs on it. With `--trt`
  the PyTorch weights are not loaded (only DA3's preprocessing) and every image runs on its own, so all cameras go in
  one call (`--window 0` without `--cameras`): ~4 s start-up once instead of ~9 s per camera.
  `fuse_depth.py` reads the next images' files in 2 background threads (both passes): identical output, ramp set
  177 -> 154 s. The GPU stays at ~60 % because each image's work is many small GPU steps with CPU syncs in between,
  not the disk; two processes would need pass 1 split by images and its fill candidates merged before pass 2.

- 2026-10-02: `tsdf_fuse.py --snap D`: own points within D of the TSDF surface are replaced by it (one layer on
  floors and walls); `--verify PLY` (`--verify_tol`, `--verify_footprint`): only depth pixels that agree with the
  multi-view checked cloud of `fuse_depth.py` (z-buffer of its points) are integrated. Without it the TSDF took every
  confident pixel of every image and built walls that the fuse check had rejected (ramp, lower levels); with it the
  TSDF can only merge what the check kept. Ramp, frames 1000-1200: fill > 0.3 m from the checked cloud 5.0 % -> 0.6 %,
  64 % of the confident pixels integrated, passes 1 and 2 ~1.8x slower (one extra projection per image).
- 2026-10-02: `consistent_depth.py` (new, after fuse): depth maps that agree with each other instead of each being
  fitted to the odometry points on its own. Per image a log-scale field (12x7 cells, bilinear); under the known poses
  a depth sample of image i lands in a neighbour j at a known pixel and must match j's scaled depth there. Scaling
  both does not help (fixed baseline), so the poses fix the metric scale; trusted odometry points only anchor it.
  No matching, no triangulation (plain floors count), robust (Tukey 6 %, pairs with median > 8 % out), all images in
  one problem. Neighbours: same camera +-15 frames, other cameras +-3, >= 10 % overlap. Then per pixel: kept if >= 2
  independent views (another camera, or >= 5 frames away: consecutive frames are 19 cm apart and share the DA3
  error) agree within 1.5 % at the exact landing point (a 3x3 search let grazing car sides agree with some sample)
  and no neighbour sees through it; depth edges (second difference of inverse depth > 2 %, grown 13 px) never count.
  The fuse confidence is not used: it measures odometry support, which plain floors and walls lack (they were holes).
  Ramp (frames 1000-1200, 6 cameras): disagreement between images median 1.65 -> 0.44 %, 90 % 6.97 -> 2.69 %;
  vs the odometry points 1.69 -> 0.81 %. Remaining: steps inside one image's DA3 depth (e.g. ramp floor) no scale
  field can remove; the TSDF of the confirmed pixels only (`tsdf_fuse.py --surface_only`: points3D = surface alone,
  no own points) gave one clean surface there. Pipeline: results/chili_6cam/run_gs_cons.sh (targets: the surface
  raycast; later variants: the confirmed points as initial points, depth_cons as targets).
  `tsdf_fuse.py --verify`/`snap_cloud.py` (checked cloud as a gate / snapping it onto the surface) were steps on the
  way: the TSDF still built surfaces near the checked one from roughly agreeing pixels.
- 2026-10-03: run_gs_cons.sh result (Chili full trajectory, 6 cameras, 24,846 training images, 732k steps): test
  PSNR 31.25 (1656 images). Speckle in views farther from the training views (e.g. B_MIDRANGECAM_C/003994): 40 % of
  the 5.4 M Gaussians had collapsed to points (largest axis < 1 mm, most < 1 um, opacity ~1). Sub-pixel in the
  training views, so nothing removed them; the rasterizer's 0.3 px dilation draws each as a dot elsewhere.
  `carve_gaussians.py --no_free --min_size 0.001` removes them (and opacity < 0.005): 5.40 -> 1.51 M Gaussians, test
  PSNR 31.25 -> 31.67, every camera and segment better, speckle gone. Free-space carving against depth_cons (confirmed
  pixels, z < 0.95 D - 0.1 m in >= 3 views) instead cost 2.4 dB: it also removed real-sized Gaussians. Under WSL a
  full GPU shows up as "CUDA driver error: device not ready" (dmesg: dxgkio_make_resident -12); run_gs_cons.sh now
  treats it as out of memory (grow_grad2d x 1.5 from the last checkpoint, checkpoints every 10 k steps).
  20 k more steps from the filtered model (same settings, no densification, gs_cons/finetune): 31.67 -> 31.76, only
  0.1 % collapsed again in that time. Small gain; the real fix belongs in training (`--antialiased`: sub-pixel
  Gaussians lose opacity, so points cannot act as opaque dots; prune Gaussians < 1 mm while densifying).
- 2026-10-04: `simple_trainer.py --prune_min_size_m M` (strategy.prune_min_size): prune Gaussians below M metres
  while densifying. Checkpoints store `rasterize_mode` ("antialiased" with `--antialiased`); eval_names,
  render_names, render_depth, render_flyby and carve_gaussians use it (older checkpoints: classic). DA3 confidence
  of chili_6cam now 8-bit PNG (`colmap/mono_s1_conf_png`, conf = 1 + png / 50). Running:
  results/chili_6cam/run_gs_cons2.sh (outputs gs_cons2/): depth_cons targets on confirmed pixels, `--antialiased`,
  `--prune_min_size_m 0.001`, collapsed Gaussians removed at the end; same TSDF surface start and schedule as run 1.
- 2026-10-05: photometric BA on our own Levenberg-Marquardt (`src/sdv/photometric_ba_solver.*`, `close_loops
  --pba-optimizer custom`, the default; `ceres` stays as the reference and for the rig stage, which refines the
  calibration): the residuals and analytic Jacobians of `TemporalCost` / `StaticCost` go straight into the normal
  equations per point, the inverse depth is eliminated, the reduced system over keyframe poses and brightness is
  solved by Eigen's sparse LDLT; Ceres' LM strategy and Huber weighting, so the costs match. Nothing per residual is
  stored. Zion garage (6 cameras, 260 keyframes, 3.2 M residuals): blocks of 100 93 s / 1.8 GB (Ceres 372 s /
  5.2 GB), joint 165 s / 2.2 GB (Ceres 172 s / 9.9 GB, the joint solve is dominated by factorising the nearly dense
  reduced system); poses within 0.04 mm (blocks) and 0.01 mm (joint) of Ceres', same rmse. close_loops keeps the
  keyframe images in grey (the point colours are decoded again one frame at a time), and the BA shares 8-bit grey
  input instead of copying it. `voxelnet 20260401T105922Z` (Zion, 16123 frames, 1222 m, 6 cameras, 2519 keyframes,
  28 M candidate residuals): Ceres ran out of memory twice (process ~29 GB; Windows' commit limit is RAM + a 4 GB
  pagefile, and a Windows process took 14.6 GB more); custom: 14.7 GB peak in the BA, coarse solve 36 s (Ceres 71 s,
  same cost), blocks 13-26 s (Ceres 49-78 s).
  Then: `pba::patternLinearization` (the TemporalCost / StaticCost model without a Ceres object, camera adjoints and
  inverse poses once per state; tested against both costs), the trial step's cost from linearising there (no extra
  evaluation pass; the points are damped with the lambda of the step before): garage blocks 93 -> 67 s, still within
  0.04 mm of Ceres' blocks. Sparse reduced system (`--pba-sparse-band N`: only keyframe pairs linked by an
  observation or odometry, or at most N apart; the dropped Schur fill-in lumped onto the diagonal as row sums of its
  magnitudes, positive definite) with `--pba-pcg K` CG iterations on the exact reduced system (applied point by point,
  not formed) preconditioned by it: exact gradient and cost, so the same optimum. Garage joint (260 keyframes; poses
  against Ceres' joint solve): exact 194 s, 0.015 mm; band 5 without CG 40 s but not converged (5.3 mm median);
  band 0 / 5 + PCG 10 40-44 s, 0.8 mm (CG at its limit every step); band 0 + PCG 30 48 s, 0.27 / 0.44 mm
  (median / max), 2.1 GB. Blocks of 100 are 2.1 / 5.1 mm from the joint optimum (Ceres or ours) and take 67 s: the
  joint sparse + PCG solve is faster and closer. Inside blocks sparsifying does not help (factorising is small
  there). `close_loops` frees the keyframe images before densify (it streams its frames); `gs_pipeline.sh` adds
  B_MIDRANGECAM_C to `GS_CAMS` only when `VO_CAMS` lacks it (listed twice, close_loops stopped after the BA with
  "--densify-cameras: unknown camera", the first voxelnet run).
- 2026-10-05: densify speed. Frames decoded camera-parallel by a background thread up to three frames ahead (the
  serial decode was 42 % of the densify time), one tracing pass per frame over all open hosts, new hosts selected
  in parallel per camera, a per-thread trace buffer: garage densify 358 s (6 cameras, byte-identical cloud; before
  1082 s, measured next to another densify). Log line `densify time:` (waiting for frames, pyramids, traces, new
  hosts, closing). Tests on the first 2000 garage frames (`--densify-max-frames`, 34 % standstill like the whole
  run), `DENSIFY_ARGS` as the pipeline: base 253 s (traces 148 s), 3.063 M points, 414 k trusted; floaters
  (cloud_consistency.py, per observation bin) 0.21-0.39 %. `--densify-drop-frames 5` (no good trace 5 frames
  after the host: not traced further): traces -23 %, wall -13 %, points -1.6 %, trusted -1.5 %, floaters the same
  (0.21-0.29 %); 10 frames: traces -14 %, points -0.6 %. `--densify-min-motion 0.05 --densify-min-rotation 0.5`
  (frames between keyframes traced only after 5 cm / 0.5 deg): 37 % of the frames skipped but traces only -6 %
  (standstill traces are cheap: the search line is too short), trusted -3.5 % (fewer observations): rejected.
  Windows ran out of commit again (15:54): another session's training job (Windows python, 13.5 GB) next to
  WSL's 29 GB with the 4 GB pagefile; the voxelnet densify died at frame 9400 of 16123.
  Trace kernel (gperftools CPU profile of frames 1000-1299, `--densify-first-frame` / `--densify-max-frames`):
  bearing rotated once per trace instead of per sample, brightness scale (an exp) once per trace, the warped pattern
  offsets once per trace, Huber functions inline, an intensity-only plane per pyramid level for the search, the
  pyramids of a frame built in parallel: traces 23.0 -> 18.1 s, densify 32.0 -> 26.9 s, cloud byte-identical. Left
  in a trace: pattern energy 32 % (bilinear lookups 12 %, Huber 8 %), EUCM projection 19 %, local warp 9 %. The
  decode thread takes ~25 % of the CPU (two INTER_AREA resizes per frame, 2896 -> 1936 -> 968 px), shared with
  the tracing.
- 2026-10-05: GPU video decoding. `run_vo`, `close_loops`, `export_colmap` and `aimrec_extract` take `--decode
  cpu|gpu|auto` (default auto: GPU when built with CUDA and a device is there). NVDEC through FFmpeg's `h264_cuvid`
  (low delay), then one CUDA pass NV12 -> BGR (BT.601/709, full/limited range) at full resolution and the
  INTER_AREA chain of the run (rig `image_width` scale, then `--scale`) on the GPU; only the 968 px image is
  copied to the host (`src/sdv/aimrec/gpu_image.cu`). Decoded luma is bit-identical to FFmpeg's software decoder
  and the resize chain bit-identical to cv::resize (tests `GpuImage.*`); the colour conversion is exact to
  rounding, swscale's was -0.6 grey levels darker on average (up to 2): images differ from the CPU path by that.
  The records are intra-only (every frame IDR), so the decoder skips forward without a flush. FFmpeg's h264
  hwaccel (NVDEC too) waits for every frame: 10 frame sets/s for 6 cameras. Decoding speed, 6 cameras, scaled to
  968 px (`aimrec_extract --dry-run`): CPU 22 sets/s (the resizes dominate: unscaled 64), GPU 51 sequential, 31
  every 7th frame. Garage frames 1000-1299: densify subset 51 -> 40 s wall (densify 24.5 -> 21.4 s, keyframe
  images 16 -> 9 s), 675733 vs 676356 points; odometry 300 frames 38 -> 24 s, poses within 2.8 mm. Six decoders
  take ~0.9 GB of GPU memory; `--decode cpu` next to a GPU job that needs all of it. The tests
  `MonoInitializerTest.ForwardMotionWithYaw/Pinhole` and `WindowOptimizerTest.MarginalizationKeepsOptimum...`
  fail already at 43e21dc (not decoding related, open).
- 2026-10-05: densify memory. voxelnet 20260401T105922Z (16123 frames, 6 cameras): the close_loops guard (WSL used
  > 28 GB) fired at the end of densify: the voxel check (a map of cells, ~150 B per point) on 75 M accepted points,
  and the merge (input copy + hash grid + output, ~420 B per point) was next. Now the voxel check and the merge sort
  instead (32 / 16 B per point), the merge compacts in place, densify appends to the cloud without a copy, and
  `malloc_trim` returns the BA's freed heap before densify (close_loops 12.5 -> 3.7 GB at densify start); clouds
  byte-identical (garage subset). Rerun: peak 18.3 GB WSL; BA 790 s, densify 1463 s (traces 1046, waiting for frames
  6.6), 74.8 M accepted -> 63.9 M after the voxel check -> merge 45.5 M (19.4 M pairs, 16 s) -> free space -2.76 M
  (337 s) -> 42.7 M points written. Trajectory: `diag/trajectory_loop.ply` (levelled, blue -> red over time).
- 2026-10-05: free-space filter 2.2x faster, result identical: points stored cell by cell (contiguous positions),
  cells beyond the radius or outside a view's field of view (the largest angle of the image border from the optical
  axis, +1 deg) skipped. voxelnet cloud (42.7 M points, 15073 views): 295 -> 137 s, support / through / floater flags
  of every point equal; smaller cells (radius/8, /16) no faster. Coarse-to-fine epipolar search
  (`--densify-coarse-step`, TraceSettings::coarseStep, default 1 = off; pipeline 2): garage frames 1000-1299 densify
  20.0 -> 16.9 s (traces 13.6 -> 11.5, new hosts 3.7 -> 2.7), 893 k vs 893 k accepted, 628.5 k vs 628.3 k after
  merge; cloud_consistency floaters 0.30 vs 0.19 % against the step-1 views but 0.17 vs 0.26 % against the step-2
  views (the reference favours its own cloud: equal), core depth rms 0.85 vs 0.84 %. Step 3: -20 %, 0.8 % fewer
  points. Odometry keeps step 1 (not tested). Joint sparse + PCG photometric BA on voxelnet (`--pba-sparse-band 0
  --pba-pcg 30` instead of blocks of 200): 705 s (CPU shared with benchmarks) vs 790 s, rmse 7.25 vs 7.28 (different
  residual sets after the outlier rounds), peak 23.4 vs ~17 GB WSL; trajectories differ by 5 cm median, 0.6 m max
  after a similarity alignment. No reference to tell which is closer: blocks stay the default.
