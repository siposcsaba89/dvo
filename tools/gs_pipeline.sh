#!/bin/bash
# Gaussian-splatting reconstruction of an aiMotive recording: sdv odometry + loops + densify, COLMAP export,
# multi-view DA3 depth, alignment, per-image fusion, TSDF fusion, gsplat with dense depth, evaluation and renders.
# See docs/GS_PIPELINE.md. Resumable: every stage leaves $R/.done_<stage>; delete it to rerun that stage.
#
#   tools/gs_pipeline.sh RESULT_DIR          (RESULT_DIR/rig/rig.yaml must exist, see the README)
#
# Settings (environment, defaults = the current best setup):
#   VO_CAMS    odometry cameras                  (F_CTCAM_L F_CTCAM_R M_NEIGHBORLANECAM_L M_NEIGHBORLANECAM_R)
#   GS_CAMS    densify, export and GS cameras    ($VO_CAMS B_MIDRANGECAM_C)
#   DA3_GROUPS multi-view DA3 passes, "window cameras:saved cameras" separated by ";" (a window holds at most
#              4 cameras x 7 frames on a 16 GB GPU)
#   FORWARD, HEIGHT  fly-by camera offset from the body origin, m (CT camera position of the vehicle)
#   INIT_VOXEL initial Gaussians thinned to this voxel, m (default 0.08 above 5 M points, else all)
#   STEPS      training steps (30000); CKPT_EVERY full checkpoint + previews every n steps (10000)
#   BALANCE_RADIUS  sample images by 1 / (same-camera images within this radius, m): every place trained about
#              equally however often it was driven past (default 0 = uniform)
#   DENSE_DEPTH, DENSE_CONF  depth targets and confidence for training (tsdf_fused, tsdf_conf; tsdf_detail,
#              tsdf_detail_conf = mono shape with TSDF scale, detail_depth.py)
#   TRAIN_CONFIG  simple_trainer.py config: default (densify + prune) or mcmc (fixed budget --strategy.cap_max)
#   GS_NAME    training run folder (gsplat_tsdf); RESUME_FROM checkpoint a new run starts from; TRAIN_ARGS extra
#              simple_trainer.py options
#   KEEP_INTERMEDIATE=1  keep the raw DA3 and aligned depth after the TSDF (default: deleted, ~10 GB each)
#   MAX_CACHE_GB  page cache of the run's files above which it is dropped (default 8, WSL)
#   STOP_AFTER stage after which to stop, e.g. vo or loops to look at the trajectory plots first (rerun to continue)
set -eo pipefail
R=$(realpath "$1")
DVO=/home/csaba/projects/dvo
G=/home/csaba/projects/gsplat/examples
PY=/home/csaba/mamba/envs/occnet/bin/python
DA3=/home/csaba/mamba/envs/depth-anything-3/bin/python
B=$DVO/build/linux/Release
V=$DVO/results/voc_k10l5.fbow
VO_CAMS=${VO_CAMS:-F_CTCAM_L F_CTCAM_R M_NEIGHBORLANECAM_L M_NEIGHBORLANECAM_R}
GS_CAMS=${GS_CAMS:-$VO_CAMS B_MIDRANGECAM_C}
DA3_GROUPS=${DA3_GROUPS:-"$VO_CAMS:$VO_CAMS;B_MIDRANGECAM_C M_NEIGHBORLANECAM_L M_NEIGHBORLANECAM_R:B_MIDRANGECAM_C"}
FORWARD=${FORWARD:-1.55}
HEIGHT=${HEIGHT:-1.50}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1
C=$R/colmap
log() { echo "[$(date '+%F %T')] $*"; }
stage() { [ -f "$R/.done_$1" ] && { log "skip $1 (done)"; return 1; }; log "start $1"; return 0; }
done_() { touch "$R/.done_$1"; log "done $1"; [ "$1" = "${STOP_AFTER:-}" ] && { log "stopped after $1 (STOP_AFTER)"; exit 0; }; return 0; }
plot() { $PY $DVO/tools/plot_trajectory.py "$@" | sed 's/^/    /'; }
test -f $R/rig/rig.yaml || { echo "no $R/rig/rig.yaml"; exit 1; }
# WSL: the page cache of the run's files is not given back to Windows while busy and took the VM down; keep it small.
$PY $DVO/tools/drop_cache.py $R --pid $$ --max_gb ${MAX_CACHE_GB:-8} >> $R/drop_cache.log 2>&1 &

if stage vo; then
    $B/run_vo --rig $R/rig/rig.yaml --rig-cameras $VO_CAMS --scale 0.5 -o $R/poses.txt \
        --keyframes-out $R/keyframes.kfr --png $R/odometry.png --trajectory-ply $R/odometry_trajectory.ply \
        > $R/run_vo.log 2>&1
    log "odometry: $R/odometry.png (top view with points), $R/odometry_trajectory.png (heights)"
    plot $R/odometry_trajectory.png $R/poses.txt
    done_ vo
fi
if stage loops; then
    $B/detect_loops --keyframes $R/keyframes.kfr --vocabulary $V --out $R/loops.txt --verbose > $R/detect_loops.log 2>&1
    mkdir -p $R/diag
    # Long runs: photometric BA in blocks of 200 keyframes after a coarse joint solve (bounded memory, ~20 GB for
    # 1400 keyframes instead of ~45 GB); up to 400 keyframes one joint problem (the reference results).
    kf=$(grep -o "[0-9]* keyframes, [0-9]* features" $R/run_vo.log | tail -1 | cut -d' ' -f1)
    blocks=""
    [ "${kf:-0}" -gt 400 ] && blocks="--pba-block-keyframes 200"
    $B/close_loops --keyframes $R/keyframes.kfr --poses $R/poses.txt --vocabulary $V --rig $R/rig/rig.yaml \
        --rig-cameras $VO_CAMS --scale 0.5 --photometric --pba-points 120 $blocks --densify --densify-cameras $GS_CAMS --merge \
        --out $R/diag/poses_loop.txt --ply $R/diag/cloud.ply --points-out $R/diag/points.ply > $R/diag/close_loops.log 2>&1
    # Without IMU "up" is the first frame's body up: level the map by the vehicle's mean up axis (ramps left out).
    $PY $DVO/tools/level_run.py $R/diag/poses_loop.txt $R/diag/cloud.ply $R/diag/points.ply | sed 's/^/    /'
    log "loops: $R/diag/trajectory.png (odometry vs loop-closed, levelled), $R/diag/cloud.ply"
    grep "odometry correction implied" $R/detect_loops.log | sed 's/^.*\] /    /'
    plot $R/diag/trajectory.png $R/poses.txt $R/diag/poses_loop.txt
    done_ loops
fi
if stage export; then
    rm -rf $C
    $B/export_colmap --rig $R/rig/rig.yaml --rig-cameras $VO_CAMS --cameras $GS_CAMS --poses $R/diag/poses_loop.txt \
        --cloud $R/diag/cloud.ply --scale 0.5 --out $C > $R/export_colmap.log 2>&1
    done_ export
fi
if stage da3; then
    IFS=';' read -ra groups <<< "$DA3_GROUPS"
    cd $G
    for g in "${groups[@]}"; do
        # Same DA3 settings as the garage runs: 7 frames (+-3 steps of 2), 756 px, the centre and its neighbours kept.
        $DA3 -u mono_depth_mv.py --data_dir $C --window 3 --step 2 --keep 1 --process_res 756 --out mono_mv \
            --cameras ${g%%:*} --save ${g#*:} 2>&1 | grep -v "INFO\|WARN" >> $C/mono_mv.log
    done
    have=$(find $C/mono_mv -name "*.npy" | wc -l)
    [ "$have" -ge "$(grep -c jpg $C/sparse/0/images.txt)" ] || { echo "DA3: only $have depth maps"; exit 1; }
    done_ da3
fi
cd $G
if stage align; then
    $PY -u align_depth.py --data_dir $C --reference $R/diag/points.ply --mono mono_mv --depth depth_mv --vis_every 20 \
        > $C/align_mv.log 2>&1
    done_ align
fi
if stage fuse; then
    $PY -u fuse_depth.py --data_dir $C --reference $R/diag/points.ply --depth depth_mv --out $R/colmap_fused \
        --fill_voxel 0.06 > $C/fuse_mv.log 2>&1
    done_ fuse
fi
if stage tsdf; then
    $PY -u tsdf_fuse.py --data_dir $C --reference $R/diag/points.ply --depth depth_mv_fused --conf depth_mv_conf \
        --out $R/colmap_tsdf --name tsdf --min_conf 0.3 --min_weight 0.02 --max_up 0.0 --iterations 2 \
        --refine quadratic > $C/tsdf.log 2>&1
    # ~10 GB per depth folder on long records: the raw DA3 and aligned depth are not needed after the TSDF
    # (KEEP_INTERMEDIATE=1 keeps them for rerunning align / fuse).
    [ "${KEEP_INTERMEDIATE:-0}" = 1 ] || rm -rf $C/mono_mv $C/mono_mv_conf $C/mono_mv_vis $C/depth_mv $C/depth_mv_vis
    done_ tsdf
fi
# A training run is $R/$GS_NAME; runs other than the default one have their own stage markers (train_<name>).
GS_NAME=${GS_NAME:-gsplat_tsdf}
T=$R/$GS_NAME
sfx=$([ "$GS_NAME" = gsplat_tsdf ] || echo "_$GS_NAME")
STEPS=${STEPS:-30000}
if stage train$sfx; then
    mkdir -p $T
    # Initial Gaussians: a 16 GB GPU trained 3.6 M (garage, all points); Chili's 11.3 M ran out of GPU memory at the
    # first densification, so larger clouds are thinned to 8 cm voxels (3.4 M on Chili).
    n=$(grep -vc "^#" $R/colmap_tsdf/sparse/0/points3D.txt)
    voxel=${INIT_VOXEL:-$([ $n -gt 5000000 ] && echo 0.08 || echo 0)}
    # Full checkpoints every CKPT_EVERY steps (with previews of views spread over the scene): after a crash the
    # rerun continues from the newest one; RESUME_FROM starts a new run from another run's checkpoint.
    # The newest checkpoint that is a complete archive (a full disk once left a truncated one).
    last=$(ls $T/ckpts/ckpt_*_rank0.pt 2>/dev/null | sort -rV | $PY -c "
import sys, zipfile
for f in sys.stdin.read().split():
    try:
        zipfile.ZipFile(f).close(); print(f); break
    except Exception:
        print('skipping broken', f, file=sys.stderr)" || true)
    resume=${last:-${RESUME_FROM:-}}
    log "train: $n points, init_voxel $voxel, $STEPS steps${resume:+, resuming from $resume}"
    $PY simple_trainer.py ${TRAIN_CONFIG:-default} --data_dir $R/colmap_tsdf --data_factor 1 --result_dir $T --disable_viewer \
        --disable_video --dense_depth ${DENSE_DEPTH:-tsdf_fused} --dense_depth_conf ${DENSE_CONF:-tsdf_conf} --init_voxel $voxel \
        --max_steps $STEPS --save_steps $STEPS --eval_steps $STEPS --ckpt_every ${CKPT_EVERY:-10000} \
        --balance_radius ${BALANCE_RADIUS:-0} ${resume:+--resume $resume} ${TRAIN_ARGS:-} \
        >> $T/stdout.log 2>> $T/train.log
    grep "previews step" $T/stdout.log | tail -1 | sed 's/^/    /'
    # The numbered checkpoints are only for resuming: keep the final one (previews stay).
    ls $T/ckpts/ckpt_*_rank0.pt | grep -v "ckpt_$((STEPS - 1))_rank0.pt" | xargs -r rm -f
    done_ train$sfx
fi
if stage eval$sfx; then
    step=$((STEPS - 1))
    $PY eval_cameras.py --data_dir $R/colmap_tsdf --result_dir $T --step $step > $T/eval_cameras.txt 2>&1
    $PY render_flyby.py --data_dir $R/colmap_tsdf --ckpt $T/ckpts/ckpt_${step}_rank0.pt --poses $R/diag/poses_loop.txt \
        --out $T/flyby_look_front.mp4 --forward $FORWARD --height $HEIGHT --yaw_amp 35 --sway_amp 0.4 --bob_amp 0.25 \
        > $T/flyby.log 2>&1
    $PY render_depth.py --data_dir $R/colmap_tsdf --ckpt $T/ckpts/ckpt_${step}_rank0.pt --out $T/depth --split val \
        --indices 20 45 80 100 140 160 200 230 260 280 > $T/render_depth.log 2>&1
    done_ eval$sfx
fi
log "all done"
