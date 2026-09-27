#!/usr/bin/env bash
# Runs run_vo on segments of KITTI 00 in parallel and prints per-segment and mean metrics.
# usage: [STARTS="0 2000"] tools/bench.sh <tag> [stereo|mono] [frames per segment] [extra run_vo args...]
set -u
tag=$1; mode=${2:-stereo}; len=${3:-500}; shift $(( $# < 3 ? $# : 3 ))
root=$(cd "$(dirname "$0")/.." && pwd)
exe="$root/build/Release/run_vo_bench_$tag-$mode.exe"
cp "$root/build/Release/run_vo.exe" "$exe"
out="$root/results/bench/$tag-$mode"
mkdir -p "$out"
seqdir=E:/records/kitti/sequences/00
gt=E:/records/kitti/poses/00.txt
total=$(wc -l < "${gt/E:/\/e}")
flag=""; [ "$mode" = stereo ] && flag="--stereo"

starts=${STARTS:-$(seq 0 "$len" $(( total - len )))}
for s in $starts; do
  "$exe" -s "$seqdir" --gt "$gt" $flag --start "$s" -n "$len" --max-distance-factor 0 \
    -o "$out/$s.txt" "$@" > "$out/$s.log" 2>&1 &
done
wait
rm -f "$exe"

printf "%6s %8s %8s %8s %8s %6s %7s %8s %6s\n" start ate_sim3 ate_se3 drift_t drift_r scale rpe_cm rpe_deg ms
for s in $starts; do
  awk -v s="$s" '
    /ATE Sim3/ { for (i=1;i<=NF;i++) { if ($i=="rmse" && !sim) sim=$(i+1); if ($i=="scale") sc=$(i+1); if ($i=="t") dt=$(i+1); if ($i=="r") dr=$(i+1) } }
    /stereo, metric/ { for (i=1;i<=NF;i++) if ($i=="rmse") se3=$(i+1) }
    /RPE 1 frame/ { for (i=1;i<=NF;i++) { if ($i=="rmse") rt=$(i+1); if ($i=="cm,") rr=$(i+1) } }
    /ms\/frame \(max/ { for (i=1;i<=NF;i++) if ($i=="ms/frame") ms=$(i-1) }
    END { if (se3=="") se3="-"; printf "%6d %8s %8s %8s %8s %6s %7s %8s %6s\n", s, sim, se3, dt, dr, sc, rt, rr, ms }' "$out/$s.log"
done | tee "$out/summary.txt"
awk '{ n++; a+=$2; if ($3!="-") b+=$3; t+=$4; r+=$5; pt+=$7; pr+=$8; m+=$9 }
     END { printf "%6s %8.3f %8.3f %8.2f %8.3f %6s %7.2f %8.4f %6.0f\n", "mean", a/n, b/n, t/n, r/n, "", pt/n, pr/n, m/n }' "$out/summary.txt"
