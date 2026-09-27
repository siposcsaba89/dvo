#!/usr/bin/env bash
# Runs three single-process profiles (KITTI mono, KITTI stereo, fisheye video) and prints accuracy and stage times.
# usage: tools/profile.sh <tag>
set -u
tag=$1
root=$(cd "$(dirname "$0")/.." && pwd)
out="$root/results/prof"; mkdir -p "$out"
exe="$root/build/Release/run_vo.exe"
seq=E:/records/kitti/sequences/00; gt=E:/records/kitti/poses/00.txt
fisheye_video=${FISHEYE_VIDEO:-e:/records/aimrec/Camera_NvMedia_05_TE06549_data.h264}
fisheye_cam=${FISHEYE_CAMERA:-}
"$exe" -s $seq --gt $gt --start 1000 -n 400 --cam-alpha=-0.03 -o "$out/${tag}_mono.txt" > "$out/${tag}_mono.log" 2>&1
"$exe" -s $seq --gt $gt --stereo --start 1000 -n 400 --cam-alpha=-0.03 -o "$out/${tag}_stereo.txt" > "$out/${tag}_stereo.log" 2>&1
modes="mono stereo"
if [ -n "$fisheye_cam" ]; then
  "$exe" --video "$fisheye_video" --camera "$fisheye_cam" --scale 0.5 -n 500 -o "$out/${tag}_fish.txt" > "$out/${tag}_fish.log" 2>&1
  modes="$modes fish"
fi
for m in $modes; do
  echo "== $m"
  grep -E "ATE Sim3|RPE 1 frame|ms/frame \(max|wall time|^\[.*\]   [a-z]" "$out/${tag}_$m.log" | sed 's/^\[[^]]*\] \[info\] //'
done
