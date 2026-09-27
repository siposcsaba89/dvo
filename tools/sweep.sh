#!/usr/bin/env bash
# Runs bench.sh for several run_vo argument sets and prints the mean row of each.
# usage: tools/sweep.sh stereo|mono "tag1:args..." "tag2:args..." ...
mode=$1; shift
dir=$(dirname "$0")
for cfg in "$@"; do
  tag=${cfg%%:*}; args=${cfg#*:}; [ "$args" = "$cfg" ] && args=""
  printf "%-24s " "$tag"; bash "$dir/bench.sh" "$tag" "$mode" 500 $args | tail -1
done
