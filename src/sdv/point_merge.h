#pragma once

#include <vector>

#include <sdv/map_point.h>

namespace sdv {

struct PointMergeSettings {
  double maxDistance = 0.03;  // world units
  int minFrameGap = 0;  // host frames at least this far apart (separate passes); 0 = any two host images
  int rounds = 2;  // a third pass can join an already merged pair in the next round
};

// Duplicates of one surface point from different passes (e.g. laps of a garage) become one point: two points are
// merged when each is the other's nearest neighbour within maxDistance among the points of hosts minFrameGap apart.
// The merged position is weighted by 1/distance^2 (depth precision falls with the distance from the host camera);
// the host (frame, camera, pixel, distance) of the closer point is kept and the observations are summed.
std::vector<MapPoint> mergeDuplicatePoints(const std::vector<MapPoint>& points, const PointMergeSettings& settings,
                                           size_t* merged = nullptr);

}  // namespace sdv
