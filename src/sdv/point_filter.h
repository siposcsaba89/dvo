#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include <sdv/map_point.h>
#include <sdv/rig.h>

namespace sdv {

// Per point: whether at least minNeighbours other points lie within radius (voxel hashing of the radius).
std::vector<char> hasNeighbours(const std::vector<Eigen::Vector3d>& points, double radius, int minNeighbours);

struct FreeSpaceSettings {
  double radius = 15.0;  // a view tests the points within this distance of its camera, m
  double tolerance = 0.05;  // relative distance for a view to confirm a point
  double margin = 0.10;  // relative distance (and minGap absolute) for a view to see through a point
  double minGap = 0.10;
  int window = 2;  // pixel radius of a view's depth lookup (the nearest of its own points)
  int minThrough = 2;
};

struct FreeSpaceResult {
  std::vector<char> floater;
  std::vector<int> support, through;
  int views = 0;
};

// Free-space test of a multi-view cloud: every host image (frame, camera) of the semi-dense points is a view whose
// depth is the distance of the points it hosts. A point is a floater when at least minThrough other views measured a
// surface clearly behind it (they saw through it) and those outnumber the views that measured it. T_w_b: body pose
// per input frame (the points' frame indices); rig: the cameras of the semi-dense points' camera indices.
FreeSpaceResult freeSpaceFloaters(const Rig& rig, const std::vector<Sophus::SE3d>& T_w_b,
                                  const std::vector<MapPoint>& points, const FreeSpaceSettings& settings = {});

// Per point: whether it is the best of the selected points in its voxel (most observations, then the smallest
// relative depth sigma). Unlike voxelThin, a kept point is an unchanged measurement with all its attributes.
std::vector<char> voxelBest(const std::vector<MapPoint>& points, const std::vector<char>& select, double voxel);

// One point per occupied voxel: the mean position and colour of its points.
void voxelThin(std::vector<Eigen::Vector3d>& points, std::vector<std::array<std::uint8_t, 3>>& colors, double voxel);

}  // namespace sdv
