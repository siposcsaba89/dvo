#pragma once

#include <vector>

#include <Eigen/Core>

namespace sdv {

// Per point: whether at least minNeighbours other points lie within radius (voxel hashing of the radius).
std::vector<char> hasNeighbours(const std::vector<Eigen::Vector3d>& points, double radius, int minNeighbours);

}  // namespace sdv
