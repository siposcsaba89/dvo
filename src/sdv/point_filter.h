#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace sdv {

// Per point: whether at least minNeighbours other points lie within radius (voxel hashing of the radius).
std::vector<char> hasNeighbours(const std::vector<Eigen::Vector3d>& points, double radius, int minNeighbours);

// One point per occupied voxel: the mean position and colour of its points.
void voxelThin(std::vector<Eigen::Vector3d>& points, std::vector<std::array<std::uint8_t, 3>>& colors, double voxel);

}  // namespace sdv
