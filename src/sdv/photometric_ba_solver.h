#pragma once

#include <array>
#include <memory>
#include <vector>

#include <sophus/se3.hpp>

#include <sdv/photometric_ba_cost.h>
#include <sdv/rig.h>

// Levenberg-Marquardt for the photometric bundle adjustment with fixed rig calibration, in place of Ceres: residuals
// and Jacobians of TemporalCost / StaticCost are accumulated straight into the normal equations per point, its
// inverse depth is eliminated (Schur complement) and the reduced system over keyframe poses and affine brightness
// is solved by a sparse Cholesky. Nothing per residual is stored, so memory grows with points and keyframes only.
namespace sdv::pba {

struct Observation {
  int point, target, cam;
};

struct OdometryEdge {
  int a;  // keyframes a and a + 1
  std::shared_ptr<const RelativePoseCost> cost;
};

struct SolverProblem {
  const Rig* rig = nullptr;
  const std::vector<PointData>* points = nullptr;
  const std::vector<const Interpolator*>* images = nullptr;  // [keyframe * cameras + camera]
  // State, updated in place: T_w_b in Sophus layout (right increments as SE3TangentManifold), affine (a, b) per
  // keyframe camera, inverse depth per point.
  std::vector<std::array<double, 7>>* poses = nullptr;
  std::vector<std::array<double, 2>>* affine = nullptr;
  std::vector<double>* rho = nullptr;
  std::vector<Observation> observations;
  std::vector<char> poseFree, affineFree, rhoFree;
  std::vector<OdometryEdge> odometry;
  double huber = 0;  // on the pattern residual norm, as ceres::HuberLoss
  double minRho = 1e-4;
};

struct SolverOptions {
  int iterations = 30;
  double functionTolerance = 1e-6;  // relative cost change of an accepted step that ends the solve
  int threads = 0;  // 0: all
};

struct SolverSummary {
  double initialCost = 0, finalCost = 0;  // 1/2 sum of the robustified squared norms, as Ceres reports it
  int iterations = 0, accepted = 0;
  double total = 0, structure = 0, linearize = 0, factorize = 0, evaluate = 0;  // s
  int reducedSize = 0;
};

SolverSummary solve(const SolverProblem& problem, const SolverOptions& options = {});

}  // namespace sdv::pba
