#include <sdv/global_ba.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <thread>

#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>

#include <sdv/loop_detector.h>

namespace sdv {

namespace {

struct ProjectionResidual {
  Eigen::Vector3d bearing;
  Sophus::SE3d T_c_b;
  double fx;

  template <typename T>
  bool operator()(const T* pose, const T* point, T* residual) const {
    const Eigen::Map<const Sophus::SE3<T>> T_w_b(pose);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> X(point);
    const Eigen::Matrix<T, 3, 1> p = T_c_b.cast<T>() * (T_w_b.inverse() * X);
    const Eigen::Matrix<T, 3, 1> e = (p / p.norm() - bearing.cast<T>()) * T(fx);
    for (int k = 0; k < 3; ++k) residual[k] = e[k];
    return true;
  }
};

struct DepthResidual {
  Sophus::SE3d T_c_b;
  double rho, sigma;  // observed inverse distance and its absolute sigma

  template <typename T>
  bool operator()(const T* pose, const T* point, T* residual) const {
    const Eigen::Map<const Sophus::SE3<T>> T_w_b(pose);
    const Eigen::Map<const Eigen::Matrix<T, 3, 1>> X(point);
    const Eigen::Matrix<T, 3, 1> p = T_c_b.cast<T>() * (T_w_b.inverse() * X);
    residual[0] = (T(1) / p.norm() - T(rho)) / T(sigma);
    return true;
  }
};

double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return std::atan2(a.cross(b).norm(), a.dot(b)); }

// Closest point between two rays (origins o, unit directions d); nullopt when nearly parallel or behind.
std::optional<Eigen::Vector3d> midpoint(const Eigen::Vector3d& o1, const Eigen::Vector3d& d1, const Eigen::Vector3d& o2,
                                        const Eigen::Vector3d& d2, double minParallaxRad) {
  if (angle(d1, d2) < minParallaxRad) return std::nullopt;
  const Eigen::Vector3d w = o1 - o2;
  const double b = d1.dot(d2), d = d1.dot(w), e = d2.dot(w), den = 1 - b * b;
  if (den < 1e-9) return std::nullopt;
  const double s = (b * e - d) / den, t = (e - b * d) / den;
  if (s <= 0 || t <= 0) return std::nullopt;
  return 0.5 * (o1 + s * d1 + o2 + t * d2);
}

struct Node {
  int kf, cam, feat;
};

int find(std::vector<int>& parent, int x) {
  while (parent[x] != x) x = parent[x] = parent[parent[x]];
  return x;
}

}  // namespace

GlobalBAResult bundleAdjustKeyframes(const Rig& rig, const std::vector<KeyframeRecord>& records,
                                     const std::vector<Sophus::SE3d>& initial_T_w_b,
                                     const std::vector<std::pair<int, int>>& loopPairs,
                                     const GlobalBASettings& settings) {
  const int nk = static_cast<int>(records.size()), nc = rig.size();
  GlobalBAResult result;
  result.T_w_b = initial_T_w_b;
  if (nk < 2) return result;

  // Feature nodes, one per (keyframe, camera, feature).
  std::vector<std::vector<int>> offset(nk, std::vector<int>(nc));
  std::vector<Node> nodes;
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c) {
      offset[k][c] = static_cast<int>(nodes.size());
      for (size_t i = 0; i < records[k].cameras[c].size(); ++i) nodes.push_back({k, c, static_cast<int>(i)});
    }
  auto cameraPose = [&](int k, int c) { return initial_T_w_b[k] * rig.T_c_b[c].inverse(); };  // T_w_c
  auto worldPoint = [&](int k, int c, int i) -> std::optional<Eigen::Vector3d> {
    const CameraFeatures& f = records[k].cameras[c];
    if (f.rho[i] <= 0) return std::nullopt;
    return cameraPose(k, c) * (f.bearings[i].cast<double>() / f.rho[i]);
  };
  auto pixelError = [&](int k, int c, int i, const Eigen::Vector3d& X) {
    return angle(cameraPose(k, c).inverse() * X, records[k].cameras[c].bearings[i].cast<double>()) * rig.cameras[c].fx;
  };

  std::set<std::pair<int, int>> pairSet;
  for (int k = 0; k < nk; ++k)
    for (int d = 1; d <= settings.sequentialNeighbours && k + d < nk; ++d) pairSet.insert({k, k + d});
  for (const auto& [q, m] : loopPairs) pairSet.insert({std::min(q, m), std::max(q, m)});
  const std::vector<std::pair<int, int>> pairs(pairSet.begin(), pairSet.end());

  // Matches that agree with the initial poses, per pair (in parallel), then joined into tracks.
  constexpr double kMinParallax = 1.0 * M_PI / 180.0;
  std::vector<std::vector<std::pair<int, int>>> links(pairs.size());
  std::vector<size_t> index(pairs.size());
  std::iota(index.begin(), index.end(), 0);
  std::for_each(std::execution::par, index.begin(), index.end(), [&](size_t p) {
    const auto [a, b] = pairs[p];
    for (const auto& m : matchKeyframeFeatures(records[a], records[b], settings.ratio, settings.maxHamming)) {
      std::optional<Eigen::Vector3d> X = worldPoint(a, m.camA, m.featA);
      if (!X) X = worldPoint(b, m.camB, m.featB);
      if (!X) {
        const Sophus::SE3d Ta = cameraPose(a, m.camA), Tb = cameraPose(b, m.camB);
        X = midpoint(Ta.translation(), Ta.so3() * records[a].cameras[m.camA].bearings[m.featA].cast<double>(),
                     Tb.translation(), Tb.so3() * records[b].cameras[m.camB].bearings[m.featB].cast<double>(),
                     kMinParallax);
      }
      if (!X || pixelError(a, m.camA, m.featA, *X) > settings.gatePixels ||
          pixelError(b, m.camB, m.featB, *X) > settings.gatePixels)
        continue;
      links[p].emplace_back(offset[a][m.camA] + m.featA, offset[b][m.camB] + m.featB);
    }
  });
  std::vector<int> parent(nodes.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::vector<char> linked(nodes.size(), 0);
  for (const auto& l : links)
    for (const auto& [x, y] : l) {
      parent[find(parent, x)] = find(parent, y);
      linked[x] = linked[y] = 1;
    }
  std::map<int, std::vector<int>> tracks;
  for (int n = 0; n < static_cast<int>(nodes.size()); ++n)
    if (linked[n]) tracks[find(parent, n)].push_back(n);

  // Points: from the map depth of the observations (mean), else triangulated; observations far from it dropped.
  struct Track {
    std::vector<int> obs;
    Eigen::Vector3d X;
    bool loop;
  };
  std::vector<Track> kept;
  for (auto& [root, obs] : tracks) {
    if (obs.size() < 2) continue;
    // One observation per keyframe camera; a track that has two is ambiguous.
    std::set<std::pair<int, int>> seen;
    bool ambiguous = false;
    for (int n : obs) ambiguous |= !seen.insert({nodes[n].kf, nodes[n].cam}).second;
    if (ambiguous) continue;
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    int count = 0;
    for (int n : obs)
      if (auto X = worldPoint(nodes[n].kf, nodes[n].cam, nodes[n].feat)) sum += *X, ++count;
    std::optional<Eigen::Vector3d> X;
    if (count > 0) X = sum / count;
    else {
      const Node &p = nodes[obs.front()], &q = nodes[obs.back()];
      const Sophus::SE3d Ta = cameraPose(p.kf, p.cam), Tb = cameraPose(q.kf, q.cam);
      X = midpoint(Ta.translation(), Ta.so3() * records[p.kf].cameras[p.cam].bearings[p.feat].cast<double>(),
                   Tb.translation(), Tb.so3() * records[q.kf].cameras[q.cam].bearings[q.feat].cast<double>(),
                   kMinParallax);
    }
    if (!X) continue;
    std::vector<int> good;
    for (int n : obs)
      if (pixelError(nodes[n].kf, nodes[n].cam, nodes[n].feat, *X) <= 2 * settings.gatePixels) good.push_back(n);
    if (good.size() < 2) continue;
    int lo = nk, hi = -1;
    for (int n : good) lo = std::min(lo, nodes[n].kf), hi = std::max(hi, nodes[n].kf);
    kept.push_back({good, *X, hi - lo > settings.sequentialNeighbours});
  }

  // Problem.
  std::vector<std::array<double, 7>> poses(nk);
  for (int k = 0; k < nk; ++k) std::copy_n(initial_T_w_b[k].data(), 7, poses[k].data());
  std::vector<std::array<double, 3>> points(kept.size());
  for (size_t t = 0; t < kept.size(); ++t) points[t] = {kept[t].X.x(), kept[t].X.y(), kept[t].X.z()};
  ceres::Problem::Options problemOptions;
  problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.enable_fast_removal = true;
  ceres::Problem problem(problemOptions);
  ceres::HuberLoss projectionLoss(settings.huberPixels), depthLoss(2.0);
  for (auto& p : poses) problem.AddParameterBlock(p.data(), 7, new Sophus::Manifold<Sophus::SE3>());
  problem.SetParameterBlockConstant(poses[0].data());
  struct Observation {
    int node, track;
    ceres::ResidualBlockId projection, depth;
  };
  std::vector<Observation> observations;
  for (size_t t = 0; t < kept.size(); ++t)
    for (int n : kept[t].obs) {
      const Node& nd = nodes[n];
      const CameraFeatures& f = records[nd.kf].cameras[nd.cam];
      Observation o{n, static_cast<int>(t), nullptr, nullptr};
      o.projection = problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<ProjectionResidual, 3, 7, 3>(
              new ProjectionResidual{f.bearings[nd.feat].cast<double>(), rig.T_c_b[nd.cam], rig.cameras[nd.cam].fx}),
          &projectionLoss, poses[nd.kf].data(), points[t].data());
      if (settings.relativeDepthSigma > 0 && f.rho[nd.feat] > 0)
        o.depth = problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DepthResidual, 1, 7, 3>(new DepthResidual{
                rig.T_c_b[nd.cam], f.rho[nd.feat], settings.relativeDepthSigma * f.rho[nd.feat]}),
            &depthLoss, poses[nd.kf].data(), points[t].data());
      observations.push_back(o);
      result.loopObservations += kept[t].loop;
    }
  if (settings.odometrySigmaFactor > 0)
    for (int k = 0; k + 1 < nk; ++k) {
      const Sophus::SE3d T_a_b = records[k].T_w_b.inverse() * records[k + 1].T_w_b;
      const double d = T_a_b.translation().norm();
      const auto& o = settings.odometry;
      problem.AddResidualBlock(
          relativePoseCost(T_a_b, settings.odometrySigmaFactor * (o.odometryTranslationBase + o.odometryTranslationRate * d),
                           settings.odometrySigmaFactor * (o.odometryRotationBaseDeg + o.odometryRotationRateDegPerMeter * d)),
          nullptr, poses[k].data(), poses[k + 1].data());
    }

  auto currentError = [&](const Observation& o) {
    const Node& nd = nodes[o.node];
    const Sophus::SE3d T_w_b = Eigen::Map<const Sophus::SE3d>(poses[nd.kf].data());
    const Eigen::Vector3d X(points[o.track][0], points[o.track][1], points[o.track][2]);
    return angle(rig.T_c_b[nd.cam] * (T_w_b.inverse() * X), records[nd.kf].cameras[nd.cam].bearings[nd.feat].cast<double>()) *
           rig.cameras[nd.cam].fx;
  };
  auto rmse = [&] {
    double sq = 0;
    size_t n = 0;
    for (const auto& o : observations)
      if (o.projection) sq += std::pow(currentError(o), 2), ++n;
    return n ? std::sqrt(sq / n) : 0.0;
  };
  result.rmseBefore = rmse();

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_SCHUR;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.max_num_iterations = settings.iterations;
  options.num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  for (int round = 0; round < settings.rounds; ++round) {
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    result.iterations += static_cast<int>(summary.iterations.size());
    if (round + 1 == settings.rounds) break;
    for (auto& o : observations) {
      if (!o.projection || currentError(o) <= settings.outlierPixels) continue;
      problem.RemoveResidualBlock(o.projection);
      if (o.depth) problem.RemoveResidualBlock(o.depth);
      o.projection = o.depth = nullptr;
    }
  }
  for (const auto& o : observations) result.observations += o.projection != nullptr;
  result.rmseAfter = rmse();
  for (int k = 0; k < nk; ++k) result.T_w_b[k] = Eigen::Map<const Sophus::SE3d>(poses[k].data());
  for (const auto& p : points) result.points.emplace_back(p[0], p[1], p[2]);
  return result;
}

}  // namespace sdv
