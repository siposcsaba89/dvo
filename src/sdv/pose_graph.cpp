#include <sdv/pose_graph.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

#include <ceres/ceres.h>
#include <sophus/ceres_manifold.hpp>

namespace sdv {

namespace {

struct RelativePoseResidual {
  Sophus::SE3d T_b_a_measured;  // inverse of the measured T_a_b
  Sophus::Vector6d weight;  // 1 / sigma per tangent component (translation, rotation)

  template <typename T>
  bool operator()(const T* a, const T* b, T* residual) const {
    const Eigen::Map<const Sophus::SE3<T>> T_w_a(a), T_w_b(b);
    const Eigen::Matrix<T, 6, 1> e = (T_b_a_measured.cast<T>() * T_w_a.inverse() * T_w_b).log();
    for (int k = 0; k < 6; ++k) residual[k] = e[k] * T(weight[k]);
    return true;
  }
};

Sophus::Vector6d weights(double sigmaT, double sigmaRDeg) {
  const double r = sigmaRDeg * M_PI / 180.0;
  return (Sophus::Vector6d() << 1 / sigmaT, 1 / sigmaT, 1 / sigmaT, 1 / r, 1 / r, 1 / r).finished();
}

}  // namespace

ceres::CostFunction* relativePoseCost(const Sophus::SE3d& T_a_b, double sigmaT, double sigmaRDeg) {
  return new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 7, 7>(
      new RelativePoseResidual{T_a_b.inverse(), weights(sigmaT, sigmaRDeg)});
}

PoseGraphResult optimizePoseGraph(const std::vector<Sophus::SE3d>& odometry_T_w_b,
                                  const std::vector<LoopConstraint>& loops, const PoseGraphSettings& settings) {
  const size_t n = odometry_T_w_b.size();
  PoseGraphResult result;
  result.T_w_b = odometry_T_w_b;
  if (n < 2) return result;
  std::vector<std::array<double, Sophus::SE3d::num_parameters>> params(n);
  for (size_t i = 0; i < n; ++i)
    std::copy_n(odometry_T_w_b[i].data(), Sophus::SE3d::num_parameters, params[i].data());

  ceres::Problem problem;
  for (auto& p : params) problem.AddParameterBlock(p.data(), Sophus::SE3d::num_parameters, new Sophus::Manifold<Sophus::SE3>());
  problem.SetParameterBlockConstant(params[0].data());
  for (size_t i = 0; i + 1 < n; ++i) {
    const Sophus::SE3d T_a_b = odometry_T_w_b[i].inverse() * odometry_T_w_b[i + 1];
    const double d = T_a_b.translation().norm();
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 7, 7>(new RelativePoseResidual{
            T_a_b.inverse(),
            weights(settings.odometryTranslationBase + settings.odometryTranslationRate * d,
                    settings.odometryRotationBaseDeg + settings.odometryRotationRateDegPerMeter * d)}),
        nullptr, params[i].data(), params[i + 1].data());
  }
  const Sophus::Vector6d loopWeight = weights(settings.loopTranslationSigma, settings.loopRotationSigmaDeg);
  std::vector<ceres::ResidualBlockId> loopBlocks;
  for (const auto& l : loops) {
    if (l.query < 0 || l.match < 0 || static_cast<size_t>(std::max(l.query, l.match)) >= n)
      throw std::invalid_argument("loop refers to an unknown keyframe");
    // T_q_m maps the match body into the query body: edge a = query, b = match.
    loopBlocks.push_back(problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<RelativePoseResidual, 6, 7, 7>(new RelativePoseResidual{l.T_q_m.inverse(), loopWeight}),
        nullptr, params[l.query].data(), params[l.match].data()));
  }

  ceres::Solver::Options options;
  options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
  options.max_num_iterations = settings.iterations;
  options.num_threads = 8;
  result.loopAccepted.assign(loops.size(), 1);
  for (int round = 0; round <= settings.maxRejectRounds; ++round) {
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (round == 0) result.initialCost = summary.initial_cost;
    result.finalCost = summary.final_cost;
    result.iterations += static_cast<int>(summary.iterations.size());

    // Worst loop, if it stands out: the sigmas are only a model of the odometry drift, so a loop is rejected when
    // it is far outside its own sigma and also much worse than the typical loop.
    std::vector<std::pair<double, int>> residual;
    for (size_t k = 0; k < loops.size(); ++k) {
      if (!result.loopAccepted[k]) continue;
      double e[6];
      RelativePoseResidual{loops[k].T_q_m.inverse(), loopWeight}(params[loops[k].query].data(),
                                                                 params[loops[k].match].data(), e);
      residual.emplace_back(Eigen::Map<Sophus::Vector6d>(e).norm(), static_cast<int>(k));
    }
    if (residual.empty() || round == settings.maxRejectRounds) break;
    std::sort(residual.begin(), residual.end());
    const double median = residual[residual.size() / 2].first;
    const auto [worstError, worst] = residual.back();
    if (worstError <= std::max(settings.loopRejectSigma, settings.loopRejectMedianFactor * median)) break;
    problem.RemoveResidualBlock(loopBlocks[worst]);
    result.loopAccepted[worst] = 0;
    ++result.loopsRejected;
  }
  for (size_t i = 0; i < n; ++i) result.T_w_b[i] = Eigen::Map<const Sophus::SE3d>(params[i].data());
  return result;
}

PoseCorrection::PoseCorrection(const std::vector<int>& keyframeFrames, const std::vector<Sophus::SE3d>& before,
                               const std::vector<Sophus::SE3d>& after)
    : m_frames(keyframeFrames) {
  if (before.size() != keyframeFrames.size() || after.size() != keyframeFrames.size())
    throw std::invalid_argument("one pose before and after per keyframe required");
  for (size_t i = 0; i < before.size(); ++i) m_corrections.push_back(after[i] * before[i].inverse());
}

Sophus::SE3d PoseCorrection::at(int frameIndex) const {
  if (m_frames.empty()) return {};
  const auto it = std::lower_bound(m_frames.begin(), m_frames.end(), frameIndex);
  if (it == m_frames.begin()) return m_corrections.front();
  if (it == m_frames.end()) return m_corrections.back();
  const size_t hi = static_cast<size_t>(it - m_frames.begin());
  if (*it == frameIndex) return m_corrections[hi];
  const double s = static_cast<double>(frameIndex - m_frames[hi - 1]) / (m_frames[hi] - m_frames[hi - 1]);
  const Sophus::SE3d& a = m_corrections[hi - 1];
  return a * Sophus::SE3d::exp(s * (a.inverse() * m_corrections[hi]).log());
}

}  // namespace sdv
