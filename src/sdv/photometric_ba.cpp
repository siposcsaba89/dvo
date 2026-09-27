#include <sdv/photometric_ba.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <execution>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <thread>

#include <ceres/ceres.h>
#include <ceres/cubic_interpolation.h>
#include <sophus/ceres_manifold.hpp>

#include <sdv/image_pyramid.h>

namespace sdv {

namespace {

using Grid = ceres::Grid2D<std::uint8_t, 1>;  // 8-bit images: a quarter of the memory of float grids
using Interpolator = ceres::BiCubicInterpolator<Grid>;

struct PointData {
  int host, hostCam;
  PatternPoint pattern;
  Eigen::Vector3d bearing;  // centre
};

// Pattern residual of a point in a target camera, relative pose T_t_h (target camera <- host camera), affine
// brightness I_t - b_t = exp(a_t - a_h) (I_h - b_h), each pixel weighted by the square root of its gradient weight.
template <typename T>
void patternResidual(const PointData& p, const Camera& cam, const Interpolator& image, const Sophus::SE3<T>& T_t_h,
                     const T& rho, const T* ah, const T* at, T* residual) {
  using std::exp;
  const T scale = exp(at[0] - ah[0]);
  for (int k = 0; k < kPatternSize; ++k) {
    const Eigen::Matrix<T, 3, 1> x = T_t_h.so3() * p.pattern.bearings[k].cast<T>() + rho * T_t_h.translation();
    Eigen::Matrix<T, 2, 1> uv;
    if (!cam.project(x, uv)) {
      residual[k] = T(0);
      continue;
    }
    T intensity;
    image.Evaluate(uv.y(), uv.x(), &intensity);
    residual[k] = T(std::sqrt(p.pattern.gradientWeights[k])) *
                  ((intensity - at[1]) - scale * (T(p.pattern.intensities[k]) - ah[1]));
  }
}

struct TemporalResidual {
  const PointData* point;
  const Camera* cam;
  const Interpolator* image;
  Sophus::SE3d T_ct_b, T_b_ch;

  template <typename T>
  bool operator()(const T* host, const T* target, const T* rho, const T* ah, const T* at, T* residual) const {
    const Eigen::Map<const Sophus::SE3<T>> T_w_h(host), T_w_t(target);
    const Sophus::SE3<T> T_t_h = T_ct_b.cast<T>() * T_w_t.inverse() * T_w_h * T_b_ch.cast<T>();
    patternResidual(*point, *cam, *image, T_t_h, rho[0], ah, at, residual);
    return true;
  }
};

// Another camera of the host keyframe: the relative pose is the fixed extrinsic.
struct StaticResidual {
  const PointData* point;
  const Camera* cam;
  const Interpolator* image;
  Sophus::SE3d T_t_h;

  template <typename T>
  bool operator()(const T* rho, const T* ah, const T* at, T* residual) const {
    patternResidual(*point, *cam, *image, T_t_h.cast<T>(), rho[0], ah, at, residual);
    return true;
  }
};

double angle(const Eigen::Vector3d& a, const Eigen::Vector3d& b) { return std::atan2(a.cross(b).norm(), a.dot(b)); }

}  // namespace

PhotometricBAResult photometricBundleAdjust(const Rig& rig, const std::vector<KeyframeRecord>& records,
                                            const std::vector<std::vector<cv::Mat>>& images,
                                            const std::vector<Sophus::SE3d>& initial_T_w_b,
                                            const std::vector<std::pair<int, int>>& loopPairs,
                                            const PhotometricBASettings& settings) {
  const int nk = static_cast<int>(records.size()), nc = rig.size();
  PhotometricBAResult result;
  result.T_w_b = initial_T_w_b;
  if (nk < 2) return result;

  // Float images for the interpolators (kept alive here), host patterns from a one-level pyramid per image.
  std::vector<cv::Mat> gray(static_cast<size_t>(nk) * nc);
  std::vector<std::unique_ptr<Grid>> grids(gray.size());
  std::vector<std::unique_ptr<Interpolator>> interpolators(gray.size());
  std::vector<PointData> points;
  std::vector<double> rho;
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c) {
      const size_t i = static_cast<size_t>(k) * nc + c;
      const cv::Mat floatGray = toFloatGray(images[k][c]);
      floatGray.convertTo(gray[i], CV_8U);
      grids[i] = std::make_unique<Grid>(gray[i].ptr<std::uint8_t>(), 0, gray[i].rows, 0, gray[i].cols);
      interpolators[i] = std::make_unique<Interpolator>(*grids[i]);
      const ImagePyramid pyr(floatGray, 1);
      const CameraFeatures& f = records[k].cameras[c];
      const size_t step = settings.maxPointsPerImage > 0
                              ? std::max<size_t>(1, (f.pointUv.size() + settings.maxPointsPerImage - 1) /
                                                        static_cast<size_t>(settings.maxPointsPerImage))
                              : 1;
      for (size_t j = 0; j < f.pointUv.size(); j += step) {
        const Eigen::Vector2d uv = f.pointUv[j].cast<double>();
        auto pattern = makePatternPoint(rig.cameras[c], pyr.level(0), uv, settings.photometric);
        Eigen::Vector3d b;
        if (!pattern || !rig.cameras[c].unproject(uv, b) || f.pointRho[j] <= 0) continue;
        points.push_back({k, c, *pattern, b});
        rho.push_back(f.pointRho[j]);
      }
    }

  std::vector<std::set<int>> partners(nk);
  for (const auto& [q, m] : loopPairs) partners[q].insert(m), partners[m].insert(q);

  std::vector<std::array<double, 7>> poses(nk);
  for (int k = 0; k < nk; ++k) std::copy_n(initial_T_w_b[k].data(), 7, poses[k].data());
  std::vector<std::array<double, 2>> affine(static_cast<size_t>(nk) * nc);
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c)
      if (c < static_cast<int>(records[k].affine.size())) affine[k * nc + c] = {records[k].affine[c].a, records[k].affine[c].b};

  // Candidate observations (point, target keyframe, target camera), checked with the initial estimate.
  struct Candidate {
    int point, target, cam;
    bool loop;
  };
  std::vector<std::vector<Candidate>> perPoint(points.size());
  std::vector<size_t> order(points.size());
  std::iota(order.begin(), order.end(), 0);
  std::for_each(std::execution::par, order.begin(), order.end(), [&](size_t pi) {
    const PointData& p = points[pi];
    std::set<int> targets;
    for (int t = std::max(0, p.host - settings.neighbours); t <= std::min(nk - 1, p.host + settings.neighbours); ++t)
      targets.insert(t);
    std::set<int> loopTargets;
    for (int m : partners[p.host])
      for (int t = std::max(0, m - settings.loopNeighbours); t <= std::min(nk - 1, m + settings.loopNeighbours); ++t)
        if (!targets.count(t)) loopTargets.insert(t);
    const Sophus::SE3d T_w_ch = initial_T_w_b[p.host] * rig.T_c_b[p.hostCam].inverse();
    const Eigen::Vector3d X = T_w_ch * (p.bearing / rho[pi]);
    auto consider = [&](int t, bool loop) {
      for (int tc = 0; tc < nc; ++tc) {
        if (t == p.host && tc == p.hostCam) continue;
        const Sophus::SE3d T_w_ct = initial_T_w_b[t] * rig.T_c_b[tc].inverse();
        const Eigen::Vector3d x = T_w_ct.inverse() * X;
        Eigen::Vector2d uv;
        if (!rig.cameras[tc].project(x, uv) || !rig.cameras[tc].isInside(uv.x(), uv.y(), 3.0)) continue;
        if (angle(X - T_w_ch.translation(), X - T_w_ct.translation()) > settings.maxViewAngleDeg * M_PI / 180.0)
          continue;
        const Sophus::SE3d T_t_h = T_w_ct.inverse() * T_w_ch;
        double r[kPatternSize];
        patternResidual(p, rig.cameras[tc], *interpolators[static_cast<size_t>(t) * nc + tc], T_t_h, rho[pi],
                        affine[p.host * nc + p.hostCam].data(), affine[t * nc + tc].data(), r);
        double sq = 0;
        for (double v : r) sq += v * v;
        if (std::sqrt(sq / kPatternSize) > settings.maxInitialPixelError) continue;
        perPoint[pi].push_back({static_cast<int>(pi), t, tc, loop});
      }
    };
    for (int t : targets) consider(t, false);
    for (int t : loopTargets) consider(t, true);
  });

  ceres::Problem::Options problemOptions;
  problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.enable_fast_removal = true;
  ceres::Problem problem(problemOptions);
  ceres::HuberLoss loss(settings.huber * std::sqrt(static_cast<double>(kPatternSize)));
  for (auto& p : poses) problem.AddParameterBlock(p.data(), 7, new Sophus::Manifold<Sophus::SE3>());
  problem.SetParameterBlockConstant(poses[0].data());
  for (auto& a : affine) problem.AddParameterBlock(a.data(), 2);
  problem.SetParameterBlockConstant(affine[0].data());
  struct Block {
    ceres::ResidualBlockId id;
    bool loop;
    int point;
  };
  std::vector<Block> blocks;
  for (const auto& list : perPoint)
    for (const Candidate& cd : list) {
      const PointData& p = points[cd.point];
      const Interpolator* image = interpolators[static_cast<size_t>(cd.target) * nc + cd.cam].get();
      double* ah = affine[p.host * nc + p.hostCam].data();
      double* at = affine[cd.target * nc + cd.cam].data();
      ceres::ResidualBlockId id;
      if (cd.target == p.host)
        id = problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<StaticResidual, kPatternSize, 1, 2, 2>(new StaticResidual{
                &p, &rig.cameras[cd.cam], image, rig.T_c_b[cd.cam] * rig.T_c_b[p.hostCam].inverse()}),
            &loss, &rho[cd.point], ah, at);
      else
        id = problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<TemporalResidual, kPatternSize, 7, 7, 1, 2, 2>(new TemporalResidual{
                &p, &rig.cameras[cd.cam], image, rig.T_c_b[cd.cam], rig.T_c_b[p.hostCam].inverse()}),
            &loss, poses[p.host].data(), poses[cd.target].data(), &rho[cd.point], ah, at);
      blocks.push_back({id, cd.loop, cd.point});
    }
  for (double& r : rho)
    if (problem.HasParameterBlock(&r)) problem.SetParameterLowerBound(&r, 0, 1e-4);
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

  auto pixelError = [&](const Block& b) {
    double cost = 0;
    problem.EvaluateResidualBlock(b.id, false, &cost, nullptr, nullptr);
    return std::sqrt(2.0 * cost / kPatternSize);
  };
  auto rmse = [&] {
    double sq = 0;
    size_t n = 0;
    for (const auto& b : blocks)
      if (b.id) sq += std::pow(pixelError(b), 2), ++n;
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
    for (auto& b : blocks)
      if (b.id && pixelError(b) > settings.outlierPixelError) {
        problem.RemoveResidualBlock(b.id);
        b.id = nullptr;
      }
  }
  result.pointResiduals.assign(points.size(), 0);
  for (const auto& b : blocks)
    if (b.id) ++result.residuals, result.loopResiduals += b.loop, ++result.pointResiduals[b.point];
  result.rmseAfter = rmse();

  for (int k = 0; k < nk; ++k) result.T_w_b[k] = Eigen::Map<const Sophus::SE3d>(poses[k].data());
  result.affine.assign(nk, std::vector<AffineBrightness>(nc));
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c) result.affine[k][c] = {affine[k * nc + c][0], affine[k * nc + c][1]};
  for (size_t i = 0; i < points.size(); ++i) {
    result.points.push_back(result.T_w_b[points[i].host] * rig.T_c_b[points[i].hostCam].inverse() *
                            (points[i].bearing / rho[i]));
    result.pointDistance.push_back(1.0 / rho[i]);
  }
  return result;
}

}  // namespace sdv
