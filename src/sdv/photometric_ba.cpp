#include <sdv/photometric_ba.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <execution>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <thread>
#include <tuple>

#include <ceres/ceres.h>
#include <spdlog/spdlog.h>

#include <sdv/image_pyramid.h>
#include <sdv/photometric_ba_cost.h>
#include <sdv/photometric_ba_solver.h>

namespace sdv {

namespace {

using pba::Grid;
using pba::Interpolator;
using pba::PointData;

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

  // 8-bit images for the interpolators (kept alive here), host patterns from a one-level float pyramid per image.
  std::vector<cv::Mat> gray(static_cast<size_t>(nk) * nc);
  std::vector<std::unique_ptr<Grid>> grids(gray.size());
  std::vector<std::unique_ptr<Interpolator>> interpolators(gray.size());
  std::vector<PointData> points;
  std::vector<double> rho;
  std::vector<Eigen::Vector2d> uvs;
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c) {
      const size_t i = static_cast<size_t>(k) * nc + c;
      const cv::Mat floatGray = toFloatGray(images[k][c]);
      if (images[k][c].type() == CV_8UC1)
        gray[i] = images[k][c];  // shared, not copied: long runs hold ~10 GB of keyframe images
      else
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
        uvs.push_back(uv);
        rho.push_back(f.pointRho[j]);
      }
    }

  std::vector<std::set<int>> partners(nk);
  for (const auto& [q, m] : loopPairs) partners[q].insert(m), partners[m].insert(q);
  std::vector<std::set<int>> spatial(nk);
  for (int h = 0; h < nk; ++h) {
    std::vector<std::pair<double, int>> near;
    for (int t = 0; t < nk; ++t) {
      if (std::abs(t - h) <= settings.neighbours) continue;
      const double d = (initial_T_w_b[t].translation() - initial_T_w_b[h].translation()).norm();
      if (d <= settings.spatialRadius) near.emplace_back(d, t);
    }
    std::sort(near.begin(), near.end());
    for (size_t i = 0; i < near.size() && i < static_cast<size_t>(settings.spatialTargets); ++i)
      spatial[h].insert(near[i].second);
  }

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
    for (int t : spatial[p.host])
      if (!targets.count(t)) loopTargets.insert(t);
    if (settings.maxCrossTargets > 0 && static_cast<int>(loopTargets.size()) > settings.maxCrossTargets) {
      std::vector<std::pair<double, int>> byDistance;
      for (int t : loopTargets)
        byDistance.emplace_back((initial_T_w_b[t].translation() - initial_T_w_b[p.host].translation()).norm(), t);
      std::sort(byDistance.begin(), byDistance.end());
      loopTargets.clear();
      for (int i = 0; i < settings.maxCrossTargets; ++i) loopTargets.insert(byDistance[i].second);
    }
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
        pba::patternResidual(p, rig.cameras[tc], *interpolators[static_cast<size_t>(t) * nc + tc], T_t_h, rho[pi],
                        affine[p.host * nc + p.hostCam].data(), affine[t * nc + tc].data(), r);
        double sq = 0;
        for (double v : r) sq += v * v;
        if (std::sqrt(sq / kPatternSize) > (loop ? settings.crossMaxInitialPixelError : settings.maxInitialPixelError))
          continue;
        perPoint[pi].push_back({static_cast<int>(pi), t, tc, loop});
      }
    };
    for (int t : targets) consider(t, false);
    for (int t : loopTargets) consider(t, true);
    // A point that agrees with too few images at the start is most likely a wrong depth; it would only be dropped
    // later as an outlier, after costing time.
    if (static_cast<int>(perPoint[pi].size()) < settings.minInitialResiduals) perPoint[pi].clear();
  });

  size_t candidates = 0;
  for (const auto& list : perPoint) candidates += list.size();
  spdlog::info("photometric bundle adjustment: {} points, {} candidate residuals", points.size(), candidates);

  // One residual (cost function and its parameter blocks) per candidate, shared by the joint and the blocked solve.
  std::vector<std::array<double, 7>> extrinsics(nc);
  for (int c = 0; c < nc; ++c) std::copy_n(rig.T_c_b[c].inverse().data(), 7, extrinsics[c].data());
  std::vector<std::array<double, 6>> intrinsics(nc);
  for (int c = 0; c < nc; ++c) {
    const Camera& cam = rig.cameras[c];
    intrinsics[c] = {cam.fx, cam.fy, cam.cx, cam.cy, cam.alpha, cam.beta};
  }
  const bool extrinsicCosts = settings.refineExtrinsics || settings.refineIntrinsics;
  struct Cost {
    std::unique_ptr<ceres::CostFunction> function;
    std::vector<double*> parameters;
    int rhoIndex;  // parameter block index of the inverse depth
  };
  auto makeCost = [&](const Candidate& cd) {
    const PointData& p = points[cd.point];
    const Interpolator* image = interpolators[static_cast<size_t>(cd.target) * nc + cd.cam].get();
    const Camera* cam = &rig.cameras[cd.cam];
    double* ah = affine[p.host * nc + p.hostCam].data();
    double* at = affine[cd.target * nc + cd.cam].data();
    double* eh = extrinsics[p.hostCam].data();
    double* et = extrinsics[cd.cam].data();
    double* r = &rho[cd.point];
    const bool sameKeyframe = cd.target == p.host;
    if (!extrinsicCosts && sameKeyframe)
      return Cost{std::make_unique<pba::StaticCost>(&p, cam, image, rig.T_c_b[cd.cam] * rig.T_c_b[p.hostCam].inverse()),
                  {r, ah, at}, 0};
    if (!extrinsicCosts)
      return Cost{std::make_unique<pba::TemporalCost>(&p, cam, image, rig.T_c_b[cd.cam], rig.T_c_b[p.hostCam].inverse()),
                  {poses[p.host].data(), poses[cd.target].data(), r, ah, at}, 2};
    if (settings.refineIntrinsics) {
      const Camera* hostCam = &rig.cameras[p.hostCam];
      double* kh = intrinsics[p.hostCam].data();
      double* kt = intrinsics[cd.cam].data();
      if (sameKeyframe)
        return Cost{std::make_unique<pba::CalibratedCost>(pba::ExtrinsicCostKind::Static, &p, hostCam, cam, image),
                    {eh, et, r, ah, at, kh, kt}, 2};
      if (cd.cam == p.hostCam)
        return Cost{std::make_unique<pba::CalibratedCost>(pba::ExtrinsicCostKind::TemporalSameCamera, &p, hostCam, cam,
                                                          image),
                    {poses[p.host].data(), poses[cd.target].data(), eh, r, ah, at, kh}, 3};
      return Cost{std::make_unique<pba::CalibratedCost>(pba::ExtrinsicCostKind::Temporal, &p, hostCam, cam, image),
                  {poses[p.host].data(), poses[cd.target].data(), eh, et, r, ah, at, kh, kt}, 4};
    }
    if (sameKeyframe) return Cost{std::make_unique<pba::StaticExtrinsicCost>(&p, cam, image), {eh, et, r, ah, at}, 2};
    if (cd.cam == p.hostCam)
      return Cost{std::make_unique<pba::TemporalSameCameraExtrinsicCost>(&p, cam, image),
                  {poses[p.host].data(), poses[cd.target].data(), eh, r, ah, at}, 3};
    return Cost{std::make_unique<pba::TemporalExtrinsicCost>(&p, cam, image),
                {poses[p.host].data(), poses[cd.target].data(), eh, et, r, ah, at}, 4};
  };
  auto odometryCost = [&](int k) {
    const Sophus::SE3d T_a_b = records[k].T_w_b.inverse() * records[k + 1].T_w_b;
    const double d = T_a_b.translation().norm();
    const auto& o = settings.odometry;
    return new pba::RelativePoseCost(
        T_a_b, settings.odometrySigmaFactor * (o.odometryTranslationBase + o.odometryTranslationRate * d),
        settings.odometrySigmaFactor * (o.odometryRotationBaseDeg + o.odometryRotationRateDegPerMeter * d));
  };

  ceres::Problem::Options problemOptions;
  problemOptions.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problemOptions.enable_fast_removal = true;
  ceres::HuberLoss loss(settings.huber * std::sqrt(static_cast<double>(kPatternSize)));
  ceres::Solver::Options options;
  if (settings.solver == PhotometricBASettings::Solver::Iterative) {
    options.linear_solver_type = ceres::ITERATIVE_SCHUR;
    options.preconditioner_type = ceres::SCHUR_JACOBI;
  } else {
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
    if (settings.solver == PhotometricBASettings::Solver::SparseNesdis)
      options.linear_solver_ordering_type = ceres::NESDIS;
  }
  options.max_num_iterations = settings.iterations;
  options.num_threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

  auto timing = [](const ceres::Solver::Summary& sm) {
    return fmt::format("{:.1f} s: preprocess {:.1f}, residuals {:.1f}, jacobians {:.1f}, linear solver {:.1f}",
                       sm.total_time_in_seconds, sm.preprocessor_time_in_seconds,
                       sm.residual_evaluation_time_in_seconds, sm.jacobian_evaluation_time_in_seconds,
                       sm.linear_solver_time_in_seconds);
  };

  // Final statistics over the kept residuals: counts, depth information (for the depth sigma), per camera pair.
  result.pointResiduals.assign(points.size(), 0);
  std::vector<double> depthInformation(points.size(), 0.0);
  std::map<std::tuple<int, int, bool>, std::array<double, 3>> pairs;  // count, squared before, squared after

  const bool custom = settings.optimizer == PhotometricBASettings::Optimizer::Custom && !extrinsicCosts;
  const bool useBlocks = settings.blockKeyframes > 0 && settings.blockKeyframes < nk;
  std::vector<const Interpolator*> imagePtrs(interpolators.size());
  for (size_t i = 0; i < interpolators.size(); ++i) imagePtrs[i] = interpolators[i].get();
  pba::SolverOptions solverOptions;
  solverOptions.iterations = settings.iterations;
  auto solverProblem = [&] {
    pba::SolverProblem sp;
    sp.rig = &rig, sp.points = &points, sp.images = &imagePtrs, sp.poses = &poses, sp.affine = &affine, sp.rho = &rho;
    sp.huber = settings.huber * std::sqrt(static_cast<double>(kPatternSize));
    return sp;
  };
  auto addOdometry = [&](pba::SolverProblem& sp, int k) {
    if (settings.odometrySigmaFactor > 0) sp.odometry.push_back({k, std::shared_ptr<const pba::RelativePoseCost>(odometryCost(k))});
  };
  auto solverTiming = [](const pba::SolverSummary& sm) {
    return fmt::format("{:.1f} s: structure {:.1f}, linearize {:.1f}, factorize {:.1f}, evaluate {:.1f}; {} iterations "
                       "({} accepted), reduced system {}",
                       sm.total, sm.structure, sm.linearize, sm.factorize, sm.evaluate, sm.iterations, sm.accepted,
                       sm.reducedSize);
  };

  if (useBlocks || custom) {
    // Block-coordinate descent for bounded memory: one problem per block of consecutive keyframes, with its poses and
    // affine brightness, the inverse depths of the points it hosts with all their residuals, and the residuals of
    // outside points into it; everything outside the block stays fixed. Block borders shift by half a block between
    // sweeps. Residuals are evaluated without a global problem (same cost functions, loss not applied).
    if (extrinsicCosts) throw std::invalid_argument("photometric BA: blocks cannot refine the rig calibration");
    std::vector<Candidate> cands;
    for (const auto& list : perPoint) cands.insert(cands.end(), list.begin(), list.end());
    perPoint = {};
    std::vector<std::vector<std::uint32_t>> byHost(nk), byTarget(nk);
    for (size_t i = 0; i < cands.size(); ++i) {
      const int host = points[cands[i].point].host;
      byHost[host].push_back(static_cast<std::uint32_t>(i));
      if (cands[i].target != host) byTarget[cands[i].target].push_back(static_cast<std::uint32_t>(i));
    }
    std::vector<char> alive(cands.size(), 1);
    std::vector<float> initialError(cands.size());
    auto pixelErrors = [&](std::vector<float>& out) {
      out.resize(cands.size());
      std::vector<size_t> idx(cands.size());
      std::iota(idx.begin(), idx.end(), 0);
      std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
        if (!alive[i]) return;
        const Cost c = makeCost(cands[i]);
        double r[kPatternSize];
        c.function->Evaluate(c.parameters.data(), r, nullptr);
        double sq = 0;
        for (double v : r) sq += v * v;
        out[i] = static_cast<float>(std::sqrt(sq / kPatternSize));
      });
    };
    auto rmseOf = [&](const std::vector<float>& e) {
      double sq = 0;
      size_t n = 0;
      for (size_t i = 0; i < e.size(); ++i)
        if (alive[i]) sq += double(e[i]) * e[i], ++n;
      return n ? std::sqrt(sq / n) : 0.0;
    };
    pixelErrors(initialError);
    result.rmseBefore = rmseOf(initialError);

    // Coarse joint solve first: blocks alone converge slowly on smooth deformations of the whole trajectory (each
    // block sees fixed neighbours); all keyframes with every m-th point (within blockCoarseResiduals) fix those, the
    // blocks then refine locally with all points.
    const size_t m = std::max<size_t>(1, (cands.size() + settings.blockCoarseResiduals - 1) /
                                             std::max<size_t>(1, settings.blockCoarseResiduals));
    if (useBlocks && custom) {
      pba::SolverProblem sp = solverProblem();
      for (const Candidate& cd : cands)
        if (cd.point % m == 0) sp.observations.push_back({cd.point, cd.target, cd.cam});
      sp.poseFree.assign(nk, 1), sp.poseFree[0] = 0;
      sp.affineFree.assign(static_cast<size_t>(nk) * nc, 0);  // as below: brightness refined by the blocks
      sp.rhoFree.assign(points.size(), 1);
      for (int k = 0; k + 1 < nk; ++k) addOdometry(sp, k);
      const pba::SolverSummary sm = pba::solve(sp, solverOptions);
      result.iterations += sm.iterations;
      spdlog::info("photometric BA coarse: all {} keyframes, every {}. point, {} residuals, cost {:.4g} -> {:.4g}, {}", nk,
                   m, sp.observations.size(), sm.initialCost, sm.finalCost, solverTiming(sm));
    } else if (useBlocks) {
      ceres::Problem problem(problemOptions);
      for (auto& p : poses) problem.AddParameterBlock(p.data(), 7, new pba::SE3TangentManifold());
      size_t added = 0;
      for (size_t i = 0; i < cands.size(); ++i) {
        if (cands[i].point % m != 0) continue;
        Cost c = makeCost(cands[i]);
        problem.AddResidualBlock(c.function.release(), &loss, c.parameters);
        ++added;
      }
      if (settings.odometrySigmaFactor > 0)
        for (int k = 0; k + 1 < nk; ++k)
          problem.AddResidualBlock(odometryCost(k), nullptr, poses[k].data(), poses[k + 1].data());
      problem.SetParameterBlockConstant(poses[0].data());
      // Brightness stays at the odometry's (the blocks refine it): 8 of the 14 reduced-system variables per keyframe
      // of four cameras, and the single-threaded Cholesky of that system dominates the coarse solve.
      for (auto& a : affine)
        if (problem.HasParameterBlock(a.data())) problem.SetParameterBlockConstant(a.data());
      for (size_t i = 0; i < points.size(); i += m)
        if (problem.HasParameterBlock(&rho[i])) problem.SetParameterLowerBound(&rho[i], 0, 1e-4);
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      result.iterations += static_cast<int>(summary.iterations.size());
      spdlog::info("photometric BA coarse: all {} keyframes, every {}. point, {} residuals, cost {:.4g} -> {:.4g}, {}", nk,
                   m, added, summary.initial_cost, summary.final_cost, timing(summary));
    }

    const int n = useBlocks ? settings.blockKeyframes : nk;
    for (int round = 0; round < settings.rounds; ++round) {
      for (int sweep = 0; sweep < (useBlocks ? settings.blockSweeps : 1); ++sweep) {
        std::vector<std::pair<int, int>> ranges;
        for (int b = 0, e = sweep % 2 ? n / 2 : n; b < nk; b = e, e += n) ranges.emplace_back(b, std::min(e, nk));
        for (const auto& [b, e] : ranges) {
          auto inside = [&, b = b, e = e](int k) { return k >= b && k < e; };
          if (custom) {
            pba::SolverProblem sp = solverProblem();
            auto add = [&](std::uint32_t i) {
              if (alive[i]) sp.observations.push_back({cands[i].point, cands[i].target, cands[i].cam});
            };
            for (int k = b; k < e; ++k) {
              for (std::uint32_t i : byHost[k]) add(i);
              for (std::uint32_t i : byTarget[k])
                if (!inside(points[cands[i].point].host)) add(i);
            }
            sp.poseFree.assign(nk, 0), sp.affineFree.assign(static_cast<size_t>(nk) * nc, 0);
            sp.rhoFree.assign(points.size(), 0);
            for (int k = b; k < e; ++k) {
              sp.poseFree[k] = k != 0;
              for (int c = 0; c < nc; ++c) sp.affineFree[static_cast<size_t>(k) * nc + c] = !(k == 0 && c == 0);
              for (std::uint32_t i : byHost[k]) sp.rhoFree[cands[i].point] = 1;
            }
            for (int k = std::max(0, b - 1); k < e && k + 1 < nk; ++k) addOdometry(sp, k);
            const pba::SolverSummary sm = pba::solve(sp, solverOptions);
            result.iterations += sm.iterations;
            spdlog::info("photometric BA round {} sweep {}: keyframes {}..{}, {} residuals, cost {:.4g} -> {:.4g}, {}",
                         round + 1, sweep + 1, b, e - 1, sp.observations.size(), sm.initialCost, sm.finalCost,
                         solverTiming(sm));
            continue;
          }
          ceres::Problem problem(problemOptions);
          for (int k = b; k < e; ++k) problem.AddParameterBlock(poses[k].data(), 7, new pba::SE3TangentManifold());
          std::vector<double*> fixed;
          size_t added = 0;
          auto add = [&](std::uint32_t i) {
            if (!alive[i]) return;
            Cost c = makeCost(cands[i]);
            problem.AddResidualBlock(c.function.release(), &loss, c.parameters);
            const PointData& p = points[cands[i].point];
            if (!inside(p.host)) fixed.push_back(&rho[cands[i].point]);
            ++added;
          };
          for (int k = b; k < e; ++k) {
            for (std::uint32_t i : byHost[k]) add(i);
            for (std::uint32_t i : byTarget[k])
              if (!inside(points[cands[i].point].host)) add(i);
          }
          if (settings.odometrySigmaFactor > 0)
            for (int k = std::max(0, b - 1); k < e && k + 1 < nk; ++k)
              problem.AddResidualBlock(odometryCost(k), nullptr, poses[k].data(), poses[k + 1].data());
          for (int k = 0; k < nk; ++k) {
            if (!inside(k) && problem.HasParameterBlock(poses[k].data())) problem.SetParameterBlockConstant(poses[k].data());
            for (int c = 0; c < nc; ++c) {
              double* a = affine[k * nc + c].data();
              if (problem.HasParameterBlock(a) && (!inside(k) || (k == 0 && c == 0))) problem.SetParameterBlockConstant(a);
            }
          }
          if (inside(0)) problem.SetParameterBlockConstant(poses[0].data());
          for (double* r : fixed) problem.SetParameterBlockConstant(r);
          for (int k = b; k < e; ++k)
            for (std::uint32_t i : byHost[k])
              if (alive[i] && problem.HasParameterBlock(&rho[cands[i].point]))
                problem.SetParameterLowerBound(&rho[cands[i].point], 0, 1e-4);
          ceres::Solver::Summary summary;
          ceres::Solve(options, &problem, &summary);
          result.iterations += static_cast<int>(summary.iterations.size());
          spdlog::info("photometric BA round {} sweep {}: keyframes {}..{}, {} residuals, cost {:.4g} -> {:.4g}, {}",
                       round + 1, sweep + 1, b, e - 1, added, summary.initial_cost, summary.final_cost, timing(summary));
        }
      }
      if (round + 1 == settings.rounds) break;
      std::vector<float> err;
      pixelErrors(err);
      for (size_t i = 0; i < cands.size(); ++i)
        if (alive[i] && err[i] > settings.outlierPixelError) alive[i] = 0;
    }

    std::vector<float> finalError;
    pixelErrors(finalError);
    result.rmseAfter = rmseOf(finalError);
    std::mutex mutex;
    std::vector<size_t> idx(cands.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::for_each(std::execution::par, idx.begin(), idx.end(), [&](size_t i) {
      if (!alive[i]) return;
      const Cost c = makeCost(cands[i]);
      double residual[kPatternSize], dRho[kPatternSize];
      std::vector<double*> jacobians(c.parameters.size(), nullptr);
      jacobians[c.rhoIndex] = dRho;
      c.function->Evaluate(c.parameters.data(), residual, jacobians.data());
      double info = 0;
      for (double j : dRho) info += j * j;
      const Candidate& cd = cands[i];
      const PointData& p = points[cd.point];
      std::lock_guard lock(mutex);
      ++result.residuals, result.loopResiduals += cd.loop, ++result.pointResiduals[cd.point];
      depthInformation[cd.point] += info;
      auto& acc = pairs[{p.hostCam, cd.cam, cd.target == p.host}];
      acc[0] += 1, acc[1] += double(initialError[i]) * initialError[i], acc[2] += double(finalError[i]) * finalError[i];
    });
  } else {
    ceres::Problem problem(problemOptions);
    for (auto& p : poses) problem.AddParameterBlock(p.data(), 7, new pba::SE3TangentManifold());
    problem.SetParameterBlockConstant(poses[0].data());
    for (auto& a : affine) problem.AddParameterBlock(a.data(), 2);
    problem.SetParameterBlockConstant(affine[0].data());

    // Extrinsics as T_b_c blocks (see pba::TemporalExtrinsicCost), used only with refineExtrinsics.
    std::array<double, 7> identity;
    std::copy_n(Sophus::SE3d().data(), 7, identity.data());
    if (extrinsicCosts) {
      for (auto& e : extrinsics) problem.AddParameterBlock(e.data(), 7, new pba::SE3TangentManifold());
      problem.SetParameterBlockConstant(extrinsics[0].data());
      if (!settings.refineExtrinsics)
        for (auto& e : extrinsics) problem.SetParameterBlockConstant(e.data());
    }
    if (settings.refineIntrinsics) {
      for (int c = 0; c < nc; ++c) {
        problem.AddParameterBlock(intrinsics[c].data(), 6);
        const auto& sel = settings.intrinsicCameras;
        if (!sel.empty() && std::find(sel.begin(), sel.end(), c) == sel.end()) {
          problem.SetParameterBlockConstant(intrinsics[c].data());
          continue;
        }
        const Camera& cam = rig.cameras[c];
        problem.AddResidualBlock(
            new pba::IntrinsicPriorCost(cam, {settings.intrinsicSigmaFocal * cam.fx, settings.intrinsicSigmaFocal * cam.fy,
                                              settings.intrinsicSigmaCenter, settings.intrinsicSigmaCenter,
                                              settings.intrinsicSigmaAlpha, settings.intrinsicSigmaBeta}),
            nullptr, intrinsics[c].data());
      }
    }
    if (settings.refineExtrinsics) {
      problem.AddParameterBlock(identity.data(), 7);
      problem.SetParameterBlockConstant(identity.data());
      for (int c = 1; c < nc; ++c)
        problem.AddResidualBlock(new pba::RelativePoseCost(rig.T_c_b[c].inverse(), settings.extrinsicSigmaT,
                                                           settings.extrinsicSigmaRDeg),
                                 nullptr, identity.data(), extrinsics[c].data());
      if (settings.extrinsicFixScale && nc > 1) {
        double sum0 = 0;
        for (int i = 0; i < nc; ++i)
          for (int j = i + 1; j < nc; ++j)
            sum0 += (rig.T_c_b[i].inverse().translation() - rig.T_c_b[j].inverse().translation()).norm();
        std::vector<double*> blocks;
        for (auto& e : extrinsics) blocks.push_back(e.data());
        // 10 um: a hard constraint next to the photometric residuals.
        problem.AddResidualBlock(new pba::RigScaleCost(nc, sum0, 1e-5), nullptr, blocks);
      }
    }

    struct Block {
      ceres::ResidualBlockId id;
      bool loop;
      int point;
      int rhoIndex;
      int hostCam, targetCam;
      bool sameKeyframe;
      double initialError;
    };
    std::vector<Block> blocks;
    for (const auto& list : perPoint)
      for (const Candidate& cd : list) {
        Cost c = makeCost(cd);
        const PointData& p = points[cd.point];
        const ceres::ResidualBlockId id = problem.AddResidualBlock(c.function.release(), &loss, c.parameters);
        blocks.push_back({id, cd.loop, cd.point, c.rhoIndex, p.hostCam, cd.cam, cd.target == p.host, 0.0});
      }
    for (double& r : rho)
      if (problem.HasParameterBlock(&r)) problem.SetParameterLowerBound(&r, 0, 1e-4);
    if (settings.odometrySigmaFactor > 0)
      for (int k = 0; k + 1 < nk; ++k)
        problem.AddResidualBlock(odometryCost(k), nullptr, poses[k].data(), poses[k + 1].data());

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
    for (auto& b : blocks) b.initialError = pixelError(b);

    for (int round = 0; round < settings.rounds; ++round) {
      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      result.iterations += static_cast<int>(summary.iterations.size());
      spdlog::info("photometric BA round {}: {}", round + 1, timing(summary));
      if (round + 1 == settings.rounds) break;
      for (auto& b : blocks)
        if (b.id && pixelError(b) > settings.outlierPixelError) {
          problem.RemoveResidualBlock(b.id);
          b.id = nullptr;
        }
    }
    for (const auto& b : blocks) {
      if (!b.id) continue;
      ++result.residuals, result.loopResiduals += b.loop, ++result.pointResiduals[b.point];
      double residual[kPatternSize], dRho[kPatternSize];
      std::array<double*, 9> jacobians{};
      jacobians[b.rhoIndex] = dRho;
      double cost = 0;
      problem.EvaluateResidualBlock(b.id, false, &cost, residual, jacobians.data());
      for (double j : dRho) depthInformation[b.point] += j * j;
    }
    result.rmseAfter = rmse();
    for (const auto& b : blocks) {
      if (!b.id) continue;
      auto& acc = pairs[{b.hostCam, b.targetCam, b.sameKeyframe}];
      acc[0] += 1, acc[1] += b.initialError * b.initialError, acc[2] += std::pow(pixelError(b), 2);
    }
  }

  // Per camera pair, over the same (kept) residuals before and after.
  for (const auto& [key, acc] : pairs)
    result.cameraPairs.push_back({std::get<0>(key), std::get<1>(key), std::get<2>(key),
                                  static_cast<size_t>(acc[0]), std::sqrt(acc[1] / acc[0]), std::sqrt(acc[2] / acc[0])});

  result.cameras.clear();
  for (int c = 0; c < nc; ++c) result.cameras.push_back(pba::withIntrinsics(rig.cameras[c], intrinsics[c].data()));
  result.T_c_b.resize(nc);
  for (int c = 0; c < nc; ++c) result.T_c_b[c] = Eigen::Map<const Sophus::SE3d>(extrinsics[c].data()).inverse();
  for (int k = 0; k < nk; ++k) result.T_w_b[k] = Eigen::Map<const Sophus::SE3d>(poses[k].data());
  result.affine.assign(nk, std::vector<AffineBrightness>(nc));
  for (int k = 0; k < nk; ++k)
    for (int c = 0; c < nc; ++c) result.affine[k][c] = {affine[k * nc + c][0], affine[k * nc + c][1]};
  for (size_t i = 0; i < points.size(); ++i) {
    Eigen::Vector3d bearing = points[i].bearing;
    if (settings.refineIntrinsics) result.cameras[points[i].hostCam].unproject(uvs[i], bearing);
    result.points.push_back(result.T_w_b[points[i].host] * result.T_c_b[points[i].hostCam].inverse() *
                            (bearing / rho[i]));
    result.pointDistance.push_back(1.0 / rho[i]);
    result.pointHost.push_back({points[i].host, points[i].hostCam});
    result.pointUv.push_back(uvs[i]);
    result.pointDepthSigma.push_back(depthInformation[i] > 0 ? 1.0 / (rho[i] * std::sqrt(depthInformation[i]))
                                                             : std::numeric_limits<double>::infinity());
  }
  return result;
}

}  // namespace sdv
