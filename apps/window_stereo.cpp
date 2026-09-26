// Validates the sliding-window BA: keyframe points start from stereo depth, then only photometric BA
// (monocular residuals) refines poses and depths. Frames are tracked against the projected window points.
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <string>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/point_selector.h>
#include <sdv/stereo_depth.h>
#include <sdv/tracker.h>
#include <sdv/window_optimizer.h>

namespace po = boost::program_options;

namespace {

sdv::ReferenceFrame makeReference(const sdv::WindowOptimizer& opt, const sdv::WindowFrame& kf, double gradientC) {
  const sdv::Camera& cam = opt.camera();
  std::vector<Eigen::Vector2i> pixels;
  std::vector<double> rhos;
  for (const auto& p : opt.points()) {
    if (p.host != kf.id && p.numGood() == 0) continue;
    const Sophus::SE3d T_kf_h = kf.params.T_c_w * opt.frame(p.host).params.T_c_w.inverse();
    const Eigen::Vector3d x = T_kf_h.so3() * p.bearing + p.rho * T_kf_h.translation();
    Eigen::Vector2d uv;
    if (!cam.project(x, uv) || !cam.isInside(uv.x(), uv.y(), 2.0)) continue;
    pixels.emplace_back(static_cast<int>(std::lround(uv.x())), static_cast<int>(std::lround(uv.y())));
    rhos.push_back(p.rho / x.norm());
  }
  return sdv::ReferenceFrame(cam, *kf.image, pixels, rhos, kf.params.affine, gradientC);
}

}  // namespace

int main(int argc, char** argv) {
  std::string sequenceDir, gtFile, outFile, plyFile;
  size_t start = 0, maxFrames = 100;
  int levels = 5, keyframeInterval = 3, windowSize = 7, numPoints = 800, iterations = 6;
  bool noBa = false, verbose = false;

  po::options_description desc("window_stereo options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir)->required(), "KITTI sequence directory")
      ("gt", po::value(&gtFile), "ground-truth poses for evaluation")
      ("out,o", po::value(&outFile), "write estimated poses (KITTI format)")
      ("ply", po::value(&plyFile), "write trajectories and window points as PLY")
      ("start", po::value(&start)->default_value(0), "first frame")
      ("max-frames,n", po::value(&maxFrames)->default_value(100), "process at most N frames")
      ("levels", po::value(&levels)->default_value(5), "pyramid levels")
      ("kf-interval", po::value(&keyframeInterval)->default_value(3), "new keyframe every N frames")
      ("window", po::value(&windowSize)->default_value(7), "keyframes in the optimisation window")
      ("points", po::value(&numPoints)->default_value(800), "new points per keyframe")
      ("iterations", po::value(&iterations)->default_value(6), "BA iterations per keyframe")
      ("no-ba", po::bool_switch(&noBa), "skip optimisation (baseline)")
      ("verbose,v", po::bool_switch(&verbose), "debug logging");
  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(vm);
  } catch (const po::error& e) {
    spdlog::error("{}", e.what());
    std::cout << desc << '\n';
    return EXIT_FAILURE;
  }
  if (verbose) spdlog::set_level(spdlog::level::debug);

  try {
    const sdv::KittiSequence seq(sequenceDir, 1 << (levels - 1));
    const sdv::Camera& cam = seq.camera();
    const size_t n = std::min(maxFrames, seq.size() - start);
    const sdv::TrackingSettings trackSettings;
    const sdv::FrameTracker tracker(trackSettings);
    sdv::WindowOptimizer opt(cam);
    sdv::PointSelectorSettings selSettings;
    selSettings.targetPoints = numPoints;
    const sdv::PointSelector selector(selSettings);

    struct FramePose {
      int keyframe;
      Sophus::SE3d T_f_kf;
    };
    std::vector<FramePose> framePoses;
    std::map<int, Sophus::SE3d> keyframePoses;  // T_c_w, last estimate
    std::optional<sdv::ReferenceFrame> ref;
    int refId = -1;
    sdv::AffineBrightness affine;
    Sophus::SE3d T_prev_ref, T_prev_prevprev;
    double trackMs = 0, baMs = 0;
    int numKeyframes = 0;
    using Clock = std::chrono::steady_clock;

    for (size_t i = 0; i < n; ++i) {
      const cv::Mat left = seq.loadImage(start + i, 0);
      auto pyr = std::make_shared<const sdv::ImagePyramid>(sdv::toFloatGray(left), levels);

      Sophus::SE3d T_f_ref;
      if (i > 0) {
        const auto t0 = Clock::now();
        const auto res = tracker.track(*ref, *pyr, sdv::makeMotionHypotheses(T_prev_ref, T_prev_prevprev), affine);
        trackMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (!res.ok) spdlog::warn("frame {}: tracking weak (inliers {:.2f})", i, res.inlierRatio);
        T_f_ref = res.T_t_h;
        T_prev_prevprev = T_f_ref * T_prev_ref.inverse();
        T_prev_ref = T_f_ref;
        affine = res.affine;
      }
      framePoses.push_back({refId, T_f_ref});

      if (i % keyframeInterval != 0) continue;
      const auto t0 = Clock::now();
      const Sophus::SE3d T_c_w = i == 0 ? Sophus::SE3d() : T_f_ref * opt.frame(refId).params.T_c_w;
      const int id = opt.addFrame(pyr, T_c_w, affine);
      ++numKeyframes;
      const cv::Mat rho = sdv::stereoInverseDistance(left, seq.loadImage(start + i, 1), cam, seq.baseline());
      int added = 0;
      for (const auto& c : selector.select(pyr->level(0))) {
        const float r = rho.at<float>(c.uv.y(), c.uv.x());
        if (r > 0 && opt.addPoint(id, c.uv.cast<double>(), r) >= 0) ++added;
      }
      if (!noBa) {
        const auto res = opt.optimize(iterations);
        spdlog::debug("kf {} (frame {}): {} new points, energy {:.0f} -> {:.0f} in {} it, good {} outliers {} oob {}",
                      id, i, added, res.initialEnergy, res.finalEnergy, res.iterations, res.numGood,
                      res.numOutliers, res.numOutOfBounds);
      }
      for (const auto& f : opt.frames()) keyframePoses[f.id] = f.params.T_c_w;
      if (static_cast<int>(opt.frames().size()) > windowSize) {
        if (noBa) {
          std::vector<int> hosted;
          for (const auto& p : opt.points())
            if (p.host == opt.frames().front().id) hosted.push_back(p.id);
          for (int pid : hosted) opt.removePoint(pid);
        }
        opt.marginalizeFrame(opt.frames().front().id);
      }
      baMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

      refId = id;
      const sdv::WindowFrame& kf = opt.frame(id);
      affine = kf.params.affine;
      ref.emplace(makeReference(opt, kf, trackSettings.gradientWeightC));
      framePoses.back() = {id, Sophus::SE3d()};
      T_prev_ref = Sophus::SE3d();
    }
    spdlog::info("tracking {:.1f} ms/frame, keyframe + BA {:.1f} ms/kf ({} keyframes)",
                 trackMs / std::max<size_t>(n - 1, 1), baMs / std::max(numKeyframes, 1), numKeyframes);

    std::vector<Sophus::SE3d> poses;  // T_w_c
    for (const auto& fp : framePoses) poses.push_back((fp.T_f_kf * keyframePoses.at(fp.keyframe)).inverse());
    if (!outFile.empty()) sdv::saveKittiPoses(outFile, poses);
    sdv::PlyScene scene;
    scene.addTrajectory(poses, {220, 0, 0});
    if (!gtFile.empty()) {
      const auto allGt = sdv::loadKittiPoses(gtFile);
      std::vector<Sophus::SE3d> gt(allGt.begin() + start, allGt.begin() + std::min(allGt.size(), start + poses.size()));
      const auto ate = sdv::absoluteTrajectoryError(gt, poses, false);
      const auto sim = sdv::absoluteTrajectoryError(gt, poses, true);
      std::vector<Sophus::SE3d> aligned = poses;
      for (auto& p : aligned) p = ate.alignment.applyToPose(p);
      const auto seg = sdv::segmentDriftError(gt, aligned);
      spdlog::info("ATE SE3 rmse {:.3f} m, max {:.3f} m | Sim3 rmse {:.3f} m, scale {:.3f} | drift t {:.2f} % "
                   "r {:.3f} deg/100m ({} segments)",
                   ate.rmse, ate.max, sim.rmse, sim.alignment.scale, seg.translationPercent, seg.rotationDegPer100m,
                   seg.numSegments);
      scene.addTrajectory(gt, {0, 200, 0});
    }
    if (!plyFile.empty()) {
      for (const auto& p : opt.points()) {
        const Sophus::SE3d T_w_h = opt.frame(p.host).params.T_c_w.inverse();
        const auto v = static_cast<uint8_t>(std::clamp(p.pattern.intensities[0], 0.f, 255.f));
        scene.addPoint(T_w_h * (p.bearing / p.rho), {v, v, v});
      }
      scene.write(plyFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
