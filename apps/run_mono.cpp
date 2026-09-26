// Full monocular pipeline on a KITTI sequence; evaluation after Sim3 alignment (scale is unobservable).
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/odometry.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string sequenceDir, gtFile, outFile, plyFile;
  size_t start = 0, maxFrames = 100;
  bool verbose = false;
  sdv::OdometrySettings settings;

  po::options_description desc("run_mono options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir)->required(), "KITTI sequence directory")
      ("gt", po::value(&gtFile), "ground-truth poses for evaluation")
      ("out,o", po::value(&outFile), "write estimated poses (KITTI format, frames with a pose)")
      ("ply", po::value(&plyFile), "write trajectory, GT and map points (Sim3-aligned to GT if given)")
      ("start", po::value(&start)->default_value(0), "first frame")
      ("max-frames,n", po::value(&maxFrames)->default_value(100), "number of frames")
      ("keyframes", po::value(&settings.maxKeyframes)->default_value(7), "keyframes in the window")
      ("points", po::value(&settings.targetActivePoints)->default_value(2000), "target active points")
      ("candidates", po::value(&settings.candidatesPerKeyframe)->default_value(1500), "candidates per keyframe")
      ("kf-flow", po::value(&settings.kfFlow)->default_value(settings.kfFlow), "keyframe flow scale (px)")
      ("kf-tflow", po::value(&settings.kfTranslationFlow)->default_value(settings.kfTranslationFlow),
       "keyframe translation flow scale (px)")
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
    const sdv::KittiSequence seq(sequenceDir, 1 << (settings.levels - 1));
    const size_t n = std::min(maxFrames, seq.size() - start);
    sdv::MonoOdometry vo(seq.camera(), settings);

    using Clock = std::chrono::steady_clock;
    double totalMs = 0, maxMs = 0;
    int keyframes = 0, weak = 0;
    for (size_t i = 0; i < n; ++i) {
      const cv::Mat image = seq.loadImage(start + i, 0);
      const auto t0 = Clock::now();
      const sdv::OdometryFrameInfo info = vo.addFrame(image);
      const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      totalMs += ms;
      maxMs = std::max(maxMs, ms);
      keyframes += info.keyframe;
      weak += info.initialized && !info.trackingOk;
      spdlog::debug("frame {}: {:.0f} ms{}{} rmse {:.2f}, active {}, candidates {}", start + i, ms,
                    info.keyframe ? " KF" : "", info.initialized && !info.trackingOk ? " WEAK" : "", info.rmse,
                    info.activePoints, info.immaturePoints);
    }
    spdlog::info("{} frames: {:.0f} ms/frame (max {:.0f}), {} keyframes, {} weak tracking frames", n, totalMs / n,
                 maxMs, keyframes, weak);

    const auto poses = vo.poses();
    std::vector<Sophus::SE3d> est, gt;
    const auto allGt = gtFile.empty() ? std::vector<Sophus::SE3d>{} : sdv::loadKittiPoses(gtFile);
    for (size_t i = 0; i < poses.size(); ++i) {
      if (!poses[i]) continue;
      est.push_back(*poses[i]);
      if (!allGt.empty()) gt.push_back(allGt[start + i]);
    }
    spdlog::info("{} of {} frames have a pose", est.size(), poses.size());
    if (!outFile.empty()) sdv::saveKittiPoses(outFile, est);

    sdv::SimilarityTransform alignment;
    if (!gt.empty() && est.size() > 2) {
      const auto ate = sdv::absoluteTrajectoryError(gt, est, true);
      alignment = ate.alignment;
      std::vector<Sophus::SE3d> aligned = est;
      for (auto& p : aligned) p = alignment.applyToPose(p);
      const auto seg = sdv::segmentDriftError(gt, aligned);
      double length = 0;
      for (size_t i = 1; i < gt.size(); ++i) length += (gt[i].translation() - gt[i - 1].translation()).norm();
      std::string localScale;
      for (size_t b = 0; b + 50 < est.size(); b += 50) {
        const double g = (gt[b + 50].translation() - gt[b].translation()).norm();
        const double e = (est[b + 50].translation() - est[b].translation()).norm();
        if (e > 0) localScale += fmt::format(" {:.2f}", g / e / alignment.scale);
      }
      spdlog::info("local scale per 50 frames relative to global:{}", localScale);
      spdlog::info("ATE Sim3 rmse {:.3f} m, max {:.3f} m, scale {:.3f} over {:.0f} m | drift t {:.2f} % "
                   "r {:.3f} deg/100m ({} segments)",
                   ate.rmse, ate.max, alignment.scale, length, seg.translationPercent, seg.rotationDegPer100m,
                   seg.numSegments);
    }

    if (!plyFile.empty()) {
      sdv::PlyScene scene;
      std::vector<Sophus::SE3d> aligned = est;
      for (auto& p : aligned) p = alignment.applyToPose(p);
      scene.addTrajectory(aligned, {220, 0, 0});
      if (!gt.empty()) scene.addTrajectory(gt, {0, 200, 0});
      for (const auto& p : vo.mapPoints()) {
        const auto v = static_cast<uint8_t>(std::clamp(p.intensity, 0.f, 255.f));
        scene.addPoint(alignment.apply(p.position), {v, v, v});
      }
      scene.write(plyFile);
      spdlog::info("wrote {} ({} map points)", plyFile, vo.mapPoints().size());
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
