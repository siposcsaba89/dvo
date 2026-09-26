// Validates monocular initialisation: motion against GT, depth against stereo after scale alignment.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <numbers>
#include <string>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/mono_initializer.h>
#include <sdv/stereo_depth.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string sequenceDir, gtFile, plyFile;
  size_t start = 0, maxFrames = 30;
  int levels = 5, numPoints = 2000, minFrames = 2, firstCandidates = 2;
  double maxSigma = 0.02, minFlow = 6.0;
  bool verbose = false, seedStereo = false;

  po::options_description desc("mono_init options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir)->required(), "KITTI sequence directory")
      ("gt", po::value(&gtFile)->required(), "ground-truth poses")
      ("start", po::value(&start)->default_value(0), "first frame")
      ("max-frames,n", po::value(&maxFrames)->default_value(30), "give up after N frames")
      ("levels", po::value(&levels)->default_value(5), "pyramid levels")
      ("points", po::value(&numPoints)->default_value(2000), "points on level 0")
      ("min-frames", po::value(&minFrames)->default_value(2), "minimum frames (incl. host) before initialising")
      ("first-candidates", po::value(&firstCandidates)->default_value(2), "first-frame starts optimised fully")
      ("min-flow", po::value(&minFlow)->default_value(6.0), "translation flow (px) required to initialise")
      ("max-sigma", po::value(&maxSigma)->default_value(0.02), "relative depth sigma for evaluated points")
      ("ply", po::value(&plyFile), "write metric-scaled points coloured by error vs stereo")
      ("seed-stereo", po::bool_switch(&seedStereo), "start from stereo depth (diagnostic)")
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
    const auto gt = sdv::loadKittiPoses(gtFile);
    const sdv::Camera& cam = seq.camera();

    sdv::MonoInitSettings settings;
    settings.pointsLevel0 = numPoints;
    settings.minTranslationFlow = minFlow;
    settings.minFrames = minFrames;
    settings.firstFrameCandidates = firstCandidates;
    sdv::MonoInitializer init(cam, levels, settings);

    size_t host = start, frame = start;
    sdv::MonoInitResult res;
    double totalMs = 0;
    using Clock = std::chrono::steady_clock;
    for (; frame < std::min(start + maxFrames, seq.size()); ++frame) {
      const sdv::ImagePyramid pyr(sdv::toFloatGray(seq.loadImage(frame, 0)), levels);
      const auto t0 = Clock::now();
      if (frame == start) {
        const cv::Mat gray = seq.loadImage(frame, 0);
        init.reset(pyr, seedStereo ? sdv::stereoInverseDistance(gray, seq.loadImage(frame, 1), cam, seq.baseline())
                                   : cv::Mat());
        totalMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        spdlog::info("host {}: points per level {} {} {}", frame, init.numPoints(0), init.numPoints(1),
                     init.numPoints(levels - 1));
        continue;
      }
      res = init.addFrame(pyr);
      const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
      totalMs += ms;
      const Sophus::SE3d T_gt = gt[frame].inverse() * gt[host];
      spdlog::info("frame {}: {:.1f} ms, inliers {:.2f}, visible {:.2f}, rmse {:.2f}, translation flow {:.1f} px, rotation {:.2f} deg "
                   "(GT {:.2f}){}",
                   frame, ms, res.inlierRatio, res.visibleRatio, res.rmse, res.translationFlow,
                   res.T_t_h.so3().log().norm() * 180.0 / std::numbers::pi,
                   T_gt.so3().log().norm() * 180.0 / std::numbers::pi, res.reset ? " -> reset" : "");
      if (res.reset) host = frame;
      if (res.initialized) break;
    }
    if (!res.initialized) {
      spdlog::error("not initialised after {} frames", maxFrames);
      return EXIT_FAILURE;
    }
    spdlog::info("initialised with host {} and frame {} ({} frames, {:.0f} ms total)", host, frame, res.numFrames,
                 totalMs);

    const Sophus::SE3d T_gt = gt[frame].inverse() * gt[host];
    const double rotErr = (res.T_t_h.so3() * T_gt.so3().inverse()).log().norm() * 180.0 / std::numbers::pi;
    const double dirErr = std::acos(std::clamp(
                              res.T_t_h.translation().normalized().dot(T_gt.translation().normalized()), -1.0, 1.0)) *
                          180.0 / std::numbers::pi;
    const double toMetricGt = res.T_t_h.translation().norm() / T_gt.translation().norm();
    spdlog::info("motion vs GT: rotation error {:.3f} deg, translation direction error {:.2f} deg, |t_gt| {:.2f} m",
                 rotErr, dirErr, T_gt.translation().norm());

    const cv::Mat hostGray = seq.loadImage(host, 0);
    const cv::Mat stereoRho = sdv::stereoInverseDistance(hostGray, seq.loadImage(host, 1), cam, seq.baseline());
    const auto points = init.points();
    std::vector<double> ratios;
    for (const auto& p : points) {
      if (p.relativeSigma > maxSigma) continue;
      const float rs = stereoRho.at<float>(static_cast<int>(p.uv.y()), static_cast<int>(p.uv.x()));
      if (rs > 0) ratios.push_back(rs / p.rho);
    }
    const size_t numGood = std::count_if(points.begin(), points.end(),
                                         [&](const auto& p) { return p.relativeSigma <= maxSigma; });
    spdlog::info("{} / {} points with relative sigma <= {}, {} with stereo reference", numGood, points.size(),
                 maxSigma, ratios.size());
    if (ratios.empty()) return EXIT_FAILURE;
    std::vector<double> sorted = ratios;
    std::nth_element(sorted.begin(), sorted.begin() + sorted.size() / 2, sorted.end());
    const double toMetricStereo = sorted[sorted.size() / 2];
    spdlog::info("scale (rho_stereo / rho_est): median {:.4f}, from GT translation {:.4f}", toMetricStereo,
                 toMetricGt);

    std::vector<double> errors;
    for (double r : ratios) errors.push_back(std::abs(toMetricStereo / r - 1.0));
    std::sort(errors.begin(), errors.end());
    auto pct = [&](double thr) {
      return 100.0 * (std::upper_bound(errors.begin(), errors.end(), thr) - errors.begin()) / errors.size();
    };
    spdlog::info("depth error vs stereo: median {:.1f} %, <5 % {:.1f} %, <10 % {:.1f} %, <20 % {:.1f} %",
                 100.0 * errors[errors.size() / 2], pct(0.05), pct(0.10), pct(0.20));

    if (!plyFile.empty()) {
      sdv::PlyScene scene;
      for (const auto& p : points) {
        if (p.relativeSigma > maxSigma) continue;
        const double rho = p.rho * toMetricStereo;
        const float rs = stereoRho.at<float>(static_cast<int>(p.uv.y()), static_cast<int>(p.uv.x()));
        const double err = rs > 0 ? std::abs(rs / rho - 1.0) : -1.0;
        const sdv::Rgb color = err < 0      ? sdv::Rgb{128, 128, 255}
                               : err < 0.05 ? sdv::Rgb{0, 200, 0}
                               : err < 0.15 ? sdv::Rgb{230, 200, 0}
                                            : sdv::Rgb{230, 0, 0};
        scene.addPoint(gt[host] * (p.bearing / rho), color);
      }
      scene.addCameraAxes({gt[host], gt[frame]}, 1.0, 1);
      scene.write(plyFile);
      spdlog::info("wrote {}", plyFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
