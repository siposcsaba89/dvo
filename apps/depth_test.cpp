// Validates candidate selection and immature-point depth estimation using GT poses and stereo depth.
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <map>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <sdv/immature_point.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/point_selector.h>
#include <sdv/stereo_depth.h>
#include <sdv/tracker.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string sequenceDir, gtFile, plyFile, pngFile;
  size_t hostIndex = 50;
  int numTargets = 8, numPoints = 2000;
  double convergedPixels = 1.0;
  bool useGt = false;

  po::options_description desc("depth_test options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir)->required(), "KITTI sequence directory")
      ("gt", po::value(&gtFile)->required(), "ground-truth poses")
      ("host", po::value(&hostIndex)->default_value(50), "host frame index")
      ("targets", po::value(&numTargets)->default_value(8), "number of following frames to trace in")
      ("points", po::value(&numPoints)->default_value(2000), "target number of candidates")
      ("converged-px", po::value(&convergedPixels)->default_value(1.0), "max matching error for 'converged'")
      ("use-gt", po::bool_switch(&useGt), "trace with GT poses instead of tracked poses (brightness fixed)")
      ("ply", po::value(&plyFile), "write estimated (coloured by error) and stereo points")
      ("png", po::value(&pngFile), "write host image with selected candidates");
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

  try {
    const sdv::KittiSequence seq(sequenceDir, 16);
    const auto gt = sdv::loadKittiPoses(gtFile);
    const sdv::Camera& cam = seq.camera();

    const cv::Mat hostGray = seq.loadImage(hostIndex, 0);
    const sdv::ImagePyramid host(sdv::toFloatGray(hostGray), 5);
    const cv::Mat stereoRho = sdv::stereoInverseDistance(hostGray, seq.loadImage(hostIndex, 1), cam, seq.baseline());

    sdv::PointSelectorSettings selSettings;
    selSettings.targetPoints = numPoints;
    auto t0 = std::chrono::steady_clock::now();
    const auto candidates = sdv::PointSelector(selSettings).select(host.level(0));
    const double selectMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::map<int, int> perPass;
    for (const auto& c : candidates) ++perPass[c.pass];
    spdlog::info("selected {} candidates in {:.1f} ms (pass0 {}, pass1 {}, pass2 {})", candidates.size(), selectMs,
                 perPass[0], perPass[1], perPass[2]);

    const sdv::TraceSettings settings;
    std::vector<sdv::ImmaturePoint> points;
    for (const auto& c : candidates)
      if (auto p = sdv::ImmaturePoint::create(cam, host.level(0), c.uv.cast<double>(), settings)) points.push_back(*p);

    std::vector<Eigen::Vector2i> refPixels;
    std::vector<double> refRho;
    for (int v = 0; v < stereoRho.rows; v += 4)
      for (int u = 0; u < stereoRho.cols; u += 4)
        if (stereoRho.at<float>(v, u) > 0 && host.level(0).gradNormSq[v * cam.width + u] > 36.f) {
          refPixels.emplace_back(u, v);
          refRho.push_back(stereoRho.at<float>(v, u));
        }
    const sdv::TrackingSettings trackSettings;
    const sdv::ReferenceFrame ref(cam, host, refPixels, refRho, {}, trackSettings.gradientWeightC);
    const sdv::FrameTracker tracker(trackSettings);
    Sophus::SE3d T_prev_h, T_vel;
    sdv::HostTargetState state;

    double traceMs = 0;
    for (int k = 1; k <= numTargets; ++k) {
      const size_t t = hostIndex + k;
      const sdv::ImagePyramid target(sdv::toFloatGray(seq.loadImage(t, 0)), 5);
      const Sophus::SE3d T_gt = gt[t].inverse() * gt[hostIndex];
      if (useGt) {
        state.T_t_h = T_gt;
      } else {
        const auto res = tracker.track(ref, target, sdv::makeMotionHypotheses(T_prev_h, T_vel), state.target);
        T_vel = res.T_t_h * T_prev_h.inverse();
        T_prev_h = res.T_t_h;
        state.T_t_h = res.T_t_h;
        state.target = res.affine[0];
        spdlog::debug("tracked vs gt: {:.3f} m", (res.T_t_h * T_gt.inverse()).translation().norm());
      }
      std::map<sdv::TraceStatus, int> status;
      t0 = std::chrono::steady_clock::now();
      for (auto& p : points) ++status[p.trace(cam, target.level(0), state, settings)];
      traceMs += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      std::string summary;
      for (const auto& [s, n] : status) summary += fmt::format("{} {}  ", sdv::toString(s), n);
      spdlog::info("target {}: {}", t, summary);
    }
    spdlog::info("tracing {:.1f} ms per frame for {} points", traceMs / numTargets, points.size());

    std::vector<double> relErrors;
    int converged = 0, withStereo = 0, contained = 0;
    sdv::PlyScene scene;
    const Sophus::SE3d& T_w_h = gt[hostIndex];
    for (const auto& p : points) {
      // rho = 0 is a valid match at infinity but has no finite position.
      if (p.numGood() == 0 || p.lastErrorPixels() > convergedPixels || p.rho() <= 1e-6) continue;
      ++converged;
      const Eigen::Vector2i uv = p.pattern().uv.cast<int>();
      const float rs = stereoRho.at<float>(uv.y(), uv.x());
      double err = -1;
      if (rs > 0) {
        ++withStereo;
        err = std::abs(1.0 / p.rho() - 1.0 / rs) * rs;
        relErrors.push_back(err);
        contained += rs >= p.rhoMin() && rs <= p.rhoMax();
      }
      const sdv::Rgb color = err < 0 ? sdv::Rgb{128, 128, 255}
                             : err < 0.05 ? sdv::Rgb{0, 200, 0}
                             : err < 0.15 ? sdv::Rgb{230, 200, 0}
                                          : sdv::Rgb{230, 0, 0};
      scene.addPoint(T_w_h * (p.bearing() / p.rho()), color);
    }
    std::sort(relErrors.begin(), relErrors.end());
    auto pct = [&](double thr) {
      return 100.0 * (std::upper_bound(relErrors.begin(), relErrors.end(), thr) - relErrors.begin()) /
             std::max<size_t>(relErrors.size(), 1);
    };
    spdlog::info("converged {} / {} ({:.1f} %), with stereo reference {}", converged, points.size(),
                 100.0 * converged / std::max<size_t>(points.size(), 1), withStereo);
    if (!relErrors.empty())
      spdlog::info("depth error vs stereo: median {:.1f} %, <5 % {:.1f} %, <10 % {:.1f} %, <20 % {:.1f} %, "
                   "stereo inside interval {:.1f} %",
                   100.0 * relErrors[relErrors.size() / 2], pct(0.05), pct(0.10), pct(0.20),
                   100.0 * contained / withStereo);

    if (!plyFile.empty()) {
      for (int v = 0; v < stereoRho.rows; v += 4)
        for (int u = 0; u < stereoRho.cols; u += 4) {
          const float rs = stereoRho.at<float>(v, u);
          Eigen::Vector3d b;
          if (rs <= 0 || !cam.unproject(Eigen::Vector2d(u, v), b)) continue;
          const uint8_t g = hostGray.at<uint8_t>(v, u) / 2;
          scene.addPoint(T_w_h * (b / rs), {g, g, g});
        }
      scene.addCameraAxes({T_w_h}, 1.0, 1);
      scene.write(plyFile);
      spdlog::info("wrote {}", plyFile);
    }
    if (!pngFile.empty()) {
      cv::Mat vis;
      cv::cvtColor(hostGray, vis, cv::COLOR_GRAY2BGR);
      const cv::Scalar colors[3] = {{0, 0, 255}, {0, 200, 0}, {255, 128, 0}};
      for (const auto& c : candidates) cv::circle(vis, {c.uv.x(), c.uv.y()}, 1, colors[std::min(c.pass, 2)], -1);
      cv::imwrite(pngFile, vis);
      spdlog::info("wrote {}", pngFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
