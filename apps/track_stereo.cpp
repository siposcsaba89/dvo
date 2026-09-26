// Validates frame tracking in isolation: keyframe depth comes from stereo instead of the VO back end.
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <optional>
#include <string>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/image_pyramid.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/stereo_depth.h>
#include <sdv/tracker.h>

namespace po = boost::program_options;

namespace {

struct Keyframe {
  size_t index;
  Sophus::SE3d T_w_c;
  std::optional<sdv::ReferenceFrame> ref;
};

void selectPixels(const sdv::ImageLevel& img, const cv::Mat& rho, int block, float minGrad,
                  std::vector<Eigen::Vector2i>& pixels, std::vector<double>& rhos) {
  pixels.clear();
  rhos.clear();
  const float minGradSq = minGrad * minGrad;
  for (int by = 0; by + block <= img.height; by += block) {
    for (int bx = 0; bx + block <= img.width; bx += block) {
      float bestGrad = minGradSq;
      Eigen::Vector2i best(-1, -1);
      for (int v = by; v < by + block; ++v)
        for (int u = bx; u < bx + block; ++u) {
          const float g = img.gradNormSq[v * img.width + u];
          if (g > bestGrad && rho.at<float>(v, u) > 0) {
            bestGrad = g;
            best = {u, v};
          }
        }
      if (best.x() < 0) continue;
      pixels.push_back(best);
      rhos.push_back(rho.at<float>(best.y(), best.x()));
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string sequenceDir, gtFile, outFile, plyFile;
  size_t maxFrames = 0;
  int levels = 5, keyframeInterval = 5, block = 8;
  float minGrad = 6.f;
  bool verbose = false;

  po::options_description desc("track_stereo options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir)->required(), "KITTI sequence directory")
      ("gt", po::value(&gtFile), "ground-truth poses for evaluation")
      ("out,o", po::value(&outFile), "write estimated poses (KITTI format)")
      ("ply", po::value(&plyFile), "write trajectories as PLY")
      ("max-frames,n", po::value(&maxFrames)->default_value(0), "process at most N frames (0 = all)")
      ("levels", po::value(&levels)->default_value(5), "pyramid levels")
      ("kf-interval", po::value(&keyframeInterval)->default_value(5), "new keyframe every N frames")
      ("block", po::value(&block)->default_value(8), "pixel selection block size")
      ("min-grad", po::value(&minGrad)->default_value(6.f), "minimum gradient for selected pixels")
      ("verbose,v", po::bool_switch(&verbose), "debug logging");
  po::positional_options_description pos;
  pos.add("sequence", 1);
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).options(desc).positional(pos).run(), vm);
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
    const size_t n = maxFrames == 0 ? seq.size() : std::min(maxFrames, seq.size());
    const sdv::TrackingSettings settings;
    const sdv::FrameTracker tracker(settings);

    std::vector<Sophus::SE3d> poses;  // T_w_c
    poses.reserve(n);
    Keyframe kf;
    std::vector<Eigen::Vector2i> pixels;
    std::vector<double> rhos;
    sdv::AffineBrightness affine;
    Sophus::SE3d T_prev_ref, T_prev_prevprev;
    double trackMs = 0, kfMs = 0;
    int failures = 0;

    using Clock = std::chrono::steady_clock;
    for (size_t i = 0; i < n; ++i) {
      const cv::Mat left = seq.loadImage(i, 0);
      const sdv::ImagePyramid pyr(sdv::toFloatGray(left), levels);

      if (i == 0) {
        poses.emplace_back();
      } else {
        const auto t0 = Clock::now();
        const auto hypotheses = sdv::makeMotionHypotheses(T_prev_ref, T_prev_prevprev);
        const auto res = tracker.track(*kf.ref, pyr, hypotheses, affine);
        trackMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (!res.ok) {
          ++failures;
          spdlog::warn("frame {}: tracking weak (inliers {:.2f}, rmse {:.2f})", i, res.inlierRatio, res.rmse);
        }
        const Sophus::SE3d T_w_c = kf.T_w_c * res.T_t_h.inverse();
        T_prev_prevprev = res.T_t_h * T_prev_ref.inverse();
        T_prev_ref = res.T_t_h;
        affine = res.affine;
        poses.push_back(T_w_c);
        spdlog::debug("frame {}: rmse {:.2f} inliers {:.2f} flow {:.1f} hyp {} a {:.3f} b {:.2f}", i, res.rmse,
                      res.inlierRatio, res.meanFlow, res.hypothesis, res.affine.a, res.affine.b);
      }

      if (i % keyframeInterval == 0) {
        const auto t0 = Clock::now();
        const cv::Mat rho = sdv::stereoInverseDistance(left, seq.loadImage(i, 1), cam, seq.baseline());
        selectPixels(pyr.level(0), rho, block, minGrad, pixels, rhos);
        const sdv::AffineBrightness kfAffine = i == 0 ? sdv::AffineBrightness{} : affine;
        kf.index = i;
        kf.T_w_c = poses.back();
        kf.ref.emplace(cam, pyr, pixels, rhos, kfAffine, settings.gradientWeightC);
        T_prev_ref = Sophus::SE3d();
        affine = kfAffine;
        kfMs += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        spdlog::debug("keyframe {}: {} points", i, pixels.size());
      }
      if (i % 500 == 0) spdlog::info("frame {}/{}", i, n);
    }
    spdlog::info("tracking {:.2f} ms/frame, keyframe creation {:.1f} ms/kf, weak frames {}",
                 trackMs / std::max<size_t>(n - 1, 1), kfMs / ((n + keyframeInterval - 1) / keyframeInterval),
                 failures);

    if (!outFile.empty()) sdv::saveKittiPoses(outFile, poses);
    sdv::PlyScene scene;
    scene.addTrajectory(poses, {220, 0, 0});
    if (!gtFile.empty()) {
      auto gt = sdv::loadKittiPoses(gtFile);
      gt.resize(std::min(gt.size(), poses.size()));
      const auto ate = sdv::absoluteTrajectoryError(gt, poses, false);
      std::vector<Sophus::SE3d> aligned = poses;
      for (auto& p : aligned) p = ate.alignment.applyToPose(p);
      const auto seg = sdv::segmentDriftError(gt, aligned);
      spdlog::info("ATE (SE3) rmse {:.3f} m, max {:.3f} m | drift t {:.2f} % r {:.3f} deg/100m ({} segments)",
                   ate.rmse, ate.max, seg.translationPercent, seg.rotationDegPer100m, seg.numSegments);
      scene.addTrajectory(gt, {0, 200, 0});
    }
    if (!plyFile.empty()) scene.write(plyFile);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
