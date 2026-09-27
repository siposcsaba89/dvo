// Direct sparse odometry on a KITTI sequence, a video, an image folder or a multi-camera rig (one video per
// camera). One camera: monocular, Sim3 evaluation against ground truth; KITTI stereo or a rig: metric scale.
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/io/camera_config.h>
#include <sdv/io/colmap.h>
#include <sdv/io/frame_source.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/io/rig_config.h>
#include <sdv/odometry.h>
#include <sdv/undistort.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string sequenceDir, videoFile, imageDir, rigFile, cameraFile, gtFile, outFile, plyFile, jsonFile, pngFile,
      colmapDir;
  std::vector<std::string> rigCameras;
  bool colmapKeyframesOnly = false, colmapAlign = false;
  double focalScale = 1.0, maxDistanceFactor = 5.0, maxDistance = 0.0, scale = 1.0, maxDepthSigma = 0.0,
         neighbourRadius = 0.2;
  int minObservations = 1, minNeighbours = 0;
  std::optional<double> camAlpha;
  size_t start = 0, stride = 1, maxFrames = 0;
  bool verbose = false, trace = false, stereo = false, checkCalibration = false;
  sdv::OdometrySettings settings;

  po::options_description desc("run_vo options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir), "input: KITTI odometry sequence directory")
      ("video", po::value(&videoFile), "input: video file (needs --camera)")
      ("images", po::value(&imageDir), "input: directory of images in file-name order (needs --camera)")
      ("rig", po::value(&rigFile), "input: rig YAML with cameras, extrinsics and one video per camera")
      ("rig-cameras", po::value(&rigCameras)->multitoken(), "use only these rig cameras (names)")
      ("camera", po::value(&cameraFile),
       "camera YAML (width, height, fx, fy, cx, cy, alpha, beta, optional mask); replaces the KITTI calibration")
      ("scale", po::value(&scale)->default_value(1.0), "resize input images (and camera) by this factor")
      ("start", po::value(&start)->default_value(0), "first input frame")
      ("stride", po::value(&stride)->default_value(1), "use every n-th input frame")
      ("max-frames,n", po::value(&maxFrames)->default_value(0), "number of frames to process (0 = all)")
      ("stereo", po::bool_switch(&stereo), "KITTI: use the right camera (image_1): metric scale")
      ("gt", po::value(&gtFile), "ground-truth poses for evaluation (KITTI format, one per input frame)")
      ("out,o", po::value(&outFile), "write estimated poses (KITTI format, frames with a pose)")
      ("ply", po::value(&plyFile), "write trajectory, GT and map points (Sim3-aligned to GT if given)")
      ("colmap", po::value(&colmapDir), "write a COLMAP text model (sparse/0) and images to this directory")
      ("colmap-keyframes", po::bool_switch(&colmapKeyframesOnly), "export keyframes only")
      ("colmap-align", po::bool_switch(&colmapAlign), "export in the GT-aligned frame (metres)")
      ("focal-scale", po::value(&focalScale)->default_value(1.0), "virtual pinhole focal scale for fisheye export")
      ("max-distance-factor", po::value(&maxDistanceFactor)->default_value(5.0),
       "drop map points farther than this times the median distance from their camera (0 = off)")
      ("max-distance", po::value(&maxDistance)->default_value(0.0),
       "drop map points farther than this from their camera, metres after alignment (0 = off)")
      ("min-observations", po::value(&minObservations)->default_value(1),
       "drop map points with fewer good residuals (other cameras and keyframes)")
      ("max-depth-sigma", po::value(&maxDepthSigma)->default_value(0.0),
       "drop map points whose relative inverse-depth sigma (unit photometric noise) exceeds this (0 = off)")
      ("min-neighbours", po::value(&minNeighbours)->default_value(0),
       "drop map points with fewer neighbours within --neighbour-radius (0 = off)")
      ("neighbour-radius", po::value(&neighbourRadius)->default_value(0.2), "neighbour radius, metres after alignment")
      ("json", po::value(&jsonFile), "write trajectory, GT, keyframes and map points as JSON (viewer data)")
      ("png", po::value(&pngFile), "write a top-down preview image")
      ("keyframes", po::value(&settings.maxKeyframes)->default_value(7), "keyframes in the window")
      ("points", po::value(&settings.targetActivePoints)->default_value(2000), "target active points")
      ("candidates", po::value(&settings.candidatesPerKeyframe)->default_value(1500), "candidates per keyframe")
      ("kf-flow", po::value(&settings.kfFlow)->default_value(settings.kfFlow), "keyframe flow scale (px)")
      ("kf-tflow", po::value(&settings.kfTranslationFlow)->default_value(settings.kfTranslationFlow),
       "keyframe translation flow scale (px)")
      ("stereo-weight", po::value(&settings.window.stereoWeight)->default_value(settings.window.stereoWeight),
       "weight of static residuals between cameras of one keyframe")
      ("ba-iterations", po::value(&settings.windowIterations)->default_value(settings.windowIterations),
       "window BA iterations per keyframe")
      ("cam-alpha", po::value<double>()->notifier([&](double a) { camAlpha = a; }),
       "override the EUCM alpha of the camera(s); a small negative value corrects residual pincushion distortion "
       "of rectified images (KITTI 00: -0.03)")
      ("check-calibration", po::bool_switch(&checkCalibration),
       "several cameras: log the temporal vs static depth bias per image radius at every keyframe")
      ("trace", po::bool_switch(&trace), "trace logging (implies verbose)")
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
  if (verbose || trace) spdlog::set_level(trace ? spdlog::level::trace : spdlog::level::debug);
  settings.checkCalibration = checkCalibration;

  try {
    if (!sequenceDir.empty() + !videoFile.empty() + !imageDir.empty() + !rigFile.empty() != 1)
      throw std::invalid_argument("give exactly one input: --sequence, --video, --images or --rig");
    if (stereo && sequenceDir.empty()) throw std::invalid_argument("--stereo needs a KITTI sequence");

    // One entry per rig camera: prepared camera, extrinsic and input stream.
    struct InputCamera {
      std::string name;
      sdv::Camera camera;
      Sophus::SE3d T_c_b;
      std::unique_ptr<sdv::SubsampledSource> source;
    };
    std::vector<InputCamera> inputs;
    const int sizeMultiple = 1 << (settings.levels - 1);
    auto prepare = [&](sdv::CameraConfig config) {
      if (camAlpha) config.camera.alpha = *camAlpha;
      return sdv::prepareCamera(config, scale, sizeMultiple);
    };
    std::optional<sdv::KittiSequence> kitti;
    if (!rigFile.empty()) {
      const sdv::RigConfig rigConfig = sdv::loadRigConfig(rigFile);
      for (const auto& c : rigConfig.cameras) {
        if (!rigCameras.empty() && std::find(rigCameras.begin(), rigCameras.end(), c.name) == rigCameras.end()) continue;
        if (c.video.empty()) throw std::invalid_argument("rig camera " + c.name + " has no video");
        inputs.push_back({c.name, prepare(c.camera), c.T_b_c.inverse(),
                          std::make_unique<sdv::SubsampledSource>(std::make_unique<sdv::VideoSource>(c.video),
                                                                  start + c.frameOffset, stride)});
      }
      if (inputs.empty()) throw std::invalid_argument("no rig camera selected");
    } else {
      sdv::CameraConfig config;
      std::unique_ptr<sdv::FrameSource> source;
      if (!sequenceDir.empty()) {
        kitti.emplace(sequenceDir, 1);
        config.camera = kitti->camera();
        source = std::make_unique<sdv::KittiSource>(*kitti, 0);
      } else if (!videoFile.empty()) {
        source = std::make_unique<sdv::VideoSource>(videoFile);
      } else {
        source = std::make_unique<sdv::ImageFolderSource>(imageDir);
      }
      if (!cameraFile.empty()) config = sdv::loadCameraConfig(cameraFile);
      else if (!kitti) throw std::invalid_argument("--video and --images need --camera");
      const sdv::Camera camera = prepare(config);
      inputs.push_back({"cam0", camera, Sophus::SE3d(), std::make_unique<sdv::SubsampledSource>(std::move(source), start, stride)});
      if (stereo) {
        // KITTI: the body frame is the left camera frame, so poses stay camera-0 poses for the ground truth.
        inputs.push_back({"cam1", camera, Sophus::SE3d(Sophus::SO3d(), Eigen::Vector3d(-kitti->baseline(), 0, 0)),
                          std::make_unique<sdv::SubsampledSource>(std::make_unique<sdv::KittiSource>(*kitti, 1), start,
                                                                  stride)});
      }
    }
    sdv::Rig rig;
    for (const auto& in : inputs) {
      const sdv::Camera& c = in.camera;
      spdlog::info("camera {}: {}x{} fx {:.2f} fy {:.2f} cx {:.2f} cy {:.2f} alpha {:.4f} beta {:.4f}{}", in.name,
                   c.width, c.height, c.fx, c.fy, c.cx, c.cy, c.alpha, c.beta,
                   c.mask ? fmt::format(", mask {:.0f} % valid", 100 * c.mask->validFraction(0)) : "");
      rig.cameras.push_back(c);
      rig.T_c_b.push_back(in.T_c_b);
    }
    const bool multiCamera = rig.size() > 1;
    auto frameIndex = [&](size_t i) { return start + i * stride; };
    sdv::Odometry vo(rig, settings);

    using Clock = std::chrono::steady_clock;
    double totalMs = 0, maxMs = 0, waitMs = 0;
    int keyframes = 0, weak = 0;
    size_t n = 0;
    // Decoding and resizing the next frame overlaps with processing the current one. Empty at the end of any stream.
    auto readFrame = [&] {
      std::vector<cv::Mat> images;
      for (const auto& in : inputs) {
        const cv::Mat input = in.source->next();
        if (input.empty()) return std::vector<cv::Mat>{};
        images.push_back(sdv::prepareImage(input, scale, in.camera));
      }
      return images;
    };
    const auto tStart = Clock::now();
    {
      std::future<std::vector<cv::Mat>> pending = std::async(std::launch::async, readFrame);
      for (; maxFrames == 0 || n < maxFrames; ++n) {
        const auto tIn = Clock::now();
        const std::vector<cv::Mat> frame = pending.get();
        if (frame.empty()) break;
        if (maxFrames == 0 || n + 1 < maxFrames) pending = std::async(std::launch::async, readFrame);
        const auto t0 = Clock::now();
        waitMs += std::chrono::duration<double, std::milli>(t0 - tIn).count();
        const sdv::OdometryFrameInfo info = vo.addFrame(frame);
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        totalMs += ms;
        maxMs = std::max(maxMs, ms);
        keyframes += info.keyframe;
        weak += info.initialized && !info.trackingOk;
        spdlog::debug("frame {}: {:.0f} ms{}{} rmse {:.2f}, active {}, candidates {}", frameIndex(n), ms,
                      info.keyframe ? " KF" : "", info.initialized && !info.trackingOk ? " WEAK" : "", info.rmse,
                      info.activePoints, info.immaturePoints);
        if (!verbose && n % 100 == 99) spdlog::info("{} frames, {:.0f} ms/frame", n + 1, totalMs / (n + 1));
      }
    }
    if (n == 0) throw std::runtime_error("no frames read");
    const double wallMs = std::chrono::duration<double, std::milli>(Clock::now() - tStart).count();
    spdlog::info("{} frames: {:.0f} ms/frame (max {:.0f}), {} keyframes, {} weak tracking frames", n, totalMs / n,
                 maxMs, keyframes, weak);
    spdlog::info("wall time {:.1f} ms/frame, of which {:.1f} ms waiting for input", wallMs / n, waitMs / n);
    for (const auto& [stage, e] : vo.profile().entries())
      spdlog::info("  {:<18} {:6.1f} ms/frame  {:7.1f} ms/call  {:5} calls", stage, e.ms / n, e.ms / e.calls, e.calls);

    const auto poses = vo.poses();
    std::vector<Sophus::SE3d> est, gt;
    std::vector<int> estFrame;  // input frame (0-based within the run) of each estimate
    const auto allGt = gtFile.empty() ? std::vector<Sophus::SE3d>{} : sdv::loadKittiPoses(gtFile);
    for (size_t i = 0; i < poses.size(); ++i) {
      if (!poses[i] || (!allGt.empty() && frameIndex(i) >= allGt.size())) continue;
      est.push_back(*poses[i]);
      estFrame.push_back(static_cast<int>(i));
      if (!allGt.empty()) gt.push_back(allGt[frameIndex(i)]);
    }
    spdlog::info("{} of {} frames have a pose", est.size(), poses.size());
    if (!outFile.empty()) sdv::saveKittiPoses(outFile, est);

    sdv::SimilarityTransform alignment;
    std::string metrics = fmt::format(R"("cameras":{},"frames":{},"start":{},"stride":{},"keyframes":{},"msPerFrame":{:.1f})",
                                      rig.size(), n, start, stride, keyframes, totalMs / n);
    if (!gt.empty() && est.size() > 2) {
      const auto ate = sdv::absoluteTrajectoryError(gt, est, true);
      alignment = ate.alignment;
      std::vector<Sophus::SE3d> aligned = est;
      for (auto& p : aligned) p = alignment.applyToPose(p);
      const auto seg = sdv::segmentDriftError(gt, aligned);
      double length = 0;
      for (size_t i = 1; i < gt.size(); ++i) length += (gt[i].translation() - gt[i - 1].translation()).norm();
      std::string localScale, localScaleJson;
      for (size_t b = 0; b + 50 < est.size(); b += 50) {
        const double g = (gt[b + 50].translation() - gt[b].translation()).norm();
        const double e = (est[b + 50].translation() - est[b].translation()).norm();
        if (e <= 0) continue;
        localScale += fmt::format(" {:.2f}", g / e / alignment.scale);
        localScaleJson += fmt::format("{}{:.3f}", localScaleJson.empty() ? "" : ",", g / e / alignment.scale);
      }
      if (multiCamera) {
        const auto metric = sdv::absoluteTrajectoryError(gt, est, false);
        std::vector<Sophus::SE3d> metricAligned = est;
        for (auto& p : metricAligned) p = metric.alignment.applyToPose(p);
        const auto metricSeg = sdv::segmentDriftError(gt, metricAligned);
        spdlog::info("metric: ATE SE3 rmse {:.3f} m, max {:.3f} m | drift t {:.2f} % r {:.3f} deg/100m",
                     metric.rmse, metric.max, metricSeg.translationPercent, metricSeg.rotationDegPer100m);
      }
      metrics += fmt::format(R"(,"scale":{:.4f},"ateRmse":{:.3f},"ateMax":{:.3f},"length":{:.1f},"driftT":{:.2f},)"
                             R"("driftR":{:.3f},)"
                             R"("localScale":[{}])",
                             alignment.scale, ate.rmse, ate.max, length, seg.translationPercent,
                             seg.rotationDegPer100m,
                             localScaleJson);
      spdlog::info("local scale per 50 frames relative to global:{}", localScale);
      // Frame-to-frame jitter; only consecutive input frames that both have a pose count.
      const auto rpe = sdv::relativePoseError(gt, est, 1, alignment.scale);
      spdlog::info("RPE 1 frame: rmse {:.2f} cm, {:.4f} deg ({} pairs)", 100 * rpe.translationRmse, rpe.rotationRmseDeg,
                   rpe.numPairs);
      {
        // Split by whether a pair involves a tracked (non-key) frame.
        const auto kfs = vo.keyframeIndices();
        auto isKf = [&](int f) { return std::find(kfs.begin(), kfs.end(), f) != kfs.end(); };
        std::vector<double> errs[2];
        for (size_t k = 0; k + 1 < est.size(); ++k) {
          if (estFrame[k + 1] != estFrame[k] + 1) continue;
          const Sophus::SO3d err = (gt[k].so3().inverse() * gt[k + 1].so3()).inverse() * (est[k].so3().inverse() * est[k + 1].so3());
          const double deg = err.logAndTheta().theta * 180.0 / M_PI;
          const int tracked = !(isKf(estFrame[k]) && isKf(estFrame[k + 1]));
          errs[tracked].push_back(deg);
        }
        // Median: KITTI ground truth has interpolated gaps (e.g. frames 2275-2290) that dominate an RMS.
        auto median = [](std::vector<double> v) {
          if (v.empty()) return 0.0;
          std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
          return v[v.size() / 2];
        };
        spdlog::info("RPE rotation median: keyframe pairs {:.4f} deg ({}), pairs with a tracked frame {:.4f} deg ({})",
                     median(errs[0]), errs[0].size(), median(errs[1]), errs[1].size());
      }
      metrics += fmt::format(R"(,"rpeT":{:.4f},"rpeR":{:.5f})", rpe.translationRmse, rpe.rotationRmseDeg);
      spdlog::info("ATE Sim3 rmse {:.3f} m, max {:.3f} m, scale {:.3f} over {:.0f} m | drift t {:.2f} % "
                   "r {:.3f} deg/100m ({} segments)",
                   ate.rmse, ate.max, alignment.scale, length, seg.translationPercent, seg.rotationDegPer100m,
                   seg.numSegments);
    }

    // A few points close to infinity would dominate any viewer; limit by distance from their host camera.
    std::vector<sdv::MapPoint> mapPoints = vo.mapPoints();
    if (!mapPoints.empty()) {
      std::vector<double> distances;
      for (const auto& p : mapPoints) distances.push_back(p.distance);
      std::nth_element(distances.begin(), distances.begin() + distances.size() / 2, distances.end());
      const double limit = maxDistanceFactor > 0 ? maxDistanceFactor * distances[distances.size() / 2]
                                                 : std::numeric_limits<double>::infinity();
      const size_t before = mapPoints.size();
      std::erase_if(mapPoints, [&](const sdv::MapPoint& p) {
        return p.distance > limit || (maxDistance > 0 && p.distance * alignment.scale > maxDistance);
      });
      spdlog::info("kept {} of {} map points (distance limit {:.1f}{})", mapPoints.size(), before,
                   std::min(limit * alignment.scale, maxDistance > 0 ? maxDistance : 1e30),
                   multiCamera || !gt.empty() ? " m" : ", unscaled units");
    }
    if (!mapPoints.empty()) {
      // Quality: observations and depth uncertainty of each point, then isolated points (few neighbours).
      auto percentiles = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return fmt::format("{:.3g} / {:.3g} / {:.3g}", v[v.size() / 10], v[v.size() / 2], v[v.size() * 9 / 10]);
      };
      std::vector<double> sigma, obs;
      for (const auto& p : mapPoints) sigma.push_back(p.relativeDepthSigma), obs.push_back(p.observations);
      spdlog::info("map points: relative depth sigma {} (10/50/90 %), observations {}", percentiles(sigma),
                   percentiles(obs));
      const size_t before = mapPoints.size();
      std::erase_if(mapPoints, [&](const sdv::MapPoint& p) {
        return p.observations < minObservations || (maxDepthSigma > 0 && p.relativeDepthSigma > maxDepthSigma);
      });
      const size_t afterQuality = mapPoints.size();
      if (minNeighbours > 0) {
        // Voxels of the neighbour radius; a point needs minNeighbours others within the radius.
        const double r = neighbourRadius / alignment.scale;
        auto key = [&](const Eigen::Vector3d& x) {
          return std::array<long long, 3>{static_cast<long long>(std::floor(x.x() / r)),
                                          static_cast<long long>(std::floor(x.y() / r)),
                                          static_cast<long long>(std::floor(x.z() / r))};
        };
        std::map<std::array<long long, 3>, std::vector<size_t>> grid;
        for (size_t i = 0; i < mapPoints.size(); ++i) grid[key(mapPoints[i].position)].push_back(i);
        std::vector<char> keep(mapPoints.size(), 0);
        for (size_t i = 0; i < mapPoints.size(); ++i) {
          const auto k = key(mapPoints[i].position);
          int count = 0;
          for (long long dx = -1; dx <= 1 && count < minNeighbours; ++dx)
            for (long long dy = -1; dy <= 1 && count < minNeighbours; ++dy)
              for (long long dz = -1; dz <= 1 && count < minNeighbours; ++dz) {
                const auto it = grid.find({k[0] + dx, k[1] + dy, k[2] + dz});
                if (it == grid.end()) continue;
                for (size_t j : it->second)
                  if (j != i && (mapPoints[j].position - mapPoints[i].position).norm() < r && ++count >= minNeighbours) break;
              }
          keep[i] = count >= minNeighbours;
        }
        size_t w = 0;
        for (size_t i = 0; i < mapPoints.size(); ++i)
          if (keep[i]) mapPoints[w++] = mapPoints[i];
        mapPoints.resize(w);
      }
      spdlog::info("point filter: {} of {} kept (quality {}, neighbours {})", mapPoints.size(), before, afterQuality,
                   mapPoints.size());
    }

    // Point colours (RGB) from the host camera's input image at the tracked pixel; grey where there is no colour.
    std::vector<std::array<std::uint8_t, 3>> pointColors(mapPoints.size());
    for (size_t i = 0; i < mapPoints.size(); ++i)
      pointColors[i].fill(static_cast<std::uint8_t>(std::clamp(mapPoints[i].intensity, 0.f, 255.f)));
    if (!plyFile.empty() || !colmapDir.empty()) {
      std::map<std::pair<int, int>, std::vector<size_t>> pointsOfImage;  // (frame, camera)
      for (size_t i = 0; i < mapPoints.size(); ++i) pointsOfImage[{mapPoints[i].frameIndex, mapPoints[i].camera}].push_back(i);
      for (auto& in : inputs) in.source->rewind();
      const int lastFrame = pointsOfImage.empty() ? -1 : pointsOfImage.rbegin()->first.first;
      for (int frame = 0; frame <= lastFrame; ++frame) {
        bool ended = false;
        for (int c = 0; c < rig.size() && !ended; ++c) {
          const cv::Mat input = inputs[c].source->next();
          if (input.empty()) ended = true;
          const auto it = pointsOfImage.find({frame, c});
          if (ended || it == pointsOfImage.end() || input.channels() != 3) continue;
          cv::Mat img = sdv::prepareImage(input, scale, inputs[c].camera);
          if (img.depth() != CV_8U) img.convertTo(img, CV_8U, img.depth() == CV_16U ? 255.0 / 65535.0 : 1.0);
          for (size_t i : it->second) {
            const cv::Vec3b bgr = img.at<cv::Vec3b>(static_cast<int>(std::lround(mapPoints[i].uv.y())),
                                                    static_cast<int>(std::lround(mapPoints[i].uv.x())));
            pointColors[i] = {bgr[2], bgr[1], bgr[0]};
          }
        }
        if (ended) break;
      }
    }

    if (!plyFile.empty()) {
      sdv::PlyScene scene;
      std::vector<Sophus::SE3d> aligned = est;
      for (auto& p : aligned) p = alignment.applyToPose(p);
      scene.addTrajectory(aligned, {220, 0, 0});
      if (!gt.empty()) scene.addTrajectory(gt, {0, 200, 0});
      for (size_t i = 0; i < mapPoints.size(); ++i) scene.addPoint(alignment.apply(mapPoints[i].position), pointColors[i]);
      scene.write(plyFile);
      spdlog::info("wrote {} ({} map points)", plyFile, mapPoints.size());
    }

    if (!jsonFile.empty()) {
      std::ofstream out(jsonFile);
      // Enough decimals for ~0.1 % of the typical point distance (monocular units are arbitrary).
      double typical = 1.0;
      if (!mapPoints.empty()) {
        std::vector<double> dist;
        for (const auto& p : mapPoints) dist.push_back(p.distance * alignment.scale);
        std::nth_element(dist.begin(), dist.begin() + dist.size() / 2, dist.end());
        typical = std::max(dist[dist.size() / 2], 1e-9);
      }
      const int decimals = std::clamp(3 - static_cast<int>(std::floor(std::log10(typical))), 2, 8);
      auto writePath = [&](const char* name, const std::vector<Sophus::SE3d>& path, bool align) {
        out << '"' << name << "\":[";
        for (size_t i = 0; i < path.size(); ++i) {
          const Eigen::Vector3d p = align ? alignment.apply(path[i].translation()) : path[i].translation();
          out << (i ? "," : "") << fmt::format("{:.{}f},{:.{}f},{:.{}f}", p.x(), decimals, p.y(), decimals, p.z(), decimals);
        }
        out << "],";
      };
      out << "{" << metrics << ",";
      writePath("estimate", est, true);
      writePath("gt", gt, false);
      std::vector<Sophus::SE3d> keyframePoses;
      for (int idx : vo.keyframeIndices())
        if (poses[idx]) keyframePoses.push_back(*poses[idx]);
      writePath("keyframePositions", keyframePoses, true);
      out << "\"points\":[";
      for (size_t i = 0; i < mapPoints.size(); ++i) {
        const Eigen::Vector3d p = alignment.apply(mapPoints[i].position);
        out << (i ? "," : "")
            << fmt::format("{:.{}f},{:.{}f},{:.{}f},{}", p.x(), decimals, p.y(), decimals, p.z(), decimals,
                           static_cast<int>(std::clamp(mapPoints[i].intensity, 0.f, 255.f)));
      }
      out << "]}";
      spdlog::info("wrote {}", jsonFile);
    }

    if (!colmapDir.empty()) {
      const std::filesystem::path root(colmapDir);
      const sdv::SimilarityTransform exportAlignment = colmapAlign ? alignment : sdv::SimilarityTransform{};
      // Per camera: pinhole export camera, undistortion, valid area (for trainers that take masks) and image folder.
      struct ExportCamera {
        sdv::Camera camera;
        sdv::UndistortMap undistortMap;
        cv::Mat mask;
        std::string prefix;
      };
      std::vector<ExportCamera> exportCams;
      std::vector<sdv::Camera> colmapCams;
      for (const auto& in : inputs) {
        const sdv::Camera& cam = in.camera;
        ExportCamera e;
        e.camera = cam.isPinhole() ? cam : sdv::virtualPinhole(cam, focalScale);
        if (!cam.isPinhole()) e.undistortMap = sdv::makeUndistortMap(cam, e.camera);
        e.prefix = multiCamera ? in.name + "/" : "";
        std::filesystem::create_directories(root / "images" / e.prefix);
        if (cam.mask) {
          e.mask = cam.isPinhole() ? cam.mask->level(0).clone()
                                   : cv::Mat(e.camera.height, e.camera.width, CV_8UC1, cv::Scalar(255));
          if (!cam.isPinhole()) e.mask.setTo(0, e.undistortMap.mapX < 0);
          std::filesystem::create_directories(root / "masks" / e.prefix);
        }
        colmapCams.push_back(e.camera);
        exportCams.push_back(std::move(e));
      }

      const auto keyframeIndices = vo.keyframeIndices();
      std::vector<sdv::ColmapImage> images;
      std::map<std::pair<int, int>, size_t> imageOf;  // (frame, camera)
      for (auto& in : inputs) in.source->rewind();
      for (size_t i = 0; i < poses.size(); ++i) {
        const int frame = static_cast<int>(i);
        const bool exported =
            poses[i] && (!colmapKeyframesOnly ||
                         std::find(keyframeIndices.begin(), keyframeIndices.end(), frame) != keyframeIndices.end());
        bool ended = false;
        for (int c = 0; c < rig.size(); ++c) {
          const cv::Mat input = inputs[c].source->next();
          if (input.empty()) {
            ended = true;
            break;
          }
          if (!exported) continue;
          cv::Mat img = sdv::prepareImage(input, scale, inputs[c].camera);
          // GS trainers expect 3-channel images.
          if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
          if (img.depth() != CV_8U) img.convertTo(img, CV_8U, img.depth() == CV_16U ? 255.0 / 65535.0 : 1.0);
          const ExportCamera& e = exportCams[c];
          const std::string name = e.prefix + fmt::format("{:06d}.png", frameIndex(i));
          cv::imwrite((root / "images" / name).string(),
                      inputs[c].camera.isPinhole() ? img : sdv::undistort(img, e.undistortMap));
          if (!e.mask.empty()) cv::imwrite((root / "masks" / name).string(), e.mask);
          imageOf[{frame, c}] = images.size();
          const Sophus::SE3d T_c_w = rig.T_c_b[c] * exportAlignment.applyToPose(*poses[i]).inverse();
          images.push_back({static_cast<int>(images.size()) + 1, name, T_c_w, {}, c + 1});
        }
        if (ended) break;
      }

      std::vector<sdv::ColmapPoint> points;
      for (size_t i = 0; i < mapPoints.size(); ++i) {
        const sdv::MapPoint& m = mapPoints[i];
        const Eigen::Vector3d X = exportAlignment.apply(m.position);
        sdv::ColmapPoint p{static_cast<std::int64_t>(points.size()) + 1, X, pointColors[i]};
        if (const auto it = imageOf.find({m.frameIndex, m.camera}); it != imageOf.end()) {
          sdv::ColmapImage& img = images[it->second];
          const sdv::Camera& exportCam = exportCams[m.camera].camera;
          Eigen::Vector2d uv;
          if (exportCam.project(img.T_c_w * X, uv) && exportCam.isInside(uv.x(), uv.y(), 0.0)) {
            p.track.emplace_back(img.id, static_cast<int>(img.points2D.size()));
            img.points2D.emplace_back(uv, p.id);
          }
        }
        points.push_back(std::move(p));
      }
      sdv::writeColmapText(root / "sparse" / "0", colmapCams, images, points);
      spdlog::info("wrote COLMAP model to {}: {} images, {} points", root.string(), images.size(), points.size());
    }

    if (!pngFile.empty()) {
      // Top-down view: camera frames (KITTI, single camera) x right, z forward; vehicle rigs x forward, y left.
      auto ground = [&](const Eigen::Vector3d& p) { return rigFile.empty() ? Eigen::Vector2d(p.x(), p.z()) : Eigen::Vector2d(p.x(), p.y()); };
      const int size = 1200;
      std::vector<Eigen::Vector3d> all;
      for (const auto& p : gt.empty() ? est : gt) all.push_back(p.translation());
      Eigen::Vector2d lo(1e9, 1e9), hi(-1e9, -1e9);
      for (const auto& p : all) {
        lo = lo.cwiseMin(ground(p));
        hi = hi.cwiseMax(ground(p));
      }
      const double margin = 0.05 * std::max(hi.x() - lo.x(), hi.y() - lo.y()) + 1e-6;
      lo.array() -= margin;
      hi.array() += margin;
      const double pxPerUnit = size / std::max(hi.x() - lo.x(), hi.y() - lo.y());
      auto px = [&](const Eigen::Vector3d& p) {
        const Eigen::Vector2d g = ground(p);
        return cv::Point(static_cast<int>((g.x() - lo.x()) * pxPerUnit),
                         size - static_cast<int>((g.y() - lo.y()) * pxPerUnit));
      };
      cv::Mat img(size, size, CV_8UC3, cv::Scalar(29, 21, 17));
      for (const auto& m : mapPoints) {
        const auto v = static_cast<uint8_t>(std::clamp(m.intensity, 0.f, 255.f));
        const cv::Point q = px(alignment.apply(m.position));
        if (q.x >= 0 && q.y >= 0 && q.x < size && q.y < size) img.at<cv::Vec3b>(q) = {v, v, v};
      }
      for (size_t i = 1; i < gt.size(); ++i)
        cv::line(img, px(gt[i - 1].translation()), px(gt[i].translation()), {204, 193, 86}, 2);
      for (size_t i = 1; i < est.size(); ++i)
        cv::line(img, px(alignment.apply(est[i - 1].translation())), px(alignment.apply(est[i].translation())),
                 {58, 163, 240}, 2);
      cv::imwrite(pngFile, img);
      spdlog::info("wrote {}", pngFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
