// Loop closure without rerunning odometry: loop detection on run_vo keyframe records, SE3 pose graph over the
// keyframes, and the correction applied to the run_vo poses of all frames. Metric runs only (stereo or rig), where
// every input frame from the first one has a pose, so line i of the pose file is run frame i.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/global_ba.h>
#include <sdv/io/camera_config.h>
#include <sdv/io/frame_source.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/io/rig_config.h>
#include <sdv/photometric_ba.h>
#include <sdv/point_filter.h>
#include <sdv/point_merge.h>
#include <sdv/loop_detector.h>
#include <sdv/pose_graph.h>
#include <sdv/semi_dense_mapper.h>

namespace po = boost::program_options;

namespace {

// Keyframe images (colour where the videos have it) of every rig camera, decoded from the run's videos (only keyframes are decoded); the rig
// cameras get their validity masks.
std::vector<std::vector<cv::Mat>> loadKeyframeImages(const std::string& rigFile, const std::vector<std::string>& names,
                                                     sdv::Rig& rig, const std::vector<sdv::KeyframeRecord>& records,
                                                     double scale, int start, int stride) {
  const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
  std::vector<const sdv::RigCameraConfig*> cams;
  for (const auto& c : config.cameras)
    if (names.empty() || std::find(names.begin(), names.end(), c.name) != names.end()) cams.push_back(&c);
  if (static_cast<int>(cams.size()) != rig.size()) throw std::invalid_argument("rig cameras do not match the records");
  std::vector<std::vector<cv::Mat>> images(records.size(), std::vector<cv::Mat>(cams.size()));
  for (size_t c = 0; c < cams.size(); ++c) {
    const sdv::Camera cam = sdv::prepareCamera(cams[c]->camera, scale, 16);
    if (cam.width != rig.cameras[c].width || std::abs(cam.fx - rig.cameras[c].fx) > 1e-6)
      throw std::invalid_argument("camera " + cams[c]->name + " does not match the records (--scale?)");
    rig.cameras[c].mask = cam.mask;
    rig.cameras[c].maskLevel = cam.maskLevel;
    sdv::VideoSource video(cams[c]->video);
    long long position = 0;  // next input frame of the video
    for (size_t k = 0; k < records.size(); ++k) {
      const long long wanted = cams[c]->frameOffset + start + static_cast<long long>(records[k].frameIndex) * stride;
      for (; position < wanted; ++position)
        if (!video.skip()) throw std::runtime_error("video ended early: " + cams[c]->video.string());
      const cv::Mat input = video.next();
      ++position;
      if (input.empty()) throw std::runtime_error("video ended early: " + cams[c]->video.string());
      cv::Mat img = sdv::prepareImage(input, scale, cam);
      images[k][c] = img;
    }
    spdlog::info("loaded {} keyframe images of {}", records.size(), cams[c]->name);
  }
  return images;
}

// Images of run frame 0, 1, 2, ... of every rig camera, prepared like the run; empty at the end.
std::function<std::vector<cv::Mat>()> rigFrames(const std::string& rigFile, const std::vector<std::string>& names,
                                                const sdv::Rig& rig, double scale, int start, int stride) {
  const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
  struct Stream {
    std::shared_ptr<sdv::VideoSource> video;
    long long position = 0, offset = 0;
  };
  std::vector<Stream> streams;
  for (const auto& c : config.cameras)
    if (names.empty() || std::find(names.begin(), names.end(), c.name) != names.end())
      streams.push_back({std::make_shared<sdv::VideoSource>(c.video), 0, c.frameOffset});
  return [=, frame = 0LL]() mutable {
    std::vector<cv::Mat> images;
    for (size_t c = 0; c < streams.size(); ++c) {
      Stream& s = streams[c];
      const long long wanted = s.offset + start + frame * stride;
      for (; s.position < wanted; ++s.position)
        if (!s.video->skip()) return std::vector<cv::Mat>{};
      const cv::Mat input = s.video->next();
      ++s.position;
      if (input.empty()) return std::vector<cv::Mat>{};
      images.push_back(sdv::prepareImage(input, scale, rig.cameras[c]));
    }
    ++frame;
    return images;
  };
}

std::array<std::uint8_t, 3> rgbAt(const cv::Mat& image, const Eigen::Vector2d& uv) {
  const int x = std::clamp(static_cast<int>(std::lround(uv.x())), 0, image.cols - 1);
  const int y = std::clamp(static_cast<int>(std::lround(uv.y())), 0, image.rows - 1);
  if (image.channels() == 3) {
    const cv::Vec3b& bgr = image.at<cv::Vec3b>(y, x);
    return {bgr[2], bgr[1], bgr[0]};
  }
  const std::uint8_t g = image.at<std::uint8_t>(y, x);
  return {g, g, g};
}

}  // namespace

int main(int argc, char** argv) {
  std::string keyframesFile, vocabularyFile, posesFile, outFile, gtFile, plyFile;
  std::vector<double> up{0, 0, 1};
  int start = 0, stride = 1;
  bool bundleAdjust = false, photometric = false, densify = false, merge = false;
  sdv::SemiDenseSettings dense{.voxelSize = 0.05, .minVoxelHosts = 2};
  sdv::PointMergeSettings mergeSettings{.minFrameGap = 100};
  std::string rigFile, sequenceDir;
  std::vector<std::string> rigCameras;
  double scale = 1.0, maxDistanceFactor = 5.0, maxDepthSigma = 0.0, neighbourRadius = 0.2;
  int minResiduals = 3, minNeighbours = 3;
  sdv::LoopSettings loopSettings;
  sdv::PoseGraphSettings graphSettings;
  sdv::GlobalBASettings baSettings;
  sdv::PhotometricBASettings pbaSettings;
  po::options_description desc("close_loops options");
  desc.add_options()
      ("help", "show help")
      ("keyframes", po::value(&keyframesFile)->required(), "run_vo --keyframes-out file")
      ("vocabulary", po::value(&vocabularyFile)->required(), "FBoW vocabulary")
      ("poses", po::value(&posesFile)->required(), "run_vo -o poses of the same run (KITTI format)")
      ("out", po::value(&outFile), "corrected poses (KITTI format)")
      ("gt", po::value(&gtFile), "ground-truth poses per input frame for evaluation")
      ("start", po::value(&start)->default_value(0), "run_vo --start")
      ("stride", po::value(&stride)->default_value(1), "run_vo --stride")
      ("up", po::value(&up)->multitoken(), "world vertical (default 0 0 1; KITTI: 0 -1 0)")
      ("min-inliers", po::value(&loopSettings.minInliers)->default_value(loopSettings.minInliers), "inliers a loop needs")
      ("loop-sigma-t", po::value(&graphSettings.loopTranslationSigma)->default_value(graphSettings.loopTranslationSigma),
       "loop edge translation sigma, m")
      ("loop-sigma-r", po::value(&graphSettings.loopRotationSigmaDeg)->default_value(graphSettings.loopRotationSigmaDeg),
       "loop edge rotation sigma, deg")
      ("bundle-adjust", po::bool_switch(&bundleAdjust), "global feature bundle adjustment after the pose graph")
      ("ba-depth-sigma", po::value(&baSettings.relativeDepthSigma)->default_value(baSettings.relativeDepthSigma),
       "relative sigma of the map-depth prior (0 = off)")
      ("ba-odometry-factor", po::value(&baSettings.odometrySigmaFactor)->default_value(baSettings.odometrySigmaFactor),
       "odometry edge sigma factor (0 = off)")
      ("ba-neighbours", po::value(&baSettings.sequentialNeighbours)->default_value(baSettings.sequentialNeighbours),
       "following keyframes each keyframe is matched with")
      ("ba-iterations", po::value(&baSettings.iterations)->default_value(baSettings.iterations), "iterations per round")
      ("photometric", po::bool_switch(&photometric),
       "global photometric bundle adjustment (needs --rig and records with map points)")
      ("rig", po::value(&rigFile), "rig YAML of the run (videos, cameras) for the keyframe images")
      ("sequence", po::value(&sequenceDir), "KITTI sequence of the run (stereo) for the keyframe images")
      ("rig-cameras", po::value(&rigCameras)->multitoken(), "rig cameras of the run, in its order")
      ("scale", po::value(&scale)->default_value(1.0), "run_vo --scale")
      ("pba-neighbours", po::value(&pbaSettings.neighbours)->default_value(pbaSettings.neighbours),
       "keyframes before and after the host a point is observed in")
      ("pba-iterations", po::value(&pbaSettings.iterations)->default_value(pbaSettings.iterations), "iterations per round")
      ("pba-points", po::value(&pbaSettings.maxPointsPerImage)->default_value(pbaSettings.maxPointsPerImage),
       "map points per keyframe image (0 = all)")
      ("pba-spatial-radius", po::value(&pbaSettings.spatialRadius)->default_value(pbaSettings.spatialRadius),
       "revisit keyframes within this distance of the host are observation targets, m (0 = off)")
      ("pba-spatial-targets", po::value(&pbaSettings.spatialTargets)->default_value(pbaSettings.spatialTargets),
       "closest revisit keyframes per host")
      ("pba-cross-targets", po::value(&pbaSettings.maxCrossTargets)->default_value(pbaSettings.maxCrossTargets),
       "loop and revisit target keyframes per point, the closest (0 = all)")
      ("pba-odometry-factor", po::value(&pbaSettings.odometrySigmaFactor)->default_value(pbaSettings.odometrySigmaFactor),
       "odometry edge sigma factor (0 = off)")
      ("ply", po::value(&plyFile), "write corrected keyframe trajectory and bundle-adjusted points")
      ("max-distance-factor", po::value(&maxDistanceFactor)->default_value(5.0),
       "PLY: drop points farther than this times the median distance from their camera")
      ("min-residuals", po::value(&minResiduals)->default_value(3), "PLY: residuals a point needs after the BA")
      ("max-depth-sigma", po::value(&maxDepthSigma)->default_value(0.0),
       "PLY: relative inverse-depth sigma limit (unit photometric noise, 0 = off)")
      ("min-neighbours", po::value(&minNeighbours)->default_value(3), "PLY: neighbours a point needs (0 = off)")
      ("neighbour-radius", po::value(&neighbourRadius)->default_value(0.2), "PLY: neighbour radius, m")
      ("pba-min-initial", po::value(&pbaSettings.minInitialResiduals)->default_value(pbaSettings.minInitialResiduals),
       "residuals a point needs to pass the initial check to take part")
      ("densify", po::bool_switch(&densify),
       "PLY: semi-dense points from all input frames with the final poses (needs --rig or --sequence)")
      ("densify-points", po::value(&dense.pointsPerImage)->default_value(dense.pointsPerImage),
       "candidate pixels per keyframe image")
      ("densify-frames", po::value(&dense.traceFrames)->default_value(dense.traceFrames),
       "following input frames each keyframe host is traced into")
      ("densify-min-depth", po::value(&dense.minDepth)->default_value(dense.minDepth), "initial search range, m")
      ("densify-min-good", po::value(&dense.minGood)->default_value(dense.minGood), "good traces a point needs")
      ("densify-interval", po::value(&dense.maxInterval)->default_value(dense.maxInterval),
       "relative inverse-depth half interval limit")
      ("densify-outliers", po::value(&dense.maxOutlierRatio)->default_value(dense.maxOutlierRatio),
       "outlier traces per good trace")
      ("densify-voxel", po::value(&dense.voxelSize)->default_value(dense.voxelSize),
       "multi-view check voxel, m (0 = off)")
      ("densify-voxel-hosts", po::value(&dense.minVoxelHosts)->default_value(dense.minVoxelHosts),
       "host images a voxel needs")
      ("merge", po::bool_switch(&merge), "PLY: merge duplicate points of separate passes (laps)")
      ("merge-distance", po::value(&mergeSettings.maxDistance)->default_value(mergeSettings.maxDistance),
       "merge radius, m")
      ("merge-frame-gap", po::value(&mergeSettings.minFrameGap)->default_value(mergeSettings.minFrameGap),
       "input frames between the hosts of merged points");
  try {
    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv)
                  .options(desc)
                  .style(po::command_line_style::unix_style ^ po::command_line_style::allow_short)
                  .run(),
              vm);
    if (vm.count("help")) {
      std::cout << desc << '\n';
      return EXIT_SUCCESS;
    }
    po::notify(vm);
    if (up.size() != 3) throw po::error("--up takes 3 values");
  } catch (const po::error& e) {
    spdlog::error("{}", e.what());
    std::cout << desc << '\n';
    return EXIT_FAILURE;
  }

  try {
    loopSettings.up = Eigen::Vector3d(up[0], up[1], up[2]).normalized();
    sdv::Rig rig;
    const auto records = sdv::loadKeyframeRecords(keyframesFile, &rig);
    auto poses = sdv::loadKittiPoses(posesFile);
    sdv::LoopDetector detector(rig, std::make_shared<const sdv::Vocabulary>(vocabularyFile), loopSettings);
    std::vector<sdv::LoopConstraint> found;
    for (const auto& r : records) {
      const auto l = detector.addKeyframe(r);
      found.insert(found.end(), l.begin(), l.end());
    }
    const auto loops = detector.temporallyConsistent(found);
    std::vector<int> frames;
    std::vector<Sophus::SE3d> before;
    for (const auto& r : records) {
      frames.push_back(r.frameIndex);
      before.push_back(poses.at(static_cast<size_t>(r.frameIndex)));
    }
    const auto graph = sdv::optimizePoseGraph(before, loops, graphSettings);
    spdlog::info("{} keyframes, {} loops ({} geometric), {} rejected by the pose graph; cost {:.1f} -> {:.1f}",
                 records.size(), loops.size(), found.size(), graph.loopsRejected, graph.initialCost, graph.finalCost);
    std::vector<Sophus::SE3d> after = graph.T_w_b;
    sdv::GlobalBAResult ba;
    if (bundleAdjust) {
      std::vector<std::pair<int, int>> pairs;
      for (size_t k = 0; k < loops.size(); ++k)
        if (graph.loopAccepted[k]) pairs.emplace_back(loops[k].query, loops[k].match);
      const auto t0 = std::chrono::steady_clock::now();
      ba = sdv::bundleAdjustKeyframes(rig, records, graph.T_w_b, pairs, baSettings);
      after = ba.T_w_b;
      spdlog::info("bundle adjustment: {} points, {} observations ({} in tracks across loops), rmse {:.2f} -> {:.2f} "
                   "px, {} iterations, {:.1f} s",
                   ba.points.size(), ba.observations, ba.loopObservations, ba.rmseBefore, ba.rmseAfter, ba.iterations,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    std::vector<Sophus::SE3d> beforePhotometric = after;
    sdv::PhotometricBAResult pba;
    std::vector<std::vector<cv::Mat>> images;
    if (photometric || (densify && !plyFile.empty())) {
      if (rigFile.empty() == sequenceDir.empty())
        throw std::invalid_argument("--photometric and --densify need --rig or --sequence");
      if (!sequenceDir.empty()) {
        // KITTI stereo: camera 0 = image_0, camera 1 = image_1, cropped like the run.
        const sdv::KittiSequence seq(sequenceDir, 1);
        for (const auto& r : records) {
          std::vector<cv::Mat> imgs;
          for (int c = 0; c < rig.size(); ++c)
            imgs.push_back(sdv::prepareImage(seq.loadImage(static_cast<size_t>(start + r.frameIndex * stride), c), scale,
                                             rig.cameras[c]));
          images.push_back(std::move(imgs));
        }
      } else {
        images = loadKeyframeImages(rigFile, rigCameras, rig, records, scale, start, stride);
      }
    }
    if (photometric) {
      std::vector<std::pair<int, int>> pairs;
      for (size_t k = 0; k < loops.size(); ++k)
        if (graph.loopAccepted[k]) pairs.emplace_back(loops[k].query, loops[k].match);
      const auto t0 = std::chrono::steady_clock::now();
      pba = sdv::photometricBundleAdjust(rig, records, images, after, pairs, pbaSettings);
      after = pba.T_w_b;
      spdlog::info("photometric bundle adjustment: {} points, {} residuals ({} across loops), rmse {:.2f} -> {:.2f}, "
                   "{} iterations, {:.1f} s",
                   pba.points.size(), pba.residuals, pba.loopResiduals, pba.rmseBefore, pba.rmseAfter, pba.iterations,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    const auto uncorrected = poses;
    auto correct = [&](const std::vector<Sophus::SE3d>& keyframePoses) {
      const sdv::PoseCorrection correction(frames, before, keyframePoses);
      std::vector<Sophus::SE3d> out = uncorrected;
      for (size_t i = 0; i < out.size(); ++i) out[i] = correction.at(static_cast<int>(i)) * out[i];
      return out;
    };
    poses = correct(after);
    if (!gtFile.empty()) {
      const auto allGt = sdv::loadKittiPoses(gtFile);
      std::vector<Sophus::SE3d> gt;
      for (size_t i = 0; i < poses.size(); ++i) gt.push_back(allGt.at(static_cast<size_t>(start) + i * stride));
      auto report = [&](const char* name, const std::vector<Sophus::SE3d>& ps) {
        const auto a = sdv::absoluteTrajectoryError(gt, ps, false);
        auto aligned = ps;
        for (auto& p : aligned) p = a.alignment.applyToPose(p);
        const auto seg = sdv::segmentDriftError(gt, aligned);
        // Split into horizontal and vertical: GPS/INS ground truth heights are the weak part (step 15).
        double sh = 0, sv = 0;
        for (size_t i = 0; i < gt.size(); ++i) {
          const Eigen::Vector3d e = aligned[i].translation() - gt[i].translation();
          const double v = e.dot(loopSettings.up);
          sv += v * v, sh += e.squaredNorm() - v * v;
        }
        spdlog::info("{:<12} ATE SE3 rmse {:.3f} m (horizontal {:.3f}, vertical {:.3f}), max {:.3f} m | drift t {:.2f} "
                     "% r {:.3f} deg/100m",
                     name, a.rmse, std::sqrt(sh / gt.size()), std::sqrt(sv / gt.size()), a.max, seg.translationPercent,
                     seg.rotationDegPer100m);
      };
      report("odometry", uncorrected);
      report("pose graph", correct(graph.T_w_b));
      if (bundleAdjust) report("feature BA", correct(beforePhotometric));
      if (photometric) report("photometric", poses);
    }
    if (!plyFile.empty()) {
      sdv::PlyScene scene;
      std::vector<Sophus::SE3d> kfs = after;
      scene.addTrajectory(kfs, {220, 0, 0});
      std::vector<sdv::MapPoint> cloud;
      if (photometric && !pba.points.empty()) {
        // As in run_vo: distance limit (points near infinity), residuals left, depth uncertainty, isolated points.
        std::vector<double> d = pba.pointDistance;
        std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
        const double limit = maxDistanceFactor * d[d.size() / 2];
        std::vector<double> sigma;
        for (size_t i = 0; i < pba.points.size(); ++i)
          if (pba.pointResiduals[i] > 0) sigma.push_back(pba.pointDepthSigma[i]);
        std::sort(sigma.begin(), sigma.end());
        if (!sigma.empty())
          spdlog::info("points with residuals: {}, relative depth sigma {:.3g} / {:.3g} / {:.3g} (10/50/90 %)",
                       sigma.size(), sigma[sigma.size() / 10], sigma[sigma.size() / 2], sigma[sigma.size() * 9 / 10]);
        for (size_t i = 0; i < pba.points.size(); ++i)
          if (pba.pointDistance[i] <= limit && pba.pointResiduals[i] >= minResiduals &&
              (maxDepthSigma <= 0 || pba.pointDepthSigma[i] <= maxDepthSigma)) {
            const auto [k, c] = pba.pointHost[i];
            const auto rgb = rgbAt(images[k][c], pba.pointUv[i]);
            cloud.push_back({pba.points[i], static_cast<float>(rgb[1]), records[k].frameIndex, c, pba.pointUv[i],
                             pba.pointDistance[i], pba.pointResiduals[i], pba.pointDepthSigma[i],
                             sdv::MapPointSource::Active, rgb});
          }
        spdlog::info("{} of {} bundle-adjusted points pass distance {:.1f} m, {}+ residuals, depth sigma", cloud.size(),
                     pba.points.size(), limit, minResiduals);
      } else {
        for (const auto& p : ba.points) scene.addPoint(p, {200, 200, 200});
      }
      if (densify) {
        const auto t0 = std::chrono::steady_clock::now();
        // All input frames with the corrected poses, keyframes as hosts: between keyframes the small baselines narrow
        // the epipolar search step by step (keyframes alone leave most traces ambiguous). Brightness between
        // keyframes is interpolated.
        auto keyframeBrightness = [&](size_t k, int c) {
          if (photometric) return pba.affine[k][c];
          return c < static_cast<int>(records[k].affine.size()) ? records[k].affine[c] : sdv::AffineBrightness{};
        };
        auto nextFrame = !sequenceDir.empty()
                             ? std::function<std::vector<cv::Mat>()>(
                                   [&, seq = std::make_shared<sdv::KittiSequence>(sequenceDir, 1), i = size_t{0}]() mutable {
                                     std::vector<cv::Mat> imgs;
                                     const size_t index = static_cast<size_t>(start) + i++ * stride;
                                     if (index >= seq->size()) return imgs;
                                     for (int c = 0; c < rig.size(); ++c)
                                       imgs.push_back(sdv::prepareImage(seq->loadImage(index, c), scale, rig.cameras[c]));
                                     return imgs;
                                   })
                             : rigFrames(rigFile, rigCameras, rig, scale, start, stride);
        sdv::SemiDenseMapper mapper(rig, dense);
        size_t k = 0;
        for (size_t i = 0; i < poses.size(); ++i) {
          const std::vector<cv::Mat> frame = nextFrame();
          if (frame.empty()) break;
          while (k + 1 < records.size() && records[k + 1].frameIndex <= static_cast<int>(i)) ++k;
          const bool host = records[k].frameIndex == static_cast<int>(i);
          std::vector<sdv::AffineBrightness> brightness(static_cast<size_t>(rig.size()));
          const int f0 = records[k].frameIndex;
          const int f1 = k + 1 < records.size() ? records[k + 1].frameIndex : f0;
          const double w = f1 > f0 ? std::clamp((static_cast<double>(i) - f0) / (f1 - f0), 0.0, 1.0) : 0.0;
          for (int c = 0; c < rig.size(); ++c) {
            const auto a = keyframeBrightness(k, c), b = keyframeBrightness(std::min(k + 1, records.size() - 1), c);
            brightness[c] = {(1 - w) * a.a + w * b.a, (1 - w) * a.b + w * b.b};
          }
          mapper.addFrame(static_cast<int>(i), frame, poses[i], brightness, host);
          if ((i + 1) % 200 == 0) spdlog::info("densify: {} frames", i + 1);
        }
        std::vector<sdv::MapPoint> densePoints = mapper.finish();
        std::vector<double> d;
        for (const auto& p : densePoints) d.push_back(p.distance);
        double limit = std::numeric_limits<double>::infinity();
        if (!d.empty()) {
          std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
          limit = maxDistanceFactor * d[d.size() / 2];
        }
        const auto& st = mapper.stats();
        size_t added = 0;
        for (const auto& p : densePoints)
          if (p.distance <= limit) cloud.push_back(p), ++added;
        spdlog::info("densify: {} candidates, {:.1f} good traces each, {} accepted (rejected: {} too few matches, {} "
                     "imprecise), {} after voxel check, {} within {:.1f} m, {:.1f} s",
                     st.candidates, static_cast<double>(st.good) / std::max<long long>(st.candidates, 1), st.accepted,
                     st.rejectMatches, st.rejectInterval, densePoints.size(), added, limit,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      if (merge) {
        size_t merged = 0;
        const size_t unmerged = cloud.size();
        cloud = sdv::mergeDuplicatePoints(cloud, mergeSettings, &merged);
        spdlog::info("merge: {} -> {} points ({} pairs within {:.3f} m, hosts {}+ frames apart)", unmerged, cloud.size(),
                     merged, mergeSettings.maxDistance, mergeSettings.minFrameGap);
      }
      std::vector<Eigen::Vector3d> positions;
      for (const auto& p : cloud) positions.push_back(p.position);
      const std::vector<char> keep = sdv::hasNeighbours(positions, neighbourRadius, minNeighbours);
      size_t kept = 0;
      for (size_t i = 0; i < cloud.size(); ++i)
        if (keep[i]) {
          const float g = std::clamp(cloud[i].intensity, 0.f, 255.f);
          scene.addPoint(cloud[i].position,
                         cloud[i].color.value_or(std::array<std::uint8_t, 3>{static_cast<std::uint8_t>(g),
                                                                             static_cast<std::uint8_t>(g),
                                                                             static_cast<std::uint8_t>(g)}));
          ++kept;
        }
      spdlog::info("{} of {} points written ({}+ neighbours within {:.2f} m)", kept, cloud.size(), minNeighbours,
                   neighbourRadius);
      scene.write(plyFile);
      spdlog::info("wrote {}", plyFile);
    }
    if (!outFile.empty()) {
      sdv::saveKittiPoses(outFile, poses);
      spdlog::info("wrote {}", outFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
