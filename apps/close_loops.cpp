// Loop closure without rerunning odometry: loop detection on run_vo keyframe records, SE3 pose graph over the
// keyframes, and the correction applied to the run_vo poses of all frames. Metric runs only (stereo or rig), where
// every input frame from the first one has a pose, so line i of the pose file is run frame i.
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
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
#include <sdv/loop_detector.h>
#include <sdv/pose_graph.h>

namespace po = boost::program_options;

namespace {

// Grey keyframe images of every rig camera, decoded from the run's videos (only keyframes are decoded); the rig
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
      if (img.channels() == 3) cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
      images[k][c] = img;
    }
    spdlog::info("loaded {} keyframe images of {}", records.size(), cams[c]->name);
  }
  return images;
}

}  // namespace

int main(int argc, char** argv) {
  std::string keyframesFile, vocabularyFile, posesFile, outFile, gtFile, plyFile;
  std::vector<double> up{0, 0, 1};
  int start = 0, stride = 1;
  bool bundleAdjust = false, photometric = false;
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
       "residuals a point needs to pass the initial check to take part");
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
    if (photometric) {
      if (rigFile.empty() == sequenceDir.empty()) throw std::invalid_argument("--photometric needs --rig or --sequence");
      std::vector<std::vector<cv::Mat>> images;
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
        std::vector<Eigen::Vector3d> candidates;
        for (size_t i = 0; i < pba.points.size(); ++i)
          if (pba.pointDistance[i] <= limit && pba.pointResiduals[i] >= minResiduals &&
              (maxDepthSigma <= 0 || pba.pointDepthSigma[i] <= maxDepthSigma))
            candidates.push_back(pba.points[i]);
        const std::vector<char> keep = sdv::hasNeighbours(candidates, neighbourRadius, minNeighbours);
        size_t kept = 0;
        for (size_t i = 0; i < candidates.size(); ++i)
          if (keep[i]) scene.addPoint(candidates[i], {200, 200, 200}), ++kept;
        spdlog::info("{} of {} points written ({} after distance {:.1f} m, {}+ residuals, depth sigma; then {}+ "
                     "neighbours within {:.2f} m)",
                     kept, pba.points.size(), candidates.size(), limit, minResiduals, minNeighbours, neighbourRadius);
      } else {
        for (const auto& p : ba.points) scene.addPoint(p, {200, 200, 200});
      }
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
