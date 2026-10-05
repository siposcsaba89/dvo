// Loop closure without rerunning odometry: loop detection on run_vo keyframe records, SE3 pose graph over the
// keyframes, and the correction applied to the run_vo poses of all frames. Metric runs only (stereo or rig), where
// every input frame from the first one has a pose, so line i of the pose file is run frame i.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <fmt/format.h>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <sdv/brightness_fit.h>
#include <sdv/eval/trajectory.h>
#include <sdv/global_ba.h>
#include <sdv/image_pyramid.h>
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

// The rig cameras of the records get the validity masks of the rig file (the records carry none).
void attachMasks(const std::string& rigFile, const std::vector<std::string>& names, sdv::Rig& rig, double scale) {
  const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
  int c = 0;
  for (const auto& cfg : config.cameras) {
    if (!names.empty() && std::find(names.begin(), names.end(), cfg.name) == names.end()) continue;
    if (c >= rig.size()) throw std::invalid_argument("rig cameras do not match the records");
    const sdv::Camera cam = sdv::prepareCamera(cfg.camera, scale, 16);
    if (cam.width != rig.cameras[c].width || std::abs(cam.fx - rig.cameras[c].fx) > 1e-6)
      throw std::invalid_argument("camera " + cfg.name + " does not match the records (--scale?)");
    rig.cameras[c].mask = cam.mask;
    rig.cameras[c].maskLevel = cam.maskLevel;
    ++c;
  }
  if (c != rig.size()) throw std::invalid_argument("rig cameras do not match the records");
}

// Keyframe images of the named rig cameras (in rig order, prepared for `cameras`; colour where the input has it,
// unless `grey`), decoded from the run's input (only keyframes are decoded). `runNames`: the cameras of the run,
// whose frames the records index.
std::vector<std::vector<cv::Mat>> loadKeyframeImages(const std::string& rigFile, const std::vector<std::string>& names,
                                                     const std::vector<std::string>& runNames,
                                                     const std::vector<sdv::Camera>& cameras,
                                                     const std::vector<sdv::KeyframeRecord>& records, double scale,
                                                     int start, int stride, bool grey = false) {
  const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
  std::vector<sdv::RigStream> streams = sdv::openRigStreams(config, names, runNames);
  if (streams.size() != cameras.size()) throw std::invalid_argument("one camera per stream required");
  std::vector<std::vector<cv::Mat>> images(records.size(), std::vector<cv::Mat>(streams.size()));
  for (size_t c = 0; c < streams.size(); ++c) {
    const std::string& name = streams[c].config->name;
    sdv::FrameSource& source = *streams[c].source;
    long long position = 0;  // next input frame of the stream
    for (size_t k = 0; k < records.size(); ++k) {
      const long long wanted = start + static_cast<long long>(records[k].frameIndex) * stride;
      for (; position < wanted; ++position)
        if (!source.skip()) throw std::runtime_error("input of " + name + " ended early");
      const cv::Mat input = source.next();
      ++position;
      if (input.empty()) throw std::runtime_error("input of " + name + " ended early");
      images[k][c] = sdv::prepareImage(input, scale, cameras[c]);
      if (grey && images[k][c].channels() == 3) cv::cvtColor(images[k][c], images[k][c], cv::COLOR_BGR2GRAY);
    }
    spdlog::info("loaded {} keyframe images of {}", records.size(), name);
  }
  return images;
}

// Images of run frame 0, 1, 2, ... of the named rig cameras (`rig`, in rig order), prepared like the run; empty at
// the end. `runNames`: the cameras of the run, whose frames these are.
std::function<std::vector<cv::Mat>()> rigFrames(const std::string& rigFile, const std::vector<std::string>& names,
                                                const std::vector<std::string>& runNames, const sdv::Rig& rig,
                                                double scale, int start, int stride) {
  auto config = std::make_shared<const sdv::RigConfig>(sdv::loadRigConfig(rigFile));
  std::vector<std::shared_ptr<sdv::FrameSource>> sources;
  for (auto& s : sdv::openRigStreams(*config, names, runNames))
    sources.push_back(std::make_shared<sdv::SubsampledSource>(std::move(s.source), start, stride));
  return [config, sources, rig, scale] {
    std::vector<cv::Mat> images;
    for (size_t c = 0; c < sources.size(); ++c) {
      const cv::Mat input = sources[c]->next();
      if (input.empty()) return std::vector<cv::Mat>{};
      images.push_back(sdv::prepareImage(input, scale, rig.cameras[c]));
    }
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

// Colours at the points' host pixels, decoded one frame at a time (the keyframe images of a long run are held in
// grey: colour would triple their memory).
std::vector<std::array<std::uint8_t, 3>> pointColours(const std::string& rigFile, const std::vector<std::string>& names,
                                                      const std::vector<std::string>& runNames,
                                                      const std::vector<sdv::Camera>& cameras,
                                                      const std::vector<sdv::KeyframeRecord>& records, double scale,
                                                      int start, int stride, const sdv::PhotometricBAResult& pba) {
  const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
  std::vector<sdv::RigStream> streams = sdv::openRigStreams(config, names, runNames);
  std::vector<std::vector<std::vector<size_t>>> byHost(streams.size(), std::vector<std::vector<size_t>>(records.size()));
  for (size_t i = 0; i < pba.points.size(); ++i) {
    const auto [k, c] = pba.pointHost[i];
    byHost[c][k].push_back(i);
  }
  std::vector<std::array<std::uint8_t, 3>> rgb(pba.points.size());
  for (size_t c = 0; c < streams.size(); ++c) {
    sdv::FrameSource& source = *streams[c].source;
    long long position = 0;
    for (size_t k = 0; k < records.size(); ++k) {
      if (byHost[c][k].empty()) continue;
      const long long wanted = start + static_cast<long long>(records[k].frameIndex) * stride;
      for (; position < wanted; ++position)
        if (!source.skip()) throw std::runtime_error("input of " + streams[c].config->name + " ended early");
      const cv::Mat input = source.next();
      ++position;
      if (input.empty()) throw std::runtime_error("input of " + streams[c].config->name + " ended early");
      const cv::Mat image = sdv::prepareImage(input, scale, cameras[c]);
      for (const size_t i : byHost[c][k]) rgb[i] = rgbAt(image, pba.pointUv[i]);
    }
  }
  return rgb;
}

}  // namespace

int main(int argc, char** argv) {
  std::string keyframesFile, vocabularyFile, posesFile, outFile, gtFile, plyFile;
  std::vector<double> up{0, 0, 1};
  int start = 0, stride = 1;
  bool bundleAdjust = false, photometric = false, densify = false, merge = false, freeSpace = false;
  bool refineIntrinsics = false;
  sdv::FreeSpaceSettings freeSpaceSettings;
  sdv::SemiDenseSettings dense{.voxelSize = 0.05, .minVoxelHosts = 2};
  sdv::PointMergeSettings mergeSettings{.minFrameGap = 100};
  std::string rigFile, sequenceDir, rigOutFile, pointsFile, brightnessFile;
  std::vector<std::string> rigCameras, densifyCameras, intrinsicNames;
  double scale = 1.0, maxDistanceFactor = 5.0, maxDepthSigma = 0.0, neighbourRadius = 0.2;
  int minResiduals = 3, minNeighbours = 3;
  sdv::LoopSettings loopSettings;
  sdv::PoseGraphSettings graphSettings;
  sdv::GlobalBASettings baSettings;
  sdv::PhotometricBASettings pbaSettings;
  std::string pbaSolver, pbaOptimizer;
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
      ("pba-block-keyframes", po::value(&pbaSettings.blockKeyframes)->default_value(pbaSettings.blockKeyframes),
       "photometric BA in blocks of this many consecutive keyframes, the rest fixed (bounded memory for long runs; "
       "0 = one joint problem)")
      ("pba-block-sweeps", po::value(&pbaSettings.blockSweeps)->default_value(pbaSettings.blockSweeps),
       "sweeps over the blocks per round, borders shifted by half a block")
      ("pba-sparse-band", po::value(&pbaSettings.sparseBand)->default_value(pbaSettings.sparseBand),
       "custom optimizer: reduced system with only the keyframe pairs linked by observations or odometry, or at most "
       "this many apart (the dropped fill-in lumped onto the diagonal; exact gradient, same optimum); -1 = exact")
      ("pba-pcg", po::value(&pbaSettings.pcgIterations)->default_value(pbaSettings.pcgIterations),
       "with --pba-sparse-band: CG iterations on the exact reduced system, preconditioned by the sparse one")
      ("pba-optimizer", po::value(&pbaOptimizer)->default_value("custom"),
       "photometric BA optimizer: custom (ours, no per-residual storage) or ceres (the reference; always with "
       "--pba-extrinsics / --pba-intrinsics)")
      ("pba-solver", po::value(&pbaSolver)->default_value("sparse"),
       "photometric BA linear solver: sparse (Eigen Cholesky, AMD), nesdis (nested dissection), iterative (CG, "
       "multi-threaded)")
      ("pba-block-coarse-residuals",
       po::value(&pbaSettings.blockCoarseResiduals)->default_value(pbaSettings.blockCoarseResiduals),
       "with blocks: first a joint solve over all keyframes with every m-th point, at most this many residuals")
      ("pba-extrinsics", po::bool_switch(&pbaSettings.refineExtrinsics),
       "photometric BA: also refine the rig extrinsics (first camera fixed); see --rig-out")
      ("pba-extrinsic-sigma-t", po::value(&pbaSettings.extrinsicSigmaT)->default_value(pbaSettings.extrinsicSigmaT),
       "prior of the refined extrinsics towards the rig, translation, m")
      ("pba-extrinsic-sigma-r",
       po::value(&pbaSettings.extrinsicSigmaRDeg)->default_value(pbaSettings.extrinsicSigmaRDeg),
       "prior of the refined extrinsics towards the rig, rotation, deg")
      ("brightness-out", po::value(&brightnessFile),
       "with --densify: write the affine brightness per keyframe of every densify camera (for export_colmap)")
      ("points-out", po::value(&pointsFile),
       "PLY: also write the points before the neighbour filter with their attributes (host camera, frame, pixel, "
       "distance, observations, sigma, kept) as vertex properties")
      ("pba-extrinsic-fix-scale", po::bool_switch(&pbaSettings.extrinsicFixScale),
       "photometric BA with --pba-extrinsics: hold the sum of the distances between the cameras (the metric scale), "
       "the translations otherwise free up to --pba-extrinsic-sigma-t")
      ("pba-intrinsics", po::value(&intrinsicNames)->multitoken()->zero_tokens(),
       "photometric BA: also refine the intrinsics of these cameras (no names: all); see --rig-out")
      ("pba-intrinsic-sigma-f", po::value(&pbaSettings.intrinsicSigmaFocal)->default_value(pbaSettings.intrinsicSigmaFocal),
       "prior of the refined intrinsics, focal length (relative)")
      ("pba-intrinsic-sigma-c", po::value(&pbaSettings.intrinsicSigmaCenter)->default_value(pbaSettings.intrinsicSigmaCenter),
       "prior of the refined intrinsics, principal point (px at the run scale)")
      ("rig-out", po::value(&rigOutFile), "write the rig (--rig) with the extrinsics of this run, for a new run_vo")
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
      ("densify-min-quality", po::value(&dense.trace.minQuality)->default_value(dense.trace.minQuality),
       "densify: second best / best energy along the epipolar line for a good trace")
      ("densify-verify", po::bool_switch(&dense.verify),
       "densify: refine each point's inverse depth over all buffered views and require photometric agreement")
      ("densify-verify-error", po::value(&dense.verifyMaxError)->default_value(dense.verifyMaxError),
       "densify verification: pattern rmse per pixel of an agreeing view")
      ("densify-verify-views", po::value(&dense.verifyMinViews)->default_value(dense.verifyMinViews),
       "densify verification: agreeing views with parallax a point needs")
      ("densify-verify-fraction", po::value(&dense.verifyMinFraction)->default_value(dense.verifyMinFraction),
       "densify verification: fraction of the views with parallax that must agree")
      ("densify-cameras", po::value(&densifyCameras)->multitoken(),
       "rig cameras to densify with (default: those of the run); cameras the run did not use keep the rig "
       "extrinsics and get no brightness estimate")
      ("merge", po::bool_switch(&merge), "PLY: merge duplicate points of separate passes (laps)")
      ("free-space", po::bool_switch(&freeSpace),
       "PLY: drop floaters, points that other densify host images saw through (needs --densify)")
      ("free-space-radius", po::value(&freeSpaceSettings.radius)->default_value(freeSpaceSettings.radius),
       "free-space test: a host image tests the points within this distance, m")
      ("free-space-min", po::value(&freeSpaceSettings.minThrough)->default_value(freeSpaceSettings.minThrough),
       "free-space test: views that must see through a point (and outnumber those that confirm it)")
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
    refineIntrinsics = vm.count("pba-intrinsics") > 0;
    if (up.size() != 3) throw po::error("--up takes 3 values");
    using Solver = sdv::PhotometricBASettings::Solver;
    if (pbaOptimizer == "custom") pbaSettings.optimizer = sdv::PhotometricBASettings::Optimizer::Custom;
    else if (pbaOptimizer == "ceres") pbaSettings.optimizer = sdv::PhotometricBASettings::Optimizer::Ceres;
    else throw po::error("--pba-optimizer: custom or ceres");
    if (pbaSolver == "sparse") pbaSettings.solver = Solver::SparseAmd;
    else if (pbaSolver == "nesdis") pbaSettings.solver = Solver::SparseNesdis;
    else if (pbaSolver == "iterative") pbaSettings.solver = Solver::Iterative;
    else throw po::error("--pba-solver: sparse, nesdis or iterative");
  } catch (const po::error& e) {
    spdlog::error("{}", e.what());
    std::cout << desc << '\n';
    return EXIT_FAILURE;
  }

  try {
    loopSettings.up = Eigen::Vector3d(up[0], up[1], up[2]).normalized();
    sdv::Rig rig;
    const auto records = sdv::loadKeyframeRecords(keyframesFile, &rig);
    std::vector<std::string> cameraNames;  // record camera order: the selected rig cameras in rig order
    if (!rigFile.empty())
      for (const auto& c : sdv::loadRigConfig(rigFile).cameras)
        if (rigCameras.empty() || std::find(rigCameras.begin(), rigCameras.end(), c.name) != rigCameras.end())
          cameraNames.push_back(c.name);
    auto cameraName = [&](int c) {
      return c < static_cast<int>(cameraNames.size()) ? cameraNames[c] : "cam" + std::to_string(c);
    };
    if (refineIntrinsics) {
      pbaSettings.refineIntrinsics = true;
      for (const auto& n : intrinsicNames) {
        const auto it = std::find(cameraNames.begin(), cameraNames.end(), n);
        if (it == cameraNames.end()) throw std::invalid_argument("--pba-intrinsics: unknown camera " + n);
        pbaSettings.intrinsicCameras.push_back(static_cast<int>(it - cameraNames.begin()));
      }
    }
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
    bool colourFromVideo = false;
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
        attachMasks(rigFile, rigCameras, rig, scale);
        images = loadKeyframeImages(rigFile, cameraNames, cameraNames, rig.cameras, records, scale, start, stride, true);
        colourFromVideo = true;
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
      spdlog::info("per camera pair (host -> target), rmse per pattern pixel before -> after:");
      for (const auto& p : pba.cameraPairs)
        spdlog::info("  {} -> {} {:8}  {:8} residuals  {:.2f} -> {:.2f}", cameraName(p.host), cameraName(p.target),
                     p.sameKeyframe ? "static" : "temporal", p.residuals, p.rmseBefore, p.rmseAfter);
      if (pbaSettings.refineExtrinsics) {
        for (int c = 1; c < rig.size(); ++c) {
          // Change of the camera in its own frame: T_c_b_new T_c_b_old^-1.
          const Sophus::SE3d d = pba.T_c_b[c] * rig.T_c_b[c].inverse();
          const Eigen::Vector3d r = d.so3().log() * 180.0 / M_PI;
          spdlog::info("extrinsic {}: rotation ({:+.3f}, {:+.3f}, {:+.3f}) deg ({:.3f}), translation of the camera "
                       "centre {:.3f} m",
                       cameraName(c), r.x(), r.y(), r.z(), r.norm(),
                       (pba.T_c_b[c].inverse().translation() - rig.T_c_b[c].inverse().translation()).norm());
        }
        for (int i = 0; i < rig.size(); ++i)
          for (int j = i + 1; j < rig.size(); ++j) {
            const double d0 = (rig.T_c_b[i].inverse().translation() - rig.T_c_b[j].inverse().translation()).norm();
            const double d1 = (pba.T_c_b[i].inverse().translation() - pba.T_c_b[j].inverse().translation()).norm();
            spdlog::info("baseline {} - {}: {:.4f} -> {:.4f} m ({:+.1f} mm)", cameraName(i), cameraName(j), d0, d1,
                         1000 * (d1 - d0));
          }
        rig.T_c_b = pba.T_c_b;  // for the densify pass
      }
      if (pbaSettings.refineIntrinsics) {
        for (int c = 0; c < rig.size(); ++c) {
          const sdv::Camera &a = rig.cameras[c], &b = pba.cameras[c];
          spdlog::info("intrinsics {}: fx {:+.3f} %, fy {:+.3f} %, cx {:+.2f} px, cy {:+.2f} px, alpha {:+.4f}, beta {:+.4f}",
                       cameraName(c), 100 * (b.fx / a.fx - 1), 100 * (b.fy / a.fy - 1), b.cx - a.cx, b.cy - a.cy,
                       b.alpha - a.alpha, b.beta - a.beta);
        }
        rig.cameras = pba.cameras;
      }
    }
    if (!rigOutFile.empty()) {
      if (rigFile.empty()) throw std::invalid_argument("--rig-out needs --rig");
      std::vector<std::pair<std::string, Sophus::SE3d>> extrinsics;
      for (int c = 0; c < rig.size(); ++c) extrinsics.emplace_back(cameraName(c), rig.T_c_b[c].inverse());
      std::vector<std::pair<std::string, std::array<double, 6>>> intrinsicsOut;
      if (pbaSettings.refineIntrinsics) {
        const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
        for (int c = 0; c < rig.size(); ++c)
          for (const auto& rc : config.cameras) {
            if (rc.name != cameraName(c)) continue;
            // Back to the camera YAML: undo the run scale and image_width (pixel centres are integers).
            const double s = scale * rc.imageScale;
            const sdv::Camera& k = rig.cameras[c];
            intrinsicsOut.push_back({rc.name, {k.fx / s, k.fy / s, (k.cx + 0.5) / s - 0.5, (k.cy + 0.5) / s - 0.5,
                                               k.alpha, k.beta}});
          }
      }
      sdv::writeRigConfig(rigFile, extrinsics, rigOutFile, intrinsicsOut);
      spdlog::info("wrote {}", rigOutFile);
    }
    const auto uncorrected = poses;
    auto correct = [&](const std::vector<Sophus::SE3d>& keyframePoses) {
      const sdv::PoseCorrection correction(frames, before, keyframePoses);
      std::vector<Sophus::SE3d> out = uncorrected;
      for (size_t i = 0; i < out.size(); ++i) out[i] = correction.at(static_cast<int>(i)) * out[i];
      return out;
    };
    poses = correct(after);
    // Final already: written before the densify, so the trajectory can be checked while that runs.
    if (!outFile.empty()) {
      sdv::saveKittiPoses(outFile, poses);
      spdlog::info("wrote {}", outFile);
    }
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
        std::vector<std::array<std::uint8_t, 3>> videoRgb;
        if (colourFromVideo)
          videoRgb = pointColours(rigFile, cameraNames, cameraNames, rig.cameras, records, scale, start, stride, pba);
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
            const auto rgb = colourFromVideo ? videoRgb[i] : rgbAt(images[k][c], pba.pointUv[i]);
            cloud.push_back({pba.points[i], static_cast<float>(rgb[1]), records[k].frameIndex, c, pba.pointUv[i],
                             pba.pointDistance[i], pba.pointResiduals[i], pba.pointDepthSigma[i],
                             sdv::MapPointSource::Active, rgb});
          }
        spdlog::info("{} of {} bundle-adjusted points pass distance {:.1f} m, {}+ residuals, depth sigma", cloud.size(),
                     pba.points.size(), limit, minResiduals);
      } else {
        for (const auto& p : ba.points) scene.addPoint(p, {200, 200, 200});
      }
      sdv::Rig denseRig = rig;
      if (densify) {
        const auto t0 = std::chrono::steady_clock::now();
        // All input frames with the corrected poses, keyframes as hosts: between keyframes the small baselines narrow
        // the epipolar search step by step (keyframes alone leave most traces ambiguous). Brightness between
        // keyframes is interpolated.
        std::vector<int> runCamera(static_cast<size_t>(rig.size()));  // record camera of each densify camera, or -1
        std::iota(runCamera.begin(), runCamera.end(), 0);
        std::vector<std::string> denseNames = rigCameras;
        if (!densifyCameras.empty()) {
          if (rigFile.empty()) throw std::invalid_argument("--densify-cameras needs --rig");
          denseRig = {};
          runCamera.clear();
          denseNames.clear();
          for (const auto& c : sdv::loadRigConfig(rigFile).cameras) {
            if (std::find(densifyCameras.begin(), densifyCameras.end(), c.name) == densifyCameras.end()) continue;
            const auto run = std::find(cameraNames.begin(), cameraNames.end(), c.name);
            if (run != cameraNames.end()) {
              const int r = static_cast<int>(run - cameraNames.begin());
              denseRig.cameras.push_back(rig.cameras[r]);
              denseRig.T_c_b.push_back(rig.T_c_b[r]);
              runCamera.push_back(r);
            } else {
              denseRig.cameras.push_back(sdv::prepareCamera(c.camera, scale, 16));
              denseRig.T_c_b.push_back(c.T_b_c.inverse());
              runCamera.push_back(-1);
            }
            denseNames.push_back(c.name);
          }
          if (denseNames.size() != densifyCameras.size()) throw std::invalid_argument("--densify-cameras: unknown camera");
          spdlog::info("densify with {} cameras, {} not in the run", denseRig.size(), std::ranges::count(runCamera, -1));
        }
        // Cameras the adjustment did not include get their brightness per keyframe from its points, in its gauge.
        std::vector<std::vector<sdv::AffineBrightness>> extraBrightness(static_cast<size_t>(denseRig.size()));
        if (std::ranges::count(runCamera, -1) > 0 && !photometric) {
          spdlog::warn("densify: without --photometric the cameras not in the run keep identity brightness");
        } else if (std::ranges::count(runCamera, -1) > 0) {
          std::vector<size_t> order;
          for (size_t i = 0; i < pba.points.size(); ++i)
            if (pba.pointResiduals[i] >= minResiduals) order.push_back(i);
          std::ranges::sort(order, {}, [&](size_t i) { return pba.pointHost[i]; });
          std::vector<sdv::IrradiancePoint> irradiance;
          cv::Mat host;
          for (size_t j = 0; j < order.size(); ++j) {
            const size_t i = order[j];
            const auto [k, c] = pba.pointHost[i];
            if (j == 0 || pba.pointHost[order[j - 1]] != pba.pointHost[i]) host = sdv::toFloatGray(images[k][c]);
            cv::Mat value;
            cv::getRectSubPix(host, {1, 1}, cv::Point2f(static_cast<float>(pba.pointUv[i].x()),
                                                       static_cast<float>(pba.pointUv[i].y())), value);
            const sdv::AffineBrightness& a = pba.affine[k][c];
            irradiance.push_back({pba.points[i], static_cast<float>(std::exp(-a.a) * (value.at<float>(0, 0) - a.b)), k});
          }
          std::vector<std::string> extraNames;
          std::vector<sdv::Camera> extraCameras;
          for (int c = 0; c < denseRig.size(); ++c)
            if (runCamera[c] < 0) extraNames.push_back(denseNames[c]), extraCameras.push_back(denseRig.cameras[c]);
          const auto extraImages =
              loadKeyframeImages(rigFile, extraNames, cameraNames, extraCameras, records, scale, start, stride, true);
          for (int c = 0, e = 0; c < denseRig.size(); ++c) {
            if (runCamera[c] >= 0) continue;
            std::vector<Sophus::SE3d> T_c_w;
            std::vector<cv::Mat> imgs;
            for (size_t k = 0; k < records.size(); ++k) {
              T_c_w.push_back(denseRig.T_c_b[c] * after[k].inverse());
              imgs.push_back(extraImages[k][e]);
            }
            ++e;
            const auto fit = sdv::fitCameraBrightness(denseRig.cameras[c], T_c_w, imgs, irradiance);
            extraBrightness[c] = fit.affine;
            std::vector<int> inliers;
            std::vector<double> as, bs;
            for (size_t k = 0; k < records.size(); ++k)
              if (fit.inliers[k] > 0) inliers.push_back(fit.inliers[k]), as.push_back(fit.affine[k].a), bs.push_back(fit.affine[k].b);
            auto median = [](auto v) {
              if (v.empty()) return 0.0;
              std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
              return static_cast<double>(v[v.size() / 2]);
            };
            spdlog::info("brightness of {}: {} of {} keyframes fitted, median {:.0f} inlier points, a {:+.3f} b {:+.1f} "
                         "(medians)",
                         denseNames[c], inliers.size(), records.size(), median(inliers), median(as), median(bs));
          }
        }
        auto keyframeBrightness = [&](size_t k, int c) {
          const int r = runCamera[c];
          if (r < 0) return extraBrightness[c].empty() ? sdv::AffineBrightness{} : extraBrightness[c][k];
          if (photometric) return pba.affine[k][r];
          return r < static_cast<int>(records[k].affine.size()) ? records[k].affine[r] : sdv::AffineBrightness{};
        };
        if (!brightnessFile.empty()) {
          std::ofstream out(brightnessFile);
          if (!out) throw std::runtime_error("cannot write " + brightnessFile);
          out << "# Affine brightness per keyframe and densify camera, I = exp(a) J + b (J: irradiance in the gauge of "
                 "the adjusted cameras)\n# frame camera a b\n";
          for (size_t k = 0; k < records.size(); ++k)
            for (int c = 0; c < denseRig.size(); ++c) {
              const auto a = keyframeBrightness(k, c);
              out << fmt::format("{} {} {:.6f} {:.4f}\n", records[k].frameIndex,
                                 denseNames.empty() ? cameraName(c) : denseNames[c], a.a, a.b);
            }
          spdlog::info("wrote {}", brightnessFile);
        }
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
                             : rigFrames(rigFile, denseNames, cameraNames, denseRig, scale, start, stride);
        images = {};  // not needed any more: densify streams its frames (~10 GB on long runs)
        sdv::SemiDenseMapper mapper(denseRig, dense);
        size_t k = 0;
        for (size_t i = 0; i < poses.size(); ++i) {
          const std::vector<cv::Mat> frame = nextFrame();
          if (frame.empty()) break;
          while (k + 1 < records.size() && records[k + 1].frameIndex <= static_cast<int>(i)) ++k;
          const bool host = records[k].frameIndex == static_cast<int>(i);
          std::vector<sdv::AffineBrightness> brightness(static_cast<size_t>(denseRig.size()));
          const int f0 = records[k].frameIndex;
          const int f1 = k + 1 < records.size() ? records[k + 1].frameIndex : f0;
          const double w = f1 > f0 ? std::clamp((static_cast<double>(i) - f0) / (f1 - f0), 0.0, 1.0) : 0.0;
          for (int c = 0; c < denseRig.size(); ++c) {
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
                     "imprecise, {} by verification), {} after voxel check, {} within {:.1f} m, {:.1f} s",
                     st.candidates, static_cast<double>(st.good) / std::max<long long>(st.candidates, 1), st.accepted,
                     st.rejectMatches, st.rejectInterval, st.rejectVerify, densePoints.size(), added, limit,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      }
      if (merge) {
        size_t merged = 0;
        const size_t unmerged = cloud.size();
        cloud = sdv::mergeDuplicatePoints(cloud, mergeSettings, &merged);
        spdlog::info("merge: {} -> {} points ({} pairs within {:.3f} m, hosts {}+ frames apart)", unmerged, cloud.size(),
                     merged, mergeSettings.maxDistance, mergeSettings.minFrameGap);
      }
      if (freeSpace && densify) {
        const auto t0 = std::chrono::steady_clock::now();
        const auto fs = sdv::freeSpaceFloaters(denseRig, poses, cloud, freeSpaceSettings);
        const size_t before = cloud.size();
        size_t w = 0;
        for (size_t i = 0; i < cloud.size(); ++i)
          if (!fs.floater[i]) cloud[w++] = cloud[i];
        cloud.resize(w);
        spdlog::info("free space: {} of {} points seen through by {}+ of {} host images, removed ({:.1f} s)",
                     before - cloud.size(), before, freeSpaceSettings.minThrough, fs.views,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
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
      if (!pointsFile.empty()) {
        sdv::writeMapPointsPly(pointsFile, cloud, &keep);
        spdlog::info("wrote {} ({} points with attributes, before the neighbour filter)", pointsFile, cloud.size());
      }
      scene.write(plyFile);
      spdlog::info("wrote {}", plyFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
