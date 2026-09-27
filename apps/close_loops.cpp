// Loop closure without rerunning odometry: loop detection on run_vo keyframe records, SE3 pose graph over the
// keyframes, and the correction applied to the run_vo poses of all frames. Metric runs only (stereo or rig), where
// every input frame from the first one has a pose, so line i of the pose file is run frame i.
#include <cstdlib>
#include <exception>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/io/kitti.h>
#include <sdv/loop_detector.h>
#include <sdv/pose_graph.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string keyframesFile, vocabularyFile, posesFile, outFile, gtFile;
  std::vector<double> up{0, 0, 1};
  int start = 0, stride = 1;
  sdv::LoopSettings loopSettings;
  sdv::PoseGraphSettings graphSettings;
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
       "loop edge rotation sigma, deg");
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
    const sdv::PoseCorrection correction(frames, before, graph.T_w_b);
    const auto uncorrected = poses;
    for (size_t i = 0; i < poses.size(); ++i) poses[i] = correction.at(static_cast<int>(i)) * poses[i];
    spdlog::info("{} keyframes, {} loops ({} geometric), {} rejected by the pose graph; cost {:.1f} -> {:.1f}",
                 records.size(), loops.size(), found.size(), graph.loopsRejected, graph.initialCost, graph.finalCost);
    if (!gtFile.empty()) {
      const auto allGt = sdv::loadKittiPoses(gtFile);
      std::vector<Sophus::SE3d> gt;
      for (size_t i = 0; i < poses.size(); ++i) gt.push_back(allGt.at(static_cast<size_t>(start) + i * stride));
      const auto a = sdv::absoluteTrajectoryError(gt, uncorrected, false);
      const auto b = sdv::absoluteTrajectoryError(gt, poses, false);
      spdlog::info("ATE SE3 rmse {:.3f} m -> {:.3f} m, max {:.3f} m -> {:.3f} m", a.rmse, b.rmse, a.max, b.max);
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
