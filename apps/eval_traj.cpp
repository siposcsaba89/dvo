#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/eval/trajectory.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string gtFile, estFile, plyFile;
  bool se3 = false;

  po::options_description desc("eval_traj options");
  desc.add_options()
      ("help,h", "show help")
      ("gt", po::value(&gtFile)->required(), "ground-truth poses (KITTI format)")
      ("est", po::value(&estFile)->required(), "estimated poses (KITTI format)")
      ("se3", po::bool_switch(&se3), "rigid alignment only (metric estimates)")
      ("ply", po::value(&plyFile), "write gt (green) and aligned estimate (red) as PLY");
  po::positional_options_description pos;
  pos.add("gt", 1).add("est", 1);

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

  try {
    const auto gt = sdv::loadKittiPoses(gtFile);
    auto est = sdv::loadKittiPoses(estFile);
    if (est.size() != gt.size())
      spdlog::warn("gt has {} poses, estimate has {}; using the first {}", gt.size(), est.size(),
                   std::min(gt.size(), est.size()));

    const auto ate = sdv::absoluteTrajectoryError(gt, est, !se3);
    spdlog::info("alignment: {} scale={:.4f}", se3 ? "SE3" : "Sim3", ate.alignment.scale);
    spdlog::info("ATE [m]: rmse={:.3f} mean={:.3f} median={:.3f} max={:.3f}", ate.rmse, ate.mean,
                 ate.median, ate.max);

    for (auto& p : est) p = ate.alignment.applyToPose(p);
    const auto seg = sdv::segmentDriftError(gt, est);
    spdlog::info("segment drift: t={:.2f} %  r={:.3f} deg/100m  ({} segments)", seg.translationPercent,
                 seg.rotationDegPer100m, seg.numSegments);

    if (!plyFile.empty()) {
      sdv::PlyScene scene;
      scene.addTrajectory(gt, {0, 200, 0});
      scene.addTrajectory(est, {220, 0, 0});
      scene.write(plyFile);
      spdlog::info("wrote {}", plyFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
