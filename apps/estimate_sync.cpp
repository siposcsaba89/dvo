// Frame offsets between cameras of a rig from independent monocular runs. All cameras of a rigid body share the
// angular velocity, which the extrinsic rotations map into the body frame; its fast part (vibration, bumps) is
// sharp in time, so the correlation of two synchronised cameras peaks at lag 0. Monocular scale does not matter.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/io/kitti.h>
#include <sdv/io/rig_config.h>

namespace po = boost::program_options;

namespace {

using Series = std::vector<Eigen::Vector3d>;

// Body-frame rotation increments between consecutive camera poses (T_w_c), minus their moving average.
Series fastAngularVelocity(const std::vector<Sophus::SE3d>& poses, const Sophus::SO3d& R_b_c, int window) {
  Series w;
  for (size_t i = 1; i < poses.size(); ++i)
    w.push_back(R_b_c * (poses[i - 1].so3().inverse() * poses[i].so3()).log());
  Series out(w.size(), Eigen::Vector3d::Zero());
  for (size_t i = 0; i < w.size(); ++i) {
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    int n = 0;
    for (int d = -window; d <= window; ++d) {
      const long long j = static_cast<long long>(i) + d;
      if (j >= 0 && j < static_cast<long long>(w.size())) mean += w[j], ++n;
    }
    out[i] = w[i] - mean / n;
  }
  return out;
}

// Normalised correlation of a[i] and b[i + lag] over their common input frames.
double correlation(const Series& a, int firstA, const Series& b, int firstB, int lag) {
  double saa = 0, sbb = 0, sab = 0;
  int n = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    const long long j = static_cast<long long>(i) + firstA + lag - firstB;
    if (j < 0 || j >= static_cast<long long>(b.size())) continue;
    saa += a[i].squaredNorm(), sbb += b[j].squaredNorm(), sab += a[i].dot(b[j]);
    ++n;
  }
  return n >= 50 && saa > 0 && sbb > 0 ? sab / std::sqrt(saa * sbb) : -1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string rigFile;
  std::vector<std::string> poseFiles;
  std::vector<int> frameCounts;
  int maxLag = 25, window = 4;
  po::options_description desc("estimate_sync options");
  desc.add_options()
      ("help,h", "show help")
      ("rig", po::value(&rigFile)->required(), "rig YAML (extrinsics)")
      ("poses", po::value(&poseFiles)->multitoken()->required(),
       "run_vo -o pose files of single-camera runs, in rig camera order (the first one is the reference)")
      ("frames", po::value(&frameCounts)->multitoken()->required(),
       "number of processed input frames per run (poses cover the last frames of each run)")
      ("max-lag", po::value(&maxLag)->default_value(25), "largest offset tried, frames")
      ("window", po::value(&window)->default_value(4), "half width of the moving average removed, frames");
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
    const sdv::RigConfig rig = sdv::loadRigConfig(rigFile);
    if (poseFiles.size() != rig.cameras.size() || frameCounts.size() != rig.cameras.size())
      throw std::invalid_argument("give one pose file and one frame count per rig camera");
    std::vector<Series> series;
    std::vector<int> first;  // input frame of the first increment
    for (size_t k = 0; k < poseFiles.size(); ++k) {
      const auto poses = sdv::loadKittiPoses(poseFiles[k]);
      first.push_back(frameCounts[k] - static_cast<int>(poses.size()));
      series.push_back(fastAngularVelocity(poses, rig.cameras[k].T_b_c.so3(), window));
    }
    for (size_t k = 1; k < series.size(); ++k) {
      int best = 0;
      double bestCorr = -2;
      std::string curve;
      for (int lag = -maxLag; lag <= maxLag; ++lag) {
        const double c = correlation(series[0], first[0], series[k], first[k], lag);
        if (std::abs(lag) <= 3) curve += fmt::format(" {:+d}:{:.2f}", lag, c);
        if (c > bestCorr) bestCorr = c, best = lag;
      }
      // Positive offset: frame i of the reference shows the same instant as frame i + offset of this camera.
      spdlog::info("{}: offset {:+d} frames relative to {} (correlation {:.3f};{})", rig.cameras[k].name, best,
                   rig.cameras[0].name, bestCorr, curve);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
