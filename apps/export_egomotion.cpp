// Writes run poses (KITTI T_world_body per run frame, e.g. close_loops' poses_loop.txt) of an aiMotive recording as
// egomotion2.json: one entry per frame id with RT_ECEF_body (4x4 row-major), enh_sep, rph_sep, time, time_host.
// The world is the run's local frame (first frame's body, levelled after close_loops), not geodetic ECEF.
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/aimrec/recording.h>
#include <sdv/io/kitti.h>
#include <sdv/io/rig_config.h>

namespace po = boost::program_options;
namespace fs = std::filesystem;

int main(int argc, char** argv) {
  std::string rigFile, posesFile, outFile;
  std::vector<std::string> runCameras;
  std::size_t start = 0, stride = 1;
  po::options_description desc("export_egomotion options");
  desc.add_options()
      ("help,h", "show help")
      ("rig", po::value(&rigFile)->required(), "rig YAML with aim_record (as given to run_vo)")
      ("rig-cameras", po::value(&runCameras)->multitoken(), "cameras of the VO run (default: all rig cameras); "
                                                            "their common frame ids are the run frames")
      ("poses", po::value(&posesFile)->required(), "KITTI poses, T_world_body per run frame")
      ("start", po::value(&start)->default_value(0), "run_vo --start")
      ("stride", po::value(&stride)->default_value(1), "run_vo --stride")
      ("out,o", po::value(&outFile)->default_value("egomotion2.json"), "output JSON");
  try {
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc << "\n";
      return EXIT_SUCCESS;
    }
    po::notify(vm);

    const sdv::RigConfig rig = sdv::loadRigConfig(rigFile);
    if (rig.aimRecord.empty()) throw std::invalid_argument(rigFile + " has no aim_record");
    if (runCameras.empty())
      for (const auto& c : rig.cameras) runCameras.push_back(c.name);
    const sdv::aimrec::Recording rec(rig.aimRecord, rig.aimCalibration, runCameras, false);
    const auto& ids = rec.syncedFrameIds();
    const auto poses = sdv::loadKittiPoses(posesFile);
    if (stride == 0 || start + (poses.size() - 1) * stride >= ids.size())
      throw std::runtime_error(fmt::format("{} poses with start {} stride {} exceed the {} synced frames",
                                           poses.size(), start, stride, ids.size()));
    if (start + poses.size() * stride < ids.size())
      spdlog::warn("{} poses cover only {} of the {} synced frames", poses.size(),
                   (poses.size() - 1) * stride + 1, ids.size() - start);

    fs::path out(outFile);
    if (out.has_parent_path()) fs::create_directories(out.parent_path());
    std::ofstream f(out);
    if (!f) throw std::runtime_error("cannot write " + outFile);
    f << "{";
    for (std::size_t i = 0; i < poses.size(); ++i) {
      const std::uint64_t id = ids[start + i * stride];
      const Eigen::Matrix4d m = poses[i].matrix();
      const double t = static_cast<double>(rec.timestampNs(id)) * 1e-9;
      f << fmt::format("{}\n  \"{}\": {{\n    \"RT_ECEF_body\": [", i ? "," : "", id);
      for (int r = 0; r < 4; ++r)
        f << fmt::format("{}[{:.9g}, {:.9g}, {:.9g}, {:.9g}]", r ? ", " : "", m(r, 0), m(r, 1), m(r, 2), m(r, 3));
      f << fmt::format("],\n    \"enh_sep\": [0.0, 0.0, 0.0],\n    \"rph_sep\": [0.0, 0.0, 0.0],\n"
                       "    \"time\": {:.9f},\n    \"time_host\": {:.9f}\n  }}",
                       t, t);
    }
    f << "\n}\n";
    spdlog::info("wrote {} frames (ids {} .. {}) to {}", poses.size(), ids[start],
                 ids[start + (poses.size() - 1) * stride], outFile);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    std::cerr << desc << "\n";
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
