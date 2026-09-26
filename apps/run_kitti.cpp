#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/image_pyramid.h>
#include <sdv/io/kitti.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string sequenceDir;
  size_t maxFrames = 0;
  int pyramidLevels = 5;
  bool verbose = false;

  po::options_description desc("run_kitti options");
  desc.add_options()
      ("help,h", "show help")
      ("sequence,s", po::value(&sequenceDir)->required(), "KITTI sequence directory (contains calib.txt, image_0)")
      ("max-frames,n", po::value(&maxFrames)->default_value(0), "process at most N frames (0 = all)")
      ("levels", po::value(&pyramidLevels)->default_value(5), "image pyramid levels")
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
    const sdv::KittiSequence seq(sequenceDir, 1 << (pyramidLevels - 1));
    const auto& cam = seq.camera();
    spdlog::info("sequence: {} frames, {}x{}, fx={:.2f} fy={:.2f} cx={:.2f} cy={:.2f}, baseline={:.4f} m",
                 seq.size(), cam.width, cam.height, cam.fx, cam.fy, cam.cx, cam.cy, seq.baseline());
    for (int l = 0; l < pyramidLevels; ++l) {
      const auto c = cam.atLevel(l);
      spdlog::debug("level {}: {}x{} fx={:.3f} cx={:.3f} cy={:.3f}", l, c.width, c.height, c.fx, c.cx, c.cy);
    }

    const size_t n = maxFrames == 0 ? seq.size() : std::min(maxFrames, seq.size());
    using Clock = std::chrono::steady_clock;
    double loadMs = 0, pyrMs = 0;
    for (size_t i = 0; i < n; ++i) {
      const auto t0 = Clock::now();
      const cv::Mat img = sdv::toFloatGray(seq.loadImage(i));
      const auto t1 = Clock::now();
      const sdv::ImagePyramid pyr(img, pyramidLevels);
      const auto t2 = Clock::now();
      loadMs += std::chrono::duration<double, std::milli>(t1 - t0).count();
      pyrMs += std::chrono::duration<double, std::milli>(t2 - t1).count();
      if (i % 500 == 0) spdlog::info("frame {}/{}", i, n);
    }
    spdlog::info("avg per frame: load {:.2f} ms, pyramid {:.2f} ms", loadMs / n, pyrMs / n);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
