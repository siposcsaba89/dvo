// Prints the properties of a video and optionally saves some frames as PNG.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <spdlog/spdlog.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string videoFile, outDir;
  std::vector<int> frames;
  bool count = false;

  po::options_description desc("video_frames options");
  desc.add_options()
      ("help,h", "show help")
      ("video", po::value(&videoFile)->required(), "video file")
      ("frames", po::value(&frames)->multitoken(), "frame indices to save")
      ("out,o", po::value(&outDir)->default_value("."), "output directory for saved frames")
      ("count", po::bool_switch(&count), "decode the whole video to count frames exactly");
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
    cv::VideoCapture cap(videoFile, cv::CAP_FFMPEG);
    if (!cap.isOpened()) throw std::runtime_error("cannot open " + videoFile);
    spdlog::info("{}: {}x{}, {:.2f} fps, {} frames (container estimate)", videoFile,
                 static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)), static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT)),
                 cap.get(cv::CAP_PROP_FPS), static_cast<long long>(cap.get(cv::CAP_PROP_FRAME_COUNT)));
    std::filesystem::create_directories(outDir);
    int last = -1;
    for (int f : frames) last = std::max(last, f);
    cv::Mat img;
    int i = 0;
    for (; (count || i <= last) && cap.read(img); ++i) {
      if (std::find(frames.begin(), frames.end(), i) == frames.end()) continue;
      const auto file = std::filesystem::path(outDir) / fmt::format("frame_{:06d}.png", i);
      cv::imwrite(file.string(), img);
      spdlog::info("wrote {} ({}x{}, {} channels)", file.string(), img.cols, img.rows, img.channels());
    }
    if (count) spdlog::info("{} frames decoded", i);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
