// Trains an FBoW vocabulary for place recognition from the ORB descriptors of run_vo keyframe records.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/keyframe_features.h>
#include <sdv/place_database.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::vector<std::string> inputs;
  std::string outFile;
  int k = 10, levels = 5;
  size_t maxImages = 0;
  po::options_description desc("train_vocabulary options");
  desc.add_options()
      ("help,h", "show help")
      ("keyframes", po::value(&inputs)->multitoken()->required(), "run_vo --keyframes-out files")
      ("out,o", po::value(&outFile)->required(), "output vocabulary (.fbow)")
      ("k", po::value(&k)->default_value(10), "branching factor")
      ("levels", po::value(&levels)->default_value(5), "tree depth (k^levels words)")
      ("max-images", po::value(&maxImages)->default_value(0), "random subset of keyframe images (0 = all)");
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
    std::vector<cv::Mat> descriptors;
    for (const auto& file : inputs)
      for (const auto& r : sdv::loadKeyframeRecords(file))
        for (const auto& f : r.cameras)
          if (!f.descriptors.empty()) descriptors.push_back(f.descriptors);
    if (maxImages > 0 && descriptors.size() > maxImages) {
      std::mt19937 rng(1);
      std::shuffle(descriptors.begin(), descriptors.end(), rng);
      descriptors.resize(maxImages);
    }
    size_t rows = 0;
    for (const auto& d : descriptors) rows += d.rows;
    spdlog::info("training k={} levels={} on {} descriptors from {} images", k, levels, rows, descriptors.size());
    sdv::Vocabulary::train(descriptors, k, levels, outFile);
    spdlog::info("wrote {}", outFile);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
