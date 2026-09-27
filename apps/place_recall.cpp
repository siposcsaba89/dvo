// Place-recognition quality on run_vo keyframe records: every keyframe queries the earlier ones (all cameras against
// all cameras, so a revisit in the opposite direction can match front against rear) and the best candidate is
// compared with the true revisits, from ground-truth positions (KITTI) or the odometry poses themselves.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/io/kitti.h>
#include <sdv/keyframe_features.h>
#include <sdv/place_database.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string keyframesFile, vocabularyFile, gtFile;
  int minGap = 150, start = 0, stride = 1;
  double radius = 5.0;
  po::options_description desc("place_recall options");
  desc.add_options()
      ("help,h", "show help")
      ("keyframes", po::value(&keyframesFile)->required(), "run_vo --keyframes-out file")
      ("vocabulary", po::value(&vocabularyFile)->required(), "FBoW vocabulary")
      ("gt", po::value(&gtFile), "ground-truth poses per input frame (KITTI format); default: odometry poses")
      ("start", po::value(&start)->default_value(0), "run_vo --start of the records")
      ("stride", po::value(&stride)->default_value(1), "run_vo --stride of the records")
      ("min-gap", po::value(&minGap)->default_value(150), "candidates are at least this many frames older")
      ("radius", po::value(&radius)->default_value(5.0), "a candidate within this distance is a true revisit");
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
    sdv::Rig rig;
    const auto records = sdv::loadKeyframeRecords(keyframesFile, &rig);
    const sdv::Vocabulary voc(vocabularyFile);
    const auto gt = gtFile.empty() ? std::vector<Sophus::SE3d>{} : sdv::loadKittiPoses(gtFile);
    const int nc = rig.size();
    std::vector<Eigen::Vector3d> position;
    for (const auto& r : records) {
      const size_t input = static_cast<size_t>(start + r.frameIndex * stride);
      position.push_back(gt.empty() ? r.T_w_b.translation() : gt.at(input).translation());
    }

    sdv::PlaceDatabase db;  // image = record * nc + camera
    for (const auto& r : records)
      for (const auto& f : r.cameras) db.add(voc.transform(f.descriptors));

    struct Query {
      bool hasRevisit;
      bool correct;
      double score;  // best candidate, normalised by the score against the previous keyframe
      int queryCam, matchCam;
    };
    std::vector<Query> queries;
    for (size_t q = 1; q < records.size(); ++q) {
      auto eligible = [&](size_t r) { return records[r].frameIndex + minGap <= records[q].frameIndex; };
      bool hasRevisit = false;
      for (size_t r = 0; r < q && !hasRevisit; ++r) hasRevisit = eligible(r) && (position[r] - position[q]).norm() < radius;

      // Normaliser (Galvez-Lopez & Tardos 2012): how similar this place is to itself a moment ago.
      double self = 0;
      for (int c = 0; c < nc; ++c)
        self = std::max(self, sdv::PlaceDatabase::score(db.vector(static_cast<int>(q * nc + c)),
                                                        db.vector(static_cast<int>((q - 1) * nc + c))));
      int best = -1, bestQc = 0, bestMc = 0;
      double bestScore = 0;
      for (int c = 0; c < nc; ++c) {
        const auto matches = db.query(db.vector(static_cast<int>(q * nc + c)), 1,
                                      [&](int image) { return eligible(static_cast<size_t>(image / nc)); });
        if (!matches.empty() && matches[0].score > bestScore)
          bestScore = matches[0].score, best = matches[0].image / nc, bestQc = c, bestMc = matches[0].image % nc;
      }
      if (best < 0) continue;
      queries.push_back({hasRevisit, (position[best] - position[q]).norm() < radius, self > 0 ? bestScore / self : 0,
                         bestQc, bestMc});
    }

    const auto revisits = std::count_if(queries.begin(), queries.end(), [](const Query& q) { return q.hasRevisit; });
    const auto top1 = std::count_if(queries.begin(), queries.end(), [](const Query& q) { return q.hasRevisit && q.correct; });
    spdlog::info("{} keyframes, {} queries with a true revisit, top-1 correct {} ({:.0f} %)", records.size(), revisits,
                 top1, revisits ? 100.0 * top1 / revisits : 0.0);
    // Threshold sweep on the normalised score: accepted candidates, of those correct.
    std::vector<Query> sorted = queries;
    std::sort(sorted.begin(), sorted.end(), [](const Query& a, const Query& b) { return a.score > b.score; });
    int accepted = 0, correct = 0, recallAtFullPrecision = 0;
    double thresholdAtFullPrecision = 0;
    for (const auto& q : sorted) {
      ++accepted;
      correct += q.correct;
      if (correct == accepted) recallAtFullPrecision = correct, thresholdAtFullPrecision = q.score;
    }
    spdlog::info("normalised score: {} correct candidates accepted before the first wrong one (threshold {:.2f}); "
                 "recall there {:.0f} % of the revisits",
                 recallAtFullPrecision, thresholdAtFullPrecision,
                 revisits ? 100.0 * recallAtFullPrecision / revisits : 0.0);
    for (double t : {0.5, 1.0, 1.5, 2.0, 3.0}) {
      int a = 0, c = 0;
      for (const auto& q : queries)
        if (q.score >= t) ++a, c += q.correct;
      spdlog::info("  threshold {:.1f}: {} accepted, {} correct, recall {:.0f} %", t, a, c,
                   revisits ? 100.0 * c / revisits : 0.0);
    }
    std::vector<int> pairs(nc * nc, 0);
    for (const auto& q : queries)
      if (q.correct) ++pairs[q.queryCam * nc + q.matchCam];
    for (int a = 0; a < nc; ++a)
      for (int b = 0; b < nc; ++b)
        if (pairs[a * nc + b] > 0) spdlog::info("  correct matches camera {} -> camera {}: {}", a, b, pairs[a * nc + b]);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
