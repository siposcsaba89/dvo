// Loop detection on run_vo keyframe records: place recognition, geometric verification, odometry-consistency and
// temporal checks. Writes the accepted loop constraints; with ground truth, every loop is checked against it.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/io/kitti.h>
#include <sdv/loop_detector.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string keyframesFile, vocabularyFile, gtFile, outFile;
  std::vector<double> up{0, 0, 1};
  int start = 0, stride = 1;
  double revisitRadius = 5.0;
  sdv::LoopSettings settings;
  po::options_description desc("detect_loops options");
  desc.add_options()
      ("help", "show help")
      ("keyframes", po::value(&keyframesFile)->required(), "run_vo --keyframes-out file")
      ("vocabulary", po::value(&vocabularyFile)->required(), "FBoW vocabulary")
      ("out", po::value(&outFile), "write accepted loops (text)")
      ("gt", po::value(&gtFile), "ground-truth body poses per input frame (KITTI format) for evaluation")
      ("start", po::value(&start)->default_value(0), "run_vo --start of the records")
      ("stride", po::value(&stride)->default_value(1), "run_vo --stride of the records")
      ("up", po::value(&up)->multitoken(), "world vertical (default 0 0 1; KITTI: 0 -1 0)")
      ("min-inliers", po::value(&settings.minInliers)->default_value(settings.minInliers), "inliers a loop needs")
      ("candidates", po::value(&settings.candidates)->default_value(settings.candidates), "candidates per keyframe")
      ("min-score", po::value(&settings.minNormalizedScore)->default_value(settings.minNormalizedScore),
       "normalised BoW score a candidate needs")
      ("min-consistent", po::value(&settings.minConsistent)->default_value(settings.minConsistent),
       "agreeing loops of neighbouring keyframes a loop needs")
      ("revisit-radius", po::value(&revisitRadius)->default_value(5.0), "ground truth: revisit distance")
      ("verbose", "debug logging");
  try {
    po::variables_map vm;
    // Without short options, negative values (--up 0 -1 0) are values.
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
    if (vm.count("verbose")) spdlog::set_level(spdlog::level::debug);
  } catch (const po::error& e) {
    spdlog::error("{}", e.what());
    std::cout << desc << '\n';
    return EXIT_FAILURE;
  }

  try {
    settings.up = Eigen::Vector3d(up[0], up[1], up[2]).normalized();
    sdv::Rig rig;
    const auto records = sdv::loadKeyframeRecords(keyframesFile, &rig);
    auto voc = std::make_shared<const sdv::Vocabulary>(vocabularyFile);
    sdv::LoopDetector detector(rig, voc, settings);
    std::vector<sdv::LoopConstraint> loops;
    for (const auto& r : records) {
      const auto found = detector.addKeyframe(r);
      loops.insert(loops.end(), found.begin(), found.end());
    }
    const auto accepted = detector.temporallyConsistent(loops);
    const auto& st = detector.stats();
    spdlog::info("{} keyframes: {} candidates verified, rejected {} few matches, {} few inliers, {} inconsistent with "
                 "odometry; {} geometric loops, {} temporally consistent",
                 records.size(), st.candidates, st.fewMatches, st.fewInliers, st.inconsistentOdometry, loops.size(),
                 accepted.size());

    auto frame = [&](int record) { return start + records[record].frameIndex * stride; };
    if (!gtFile.empty()) {
      const auto gt = sdv::loadKittiPoses(gtFile);
      auto check = [&](const std::vector<sdv::LoopConstraint>& ls, const char* name) {
        int good = 0;
        std::vector<double> errT, errH, errV, errR;
        for (const auto& l : ls) {
          const Sophus::SE3d truth = gt.at(frame(l.query)).inverse() * gt.at(frame(l.match));
          const Sophus::SE3d err = l.T_q_m * truth.inverse();
          const double t = err.translation().norm(), r = err.so3().log().norm() * 180.0 / M_PI;
          const Eigen::Vector3d e = gt.at(frame(l.query)).so3() * err.translation();
          const double v = std::abs(e.dot(settings.up));
          const bool ok = t < 1.0 && r < 2.0;
          good += ok;
          errT.push_back(t), errR.push_back(r), errV.push_back(v), errH.push_back(std::sqrt(std::max(0.0, t * t - v * v)));
          if (t > 3.0 || r > 5.0)
            spdlog::info("  {} wrong: frame {} -> {}: {:.2f} m {:.2f} deg ({} inliers)", name, frame(l.query),
                         frame(l.match), t, r, l.inliers);
        }
        auto pct = [](std::vector<double> v, double p) {
          if (v.empty()) return 0.0;
          std::sort(v.begin(), v.end());
          return v[static_cast<size_t>(p * (v.size() - 1))];
        };
        spdlog::info("{}: {} of {} loops within 1 m, 2 deg of ground truth; error 50/90/max % {:.2f} / {:.2f} / "
                     "{:.2f} m, {:.2f} / {:.2f} / {:.2f} deg",
                     name, good, ls.size(), pct(errT, 0.5), pct(errT, 0.9), pct(errT, 1.0), pct(errR, 0.5),
                     pct(errR, 0.9), pct(errR, 1.0));
        spdlog::info("{}: horizontal error 50/90 % {:.2f} / {:.2f} m, vertical {:.2f} / {:.2f} m", name, pct(errH, 0.5),
                     pct(errH, 0.9), pct(errV, 0.5), pct(errV, 0.9));
      };
      check(loops, "geometric");
      check(accepted, "consistent");
      int revisits = 0, covered = 0;
      for (size_t q = 0; q < records.size(); ++q) {
        bool revisit = false;
        for (size_t m = 0; m < q && !revisit; ++m)
          revisit = records[m].frameIndex + settings.minGapFrames <= records[q].frameIndex &&
                    (gt.at(frame(static_cast<int>(m))).translation() - gt.at(frame(static_cast<int>(q))).translation())
                            .norm() < revisitRadius;
        if (!revisit) continue;
        ++revisits;
        covered += std::any_of(accepted.begin(), accepted.end(),
                               [&](const sdv::LoopConstraint& l) { return l.query == static_cast<int>(q); });
      }
      spdlog::info("keyframes with a true revisit: {}, with an accepted loop: {} ({:.0f} %)", revisits, covered,
                   revisits ? 100.0 * covered / revisits : 0.0);
    }
    if (!accepted.empty()) {
      std::vector<double> shift;
      for (const auto& l : accepted) {
        const Sophus::SE3d& T_w_q = records[l.query].T_w_b;
        const Sophus::SE3d c = detector.correction(l);
        shift.push_back(((c * T_w_q).translation() - T_w_q.translation()).norm());
        spdlog::info("  loop frame {} -> {}: {} inliers, {:.2f} px, score {:.2f}, correction {:.2f} m {:.2f} deg",
                     frame(l.query), frame(l.match), l.inliers, l.rmsePixels, l.score, shift.back(),
                     c.so3().log().norm() * 180.0 / M_PI);
      }
      std::sort(shift.begin(), shift.end());
      spdlog::info("odometry correction implied by accepted loops: median {:.2f} m, max {:.2f} m", shift[shift.size() / 2],
                   shift.back());
    }
    if (!outFile.empty()) {
      std::ofstream out(outFile);
      out << "# query_frame match_frame tx ty tz qx qy qz qw inliers rmse_px (T_q_m: match body -> query body)\n";
      for (const auto& l : accepted) {
        const Eigen::Quaterniond q = l.T_q_m.unit_quaternion();
        const Eigen::Vector3d t = l.T_q_m.translation();
        out << fmt::format("{} {} {:.6f} {:.6f} {:.6f} {:.9f} {:.9f} {:.9f} {:.9f} {} {:.3f}\n", frame(l.query),
                           frame(l.match), t.x(), t.y(), t.z(), q.x(), q.y(), q.z(), q.w(), l.inliers, l.rmsePixels);
      }
      spdlog::info("wrote {}", outFile);
    }
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
