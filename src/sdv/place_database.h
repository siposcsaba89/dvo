#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include <opencv2/core.hpp>

namespace fbow {
class Vocabulary;
}

namespace sdv {

// Bag-of-words vector: word id -> weight, L1-normalised.
using BowVector = std::vector<std::pair<std::uint32_t, float>>;

class Vocabulary {
 public:
  explicit Vocabulary(const std::filesystem::path& file);
  // Hierarchical k-means (FBoW) on descriptor rows of all images: branching factor k, depth levels.
  static void train(const std::vector<cv::Mat>& descriptors, int k, int levels, const std::filesystem::path& out);
  ~Vocabulary();

  BowVector transform(const cv::Mat& descriptors) const;

 private:
  std::unique_ptr<fbow::Vocabulary> m_voc;
};

struct PlaceMatch {
  int image;
  double score;
};

// Inverted-file image retrieval (Nister & Stewenius, CVPR 2006) with the L1 score of Galvez-Lopez & Tardos,
// T-RO 2012: s = 1 - |v/|v| - w/|w||_1 / 2, in [0, 1].
class PlaceDatabase {
 public:
  int add(BowVector v);
  // Best images first; `accept` filters candidates (e.g. excludes recent keyframes).
  std::vector<PlaceMatch> query(const BowVector& v, size_t maxResults,
                                const std::function<bool(int)>& accept = {}) const;
  const BowVector& vector(int image) const { return m_vectors[image]; }
  size_t size() const { return m_vectors.size(); }

  static double score(const BowVector& a, const BowVector& b);

 private:
  std::vector<BowVector> m_vectors;
  std::unordered_map<std::uint32_t, std::vector<std::pair<int, float>>> m_inverted;
};

}  // namespace sdv
