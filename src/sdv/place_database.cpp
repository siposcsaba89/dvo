#include <sdv/place_database.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <thread>

#include <fbow/vocabulary_creator.h>

namespace sdv {

namespace {

BowVector normalized(const fbow::BoWVector& v) {
  BowVector out(v.begin(), v.end());
  double sum = 0;
  for (const auto& [w, x] : out) sum += std::abs(x);
  if (sum > 0)
    for (auto& [w, x] : out) x = static_cast<float>(x / sum);
  return out;
}

}  // namespace

Vocabulary::Vocabulary(const std::filesystem::path& file) : m_voc(std::make_unique<fbow::Vocabulary>()) {
  if (!std::filesystem::exists(file)) throw std::runtime_error("vocabulary not found: " + file.string());
  m_voc->readFromFile(file.string());
  if (!m_voc->isValid()) throw std::runtime_error("invalid vocabulary: " + file.string());
}

Vocabulary::~Vocabulary() = default;

void Vocabulary::train(const std::vector<cv::Mat>& descriptors, int k, int levels, const std::filesystem::path& out) {
  fbow::Vocabulary voc;
  fbow::VocabularyCreator creator;
  fbow::VocabularyCreator::Params params(static_cast<uint32_t>(k), levels,
                                         std::max(1u, std::thread::hardware_concurrency()));
  creator.create(voc, descriptors, "orb", params);
  voc.saveToFile(out.string());
}

BowVector Vocabulary::transform(const cv::Mat& descriptors) const {
  if (descriptors.empty()) return {};
  return normalized(m_voc->transform(descriptors));
}

int PlaceDatabase::add(BowVector v) {
  const int id = static_cast<int>(m_vectors.size());
  for (const auto& [w, x] : v) m_inverted[w].emplace_back(id, x);
  m_vectors.push_back(std::move(v));
  return id;
}

std::vector<PlaceMatch> PlaceDatabase::query(const BowVector& v, size_t maxResults,
                                             const std::function<bool(int)>& accept) const {
  // For L1-normalised vectors, |a - b|_1 = 2 - sum over shared words of (|a_i| + |b_i| - |a_i - b_i|).
  std::vector<double> acc(m_vectors.size(), 0.0);
  for (const auto& [w, x] : v) {
    const auto it = m_inverted.find(w);
    if (it == m_inverted.end()) continue;
    for (const auto& [image, y] : it->second) acc[image] += std::abs(x) + std::abs(y) - std::abs(x - y);
  }
  std::vector<PlaceMatch> out;
  for (size_t i = 0; i < acc.size(); ++i)
    if (acc[i] > 0 && (!accept || accept(static_cast<int>(i)))) out.push_back({static_cast<int>(i), 0.5 * acc[i]});
  const size_t n = std::min(maxResults, out.size());
  std::partial_sort(out.begin(), out.begin() + n, out.end(),
                    [](const PlaceMatch& a, const PlaceMatch& b) { return a.score > b.score; });
  out.resize(n);
  return out;
}

double PlaceDatabase::score(const BowVector& a, const BowVector& b) {
  double s = 0;
  auto i = a.begin(), j = b.begin();
  while (i != a.end() && j != b.end()) {
    if (i->first < j->first) ++i;
    else if (j->first < i->first) ++j;
    else {
      s += std::abs(i->second) + std::abs(j->second) - std::abs(i->second - j->second);
      ++i, ++j;
    }
  }
  return 0.5 * s;
}

}  // namespace sdv
