#include <filesystem>
#include <random>

#include <gtest/gtest.h>

#include <sdv/place_database.h>

namespace {

// Descriptor "places": each image is a noisy copy of its place's descriptor set.
cv::Mat noisyCopy(const cv::Mat& base, std::mt19937& rng, int flips) {
  cv::Mat out = base.clone();
  std::uniform_int_distribution<int> bit(0, 255);
  for (int r = 0; r < out.rows; ++r)
    for (int k = 0; k < flips; ++k) out.at<uint8_t>(r, bit(rng) % 32) ^= static_cast<uint8_t>(1 << (bit(rng) % 8));
  return out;
}

}  // namespace

TEST(PlaceDatabase, L1ScoreOfNormalisedVectors) {
  const sdv::BowVector a = {{1, 0.5f}, {2, 0.5f}}, b = {{2, 0.5f}, {3, 0.5f}};
  EXPECT_NEAR(sdv::PlaceDatabase::score(a, a), 1.0, 1e-9);
  EXPECT_NEAR(sdv::PlaceDatabase::score(a, b), 0.5, 1e-9);
  sdv::PlaceDatabase db;
  db.add(a);
  db.add(b);
  const auto m = db.query(a, 2);
  ASSERT_EQ(m.size(), 2u);
  EXPECT_EQ(m[0].image, 0);
  EXPECT_NEAR(m[1].score, 0.5, 1e-9);
  EXPECT_TRUE(db.query(a, 2, [](int image) { return image != 0; }).size() == 1);
}

TEST(PlaceDatabase, RetrievesTheSamePlace) {
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> byte(0, 255);
  std::vector<cv::Mat> places;
  for (int p = 0; p < 20; ++p) {
    cv::Mat d(150, 32, CV_8UC1);
    for (int i = 0; i < d.rows * d.cols; ++i) d.data[i] = static_cast<uint8_t>(byte(rng));
    places.push_back(d);
  }
  std::vector<cv::Mat> training;
  for (const auto& p : places) training.push_back(noisyCopy(p, rng, 4));
  const auto file = std::filesystem::temp_directory_path() / "sdv_voc_test.fbow";
  sdv::Vocabulary::train(training, 8, 3, file);
  const sdv::Vocabulary voc(file);
  std::filesystem::remove(file);

  sdv::PlaceDatabase db;
  for (const auto& p : places) db.add(voc.transform(noisyCopy(p, rng, 4)));
  int correct = 0;
  for (int p = 0; p < 20; ++p) {
    const auto m = db.query(voc.transform(noisyCopy(places[p], rng, 4)), 1);
    correct += !m.empty() && m[0].image == p;
  }
  EXPECT_GE(correct, 18);
}
