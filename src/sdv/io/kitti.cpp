#include <sdv/io/kitti.h>

#include <array>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgcodecs.hpp>

namespace sdv {

namespace {

std::array<double, 12> parseCalibRow(const std::filesystem::path& file, const std::string& key) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file.string());
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    std::string k;
    ss >> k;
    if (k != key + ":") continue;
    std::array<double, 12> p{};
    for (double& x : p) ss >> x;
    return p;
  }
  throw std::runtime_error("key " + key + " not found in " + file.string());
}

Sophus::SE3d makePose(const Eigen::Matrix3d& r, const Eigen::Vector3d& t) {
  // Stored rotations are only orthonormal to ~1e-9; re-project via quaternion.
  Eigen::Quaterniond q(r);
  q.normalize();
  return Sophus::SE3d(q, t);
}

}  // namespace

KittiSequence::KittiSequence(const std::filesystem::path& sequenceDir, int sizeMultiple)
    : m_dir(sequenceDir) {
  const auto p0 = parseCalibRow(m_dir / "calib.txt", "P0");
  const auto p1 = parseCalibRow(m_dir / "calib.txt", "P1");
  m_camera.fx = p0[0];
  m_camera.cx = p0[2];
  m_camera.fy = p0[5];
  m_camera.cy = p0[6];
  m_baseline = -p1[3] / p1[0];

  std::ifstream times(m_dir / "times.txt");
  if (!times) throw std::runtime_error("cannot open times.txt in " + m_dir.string());
  for (double t; times >> t;) m_timestamps.push_back(t);
  if (m_timestamps.empty()) throw std::runtime_error("no timestamps in " + m_dir.string());

  const cv::Mat first = cv::imread((m_dir / "image_0" / "000000.png").string(), cv::IMREAD_UNCHANGED);
  if (first.empty()) throw std::runtime_error("cannot read first image in " + m_dir.string());
  m_camera.width = first.cols - first.cols % sizeMultiple;
  m_camera.height = first.rows - first.rows % sizeMultiple;
}

cv::Mat KittiSequence::loadImage(size_t i, int camIndex) const {
  char name[32];
  std::snprintf(name, sizeof(name), "%06zu.png", i);
  const auto path = m_dir / (camIndex == 0 ? "image_0" : "image_1") / name;
  cv::Mat img = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
  if (img.empty()) throw std::runtime_error("cannot read " + path.string());
  return img(cv::Rect(0, 0, m_camera.width, m_camera.height));
}

std::vector<Sophus::SE3d> loadKittiPoses(const std::filesystem::path& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot open " + file.string());
  std::vector<Sophus::SE3d> poses;
  std::string line;
  while (std::getline(in, line)) {
    std::istringstream ss(line);
    Eigen::Matrix3d r;
    Eigen::Vector3d t;
    for (int row = 0; row < 3; ++row) {
      ss >> r(row, 0) >> r(row, 1) >> r(row, 2) >> t(row);
    }
    if (!ss) continue;
    poses.push_back(makePose(r, t));
  }
  return poses;
}

void saveKittiPoses(const std::filesystem::path& file, const std::vector<Sophus::SE3d>& poses) {
  std::ofstream out(file);
  if (!out) throw std::runtime_error("cannot write " + file.string());
  out.precision(9);
  for (const auto& p : poses) {
    const Eigen::Matrix<double, 3, 4> m = p.matrix3x4();
    for (int row = 0; row < 3; ++row)
      for (int col = 0; col < 4; ++col) out << m(row, col) << ((row == 2 && col == 3) ? '\n' : ' ');
  }
}

}  // namespace sdv
