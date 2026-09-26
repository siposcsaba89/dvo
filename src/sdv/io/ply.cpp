#include <sdv/io/ply.h>

#include <fstream>
#include <stdexcept>

#include <tinyply.h>

namespace sdv {

int32_t PlyScene::addVertex(const Eigen::Vector3d& p, const Rgb& color) {
  m_xyz.push_back(p.cast<float>());
  m_rgb.push_back(color);
  return static_cast<int32_t>(m_rgb.size() - 1);
}

void PlyScene::addPoint(const Eigen::Vector3d& p, const Rgb& color) { addVertex(p, color); }

void PlyScene::addTrajectory(const std::vector<Sophus::SE3d>& T_w_c, const Rgb& color) {
  int32_t prev = -1;
  for (const auto& T : T_w_c) {
    const int32_t v = addVertex(T.translation(), color);
    if (prev >= 0) addEdge(prev, v);
    prev = v;
  }
}

void PlyScene::addCameraAxes(const std::vector<Sophus::SE3d>& T_w_c, double length, size_t stride) {
  constexpr Rgb kAxisColors[3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}};
  for (size_t i = 0; i < T_w_c.size(); i += stride) {
    const Eigen::Vector3d c = T_w_c[i].translation();
    for (int k = 0; k < 3; ++k) {
      const int32_t a = addVertex(c, kAxisColors[k]);
      const int32_t b = addVertex(T_w_c[i] * (length * Eigen::Vector3d::Unit(k)), kAxisColors[k]);
      addEdge(a, b);
    }
  }
}

void PlyScene::write(const std::filesystem::path& file, bool binary) const {
  std::ofstream out(file, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + file.string());

  tinyply::PlyFile ply;
  const size_t n = m_rgb.size();
  ply.add_properties_to_element("vertex", {"x", "y", "z"}, tinyply::Type::FLOAT32, n,
                                reinterpret_cast<const uint8_t*>(m_xyz.data()), tinyply::Type::INVALID, 0);
  ply.add_properties_to_element("vertex", {"red", "green", "blue"}, tinyply::Type::UINT8, n,
                                reinterpret_cast<const uint8_t*>(m_rgb.data()), tinyply::Type::INVALID, 0);
  if (!m_edges.empty()) {
    ply.add_properties_to_element("edge", {"vertex1", "vertex2"}, tinyply::Type::INT32, m_edges.size() / 2,
                                  reinterpret_cast<const uint8_t*>(m_edges.data()), tinyply::Type::INVALID, 0);
  }
  ply.write(out, binary);
}

}  // namespace sdv
