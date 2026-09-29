#include <sdv/io/ply.h>

#include <algorithm>
#include <fstream>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <tinyply.h>

namespace sdv {

int32_t PlyScene::addVertex(const Eigen::Vector3d& p, const Rgb& color) {
  const Eigen::Vector3f f = p.cast<float>();
  if (!f.allFinite()) return -1;
  m_xyz.push_back(f);
  m_rgb.push_back(color);
  return static_cast<int32_t>(m_rgb.size() - 1);
}

bool PlyScene::addPoint(const Eigen::Vector3d& p, const Rgb& color) { return addVertex(p, color) >= 0; }

void PlyScene::addTrajectory(const std::vector<Sophus::SE3d>& T_w_c, const Rgb& color) {
  int32_t prev = -1;
  for (const auto& T : T_w_c) {
    const int32_t v = addVertex(T.translation(), color);
    if (prev >= 0 && v >= 0) addEdge(prev, v);
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
      if (a >= 0 && b >= 0) addEdge(a, b);
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

void writeMapPointsPly(const std::filesystem::path& file, const std::vector<MapPoint>& points,
                       const std::vector<char>* keep) {
  std::ofstream out(file, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + file.string());
  const size_t n = points.size();
  std::vector<Eigen::Vector3f> xyz(n);
  std::vector<Rgb> rgb(n);
  std::vector<uint8_t> source(n), kept(n);
  std::vector<int32_t> camera(n), frame(n), observations(n);
  std::vector<float> u(n), v(n), distance(n), sigma(n);
  for (size_t i = 0; i < n; ++i) {
    const MapPoint& p = points[i];
    xyz[i] = p.position.cast<float>();
    const auto g = static_cast<uint8_t>(std::clamp(p.intensity, 0.f, 255.f));
    rgb[i] = p.color.value_or(Rgb{g, g, g});
    source[i] = static_cast<uint8_t>(p.source);
    camera[i] = p.camera;
    frame[i] = p.frameIndex;
    u[i] = static_cast<float>(p.uv.x());
    v[i] = static_cast<float>(p.uv.y());
    distance[i] = static_cast<float>(p.distance);
    observations[i] = p.observations;
    sigma[i] = static_cast<float>(p.relativeDepthSigma);
    kept[i] = keep ? static_cast<uint8_t>((*keep)[i] != 0) : 1;
  }
  tinyply::PlyFile ply;
  auto add = [&](const std::vector<std::string>& names, tinyply::Type type, const void* data) {
    ply.add_properties_to_element("vertex", names, type, n, static_cast<const uint8_t*>(data), tinyply::Type::INVALID,
                                  0);
  };
  add({"x", "y", "z"}, tinyply::Type::FLOAT32, xyz.data());
  add({"red", "green", "blue"}, tinyply::Type::UINT8, rgb.data());
  add({"source"}, tinyply::Type::UINT8, source.data());
  add({"camera"}, tinyply::Type::INT32, camera.data());
  add({"frame"}, tinyply::Type::INT32, frame.data());
  add({"u"}, tinyply::Type::FLOAT32, u.data());
  add({"v"}, tinyply::Type::FLOAT32, v.data());
  add({"distance"}, tinyply::Type::FLOAT32, distance.data());
  add({"observations"}, tinyply::Type::INT32, observations.data());
  add({"sigma"}, tinyply::Type::FLOAT32, sigma.data());
  add({"kept"}, tinyply::Type::UINT8, kept.data());
  ply.write(out, true);
}

void readPlyPoints(const std::filesystem::path& file, std::vector<Eigen::Vector3d>& points, std::vector<Rgb>& colors) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + file.string());
  tinyply::PlyFile ply;
  ply.parse_header(in);
  std::shared_ptr<tinyply::PlyData> xyz, rgb, edges;
  try {
    xyz = ply.request_properties_from_element("vertex", {"x", "y", "z"});
    rgb = ply.request_properties_from_element("vertex", {"red", "green", "blue"});
  } catch (const std::exception& e) {
    throw std::runtime_error(file.string() + ": " + e.what());
  }
  try {
    edges = ply.request_properties_from_element("edge", {"vertex1", "vertex2"});
  } catch (const std::exception&) {
  }
  ply.read(in);
  if (xyz->t != tinyply::Type::FLOAT32 || rgb->t != tinyply::Type::UINT8)
    throw std::runtime_error(file.string() + ": expected float x y z and uchar red green blue");
  const size_t n = xyz->count;
  std::vector<char> used(n, 0);
  if (edges && edges->t == tinyply::Type::INT32) {
    std::vector<int32_t> e(edges->count * 2);
    std::memcpy(e.data(), edges->buffer.get(), e.size() * sizeof(int32_t));
    for (int32_t v : e)
      if (v >= 0 && static_cast<size_t>(v) < n) used[v] = 1;
  }
  const auto* f = reinterpret_cast<const float*>(xyz->buffer.get());
  const auto* c = reinterpret_cast<const uint8_t*>(rgb->buffer.get());
  points.clear();
  colors.clear();
  for (size_t i = 0; i < n; ++i) {
    if (used[i]) continue;
    points.emplace_back(f[3 * i], f[3 * i + 1], f[3 * i + 2]);
    colors.push_back({c[3 * i], c[3 * i + 1], c[3 * i + 2]});
  }
}

}  // namespace sdv
