#include <sdv/keyframe_features.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <opencv2/features2d.hpp>

namespace sdv {

namespace {

// Version 02 adds the affine brightness and the hosted map points per camera.
constexpr char kMagic[8] = {'S', 'D', 'V', 'K', 'F', 'R', '0', '2'};
constexpr char kMagicV1[8] = {'S', 'D', 'V', 'K', 'F', 'R', '0', '1'};

template <typename T>
void put(std::ostream& out, const T& v) {
  out.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

template <typename T>
T get(std::istream& in) {
  T v;
  in.read(reinterpret_cast<char*>(&v), sizeof(T));
  if (!in) throw std::runtime_error("truncated keyframe record file");
  return v;
}

void putPose(std::ostream& out, const Sophus::SE3d& T) {
  const Eigen::Quaterniond q = T.unit_quaternion();
  for (double v : {q.w(), q.x(), q.y(), q.z(), T.translation().x(), T.translation().y(), T.translation().z()})
    put(out, v);
}

Sophus::SE3d getPose(std::istream& in) {
  double v[7];
  for (double& x : v) x = get<double>(in);
  return Sophus::SE3d(Eigen::Quaterniond(v[0], v[1], v[2], v[3]).normalized(), Eigen::Vector3d(v[4], v[5], v[6]));
}

}  // namespace

int CameraFeatures::numWithDepth() const {
  return static_cast<int>(std::count_if(rho.begin(), rho.end(), [](float r) { return r > 0; }));
}

cv::Mat toGray8(const ImageLevel& img) {
  cv::Mat out(img.height, img.width, CV_8UC1);
  for (int v = 0; v < img.height; ++v)
    for (int u = 0; u < img.width; ++u)
      out.at<std::uint8_t>(v, u) = cv::saturate_cast<std::uint8_t>(img.at(u, v)[0]);
  return out;
}

CameraFeatures extractFeatures(const Camera& cam, const cv::Mat& gray8, const FeatureSettings& settings) {
  // Detect more than needed, then keep the strongest per grid cell so that keypoints cover the whole image.
  const int cells = settings.gridColumns * settings.gridRows;
  auto orb = cv::ORB::create(4 * settings.featuresPerImage, 1.2f, 8, 19, 0, 2, cv::ORB::HARRIS_SCORE, 31,
                             static_cast<int>(settings.fastThreshold));
  std::vector<cv::KeyPoint> detected;
  orb->detect(gray8, detected, cam.maskImage());
  std::vector<std::vector<cv::KeyPoint>> grid(cells);
  for (const auto& k : detected) {
    const int gx = std::clamp(static_cast<int>(k.pt.x * settings.gridColumns / gray8.cols), 0, settings.gridColumns - 1);
    const int gy = std::clamp(static_cast<int>(k.pt.y * settings.gridRows / gray8.rows), 0, settings.gridRows - 1);
    grid[gy * settings.gridColumns + gx].push_back(k);
  }
  const size_t perCell = static_cast<size_t>(std::max(1, settings.featuresPerImage / cells));
  std::vector<cv::KeyPoint> kept;
  for (auto& cell : grid) {
    std::sort(cell.begin(), cell.end(), [](const cv::KeyPoint& a, const cv::KeyPoint& b) { return a.response > b.response; });
    kept.insert(kept.end(), cell.begin(), cell.begin() + std::min(perCell, cell.size()));
  }

  CameraFeatures f;
  cv::Mat descriptors;
  orb->compute(gray8, kept, descriptors);  // may drop keypoints near the border
  for (size_t i = 0; i < kept.size(); ++i) {
    Eigen::Vector3d b;
    if (!cam.unproject(Eigen::Vector2d(kept[i].pt.x, kept[i].pt.y), b)) continue;
    f.keypoints.push_back(kept[i]);
    f.bearings.push_back(b.normalized().cast<float>());
    f.rho.push_back(0.f);
    f.descriptors.push_back(descriptors.row(static_cast<int>(i)));
  }
  return f;
}

void assignDepth(CameraFeatures& features, const std::vector<Eigen::Vector2f>& uv, const std::vector<float>& rho,
                 const FeatureSettings& settings) {
  const double r = settings.depthRadius;
  const int cell = std::max(1, static_cast<int>(std::ceil(r)));
  std::unordered_multimap<long long, size_t> grid;
  auto key = [](int x, int y) { return (static_cast<long long>(y) << 32) ^ static_cast<unsigned>(x); };
  for (size_t i = 0; i < uv.size(); ++i)
    if (rho[i] > 0) grid.emplace(key(static_cast<int>(uv[i].x()) / cell, static_cast<int>(uv[i].y()) / cell), i);
  std::vector<float> near;
  for (size_t k = 0; k < features.size(); ++k) {
    const cv::Point2f& p = features.keypoints[k].pt;
    const int cx = static_cast<int>(p.x) / cell, cy = static_cast<int>(p.y) / cell;
    near.clear();
    for (int dy = -1; dy <= 1; ++dy)
      for (int dx = -1; dx <= 1; ++dx) {
        const auto [b, e] = grid.equal_range(key(cx + dx, cy + dy));
        for (auto it = b; it != e; ++it)
          if ((uv[it->second] - Eigen::Vector2f(p.x, p.y)).norm() <= r) near.push_back(rho[it->second]);
      }
    if (near.empty()) continue;
    // Keypoints sit on corners, often at depth edges: only a consistent neighbourhood gives a depth.
    std::sort(near.begin(), near.end());
    const float median = near[near.size() / 2];
    if (near.back() - near.front() > settings.depthConsistency * median) continue;
    features.rho[k] = median;
  }
}

void saveKeyframeRecords(const std::filesystem::path& file, const Rig& rig, const std::vector<KeyframeRecord>& records) {
  std::ofstream out(file, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + file.string());
  out.write(kMagic, sizeof(kMagic));
  put<std::int32_t>(out, rig.size());
  for (int c = 0; c < rig.size(); ++c) {
    const Camera& cam = rig.cameras[c];
    for (double v : {cam.fx, cam.fy, cam.cx, cam.cy, cam.alpha, cam.beta}) put(out, v);
    put<std::int32_t>(out, cam.width);
    put<std::int32_t>(out, cam.height);
    putPose(out, rig.T_c_b[c]);
  }
  put<std::int64_t>(out, static_cast<std::int64_t>(records.size()));
  for (const auto& r : records) {
    put<std::int32_t>(out, r.frameIndex);
    putPose(out, r.T_w_b);
    for (int c = 0; c < rig.size(); ++c) {
      const AffineBrightness a = c < static_cast<int>(r.affine.size()) ? r.affine[c] : AffineBrightness{};
      put(out, a.a);
      put(out, a.b);
    }
    for (const auto& f : r.cameras) {
      put<std::int32_t>(out, static_cast<std::int32_t>(f.size()));
      put<std::int32_t>(out, f.descriptors.cols);
      for (size_t i = 0; i < f.size(); ++i) {
        const cv::KeyPoint& k = f.keypoints[i];
        for (float v : {k.pt.x, k.pt.y, k.size, k.angle, k.response}) put(out, v);
        put<std::int32_t>(out, k.octave);
        for (int j = 0; j < 3; ++j) put(out, f.bearings[i][j]);
        put(out, f.rho[i]);
        out.write(reinterpret_cast<const char*>(f.descriptors.ptr(static_cast<int>(i))), f.descriptors.cols);
      }
      put<std::int32_t>(out, static_cast<std::int32_t>(f.pointUv.size()));
      for (size_t i = 0; i < f.pointUv.size(); ++i) {
        put(out, f.pointUv[i].x());
        put(out, f.pointUv[i].y());
        put(out, f.pointRho[i]);
      }
    }
  }
}

std::vector<KeyframeRecord> loadKeyframeRecords(const std::filesystem::path& file, Rig* rig) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + file.string());
  char magic[sizeof(kMagic)];
  in.read(magic, sizeof(magic));
  const bool v1 = in && std::equal(magic, magic + sizeof(magic), kMagicV1);
  if (!in || (!v1 && !std::equal(magic, magic + sizeof(magic), kMagic)))
    throw std::runtime_error(file.string() + ": not a keyframe record file");
  const int nc = get<std::int32_t>(in);
  Rig r;
  for (int c = 0; c < nc; ++c) {
    double v[6];
    for (double& x : v) x = get<double>(in);
    const int w = get<std::int32_t>(in), h = get<std::int32_t>(in);
    r.cameras.push_back(Camera::eucm(v[0], v[1], v[2], v[3], v[4], v[5], w, h));
    r.T_c_b.push_back(getPose(in));
  }
  if (rig) *rig = r;
  const auto n = get<std::int64_t>(in);
  std::vector<KeyframeRecord> records(static_cast<size_t>(n));
  for (auto& rec : records) {
    rec.frameIndex = get<std::int32_t>(in);
    rec.T_w_b = getPose(in);
    if (!v1)
      for (int c = 0; c < nc; ++c) {
        const double a = get<double>(in), b = get<double>(in);
        rec.affine.push_back({a, b});
      }
    rec.cameras.resize(nc);
    for (auto& f : rec.cameras) {
      const int count = get<std::int32_t>(in), cols = get<std::int32_t>(in);
      f.descriptors.create(count, cols, CV_8UC1);
      for (int i = 0; i < count; ++i) {
        float v[5];
        for (float& x : v) x = get<float>(in);
        const int octave = get<std::int32_t>(in);
        f.keypoints.emplace_back(cv::Point2f(v[0], v[1]), v[2], v[3], v[4], octave);
        Eigen::Vector3f b;
        for (int j = 0; j < 3; ++j) b[j] = get<float>(in);
        f.bearings.push_back(b);
        f.rho.push_back(get<float>(in));
        in.read(reinterpret_cast<char*>(f.descriptors.ptr(i)), cols);
      }
      if (v1) continue;
      const int points = get<std::int32_t>(in);
      for (int i = 0; i < points; ++i) {
        const float u = get<float>(in), v = get<float>(in);
        f.pointUv.emplace_back(u, v);
        f.pointRho.push_back(get<float>(in));
      }
    }
  }
  if (!in) throw std::runtime_error("truncated keyframe record file");
  return records;
}

}  // namespace sdv
