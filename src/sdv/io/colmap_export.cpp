#include <sdv/io/colmap_export.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

#include <fmt/format.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace sdv {

ColmapExporter::ColmapExporter(const std::filesystem::path& root, std::vector<ColmapExportCamera> cameras,
                               ColmapExportSettings settings)
    : m_root(root), m_settings(std::move(settings)) {
  for (auto& in : cameras) {
    Output o;
    const Camera& cam = in.camera;
    o.pinhole = cam.isPinhole() ? cam : virtualPinhole(cam, m_settings.focalScale);
    if (!cam.isPinhole() && m_settings.maxFovDeg > 0) {
      const double t = std::tan(m_settings.maxFovDeg * M_PI / 360.0);
      // Centred: pixel centres at +-(size - 1) / 2 around the principal point.
      o.pinhole.width = std::min(o.pinhole.width, 2 * static_cast<int>(std::floor(o.pinhole.fx * t + 1e-6)) + 1);
      o.pinhole.height = std::min(o.pinhole.height, 2 * static_cast<int>(std::floor(o.pinhole.fy * t + 1e-6)) + 1);
      o.pinhole.cx = (o.pinhole.width - 1) / 2.0;
      o.pinhole.cy = (o.pinhole.height - 1) / 2.0;
    }
    o.pinhole.mask.reset();
    o.pinhole.maskLevel = 0;
    if (!cam.isPinhole()) o.map = makeUndistortMap(cam, o.pinhole);
    // Valid where the source pixel exists and the input mask is valid.
    cv::Mat valid = cam.mask ? cam.mask->level(cam.maskLevel).clone() : cv::Mat(cam.height, cam.width, CV_8UC1, 255);
    if (!cam.isPinhole()) {
      cv::remap(valid, o.mask, o.map.mapX, o.map.mapY, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
      o.mask.setTo(0, o.map.mapX < 0);
    } else if (cam.mask) {
      o.mask = valid;
    }
    std::filesystem::create_directories(m_root / "images" / in.name);
    if (!o.mask.empty()) std::filesystem::create_directories(m_root / "masks" / in.name);
    o.input = std::move(in);
    m_outputs.push_back(std::move(o));
  }
}

void ColmapExporter::addFrame(int frameIndex, const std::vector<cv::Mat>& images, const Sophus::SE3d& T_w_b,
                              const std::vector<AffineBrightness>& brightness) {
  if (images.size() != m_outputs.size()) throw std::invalid_argument("one image per export camera required");
  if (!brightness.empty() && brightness.size() != m_outputs.size())
    throw std::invalid_argument("one brightness per export camera required");
  for (size_t c = 0; c < m_outputs.size(); ++c) {
    const Output& o = m_outputs[c];
    cv::Mat img = images[c];
    // GS trainers expect 3-channel 8-bit images.
    if (img.channels() == 1) cv::cvtColor(img, img, cv::COLOR_GRAY2BGR);
    const double depthScale = img.depth() == CV_16U ? 255.0 / 65535.0 : 1.0;
    if (m_settings.correctBrightness && !brightness.empty()) {
      const double s = std::exp(-brightness[c].a);
      img.convertTo(img, CV_8U, s * depthScale, -s * brightness[c].b);
    } else if (img.depth() != CV_8U) {
      img.convertTo(img, CV_8U, depthScale);
    }
    if (!o.input.camera.isPinhole()) img = undistort(img, o.map);
    const std::filesystem::path name =
        std::filesystem::path(o.input.name) / fmt::format("{:06d}{}", frameIndex, m_settings.imageExtension);
    const std::string nameText = name.generic_string();
    if (!cv::imwrite((m_root / "images" / name).string(), img))
      throw std::runtime_error("cannot write " + (m_root / "images" / name).string());
    if (!o.mask.empty()) {
      std::filesystem::path maskName = name;
      maskName.replace_extension(".png");
      cv::imwrite((m_root / "masks" / maskName).string(), o.mask);
    }
    m_imageOf[{frameIndex, static_cast<int>(c)}] = m_images.size();
    m_images.push_back({static_cast<int>(m_images.size()) + 1, nameText, o.input.T_c_b * T_w_b.inverse(), {},
                        static_cast<int>(c) + 1});
    if (!brightness.empty()) m_exposure.emplace_back(nameText, brightness[c]);
  }
}

void ColmapExporter::write(const std::vector<ColmapExportPoint>& input) {
  std::vector<ColmapPoint> points;
  points.reserve(input.size());
  for (const auto& m : input) {
    ColmapPoint p{static_cast<std::int64_t>(points.size()) + 1, m.position, m.rgb};
    if (const auto it = m_imageOf.find({m.frameIndex, m.camera}); it != m_imageOf.end()) {
      ColmapImage& img = m_images[it->second];
      const Camera& cam = m_outputs[m.camera].pinhole;
      Eigen::Vector2d uv;
      if (cam.project(Eigen::Vector3d(img.T_c_w * m.position), uv) && cam.isInside(uv.x(), uv.y(), 0.0)) {
        p.track.emplace_back(img.id, static_cast<int>(img.points2D.size()));
        img.points2D.emplace_back(uv, p.id);
      }
    }
    points.push_back(std::move(p));
  }
  std::vector<Camera> cams;
  for (const auto& o : m_outputs) cams.push_back(o.pinhole);
  writeColmapText(m_root / "sparse" / "0", cams, m_images, points);
  if (!m_exposure.empty()) {
    std::ofstream out(m_root / "exposure.txt");
    if (!out) throw std::runtime_error("cannot write " + (m_root / "exposure.txt").string());
    out << "# Affine brightness per image, I = exp(a) J + b (J: irradiance in the gauge of the adjusted cameras)"
        << (m_settings.correctBrightness ? "; the images are corrected to J" : "") << "\n# image a b\n";
    for (const auto& [name, a] : m_exposure) out << fmt::format("{} {:.6f} {:.4f}\n", name, a.a, a.b);
  }
}

}  // namespace sdv
