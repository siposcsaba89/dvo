#include <sdv/io/camera_config.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <yaml-cpp/yaml.h>

namespace sdv {

namespace {

template <typename T>
T required(const YAML::Node& node, const char* key, const std::filesystem::path& file) {
  if (!node[key]) throw std::runtime_error(std::string("camera config ") + file.string() + " lacks '" + key + "'");
  return node[key].as<T>();
}

cv::Mat resizeAndCrop(const cv::Mat& image, double scale, int width, int height, int interpolation) {
  cv::Mat out = image;
  if (scale != 1.0) cv::resize(image, out, {}, scale, scale, interpolation);
  if (out.cols < width || out.rows < height) throw std::runtime_error("image smaller than the camera");
  return out(cv::Rect(0, 0, width, height)).clone();
}

}  // namespace

CameraConfig loadCameraConfig(const std::filesystem::path& file) {
  YAML::Node node;
  try {
    node = YAML::LoadFile(file.string());
  } catch (const YAML::Exception& e) {
    throw std::runtime_error("cannot read camera config " + file.string() + ": " + e.what());
  }
  return parseCameraConfig(node, file);
}

CameraConfig parseCameraConfig(const YAML::Node& node, const std::filesystem::path& file) {
  CameraConfig config;
  Camera& c = config.camera;
  c.width = required<int>(node, "width", file);
  c.height = required<int>(node, "height", file);
  c.fx = required<double>(node, "fx", file);
  c.fy = required<double>(node, "fy", file);
  c.cx = required<double>(node, "cx", file);
  c.cy = required<double>(node, "cy", file);
  c.alpha = node["alpha"] ? node["alpha"].as<double>() : 0.0;
  c.beta = node["beta"] ? node["beta"].as<double>() : 1.0;

  if (node["mask"]) {
    std::filesystem::path maskFile = node["mask"].as<std::string>();
    if (maskFile.is_relative()) maskFile = file.parent_path() / maskFile;
    config.mask = cv::imread(maskFile.string(), cv::IMREAD_GRAYSCALE);
    if (config.mask.empty()) throw std::runtime_error("cannot read mask " + maskFile.string());
    if (config.mask.cols != c.width || config.mask.rows != c.height)
      throw std::runtime_error("mask " + maskFile.string() + " does not match the camera size");
  }
  return config;
}

Camera prepareCamera(const CameraConfig& config, double scale, int sizeMultiple) {
  Camera c = config.camera;
  // Pixel centres are integers: u' = (u + 0.5) * scale - 0.5.
  c.fx *= scale;
  c.fy *= scale;
  c.cx = (c.cx + 0.5) * scale - 0.5;
  c.cy = (c.cy + 0.5) * scale - 0.5;
  const int w = static_cast<int>(std::lround(config.camera.width * scale));
  const int h = static_cast<int>(std::lround(config.camera.height * scale));
  c.width = w - w % sizeMultiple;
  c.height = h - h % sizeMultiple;
  if (!config.mask.empty())
    c.mask = std::make_shared<const ValidityMask>(resizeAndCrop(config.mask, scale, c.width, c.height, cv::INTER_AREA));
  return c;
}

cv::Mat prepareImage(const cv::Mat& image, double scale, const Camera& prepared) {
  return resizeAndCrop(image, scale, prepared.width, prepared.height, cv::INTER_AREA);
}

}  // namespace sdv
