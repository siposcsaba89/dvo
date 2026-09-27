// Renders the coloured points of a PLY file from a virtual pinhole camera (z-buffered square splats) into a PNG, for
// quick looks at point clouds without a viewer. Up is +z (vehicle rigs); --up changes it.
#include <cmath>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <Eigen/Geometry>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>
#include <tinyply.h>

namespace po = boost::program_options;

int main(int argc, char** argv) {
  std::string plyFile, outFile;
  std::vector<double> eye, target, up{0, 0, 1};
  double fovDeg = 70;
  int width = 1600, height = 1000, splat = 1;
  po::options_description desc("render_cloud options");
  desc.add_options()
      ("help", "show help")
      ("ply", po::value(&plyFile)->required(), "input PLY (vertex x, y, z, red, green, blue)")
      ("out", po::value(&outFile)->required(), "output PNG")
      ("eye", po::value(&eye)->multitoken()->required(), "camera position x y z")
      ("target", po::value(&target)->multitoken()->required(), "point looked at x y z")
      ("up", po::value(&up)->multitoken(), "up direction (default 0 0 1)")
      ("fov", po::value(&fovDeg)->default_value(70), "horizontal field of view, degrees")
      ("width", po::value(&width)->default_value(1600), "image width")
      ("height", po::value(&height)->default_value(1000), "image height")
      ("splat", po::value(&splat)->default_value(1), "point half size, pixels");
  try {
    po::variables_map vm;
    // Without short options, negative coordinates are values.
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
    if (eye.size() != 3 || target.size() != 3 || up.size() != 3) throw po::error("--eye, --target, --up take 3 values");
  } catch (const po::error& e) {
    spdlog::error("{}", e.what());
    std::cout << desc << '\n';
    return EXIT_FAILURE;
  }

  try {
    std::ifstream in(plyFile, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + plyFile);
    tinyply::PlyFile ply;
    ply.parse_header(in);
    auto xyz = ply.request_properties_from_element("vertex", {"x", "y", "z"});
    auto rgb = ply.request_properties_from_element("vertex", {"red", "green", "blue"});
    ply.read(in);
    if (xyz->t != tinyply::Type::FLOAT32 || rgb->t != tinyply::Type::UINT8)
      throw std::runtime_error("expected float xyz and uchar rgb");
    const auto* p = reinterpret_cast<const float*>(xyz->buffer.get());
    const auto* c = rgb->buffer.get();

    // OpenCV camera looking from eye to target.
    const Eigen::Vector3d e(eye[0], eye[1], eye[2]), t(target[0], target[1], target[2]), u(up[0], up[1], up[2]);
    const Eigen::Vector3d z = (t - e).normalized(), x = z.cross(u).normalized(), y = z.cross(x);
    Eigen::Matrix3d R_c_w;
    R_c_w << x.transpose(), y.transpose(), z.transpose();
    const double f = 0.5 * width / std::tan(0.5 * fovDeg * M_PI / 180.0);

    cv::Mat img(height, width, CV_8UC3, cv::Scalar(40, 32, 28));
    std::vector<float> depth(static_cast<size_t>(width) * height, std::numeric_limits<float>::infinity());
    size_t drawn = 0;
    for (size_t i = 0; i < xyz->count; ++i) {
      const Eigen::Vector3d X = R_c_w * (Eigen::Vector3d(p[3 * i], p[3 * i + 1], p[3 * i + 2]) - e);
      if (X.z() < 0.1) continue;
      const int cu = static_cast<int>(std::lround(f * X.x() / X.z() + 0.5 * width));
      const int cv_ = static_cast<int>(std::lround(f * X.y() / X.z() + 0.5 * height));
      bool any = false;
      for (int dv = -splat + 1; dv < splat; ++dv)
        for (int du = -splat + 1; du < splat; ++du) {
          const int uu = cu + du, vv = cv_ + dv;
          if (uu < 0 || vv < 0 || uu >= width || vv >= height) continue;
          float& d = depth[static_cast<size_t>(vv) * width + uu];
          if (X.z() >= d) continue;
          d = static_cast<float>(X.z());
          img.at<cv::Vec3b>(vv, uu) = {c[3 * i + 2], c[3 * i + 1], c[3 * i]};
          any = true;
        }
      drawn += any;
    }
    cv::imwrite(outFile, img);
    spdlog::info("wrote {} ({} of {} vertices visible)", outFile, drawn, xyz->count);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
