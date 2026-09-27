#include <sdv/io/colmap.h>

#include <fstream>
#include <stdexcept>

#include <fmt/format.h>

namespace sdv {

namespace {

std::ofstream openFile(const std::filesystem::path& path) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path.string());
  return out;
}

}  // namespace

void writeColmapText(const std::filesystem::path& dir, const std::vector<Camera>& cams,
                     const std::vector<ColmapImage>& images, const std::vector<ColmapPoint>& points) {
  for (const auto& cam : cams)
    if (!cam.isPinhole()) throw std::invalid_argument("COLMAP export expects pinhole cameras; undistort first");
  std::filesystem::create_directories(dir);

  auto cameras = openFile(dir / "cameras.txt");
  cameras << "# Camera list with one line of data per camera:\n"
          << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
          << fmt::format("# Number of cameras: {}\n", cams.size());
  for (size_t i = 0; i < cams.size(); ++i)
    cameras << fmt::format("{} PINHOLE {} {} {:.6f} {:.6f} {:.6f} {:.6f}\n", i + 1, cams[i].width, cams[i].height,
                           cams[i].fx, cams[i].fy, cams[i].cx, cams[i].cy);

  size_t numObservations = 0;
  for (const auto& img : images) numObservations += img.points2D.size();
  auto imageFile = openFile(dir / "images.txt");
  imageFile << "# Image list with two lines of data per image:\n"
            << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
            << "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            << fmt::format("# Number of images: {}, mean observations per image: {:.1f}\n", images.size(),
                           images.empty() ? 0.0 : double(numObservations) / images.size());
  for (const auto& img : images) {
    // COLMAP stores world-to-camera rotation as a Hamilton quaternion (w first) and translation.
    const Eigen::Quaterniond q = img.T_c_w.unit_quaternion();
    const Eigen::Vector3d& t = img.T_c_w.translation();
    imageFile << fmt::format("{} {:.9f} {:.9f} {:.9f} {:.9f} {:.6f} {:.6f} {:.6f} {} {}\n", img.id, q.w(), q.x(), q.y(),
                             q.z(), t.x(), t.y(), t.z(), img.cameraId, img.name);
    for (size_t i = 0; i < img.points2D.size(); ++i) {
      const auto& [uv, id] = img.points2D[i];
      imageFile << fmt::format("{}{:.2f} {:.2f} {}", i ? " " : "", uv.x(), uv.y(), id);
    }
    imageFile << '\n';
  }

  size_t trackLength = 0;
  for (const auto& p : points) trackLength += p.track.size();
  auto pointFile = openFile(dir / "points3D.txt");
  pointFile << "# 3D point list with one line of data per point:\n"
            << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
            << fmt::format("# Number of points: {}, mean track length: {:.2f}\n", points.size(),
                           points.empty() ? 0.0 : double(trackLength) / points.size());
  for (const auto& p : points) {
    pointFile << fmt::format("{} {:.4f} {:.4f} {:.4f} {} {} {} {:.3f}", p.id, p.position.x(), p.position.y(),
                             p.position.z(), p.rgb[0], p.rgb[1], p.rgb[2], p.error);
    for (const auto& [image, index] : p.track) pointFile << ' ' << image << ' ' << index;
    pointFile << '\n';
  }
}

}  // namespace sdv
