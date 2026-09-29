// COLMAP text model of a finished rig run for Gaussian Splatting (gsplat): the frames where the vehicle moved or turned
// enough since the last exported one (standing frames repeat a view), for the chosen rig cameras, the corrected body poses (close_loops --out) with the rig extrinsics, points3D from a point cloud
// (close_loops --ply; neighbour-filtered against floating points, voxel-thinned) and, with close_loops --brightness-out, exposure.txt.
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <iostream>
#include <cmath>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>

#include <sdv/io/camera_config.h>
#include <sdv/io/colmap_export.h>
#include <sdv/io/kitti.h>
#include <sdv/io/ply.h>
#include <sdv/io/rig_config.h>
#include <sdv/point_filter.h>

namespace po = boost::program_options;

namespace {

// close_loops --brightness-out: camera -> keyframe index -> brightness.
std::map<std::string, std::map<int, sdv::AffineBrightness>> loadBrightness(const std::string& file) {
  std::ifstream in(file);
  if (!in) throw std::runtime_error("cannot read " + file);
  std::map<std::string, std::map<int, sdv::AffineBrightness>> out;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    int frame;
    std::string camera;
    sdv::AffineBrightness a;
    if (!(ss >> frame >> camera >> a.a >> a.b)) throw std::runtime_error(file + ": bad line '" + line + "'");
    out[camera][frame] = a;
  }
  return out;
}

// Linear between the keyframes around `frame`, as in the densify pass.
sdv::AffineBrightness interpolate(const std::map<int, sdv::AffineBrightness>& keyframes, int frame) {
  if (keyframes.empty()) return {};
  const auto next = keyframes.lower_bound(frame);
  if (next == keyframes.end()) return std::prev(next)->second;
  if (next->first == frame || next == keyframes.begin()) return next->second;
  const auto prev = std::prev(next);
  const double w = static_cast<double>(frame - prev->first) / (next->first - prev->first);
  return {(1 - w) * prev->second.a + w * next->second.a, (1 - w) * prev->second.b + w * next->second.b};
}

}  // namespace

int main(int argc, char** argv) {
  std::string rigFile, posesFile, cloudFile, outDir, brightnessFile, format = "jpg";
  std::vector<std::string> runCameras, exportCameras;
  double scale = 1.0, focalScale = 1.0, maxFov = 100.0, voxel = 0.03, minTravel = 0.25, minRotationDeg = 5.0;
  int start = 0, stride = 1, frameStride = 1, minNeighbours = 10;
  double neighbourRadius = 0.2;
  bool correctBrightness = false;

  po::options_description desc("export_colmap options");
  desc.add_options()
      ("help,h", "show help")
      ("rig", po::value(&rigFile)->required(), "rig YAML of the run")
      ("rig-cameras", po::value(&runCameras)->multitoken(), "rig cameras of the run (run_vo --rig-cameras)")
      ("cameras", po::value(&exportCameras)->multitoken(), "rig cameras to export (default: those of the run)")
      ("poses", po::value(&posesFile)->required(), "body poses of all run frames (close_loops --out)")
      ("cloud", po::value(&cloudFile), "point cloud PLY for points3D (close_loops --ply)")
      ("min-neighbours", po::value(&minNeighbours)->default_value(minNeighbours),
       "points3D: keep points with this many others within --neighbour-radius (0 = off); stricter than close_loops' "
       "3, against floating points that would seed floaters")
      ("neighbour-radius", po::value(&neighbourRadius)->default_value(neighbourRadius), "neighbour radius, m")
      ("voxel", po::value(&voxel)->default_value(voxel), "thin points3D to one point per voxel, m (0 = off)")
      ("brightness", po::value(&brightnessFile), "close_loops --brightness-out: writes exposure.txt")
      ("correct-brightness", po::bool_switch(&correctBrightness),
       "write brightness-corrected images e^-a (I - b) (needs --brightness; clips at 255)")
      ("out", po::value(&outDir)->required(), "output directory (images/, masks/, sparse/0, exposure.txt)")
      ("scale", po::value(&scale)->default_value(scale), "image scale (run_vo --scale for the run resolution)")
      ("focal-scale", po::value(&focalScale)->default_value(focalScale),
       "virtual pinhole focal length relative to the camera's (< 1 keeps more of a fisheye)")
      ("min-travel", po::value(&minTravel)->default_value(minTravel),
       "export a frame once the body moved this far since the last exported one, m")
      ("min-rotation", po::value(&minRotationDeg)->default_value(minRotationDeg), "... or turned this much, deg")
      ("max-fov", po::value(&maxFov)->default_value(maxFov),
       "crop virtual pinholes to this field of view per axis, deg (0 = off)")
      ("frame-stride", po::value(&frameStride)->default_value(frameStride),
       "consider only every n-th run frame (with --min-travel 0 --min-rotation 0: every n-th)")
      ("format", po::value(&format)->default_value(format), "image format: jpg or png")
      ("start", po::value(&start)->default_value(start), "run_vo --start")
      ("stride", po::value(&stride)->default_value(stride), "run_vo --stride");
  po::variables_map vm;
  try {
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
    if (format != "jpg" && format != "png") throw po::error("--format must be jpg or png");
    if (correctBrightness && brightnessFile.empty()) throw po::error("--correct-brightness needs --brightness");
    if (frameStride < 1) throw po::error("--frame-stride must be positive");
  } catch (const po::error& e) {
    spdlog::error("{}", e.what());
    std::cout << desc << '\n';
    return EXIT_FAILURE;
  }

  try {
    const sdv::RigConfig config = sdv::loadRigConfig(rigFile);
    std::vector<std::string> runNames, names;
    for (const auto& c : config.cameras) {
      if (runCameras.empty() || std::ranges::find(runCameras, c.name) != runCameras.end()) runNames.push_back(c.name);
      const auto& wanted = exportCameras.empty() ? runCameras : exportCameras;
      if (wanted.empty() || std::ranges::find(wanted, c.name) != wanted.end()) names.push_back(c.name);
    }
    auto streams = sdv::openRigStreams(config, names, runNames);
    std::vector<sdv::ColmapExportCamera> cameras;
    std::vector<sdv::Camera> prepared;
    std::vector<std::unique_ptr<sdv::FrameSource>> sources;
    for (auto& s : streams) {
      prepared.push_back(sdv::prepareCamera(s.config->camera, scale, 1));
      cameras.push_back({s.config->name, prepared.back(), s.config->T_b_c.inverse()});
      sources.push_back(std::make_unique<sdv::SubsampledSource>(std::move(s.source), start, stride));
    }
    const auto brightness =
        brightnessFile.empty() ? std::map<std::string, std::map<int, sdv::AffineBrightness>>{} : loadBrightness(brightnessFile);
    for (const auto& n : names)
      if (!brightnessFile.empty() && !brightness.contains(n))
        throw std::runtime_error(brightnessFile + " has no brightness of " + n + " (not a densify camera?)");

    sdv::ColmapExportSettings settings;
    settings.focalScale = focalScale;
    settings.maxFovDeg = maxFov;
    settings.imageExtension = "." + format;
    settings.correctBrightness = correctBrightness;
    sdv::ColmapExporter exporter(outDir, cameras, settings);
    const auto poses = sdv::loadKittiPoses(posesFile);
    std::optional<Sophus::SE3d> last;
    size_t numExported = 0;
    for (size_t i = 0; i < poses.size(); ++i) {
      const int frame = static_cast<int>(i);
      const Sophus::SE3d moved = last ? last->inverse() * poses[i] : Sophus::SE3d();
      const bool exported = frame % frameStride == 0 &&
                            (!last || moved.translation().norm() >= minTravel ||
                             moved.so3().log().norm() * 180.0 / M_PI >= minRotationDeg);
      std::vector<cv::Mat> images;
      std::vector<sdv::AffineBrightness> affine;
      bool ended = false;
      for (size_t c = 0; c < sources.size() && !ended; ++c) {
        if (!exported) {
          ended = !sources[c]->skip();
          continue;
        }
        const cv::Mat input = sources[c]->next();
        if (input.empty()) ended = true;
        else images.push_back(sdv::prepareImage(input, scale, prepared[c]));
        if (!brightnessFile.empty()) affine.push_back(interpolate(brightness.at(names[c]), frame));
      }
      if (ended) {
        spdlog::warn("input ended at run frame {} of {}", i, poses.size());
        break;
      }
      if (!exported) continue;
      exporter.addFrame(frame, images, poses[i], affine);
      last = poses[i];
      if (++numExported % 50 == 0) spdlog::info("exported {} frames (run frame {} of {})", numExported, i, poses.size());
    }

    std::vector<sdv::ColmapExportPoint> points;
    if (!cloudFile.empty()) {
      std::vector<Eigen::Vector3d> xyz;
      std::vector<sdv::Rgb> rgb;
      sdv::readPlyPoints(cloudFile, xyz, rgb);
      const size_t read = xyz.size();
      const std::vector<char> keep = sdv::hasNeighbours(xyz, neighbourRadius, minNeighbours);
      size_t kept = 0;
      for (size_t i = 0; i < xyz.size(); ++i)
        if (keep[i]) xyz[kept] = xyz[i], rgb[kept] = rgb[i], ++kept;
      xyz.resize(kept);
      rgb.resize(kept);
      sdv::voxelThin(xyz, rgb, voxel);
      spdlog::info("points3D: {} of {} points ({}+ neighbours within {:.2f} m: {}; {:.3f} m voxels)", xyz.size(), read,
                   minNeighbours, neighbourRadius, kept, voxel);
      for (size_t i = 0; i < xyz.size(); ++i) points.push_back({xyz[i], rgb[i]});
    }
    exporter.write(points);
    spdlog::info("wrote COLMAP model to {}: {} cameras, {} images, {} points", outDir, cameras.size(),
                 exporter.numImages(), points.size());
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
