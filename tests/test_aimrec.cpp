#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>
#include <zip.h>

#include <sdv/aimrec/calibration.h>
#include <sdv/aimrec/camera_meta.h>

namespace {

using namespace sdv::aimrec;
namespace fs = std::filesystem;

template <typename T>
void put(std::vector<std::uint8_t>& b, std::size_t at, T v) {
  if (b.size() < at + sizeof(T)) b.resize(at + sizeof(T));
  std::memcpy(b.data() + at, &v, sizeof(T));
}

struct Record {
  std::uint64_t offset, frameId;
  std::uint32_t size;
  std::int64_t t;
};

// Byte layout of a version 8 camera meta file as the recorder writes it.
std::vector<std::uint8_t> metaFileV8(const std::vector<Record>& records, std::int32_t triggerBoxId = -1) {
  std::vector<std::uint8_t> b(15 + 222, 0);
  put<std::uint16_t>(b, 0, 15);
  put<std::uint16_t>(b, 2, 1);
  put<std::uint32_t>(b, 4, 222);
  put<std::uint32_t>(b, 8, 77);
  const std::size_t h = 15;
  put<std::uint32_t>(b, h, 8);
  b[h + 5] = 1;                           // H.264
  put<std::uint16_t>(b, h + 14, 0x8001);  // NvMedia
  put<std::int32_t>(b, h + 22, triggerBoxId);
  put<float>(b, h + 26, 15.0f);
  put<std::int32_t>(b, h + 86, 1936);
  put<std::int32_t>(b, h + 90, 1220);
  std::memcpy(b.data() + h + 94, "TE05232", 7);
  for (const Record& r : records) {
    const std::size_t o = b.size();
    b.resize(o + 77, 0);
    put<std::uint64_t>(b, o, r.offset);
    put<std::uint32_t>(b, o + 8, r.size);
    put<std::uint64_t>(b, o + 12, r.frameId);
    put<std::int64_t>(b, o + 20, r.t);
    put<std::int64_t>(b, o + 28, r.t + 5'000'000);
    put<float>(b, o + 68, 3.5f);
    put<float>(b, o + 72, 11000.0f);
  }
  return b;
}

const std::vector<Record> kRecords = {
    {0, 77, 1000, 1'000'000'000}, {1000, 78, 900, 1'066'666'666}, {1900, 80, 950, 1'200'000'000}};

TEST(AimRecMeta, ParsesVersion8) {
  std::vector<std::uint8_t> bytes = metaFileV8(kRecords);
  bytes.resize(bytes.size() + 30);  // partially written trailing record
  const CameraMeta meta = parseCameraMeta(bytes, "test");
  EXPECT_EQ(meta.version, 8);
  EXPECT_EQ(meta.serial, "TE05232");
  EXPECT_EQ(meta.width, 1936);
  EXPECT_EQ(meta.height, 1220);
  EXPECT_FLOAT_EQ(meta.fps, 15.0f);
  EXPECT_TRUE(meta.h264);
  EXPECT_EQ(meta.gopLength, 0);
  EXPECT_EQ(meta.colorStandard, ColorStandard::Bt709Full);
  ASSERT_EQ(meta.frames.size(), 3u);
  EXPECT_EQ(meta.frames[1].offset, 1000u);
  EXPECT_EQ(meta.frames[1].size, 900u);
  EXPECT_EQ(meta.frames[1].frameId, 78u);
  EXPECT_EQ(meta.frames[1].timestampNs, 1'066'666'666);
  EXPECT_EQ(meta.frames[1].receivedNs, 1'071'666'666);
  EXPECT_FLOAT_EQ(meta.frames[1].gain, 3.5f);
  EXPECT_FLOAT_EQ(meta.frames[1].exposureUs, 11000.0f);
}

TEST(AimRecMeta, GopLengthFromTriggerBoxId) {
  EXPECT_EQ(parseCameraMeta(metaFileV8(kRecords, -30), "test").gopLength, 30);
}

TEST(AimRecMeta, Lookup) {
  const CameraMeta meta = parseCameraMeta(metaFileV8(kRecords), "test");
  EXPECT_EQ(meta.indexOf(78), 1u);
  EXPECT_EQ(meta.indexOf(79), std::nullopt);
  EXPECT_EQ(meta.nearestIndex(0), 0u);
  EXPECT_EQ(meta.nearestIndex(1'040'000'000), 1u);
  EXPECT_EQ(meta.nearestIndex(1'150'000'000), 2u);
  EXPECT_EQ(meta.nearestIndex(5'000'000'000), 2u);
}

TEST(AimRecMeta, RejectsBadFiles) {
  std::vector<std::uint8_t> bytes = metaFileV8(kRecords);
  put<std::uint8_t>(bytes, 14, 8);  // compressed
  EXPECT_THROW(parseCameraMeta(bytes, "test"), std::runtime_error);
  EXPECT_THROW(parseCameraMeta(metaFileV8({kRecords[1], kRecords[0]}), "test"), std::runtime_error);
  EXPECT_THROW(parseCameraMeta(std::vector<std::uint8_t>(10), "test"), std::runtime_error);
}

Eigen::Vector3d opticalAxis(double yaw, double pitch, double roll) {
  return vehicleFromCamera(Eigen::Vector3d::Zero(), {yaw, pitch, roll}).so3() * Eigen::Vector3d::UnitZ();
}

TEST(AimRecCalibration, ExtrinsicConvention) {
  const Sophus::SE3d T = vehicleFromCamera({2.0, 0.5, 1.2}, {0, 0, 0});
  EXPECT_TRUE(T.translation().isApprox(Eigen::Vector3d(2.0, 0.5, 1.2)));
  EXPECT_TRUE((T.so3() * Eigen::Vector3d::UnitZ()).isApprox(Eigen::Vector3d::UnitX()));   // forward
  EXPECT_TRUE((T.so3() * Eigen::Vector3d::UnitX()).isApprox(-Eigen::Vector3d::UnitY()));  // right
  EXPECT_TRUE((T.so3() * Eigen::Vector3d::UnitY()).isApprox(-Eigen::Vector3d::UnitZ()));  // down
  // Left fisheye (sensorconfig yaw -90, pitch -24): looks left and down.
  const Eigen::Vector3d left = opticalAxis(-90, -24, 0);
  EXPECT_NEAR(left.y(), std::cos(24 * std::numbers::pi / 180), 1e-9);
  EXPECT_NEAR(left.z(), -std::sin(24 * std::numbers::pi / 180), 1e-9);
  EXPECT_TRUE(opticalAxis(180, 0, 0).isApprox(-Eigen::Vector3d::UnitX()));
}

const char* kSensorConfig = R"(sensors:
  - label: novatel
    device_id: DMMU
    sensor_type: gpsimu
  - label: F_FISHEYE_C
    device_id: TE05232
    sensor_type: camera
    custom_vars:
      obstruction_mask_file: obstruction_masks/F_FISHEYE_C_mask.png
    image_resolution_px: [40, 20]
    model: eucm
    principal_point_px: [19.5, 9.5]
    focal_length_px: [10.0, 11.0]
    alpha: 0.58
    beta: 0.83
    pos_meter: [3.87, 0.03, 0.84]
    yaw_pitch_roll_deg: [-0.2, -27.3, -0.7]
  - label: F_PINHOLE
    device_id: TE00000
    sensor_type: camera
    model: pinhole
)";

TEST(AimRecCalibration, ParsesSensorConfig) {
  for (const std::string& yaml : {std::string(kSensorConfig), std::string(kSensorConfig).substr(9)}) {
    const auto cams = parseSensorConfig(yaml, "test");  // `sensors:` map and bare list
    ASSERT_EQ(cams.size(), 1u);
    const CameraCalibration& c = cams[0];
    EXPECT_EQ(c.label, "F_FISHEYE_C");
    EXPECT_EQ(c.deviceId, "TE05232");
    EXPECT_EQ(c.width, 40);
    EXPECT_EQ(c.height, 20);
    EXPECT_DOUBLE_EQ(c.fy, 11.0);
    EXPECT_DOUBLE_EQ(c.cx, 19.5);
    EXPECT_DOUBLE_EQ(c.beta, 0.83);
    EXPECT_EQ(c.maskFile, "obstruction_masks/F_FISHEYE_C_mask.png");
    EXPECT_TRUE(c.T_vehicle_camera.translation().isApprox(Eigen::Vector3d(3.87, 0.03, 0.84)));
  }
}

class AimRecFiles : public ::testing::Test {
 protected:
  void SetUp() override {
    m_dir = fs::temp_directory_path() / ("sdv_aimrec_" + std::to_string(::testing::UnitTest::GetInstance()->random_seed()) +
                                         ::testing::UnitTest::GetInstance()->current_test_info()->name());
    fs::create_directories(m_dir / "obstruction_masks");
    std::ofstream(m_dir / "sensorconfig.yaml") << kSensorConfig;
    // Half resolution: the loader scales it to the camera size.
    cv::Mat mask(10, 20, CV_8UC1, cv::Scalar(255));
    mask.rowRange(8, 10).setTo(0);
    cv::imwrite((m_dir / "obstruction_masks/F_FISHEYE_C_mask.png").string(), mask);
  }
  void TearDown() override { fs::remove_all(m_dir); }

  static void checkMask(const CameraCalibration& c) {
    ASSERT_EQ(c.mask.size(), cv::Size(40, 20));
    EXPECT_EQ(c.mask.type(), CV_8UC1);
    EXPECT_EQ(c.mask.at<std::uint8_t>(0, 0), 255);
    EXPECT_EQ(c.mask.at<std::uint8_t>(19, 39), 0);
  }

  fs::path m_dir;
};

TEST_F(AimRecFiles, LoadsSensorConfigFromDirectory) {
  const VehicleCalibration calib = loadCalibration(m_dir / "sensorconfig.yaml");
  ASSERT_NE(calib.find("F_FISHEYE_C"), nullptr);
  checkMask(*calib.find("F_FISHEYE_C"));
  // Masks only for the selected cameras.
  EXPECT_TRUE(loadCalibration(m_dir / "sensorconfig.yaml", {}, std::vector<std::string>{}).cameras[0].mask.empty());
  checkMask(loadCalibration(m_dir / "sensorconfig.yaml", {}, std::vector<std::string>{"F_FISHEYE_C"}).cameras[0]);
}

TEST_F(AimRecFiles, LoadsCalibrationZip) {
  const fs::path zipFile = m_dir / "aimprototype_config.zip";
  int err = 0;
  zip_t* z = zip_open(zipFile.string().c_str(), ZIP_CREATE | ZIP_TRUNCATE, &err);
  ASSERT_NE(z, nullptr);
  const std::string vdir = "vehicle_database/VEHICLES/Zion/";
  for (const std::string f : {"sensorconfig.yaml", "obstruction_masks/F_FISHEYE_C_mask.png"}) {
    zip_source_t* src = zip_source_file(z, (m_dir / f).string().c_str(), 0, -1);
    ASSERT_NE(src, nullptr);
    ASSERT_GE(zip_file_add(z, (vdir + f).c_str(), src, ZIP_FL_OVERWRITE), 0);
  }
  ASSERT_EQ(zip_close(z), 0);

  const VehicleCalibration calib = loadCalibration(zipFile);
  EXPECT_EQ(calib.vehicle, "Zion");
  ASSERT_EQ(calib.cameras.size(), 1u);
  checkMask(calib.cameras[0]);
  EXPECT_THROW(loadCalibration(zipFile, "Other"), std::runtime_error);
}

}  // namespace
