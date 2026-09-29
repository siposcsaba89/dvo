#include <cmath>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/image_pyramid.h>
#include <sdv/photometric_ba_cost.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;
constexpr double kStep = 1e-6;

struct Scene {
  sdv::Camera cam = sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH);
  Sophus::SE3d T_c_b = synthetic::room::cameraFromBody(0.3, {0.8, 0.2, 0.5});
  cv::Mat hostImage, targetImage;
  std::unique_ptr<sdv::pba::Grid> grid;
  std::unique_ptr<sdv::pba::Interpolator> interpolator;
  sdv::pba::PointData point;
  double rho = 0;

  Scene(const Sophus::SE3d& T_w_h, const Sophus::SE3d& T_w_t) {
    cv::Mat h = synthetic::room::render(cam, T_c_b * T_w_h.inverse());
    cv::Mat t = synthetic::room::render(cam, T_c_b * T_w_t.inverse());
    cv::GaussianBlur(h, h, cv::Size(0, 0), 1.5);
    cv::GaussianBlur(t, t, cv::Size(0, 0), 1.5);
    hostImage = h;
    t.convertTo(targetImage, CV_8U);
    grid = std::make_unique<sdv::pba::Grid>(targetImage.ptr<std::uint8_t>(), 0, kH, 0, kW);
    interpolator = std::make_unique<sdv::pba::Interpolator>(*grid);
    const sdv::ImagePyramid pyr(hostImage, 1);
    const Eigen::Vector2d uv(150.3, 110.7);
    point = {0, 0, *sdv::makePatternPoint(cam, pyr.level(0), uv, {}), {}};
    cam.unproject(uv, point.bearing);
    rho = synthetic::room::trueRho(T_c_b * T_w_h.inverse(), point.bearing) * 1.05;
  }
};

// Central differences of a cost with respect to one parameter block; pose blocks through T * exp(delta).
Eigen::MatrixXd numericJacobian(const ceres::CostFunction& cost, std::vector<std::vector<double>> params, int block,
                                bool pose) {
  const int rows = cost.num_residuals(), cols = pose ? 6 : static_cast<int>(params[block].size());
  Eigen::MatrixXd J(rows, cols);
  for (int i = 0; i < cols; ++i) {
    std::vector<double> plus = params[block], minus = params[block];
    if (pose) {
      Sophus::Vector6d d = Sophus::Vector6d::Zero();
      d[i] = kStep;
      Eigen::Map<Sophus::SE3d>(plus.data()) = Eigen::Map<const Sophus::SE3d>(params[block].data()) * Sophus::SE3d::exp(d);
      Eigen::Map<Sophus::SE3d>(minus.data()) = Eigen::Map<const Sophus::SE3d>(params[block].data()) * Sophus::SE3d::exp(-d);
    } else {
      plus[i] += kStep, minus[i] -= kStep;
    }
    Eigen::VectorXd rp(rows), rm(rows);
    auto eval = [&](const std::vector<double>& value, Eigen::VectorXd& r) {
      auto p = params;
      p[block] = value;
      std::vector<const double*> ptrs;
      for (const auto& v : p) ptrs.push_back(v.data());
      cost.Evaluate(ptrs.data(), r.data(), nullptr);
    };
    eval(plus, rp), eval(minus, rm);
    J.col(i) = (rp - rm) / (2 * kStep);
  }
  return J;
}

Eigen::MatrixXd analyticJacobian(const ceres::CostFunction& cost, const std::vector<std::vector<double>>& params,
                                 int block, bool pose) {
  const int rows = cost.num_residuals();
  std::vector<std::vector<double>> storage;
  std::vector<double*> jacobians;
  for (const auto& p : params) storage.emplace_back(rows * p.size());
  for (auto& s : storage) jacobians.push_back(s.data());
  std::vector<const double*> ptrs;
  for (const auto& v : params) ptrs.push_back(v.data());
  Eigen::VectorXd r(rows);
  cost.Evaluate(ptrs.data(), r.data(), jacobians.data());
  const int ambient = static_cast<int>(params[block].size());
  const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> J(storage[block].data(),
                                                                                                   rows, ambient);
  return pose ? Eigen::MatrixXd(J.leftCols(6)) : Eigen::MatrixXd(J);
}

std::vector<double> poseParams(const Sophus::SE3d& T) { return {T.data(), T.data() + 7}; }

void expectClose(const Eigen::MatrixXd& analytic, const Eigen::MatrixXd& numeric, const char* name) {
  const double scale = std::max(1.0, numeric.cwiseAbs().maxCoeff());
  EXPECT_LT((analytic - numeric).cwiseAbs().maxCoeff() / scale, 2e-3) << name << "\nanalytic\n"
                                                                       << analytic << "\nnumeric\n"
                                                                       << numeric;
}

}  // namespace

TEST(PhotometricBACost, TemporalJacobiansMatchNumeric) {
  const Sophus::SE3d T_w_h(Sophus::SO3d::rotZ(0.1), Eigen::Vector3d(0.2, -0.1, 0.05));
  const Sophus::SE3d T_w_t(Sophus::SO3d::rotZ(0.15) * Sophus::SO3d::rotX(0.02), Eigen::Vector3d(0.5, 0.05, 0.0));
  const Scene s(T_w_h, T_w_t);
  const sdv::pba::TemporalCost cost(&s.point, &s.cam, s.interpolator.get(), s.T_c_b, s.T_c_b.inverse());
  const std::vector<std::vector<double>> params = {poseParams(T_w_h), poseParams(T_w_t), {s.rho}, {0.05, 2.0}, {-0.03, -1.0}};
  const char* names[] = {"host pose", "target pose", "inverse depth", "host affine", "target affine"};
  for (int b = 0; b < 5; ++b)
    expectClose(analyticJacobian(cost, params, b, b < 2), numericJacobian(cost, params, b, b < 2), names[b]);
}

TEST(PhotometricBACost, StaticJacobiansMatchNumeric) {
  const Scene s(Sophus::SE3d(), Sophus::SE3d(Sophus::SO3d::rotZ(0.05), Eigen::Vector3d(0.1, 0.3, 0.0)));
  const Sophus::SE3d T_t_h = s.T_c_b * Sophus::SE3d(Sophus::SO3d::rotZ(0.05), Eigen::Vector3d(0.1, 0.3, 0.0)).inverse() *
                             s.T_c_b.inverse();
  const sdv::pba::StaticCost cost(&s.point, &s.cam, s.interpolator.get(), T_t_h);
  const std::vector<std::vector<double>> params = {{s.rho}, {0.05, 2.0}, {-0.03, -1.0}};
  for (int b = 0; b < 3; ++b) expectClose(analyticJacobian(cost, params, b, false), numericJacobian(cost, params, b, false), "static");
}

TEST(PhotometricBACost, TemporalExtrinsicJacobiansMatchNumeric) {
  const Sophus::SE3d T_w_h(Sophus::SO3d::rotZ(0.1), Eigen::Vector3d(0.2, -0.1, 0.05));
  const Sophus::SE3d T_w_t(Sophus::SO3d::rotZ(0.15) * Sophus::SO3d::rotX(0.02), Eigen::Vector3d(0.5, 0.05, 0.0));
  const Scene s(T_w_h, T_w_t);
  // The target image was rendered with s.T_c_b; the host camera gets a slightly different extrinsic, so that the
  // adjoints differ between the blocks.
  const Sophus::SE3d T_b_ch = s.T_c_b.inverse() * Sophus::SE3d(Sophus::SO3d::rotY(0.01), Eigen::Vector3d(0.01, 0, 0));
  const sdv::pba::TemporalExtrinsicCost cost(&s.point, &s.cam, s.interpolator.get());
  const std::vector<std::vector<double>> params = {poseParams(T_w_h),          poseParams(T_w_t), poseParams(T_b_ch),
                                                   poseParams(s.T_c_b.inverse()), {s.rho}, {0.05, 2.0}, {-0.03, -1.0}};
  const char* names[] = {"host pose", "target pose", "host extrinsic", "target extrinsic", "inverse depth",
                         "host affine", "target affine"};
  for (int b = 0; b < 7; ++b)
    expectClose(analyticJacobian(cost, params, b, b < 4), numericJacobian(cost, params, b, b < 4), names[b]);
}

TEST(PhotometricBACost, TemporalSameCameraExtrinsicJacobiansMatchNumeric) {
  const Sophus::SE3d T_w_h(Sophus::SO3d::rotZ(0.1), Eigen::Vector3d(0.2, -0.1, 0.05));
  const Sophus::SE3d T_w_t(Sophus::SO3d::rotZ(0.15) * Sophus::SO3d::rotX(0.02), Eigen::Vector3d(0.5, 0.05, 0.0));
  const Scene s(T_w_h, T_w_t);
  const sdv::pba::TemporalSameCameraExtrinsicCost cost(&s.point, &s.cam, s.interpolator.get());
  const std::vector<std::vector<double>> params = {poseParams(T_w_h), poseParams(T_w_t), poseParams(s.T_c_b.inverse()),
                                                   {s.rho}, {0.05, 2.0}, {-0.03, -1.0}};
  const char* names[] = {"host pose", "target pose", "extrinsic", "inverse depth", "host affine", "target affine"};
  for (int b = 0; b < 6; ++b)
    expectClose(analyticJacobian(cost, params, b, b < 3), numericJacobian(cost, params, b, b < 3), names[b]);
}

TEST(PhotometricBACost, StaticExtrinsicJacobiansMatchNumeric) {
  const Sophus::SE3d T_b_t(Sophus::SO3d::rotZ(0.05), Eigen::Vector3d(0.1, 0.3, 0.0));
  const Scene s(Sophus::SE3d(), T_b_t);
  // Host camera at body pose identity, target camera: body T_b_t with the same mounting.
  const Sophus::SE3d T_b_ch = s.T_c_b.inverse(), T_b_ct = T_b_t * s.T_c_b.inverse();
  const sdv::pba::StaticExtrinsicCost cost(&s.point, &s.cam, s.interpolator.get());
  const std::vector<std::vector<double>> params = {poseParams(T_b_ch), poseParams(T_b_ct), {s.rho}, {0.05, 2.0},
                                                   {-0.03, -1.0}};
  const char* names[] = {"host extrinsic", "target extrinsic", "inverse depth", "host affine", "target affine"};
  for (int b = 0; b < 5; ++b)
    expectClose(analyticJacobian(cost, params, b, b < 2), numericJacobian(cost, params, b, b < 2), names[b]);
}

TEST(PhotometricBACost, RelativePoseJacobiansMatchNumericNearTheMeasurement) {
  const Sophus::SE3d T_w_a(Sophus::SO3d::rotZ(0.3), Eigen::Vector3d(1, 2, 0));
  const Sophus::SE3d T_a_b(Sophus::SO3d::rotY(0.1), Eigen::Vector3d(1.5, 0.2, 0.1));
  // Small residual: the identity right Jacobian is exact to first order.
  const Sophus::SE3d T_w_b = T_w_a * T_a_b * Sophus::SE3d::exp((Sophus::Vector6d() << 1e-4, -2e-4, 1e-4, 1e-5, 2e-5, -1e-5).finished());
  const sdv::pba::RelativePoseCost cost(T_a_b, 0.01, 0.1);
  const std::vector<std::vector<double>> params = {poseParams(T_w_a), poseParams(T_w_b)};
  for (int b = 0; b < 2; ++b) expectClose(analyticJacobian(cost, params, b, true), numericJacobian(cost, params, b, true), "relative pose");
}
