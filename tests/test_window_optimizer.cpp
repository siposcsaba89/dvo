#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <random>

#include <gtest/gtest.h>
#include <opencv2/imgproc.hpp>

#include <sdv/window_optimizer.h>

#include <synthetic_scene.h>

namespace {

constexpr int kW = 320, kH = 240;

class WindowOptimizerTest : public ::testing::TestWithParam<sdv::Camera> {};

cv::Mat renderSmooth(const sdv::Camera& cam, const Sophus::SE3d& T_c_w, double gain = 1.0, double offset = 0.0) {
  cv::Mat img = synthetic::renderTarget(cam, T_c_w, gain, offset);
  cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
  return img;
}

struct Scene {
  std::vector<Sophus::SE3d> poses;  // T_c_w
  std::vector<sdv::AffineBrightness> affine;
  std::vector<std::shared_ptr<const sdv::ImagePyramid>> images;
};

Scene makeScene(const sdv::Camera& cam, int numFrames) {
  Scene s;
  const Sophus::Vector6d v = (Sophus::Vector6d() << -0.08, 0.01, -0.1, 0.003, 0.01, -0.002).finished();
  for (int k = 0; k < numFrames; ++k) {
    s.poses.push_back(Sophus::SE3d::exp(k * v));
    const double gain = 1.0 + 0.03 * k, offset = -3.0 * k;
    s.affine.push_back({std::log(gain), offset});
    s.images.push_back(std::make_shared<sdv::ImagePyramid>(renderSmooth(cam, s.poses[k], gain, offset), 1));
  }
  return s;
}

// Frames at perturbed poses, points at perturbed depth; returns true inverse distance per point id.
std::vector<double> buildWindow(const sdv::Camera& cam, const Scene& s, sdv::WindowOptimizer& opt,
                                bool perturb) {
  std::mt19937 rng(7);
  std::normal_distribution<double> n01(0.0, 1.0);
  for (size_t k = 0; k < s.poses.size(); ++k) {
    Sophus::SE3d T = s.poses[k];
    if (perturb && k > 0) {
      Sophus::Vector6d d;
      d << 0.02 * n01(rng), 0.02 * n01(rng), 0.02 * n01(rng), 0.004 * n01(rng), 0.004 * n01(rng), 0.004 * n01(rng);
      T = Sophus::SE3d::exp(d) * T;
    }
    opt.addFrame(s.images[k], T, perturb && k > 0 ? sdv::AffineBrightness{} : s.affine[k]);
  }
  std::vector<double> truth;
  for (size_t k = 0; k < s.poses.size(); ++k)
    for (int v = 12; v < kH - 12; v += 14)
      for (int u = 12; u < kW - 12; u += 14) {
        Eigen::Vector3d b;
        if (!cam.unproject(Eigen::Vector2d(u, v), b)) continue;
        const double rho = synthetic::trueRho(s.poses[k], b);
        if (rho <= 0) continue;
        const double start = perturb ? rho * (1.0 + 0.1 * std::clamp(n01(rng), -2.0, 2.0)) : rho;
        const int id = opt.addPoint(opt.frames()[k].id, Eigen::Vector2d(u, v), start);
        if (id < 0) continue;
        truth.resize(id + 1, 0.0);
        truth[id] = rho;
      }
  return truth;
}

// The gauge frame fixes the world origin, so a monocular scale change scales every translation of T_c_w.
void expectSamePosesUpToScale(const sdv::WindowOptimizer& opt, const std::vector<Sophus::SE3d>& expected,
                              double tolerance) {
  ASSERT_EQ(opt.frames().size(), expected.size());
  double num = 0, den = 0;
  for (size_t k = 0; k < expected.size(); ++k) {
    const Eigen::Vector3d& t = opt.frames()[k].params.T_b_w.translation();
    num += t.dot(expected[k].translation());
    den += t.squaredNorm();
  }
  const double s = den > 0 ? num / den : 1.0;
  for (size_t k = 0; k < expected.size(); ++k) {
    const Sophus::SE3d& T = opt.frames()[k].params.T_b_w;
    EXPECT_LT((T.so3() * expected[k].so3().inverse()).log().norm(), tolerance) << "frame " << k;
    EXPECT_LT((s * T.translation() - expected[k].translation()).norm(), tolerance) << "frame " << k;
  }
}

}  // namespace

TEST_P(WindowOptimizerTest, ResidualJacobiansMatchNumeric) {
  const sdv::Camera& cam = GetParam();
  const Sophus::SE3d T_h_w = Sophus::SE3d::exp((Sophus::Vector6d() << 0.05, -0.02, 0.1, 0.01, -0.02, 0.005).finished());
  const Sophus::SE3d T_t_w = Sophus::SE3d::exp((Sophus::Vector6d() << -0.2, 0.03, -0.15, 0.0, 0.02, 0.01).finished());
  const sdv::ImagePyramid host(renderSmooth(cam, T_h_w), 1);
  const sdv::ImagePyramid target(renderSmooth(cam, T_t_w, 1.1, -5.0), 1);
  const sdv::FrameParams hp{T_h_w, {0.05, 3.0}, 1.0};
  const sdv::FrameParams tp{T_t_w, {0.12, -2.0}, 1.2};
  const sdv::PhotometricSettings ps;

  int checked = 0;
  for (int v = 60; v < kH - 60; v += 30)
    for (int u = 60; u < kW - 60; u += 30) {
      const auto pattern = sdv::makePatternPoint(cam, host.level(0), Eigen::Vector2d(u, v), ps);
      Eigen::Vector3d b;
      ASSERT_TRUE(pattern && cam.unproject(Eigen::Vector2d(u, v), b));
      const double rho = synthetic::trueRho(T_h_w, b);
      sdv::WindowPatternResidual res;
      if (!sdv::evaluateWindowResidual(*pattern, rho, hp, tp, hp, tp, cam, target.level(0), ps, res)) continue;
      ++checked;

      auto residuals = [&](const sdv::FrameParams& h, const sdv::FrameParams& t, double r) {
        sdv::WindowPatternResidual out;
        EXPECT_TRUE(sdv::evaluateWindowResidual(*pattern, r, h, t, h, t, cam, target.level(0), ps, out));
        Eigen::Matrix<double, sdv::kPatternSize, 1> vec;
        for (int k = 0; k < sdv::kPatternSize; ++k) vec[k] = out.pixels[k].r;
        return vec;
      };
      auto perturbed = [](sdv::FrameParams f, int i, double eps) {
        if (i < 6) {
          Sophus::Vector6d d = Sophus::Vector6d::Zero();
          d[i] = eps;
          f.T_c_w = Sophus::SE3d::exp(d) * f.T_c_w;
        } else if (i == 6) {
          f.affine.a += eps;
        } else {
          f.affine.b += eps;
        }
        return f;
      };
      for (int i = 0; i < 8; ++i) {
        // Steps move the projection by ~2 px: bilinear interpolation is only piecewise linear, so smaller
        // steps measure the slope of a single pixel cell instead of the image gradient.
        const double eps = i < 3 ? 4e-2 : i < 6 ? 8e-3 : 1e-4;
        using Vec = Eigen::Matrix<double, sdv::kPatternSize, 1>;
        const Vec dh = (residuals(perturbed(hp, i, eps), tp, rho) - residuals(perturbed(hp, i, -eps), tp, rho)) / (2 * eps);
        const Vec dt = (residuals(hp, perturbed(tp, i, eps), rho) - residuals(hp, perturbed(tp, i, -eps), rho)) / (2 * eps);
        for (int k = 0; k < sdv::kPatternSize; ++k) {
          const double tolH = 0.1 * res.pixels[k].dHost.norm() + 1e-3;
          const double tolT = 0.1 * res.pixels[k].dTarget.norm() + 1e-3;
          EXPECT_NEAR(res.pixels[k].dHost[i], dh[k], tolH) << "host param " << i;
          EXPECT_NEAR(res.pixels[k].dTarget[i], dt[k], tolT) << "target param " << i;
        }
      }
      const double eps = 0.03;
      const Eigen::Matrix<double, sdv::kPatternSize, 1> dr = (residuals(hp, tp, rho + eps) - residuals(hp, tp, rho - eps)) / (2 * eps);
      double maxRho = 0;
      for (const auto& px : res.pixels) maxRho = std::max(maxRho, std::abs(px.dRho));
      for (int k = 0; k < sdv::kPatternSize; ++k) EXPECT_NEAR(res.pixels[k].dRho, dr[k], 0.1 * maxRho + 1e-2);

      // FEJ: a different linearisation point changes the Jacobians but not the residual.
      sdv::FrameParams h0 = perturbed(hp, 2, 0.01);
      sdv::WindowPatternResidual fej;
      ASSERT_TRUE(sdv::evaluateWindowResidual(*pattern, rho, hp, tp, h0, tp, cam, target.level(0), ps, fej));
      for (int k = 0; k < sdv::kPatternSize; ++k) EXPECT_DOUBLE_EQ(fej.pixels[k].r, res.pixels[k].r);
    }
  EXPECT_GT(checked, 10);
}

TEST_P(WindowOptimizerTest, ConvergesFromPerturbedState) {
  const sdv::Camera& cam = GetParam();
  const Scene scene = makeScene(cam, 5);
  sdv::WindowOptimizer opt(cam);
  const std::vector<double> truth = buildWindow(cam, scene, opt, true);
  ASSERT_GT(opt.points().size(), 500u);

  const auto res = opt.optimize(30);
  EXPECT_LT(res.finalEnergy, 0.1 * res.initialEnergy);

  std::vector<double> ratios;
  for (const auto& p : opt.points()) ratios.push_back(truth[p.id] / p.rho);
  std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
  const double s = ratios[ratios.size() / 2];
  EXPECT_NEAR(s, 1.0, 0.1);  // monocular scale is a free gauge

  int accurate = 0;
  for (const auto& p : opt.points()) accurate += std::abs(p.rho * s - truth[p.id]) < 0.02 * truth[p.id];
  // Blurring each rendered frame in its own pixels is slightly inconsistent under fisheye distortion, which caps
  // this at ~90 % for the fisheye camera.
  EXPECT_GT(accurate, 0.88 * opt.points().size());

  for (size_t k = 0; k < scene.poses.size(); ++k) {
    const auto& f = opt.frames()[k].params;
    Sophus::SE3d T = f.T_b_w;
    T.translation() /= s;
    const Sophus::Vector6d err = (T * scene.poses[k].inverse()).log();
    EXPECT_LT(err.head<3>().norm(), 3e-3) << "frame " << k;
    EXPECT_LT(err.tail<3>().norm(), 1e-3) << "frame " << k;
    // a and b are strongly correlated; compare the brightness mapping at mid-grey.
    auto mapped = [](const sdv::AffineBrightness& a) { return std::exp(a.a) * 128.0 + a.b; };
    EXPECT_NEAR(mapped(f.affine[0]), mapped(scene.affine[k]), 1.0) << "frame " << k;
  }
}

TEST_P(WindowOptimizerTest, MarginalizationKeepsOptimumAndInformation) {
  const sdv::Camera& cam = GetParam();
  const Scene scene = makeScene(cam, 5);
  sdv::WindowOptimizer opt(cam);
  buildWindow(cam, scene, opt, true);
  opt.optimize(30);

  std::vector<Sophus::SE3d> before;
  for (size_t k = 1; k < opt.frames().size(); ++k) before.push_back(opt.frames()[k].params.T_b_w);
  opt.marginalizeFrame(opt.frames()[0].id);
  ASSERT_EQ(opt.frames().size(), 4u);
  opt.optimize(10);
  // Residuals of other points into the removed frame are dropped, so the optimum shifts slightly.
  expectSamePosesUpToScale(opt, before, 1e-3);

  // The prior now carries the gauge and the removed frame's constraints: a perturbed window returns.
  opt.marginalizeFrame(opt.frames()[0].id);
  opt.optimize(10);
  std::vector<Sophus::SE3d> kept;
  for (const auto& f : opt.frames()) kept.push_back(f.params.T_b_w);
  for (const auto& f : opt.frames()) {
    sdv::KeyframeParams p = f.params;
    p.T_b_w = Sophus::SE3d::exp((Sophus::Vector6d() << 0.01, -0.01, 0.01, 0.002, -0.002, 0.001).finished()) * p.T_b_w;
    opt.setFrameParams(f.id, p);
  }
  opt.optimize(20);
  expectSamePosesUpToScale(opt, kept, 1e-3);
}

TEST_P(WindowOptimizerTest, SlidingWindowStaysAtTruthInTurn) {
  const sdv::Camera& cam = GetParam();
  sdv::WindowOptimizer opt(cam);
  const Sophus::Vector6d v = (Sophus::Vector6d() << -0.12, 0.0, -0.05, 0.0, 0.02, 0.0).finished();
  std::map<int, Sophus::SE3d> truth;
  std::map<int, double> trueRho;
  for (int k = 0; k < 12; ++k) {
    const Sophus::SE3d T_c_w = Sophus::SE3d::exp(k * v);
    auto image = std::make_shared<sdv::ImagePyramid>(renderSmooth(cam, T_c_w), 1);
    const int id = opt.addFrame(image, T_c_w);
    truth[id] = T_c_w;
    for (int y = 16; y < kH - 16; y += 16)
      for (int x = 16; x < kW - 16; x += 16) {
        Eigen::Vector3d b;
        if (!cam.unproject(Eigen::Vector2d(x, y), b)) continue;
        const double rho = synthetic::trueRho(T_c_w, b);
        if (rho <= 0) continue;
        if (const int pid = opt.addPoint(id, Eigen::Vector2d(x, y), rho); pid >= 0) trueRho[pid] = rho;
      }
    opt.optimize(6);
    if (opt.frames().size() > 5) opt.marginalizeFrame(opt.frames().front().id);
  }
  for (const auto& f : opt.frames()) {
    std::vector<double> r;
    for (const auto& p : opt.points())
      if (p.host == f.id) r.push_back(p.rho / trueRho.at(p.id));
    std::nth_element(r.begin(), r.begin() + r.size() / 2, r.end());
    EXPECT_NEAR(r[r.size() / 2], 1.0, 0.01) << "host " << f.id;
    const Sophus::Vector6d err = (f.params.T_b_w * truth.at(f.id).inverse()).log();
    EXPECT_LT(err.head<3>().norm(), 0.01) << "frame " << f.id;
    EXPECT_LT(err.tail<3>().norm(), 1e-3) << "frame " << f.id;
  }
}

TEST_P(WindowOptimizerTest, StereoResidualsRecoverMetricScale) {
  const sdv::Camera& cam = GetParam();
  const Scene scene = makeScene(cam, 4);
  const Sophus::SE3d T_r_l(Sophus::SO3d(), Eigen::Vector3d(-0.3, 0, 0));
  sdv::WindowOptimizer opt(sdv::Rig{{cam, cam}, {Sophus::SE3d(), T_r_l}});
  constexpr double wrongScale = 1.2;
  std::map<int, double> truth;
  for (size_t k = 0; k < scene.poses.size(); ++k) {
    Sophus::SE3d T = scene.poses[k];
    T.translation() *= wrongScale;
    const auto right = std::make_shared<sdv::ImagePyramid>(
        renderSmooth(cam, T_r_l * scene.poses[k], std::exp(scene.affine[k].a), scene.affine[k].b), 1);
    const int id = opt.addFrame({scene.images[k], right}, T, {scene.affine[k], scene.affine[k]});
    for (int v = 16; v < kH - 16; v += 16)
      for (int u = 16; u < kW - 16; u += 16) {
        Eigen::Vector3d b;
        if (!cam.unproject(Eigen::Vector2d(u, v), b)) continue;
        const double rho = synthetic::trueRho(scene.poses[k], b);
        if (rho <= 0) continue;
        if (const int pid = opt.addPoint(id, Eigen::Vector2d(u, v), rho / wrongScale); pid >= 0) truth[pid] = rho;
      }
  }
  opt.optimize(30);
  std::vector<double> ratios;
  for (const auto& p : opt.points()) ratios.push_back(p.rho / truth.at(p.id));
  std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
  EXPECT_NEAR(ratios[ratios.size() / 2], 1.0, 0.01);
  for (size_t k = 1; k < scene.poses.size(); ++k)
    EXPECT_LT((opt.frames()[k].params.T_b_w.translation() - scene.poses[k].translation()).norm(), 5e-3) << k;
}

TEST_P(WindowOptimizerTest, RigJacobiansMatchNumeric) {
  const sdv::Camera& cam = GetParam();
  const Sophus::SE3d T_ch_b = Sophus::SE3d::exp((Sophus::Vector6d() << 0.2, -0.1, 0.05, 0.03, -0.05, 0.02).finished());
  const Sophus::SE3d T_ct_b = Sophus::SE3d::exp((Sophus::Vector6d() << -0.3, 0.05, 0.1, -0.02, 0.04, -0.03).finished());
  const Sophus::SE3d T_bh_w = T_ch_b.inverse() * Sophus::SE3d::exp((Sophus::Vector6d() << 0.05, -0.02, 0.1, 0.01, -0.02, 0.005).finished());
  const Sophus::SE3d T_bt_w = T_ct_b.inverse() * Sophus::SE3d::exp((Sophus::Vector6d() << -0.2, 0.03, -0.15, 0.0, 0.02, 0.01).finished());
  const sdv::AffineBrightness ah{0.05, 3.0}, at{0.12, -2.0};
  const sdv::PhotometricSettings ps;
  auto params = [](const Sophus::SE3d& T_c_b, const Sophus::SE3d& T_b_w, const sdv::AffineBrightness& a) {
    return sdv::FrameParams{T_c_b * T_b_w, a, 1.0};
  };
  const sdv::ImagePyramid host(renderSmooth(cam, T_ch_b * T_bh_w), 1);
  const sdv::ImagePyramid target(renderSmooth(cam, T_ct_b * T_bt_w, 1.1, -5.0), 1);
  // Static pair: host and target camera in the same keyframe (body pose T_bh_w).
  const sdv::ImagePyramid staticTarget(renderSmooth(cam, T_ct_b * T_bh_w, 1.1, -5.0), 1);

  int checked = 0;
  for (int v = 60; v < kH - 60; v += 30)
    for (int u = 60; u < kW - 60; u += 30) {
      const auto pattern = sdv::makePatternPoint(cam, host.level(0), Eigen::Vector2d(u, v), ps);
      Eigen::Vector3d b;
      ASSERT_TRUE(pattern && cam.unproject(Eigen::Vector2d(u, v), b));
      const double rho = synthetic::trueRho(T_ch_b * T_bh_w, b);
      for (bool sameFrame : {false, true}) {
        const Sophus::SE3d& T_bt = sameFrame ? T_bh_w : T_bt_w;
        const sdv::ImageLevel& img = sameFrame ? staticTarget.level(0) : target.level(0);
        auto residuals = [&](const Sophus::SE3d& Th, const sdv::AffineBrightness& a_h, const Sophus::SE3d& Tt,
                             const sdv::AffineBrightness& a_t, bool& ok) {
          const sdv::FrameParams h = params(T_ch_b, Th, a_h), t = params(T_ct_b, sameFrame ? Th : Tt, a_t);
          sdv::WindowPatternResidual out;
          ok = sdv::evaluateWindowResidual(*pattern, rho, sdv::makeWindowPairState(h, t, h, t, T_ch_b, T_ct_b, sameFrame),
                                           cam, img, ps, out);
          Eigen::Matrix<double, sdv::kPatternSize, 1> vec;
          for (int k = 0; k < sdv::kPatternSize; ++k) vec[k] = out.pixels[k].r;
          return vec;
        };
        bool ok;
        residuals(T_bh_w, ah, T_bt, at, ok);
        if (!ok) continue;
        const sdv::FrameParams h = params(T_ch_b, T_bh_w, ah), t = params(T_ct_b, T_bt, at);
        sdv::WindowPatternResidual res;
        ASSERT_TRUE(sdv::evaluateWindowResidual(*pattern, rho, sdv::makeWindowPairState(h, t, h, t, T_ch_b, T_ct_b, sameFrame),
                                                cam, img, ps, res));
        ++checked;
        for (int i = 0; i < 8; ++i) {
          const double eps = i < 3 ? 4e-2 : i < 6 ? 8e-3 : 1e-4;
          auto pose = [&](const Sophus::SE3d& T, double e) {
            Sophus::Vector6d d = Sophus::Vector6d::Zero();
            if (i < 6) d[i] = e;
            return Sophus::SE3d::exp(d) * T;
          };
          auto affine = [&](sdv::AffineBrightness a, double e) {
            if (i == 6) a.a += e;
            if (i == 7) a.b += e;
            return a;
          };
          bool okp, okm;
          using Vec = Eigen::Matrix<double, sdv::kPatternSize, 1>;
          const Vec dh = (residuals(pose(T_bh_w, eps), affine(ah, eps), T_bt, at, okp) -
                          residuals(pose(T_bh_w, -eps), affine(ah, -eps), T_bt, at, okm)) / (2 * eps);
          const Vec dt = (residuals(T_bh_w, ah, pose(T_bt, eps), affine(at, eps), okp) -
                          residuals(T_bh_w, ah, pose(T_bt, -eps), affine(at, -eps), okm)) / (2 * eps);
          for (int k = 0; k < sdv::kPatternSize; ++k) {
            EXPECT_NEAR(res.pixels[k].dHost[i], dh[k], 0.1 * res.pixels[k].dHost.norm() + 1e-3) << "host " << i;
            // Static: the target pose is the host pose, so its numeric derivative is part of dh.
            if (!sameFrame) EXPECT_NEAR(res.pixels[k].dTarget[i], dt[k], 0.1 * res.pixels[k].dTarget.norm() + 1e-3) << "target " << i;
            else if (i < 6) EXPECT_EQ(res.pixels[k].dTarget[i], 0.0);
            else EXPECT_NEAR(res.pixels[k].dTarget[i], dt[k], 0.1 * res.pixels[k].dTarget.norm() + 1e-3) << "target " << i;
          }
        }
      }
    }
  EXPECT_GT(checked, 10);
}

INSTANTIATE_TEST_SUITE_P(Cameras, WindowOptimizerTest,
                         ::testing::Values(sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH),
                                           sdv::Camera::eucm(140, 140, 160.2, 119.7, 0.6, 1.1, kW, kH)),
                         [](const auto& info) { return info.param.isPinhole() ? "Pinhole" : "Fisheye"; });

// Front and rear camera without a common view: metric scale comes from the lever arm of the extrinsics, which is
// only observable when the body rotates (Clipp et al., 2008). With 0.04 rad per keyframe instead of 0.1 the
// photometric noise of the rendering already biases the scale by ~3 %.
TEST(WindowOptimizerRig, NonOverlappingRigRecoversScaleInTurn) {
  const sdv::Camera cam = sdv::Camera::pinhole(250, 250, 159.5, 119.5, kW, kH);
  const sdv::Rig rig{{cam, cam},
                     {synthetic::room::cameraFromBody(0.0, {1.5, 0.0, 0.3}),
                      synthetic::room::cameraFromBody(M_PI, {-1.0, 0.0, 0.3})}};
  sdv::WindowOptimizer opt(rig);
  constexpr double wrongScale = 1.2;
  std::map<int, double> truth;
  std::vector<Sophus::SE3d> trueBody;
  for (int k = 0; k < 5; ++k) {
    const Sophus::SE3d T_w_b = Sophus::SE3d::exp((Sophus::Vector6d() << 0.25 * k, 0, 0, 0, 0, 0.1 * k).finished());
    const Sophus::SE3d T_b_w = T_w_b.inverse();
    trueBody.push_back(T_b_w);
    std::vector<std::shared_ptr<const sdv::ImagePyramid>> images;
    for (int c = 0; c < 2; ++c) {
      cv::Mat img = synthetic::room::render(cam, rig.T_c_b[c] * T_b_w);
      cv::GaussianBlur(img, img, cv::Size(0, 0), 1.0);
      images.push_back(std::make_shared<sdv::ImagePyramid>(img, 1));
    }
    Sophus::SE3d start = T_w_b;
    start.translation() *= wrongScale;
    const int id = opt.addFrame(images, start.inverse());
    for (int c = 0; c < 2; ++c)
      for (int v = 16; v < kH - 16; v += 16)
        for (int u = 16; u < kW - 16; u += 16) {
          Eigen::Vector3d b;
          if (!cam.unproject(Eigen::Vector2d(u, v), b)) continue;
          const double rho = synthetic::room::trueRho(rig.T_c_b[c] * T_b_w, b);
          if (const int pid = opt.addPoint(id, Eigen::Vector2d(u, v), rho / wrongScale, c); pid >= 0) truth[pid] = rho;
        }
  }
  opt.optimize(40);
  std::vector<double> ratios;
  for (const auto& p : opt.points()) ratios.push_back(p.rho / truth.at(p.id));
  std::nth_element(ratios.begin(), ratios.begin() + ratios.size() / 2, ratios.end());
  EXPECT_NEAR(ratios[ratios.size() / 2], 1.0, 0.01);
  for (size_t k = 1; k < trueBody.size(); ++k) {
    const Sophus::SE3d& T = opt.frames()[k].params.T_b_w;
    EXPECT_LT((T.inverse().translation() - trueBody[k].inverse().translation()).norm(), 0.02) << k;
  }
}
