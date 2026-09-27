#include <sdv/eval/trajectory.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <Eigen/SVD>

namespace sdv {

Sophus::SE3d SimilarityTransform::applyToPose(const Sophus::SE3d& T_w_c) const {
  Eigen::Quaterniond q(R * T_w_c.so3().matrix());
  q.normalize();
  return Sophus::SE3d(q, apply(T_w_c.translation()));
}

SimilarityTransform alignPoints(const std::vector<Eigen::Vector3d>& src,
                                const std::vector<Eigen::Vector3d>& dst, bool estimateScale) {
  if (src.size() != dst.size() || src.size() < 3)
    throw std::invalid_argument("alignPoints needs >= 3 corresponding points");
  const double n = static_cast<double>(src.size());

  Eigen::Vector3d muS = Eigen::Vector3d::Zero(), muD = Eigen::Vector3d::Zero();
  for (size_t i = 0; i < src.size(); ++i) {
    muS += src[i];
    muD += dst[i];
  }
  muS /= n;
  muD /= n;

  Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
  double varS = 0;
  for (size_t i = 0; i < src.size(); ++i) {
    const Eigen::Vector3d s = src[i] - muS;
    cov += (dst[i] - muD) * s.transpose();
    varS += s.squaredNorm();
  }
  cov /= n;
  varS /= n;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d S = Eigen::Matrix3d::Identity();
  if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2, 2) = -1;

  SimilarityTransform out;
  out.R = svd.matrixU() * S * svd.matrixV().transpose();
  out.scale = estimateScale ? (svd.singularValues().asDiagonal() * S).trace() / varS : 1.0;
  out.t = muD - out.scale * out.R * muS;
  return out;
}

AteResult absoluteTrajectoryError(const std::vector<Sophus::SE3d>& gt,
                                  const std::vector<Sophus::SE3d>& est, bool estimateScale) {
  const size_t n = std::min(gt.size(), est.size());
  std::vector<Eigen::Vector3d> src(n), dst(n);
  for (size_t i = 0; i < n; ++i) {
    src[i] = est[i].translation();
    dst[i] = gt[i].translation();
  }
  AteResult r;
  r.alignment = alignPoints(src, dst, estimateScale);

  std::vector<double> err(n);
  double sumSq = 0, sum = 0;
  for (size_t i = 0; i < n; ++i) {
    err[i] = (r.alignment.apply(src[i]) - dst[i]).norm();
    sumSq += err[i] * err[i];
    sum += err[i];
  }
  r.rmse = std::sqrt(sumSq / n);
  r.mean = sum / n;
  r.max = *std::max_element(err.begin(), err.end());
  std::nth_element(err.begin(), err.begin() + n / 2, err.end());
  r.median = err[n / 2];
  return r;
}

SegmentErrorResult segmentDriftError(const std::vector<Sophus::SE3d>& gt,
                                     const std::vector<Sophus::SE3d>& est) {
  const size_t n = std::min(gt.size(), est.size());
  std::vector<double> dist(n, 0.0);
  for (size_t i = 1; i < n; ++i)
    dist[i] = dist[i - 1] + (gt[i].translation() - gt[i - 1].translation()).norm();

  constexpr double kLengths[] = {100, 200, 300, 400, 500, 600, 700, 800};
  constexpr size_t kStep = 10;

  SegmentErrorResult r;
  double sumT = 0, sumR = 0;
  for (size_t first = 0; first < n; first += kStep) {
    for (double len : kLengths) {
      const auto it = std::lower_bound(dist.begin() + first, dist.end(), dist[first] + len);
      if (it == dist.end()) continue;
      const size_t last = static_cast<size_t>(it - dist.begin());
      const Sophus::SE3d relGt = gt[first].inverse() * gt[last];
      const Sophus::SE3d relEst = est[first].inverse() * est[last];
      const Sophus::SE3d delta = relEst.inverse() * relGt;
      sumT += delta.translation().norm() / len;
      sumR += delta.so3().logAndTheta().theta / len;
      ++r.numSegments;
    }
  }
  if (r.numSegments > 0) {
    r.translationPercent = 100.0 * sumT / r.numSegments;
    r.rotationDegPer100m = 100.0 * (180.0 / M_PI) * sumR / r.numSegments;
  }
  return r;
}


RelativePoseErrorResult relativePoseError(const std::vector<Sophus::SE3d>& gt, const std::vector<Sophus::SE3d>& est,
                                          size_t delta, double scale) {
  RelativePoseErrorResult r;
  const size_t n = std::min(gt.size(), est.size());
  double sumT = 0, sumR = 0;
  for (size_t i = 0; i + delta < n; ++i) {
    Sophus::SE3d relEst = est[i].inverse() * est[i + delta];
    relEst.translation() *= scale;
    const Sophus::SE3d err = (gt[i].inverse() * gt[i + delta]).inverse() * relEst;
    sumT += err.translation().squaredNorm();
    const double deg = err.so3().logAndTheta().theta * 180.0 / M_PI;
    sumR += deg * deg;
    ++r.numPairs;
  }
  if (r.numPairs > 0) {
    r.translationRmse = std::sqrt(sumT / r.numPairs);
    r.rotationRmseDeg = std::sqrt(sumR / r.numPairs);
  }
  return r;
}

}  // namespace sdv
