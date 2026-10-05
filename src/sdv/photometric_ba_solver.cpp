#include <sdv/photometric_ba_solver.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <thread>

#include <Eigen/OrderingMethods>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseCore>

namespace sdv::pba {

namespace {

using Clock = std::chrono::steady_clock;
double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

template <typename F>
void parallelFor(size_t n, int threads, size_t chunk, F f) {
  std::atomic<size_t> next{0};
  auto work = [&](int t) {
    for (size_t b; (b = next.fetch_add(chunk)) < n;) f(b, std::min(n, b + chunk), t);
  };
  std::vector<std::thread> pool;
  for (int t = 1; t < threads; ++t) pool.emplace_back(work, t);
  work(0);
  for (auto& th : pool) th.join();
}

// Ceres' Levenberg-Marquardt defaults: diagonal of J^T J clamped to these, trust region radius 1e4 at the start.
double lmDiagonal(double d) { return std::clamp(d, 1e-6, 1e32); }

// Compact Jacobian of one pattern residual: columns host pose (6), target pose (6), host affine (2), target affine
// (2); poses in the tangent space of SE3TangentManifold.
using CompactJacobian = Eigen::Matrix<double, kPatternSize, 16>;
constexpr std::array<int, 4> kGroupColumn = {0, 6, 12, 14}, kGroupDim = {6, 6, 2, 2};

void evaluateObservation(const SolverProblem& pr, const Observation& o, double* r, CompactJacobian* J,
                         Eigen::Matrix<double, kPatternSize, 1>* jr) {
  const PointData& p = (*pr.points)[o.point];
  const Rig& rig = *pr.rig;
  const int nc = rig.size();
  const Interpolator& image = *(*pr.images)[static_cast<size_t>(o.target) * nc + o.cam];
  const double* rho = &(*pr.rho)[o.point];
  const double* ah = (*pr.affine)[static_cast<size_t>(p.host) * nc + p.hostCam].data();
  const double* at = (*pr.affine)[static_cast<size_t>(o.target) * nc + o.cam].data();
  double dRho[kPatternSize], dAh[2 * kPatternSize], dAt[2 * kPatternSize];
  if (o.target == p.host) {
    const StaticCost cost(&p, &rig.cameras[o.cam], &image, rig.T_c_b[o.cam] * rig.T_c_b[p.hostCam].inverse());
    const double* params[] = {rho, ah, at};
    double* jacobians[] = {dRho, dAh, dAt};
    cost.Evaluate(params, r, J ? jacobians : nullptr);
    if (J) J->leftCols<12>().setZero();
  } else {
    const TemporalCost cost(&p, &rig.cameras[o.cam], &image, rig.T_c_b[o.cam], rig.T_c_b[p.hostCam].inverse());
    const double* params[] = {(*pr.poses)[p.host].data(), (*pr.poses)[o.target].data(), rho, ah, at};
    double dH[7 * kPatternSize], dT[7 * kPatternSize];
    double* jacobians[] = {dH, dT, dRho, dAh, dAt};
    cost.Evaluate(params, r, J ? jacobians : nullptr);
    if (J)
      for (int k = 0; k < kPatternSize; ++k)
        for (int i = 0; i < 6; ++i) (*J)(k, i) = dH[7 * k + i], (*J)(k, 6 + i) = dT[7 * k + i];
  }
  if (!J) return;
  for (int k = 0; k < kPatternSize; ++k) {
    (*jr)(k) = dRho[k];
    (*J)(k, 12) = dAh[2 * k], (*J)(k, 13) = dAh[2 * k + 1];
    (*J)(k, 14) = dAt[2 * k], (*J)(k, 15) = dAt[2 * k + 1];
  }
}

// ceres::HuberLoss: rho(s) = s inside, 2 a sqrt(s) - a^2 outside; its first derivative weights the normal equations
// (Ceres' corrector does the same where rho'' <= 0).
double robust(double s, double a) { return a <= 0 || s <= a * a ? s : 2 * a * std::sqrt(s) - a * a; }
double robustWeight(double s, double a) { return a <= 0 || s <= a * a ? 1.0 : a / std::sqrt(s); }

// Variables (slots): free poses (6) and free affine brightness (2), grouped per keyframe into one block row of the
// reduced system. Per point the slots its observations touch, in keyframe order.
struct Structure {
  int nk = 0, nc = 0, n = 0;
  std::vector<int> poseSlot, affineSlot;
  std::vector<int> slotKf, slotOffset, slotDim;
  std::vector<int> kfDim, kfStart;
  std::vector<Observation> obs;  // sorted by point
  std::vector<int> points;       // with observations
  std::vector<size_t> obsBegin;  // per listed point
  std::vector<size_t> localBegin;
  std::vector<int> localSlot, localOffset;  // per point slot list, offset in the point's local vector
  std::vector<int> localDim;                // per listed point
  std::vector<std::array<int, 4>> obsLocal;  // per observation: local offsets of its slot groups (-1: fixed)
  std::vector<size_t> hpcBegin;
  // Upper block-sparse pattern of keyframe pairs, dense kfDim x kfDim blocks (diagonal blocks full).
  std::vector<size_t> rowBegin;
  std::vector<int> colKf;
  std::vector<size_t> blockOffset;
  size_t values = 0;

  size_t block(int i, int j) const {
    const auto b = colKf.begin() + static_cast<std::ptrdiff_t>(rowBegin[i]);
    const auto e = colKf.begin() + static_cast<std::ptrdiff_t>(rowBegin[i + 1]);
    return blockOffset[static_cast<size_t>(std::lower_bound(b, e, j) - colKf.begin())];
  }
};

Structure buildStructure(const SolverProblem& pr, int threads) {
  Structure s;
  s.nc = pr.rig->size();
  s.nk = static_cast<int>(pr.poses->size());
  const int nk = s.nk, nc = s.nc;
  s.obs = pr.observations;
  std::stable_sort(s.obs.begin(), s.obs.end(), [](const Observation& a, const Observation& b) { return a.point < b.point; });
  auto used = [&](std::vector<char>& pose, std::vector<char>& aff) {
    for (const auto& o : s.obs) {
      const PointData& p = (*pr.points)[o.point];
      if (o.target != p.host) pose[p.host] = pose[o.target] = 1;
      aff[static_cast<size_t>(p.host) * nc + p.hostCam] = aff[static_cast<size_t>(o.target) * nc + o.cam] = 1;
    }
    for (const auto& e : pr.odometry) pose[e.a] = pose[e.a + 1] = 1;
  };
  std::vector<char> poseUsed(nk, 0), affUsed(static_cast<size_t>(nk) * nc, 0);
  used(poseUsed, affUsed);
  s.poseSlot.assign(nk, -1);
  s.affineSlot.assign(static_cast<size_t>(nk) * nc, -1);
  s.kfDim.assign(nk, 0);
  for (int k = 0; k < nk; ++k) {
    auto add = [&](int dim) {
      s.slotKf.push_back(k), s.slotOffset.push_back(s.kfDim[k]), s.slotDim.push_back(dim);
      s.kfDim[k] += dim;
      return static_cast<int>(s.slotKf.size()) - 1;
    };
    if (pr.poseFree[k] && poseUsed[k]) s.poseSlot[k] = add(6);
    for (int c = 0; c < nc; ++c) {
      const size_t i = static_cast<size_t>(k) * nc + c;
      if (pr.affineFree[i] && affUsed[i]) s.affineSlot[i] = add(2);
    }
  }
  s.kfStart.assign(nk, -1);
  for (int k = 0; k < nk; ++k)
    if (s.kfDim[k] > 0) s.kfStart[k] = s.n, s.n += s.kfDim[k];

  auto groups = [&](const Observation& o) {
    const PointData& p = (*pr.points)[o.point];
    const bool temporal = o.target != p.host;
    return std::array<int, 4>{temporal ? s.poseSlot[p.host] : -1, temporal ? s.poseSlot[o.target] : -1,
                              s.affineSlot[static_cast<size_t>(p.host) * nc + p.hostCam],
                              s.affineSlot[static_cast<size_t>(o.target) * nc + o.cam]};
  };
  s.obsLocal.resize(s.obs.size());
  for (size_t i = 0; i < s.obs.size();) {
    const int point = s.obs[i].point;
    size_t j = i;
    while (j < s.obs.size() && s.obs[j].point == point) ++j;
    s.points.push_back(point);
    s.obsBegin.push_back(i);
    s.localBegin.push_back(s.localSlot.size());
    std::vector<int> slots;
    for (size_t q = i; q < j; ++q)
      for (int sl : groups(s.obs[q]))
        if (sl >= 0) slots.push_back(sl);
    std::sort(slots.begin(), slots.end());  // slot ids grow with the keyframe
    slots.erase(std::unique(slots.begin(), slots.end()), slots.end());
    int dim = 0;
    for (int sl : slots) s.localSlot.push_back(sl), s.localOffset.push_back(dim), dim += s.slotDim[sl];
    s.localDim.push_back(dim);
    for (size_t q = i; q < j; ++q) {
      const auto g = groups(s.obs[q]);
      for (int t = 0; t < 4; ++t) {
        s.obsLocal[q][t] = -1;
        if (g[t] < 0) continue;
        const auto it = std::lower_bound(slots.begin(), slots.end(), g[t]);
        s.obsLocal[q][t] = s.localOffset[s.localBegin.back() + static_cast<size_t>(it - slots.begin())];
      }
    }
    i = j;
  }
  s.obsBegin.push_back(s.obs.size());
  s.localBegin.push_back(s.localSlot.size());
  s.hpcBegin.assign(s.points.size() + 1, 0);
  for (size_t q = 0; q < s.points.size(); ++q)
    s.hpcBegin[q + 1] = s.hpcBegin[q] + (pr.rhoFree[s.points[q]] ? static_cast<size_t>(s.localDim[q]) : 0);

  // Keyframe pairs coupled by a point (all pairs of its slots when its inverse depth is eliminated, else per
  // observation) or by an odometry edge: a bit matrix, filled in parallel.
  const size_t words = (static_cast<size_t>(nk) * nk + 63) / 64;
  std::vector<std::atomic<std::uint64_t>> bits(words);
  auto mark = [&](int a, int b) {
    if (a > b) std::swap(a, b);
    const size_t bit = static_cast<size_t>(a) * nk + b;
    bits[bit / 64].fetch_or(std::uint64_t{1} << (bit % 64), std::memory_order_relaxed);
  };
  parallelFor(s.points.size(), threads, 256, [&](size_t b, size_t e, int) {
    std::vector<int> kfs;
    for (size_t q = b; q < e; ++q) {
      if (pr.rhoFree[s.points[q]]) {
        kfs.clear();
        for (size_t l = s.localBegin[q]; l < s.localBegin[q + 1]; ++l) kfs.push_back(s.slotKf[s.localSlot[l]]);
        kfs.erase(std::unique(kfs.begin(), kfs.end()), kfs.end());
        for (size_t x = 0; x < kfs.size(); ++x)
          for (size_t y = x; y < kfs.size(); ++y) mark(kfs[x], kfs[y]);
      } else {
        for (size_t o = s.obsBegin[q]; o < s.obsBegin[q + 1]; ++o) {
          const auto g = groups(s.obs[o]);
          for (int x = 0; x < 4; ++x)
            for (int y = x; y < 4; ++y)
              if (g[x] >= 0 && g[y] >= 0) mark(s.slotKf[g[x]], s.slotKf[g[y]]);
        }
      }
    }
  });
  for (const auto& e : pr.odometry)
    for (int a : {e.a, e.a + 1})
      for (int b : {e.a, e.a + 1})
        if (s.poseSlot[a] >= 0 && s.poseSlot[b] >= 0) mark(a, b);
  s.rowBegin.assign(nk + 1, 0);
  for (int i = 0; i < nk; ++i) {
    s.rowBegin[i] = s.colKf.size();
    if (s.kfDim[i] == 0) continue;
    for (int j = i; j < nk; ++j) {
      const size_t bit = static_cast<size_t>(i) * nk + j;
      if (s.kfDim[j] > 0 && (bits[bit / 64].load(std::memory_order_relaxed) >> (bit % 64) & 1)) {
        s.colKf.push_back(j);
        s.blockOffset.push_back(s.values);
        s.values += static_cast<size_t>(s.kfDim[i]) * s.kfDim[j];
      }
    }
  }
  s.rowBegin[nk] = s.colKf.size();
  return s;
}

struct Accumulator {
  std::vector<double> H, b, diag;  // reduced system (Schur complement), undamped diagonal and gradient of J^T J
  std::vector<double> gradient;
  double cost = 0;
  void reset(const Structure& s) {
    H.assign(s.values, 0.0), b.assign(s.n, 0.0), diag.assign(s.n, 0.0), gradient.assign(s.n, 0.0);
    cost = 0;
  }
};

class Solver {
 public:
  Solver(const SolverProblem& problem, const SolverOptions& options) : m_pr(problem), m_options(options) {
    m_threads = options.threads > 0 ? options.threads : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
  }

  SolverSummary run() {
    const auto t0 = Clock::now();
    m_s = buildStructure(m_pr, m_threads);
    m_summary.structure = since(t0);
    m_summary.reducedSize = m_s.n;
    buildSparsePattern();
    m_hpc.assign(m_s.hpcBegin.back(), 0.0);
    m_hpp.assign(m_s.points.size(), 0.0), m_hd.assign(m_s.points.size(), 0.0), m_bp.assign(m_s.points.size(), 0.0);
    m_dp.assign(m_s.points.size(), 0.0);

    double cost = evaluate();
    m_summary.initialCost = cost;
    double radius = 1e4, decrease = 2.0;
    for (int it = 0; it < m_options.iterations; ++it) {
      ++m_summary.iterations;
      const double lambda = 1.0 / radius;
      linearize(lambda);
      Eigen::VectorXd dc;
      if (!solveReduced(lambda, dc)) {
        radius /= decrease, decrease *= 2;
        continue;
      }
      double bDelta = 0, damp = 0;
      backSubstitute(dc, bDelta, damp);
      const double model = -0.5 * bDelta + 0.5 * lambda * damp;
      if (!(model > 0)) break;  // no decrease predicted: converged
      saveState();
      apply(dc);
      const double next = evaluate();
      const double ratio = (cost - next) / model;
      if (ratio > 1e-3) {
        ++m_summary.accepted;
        const bool small = cost - next <= m_options.functionTolerance * cost;
        cost = next;
        radius = std::min(1e16, radius / std::max(1.0 / 3.0, 1.0 - std::pow(2.0 * ratio - 1.0, 3)));
        decrease = 2.0;
        if (small) break;
      } else {
        restoreState();
        radius /= decrease, decrease *= 2;
        if (radius < 1e-32) break;
      }
    }
    m_summary.finalCost = cost;
    m_summary.total = since(t0);
    return m_summary;
  }

 private:
  void buildSparsePattern() {
    const Structure& s = m_s;
    std::vector<Eigen::Triplet<double, int>> triplets;
    triplets.reserve(s.values);
    for (int i = 0; i < s.nk; ++i)
      for (size_t q = s.rowBegin[i]; q < s.rowBegin[i + 1]; ++q) {
        const int j = s.colKf[q];
        for (int r = 0; r < s.kfDim[i]; ++r)
          for (int c = 0; c < s.kfDim[j]; ++c)
            if (i != j || r <= c) triplets.emplace_back(s.kfStart[i] + r, s.kfStart[j] + c, 0.0);
      }
    m_S.resize(s.n, s.n);
    m_S.setFromTriplets(triplets.begin(), triplets.end());
    m_S.makeCompressed();
    triplets = {};
    m_map.assign(s.values, -1);
    for (int i = 0; i < s.nk; ++i)
      for (size_t q = s.rowBegin[i]; q < s.rowBegin[i + 1]; ++q) {
        const int j = s.colKf[q];
        for (int r = 0; r < s.kfDim[i]; ++r)
          for (int c = 0; c < s.kfDim[j]; ++c)
            if (i != j || r <= c)
              m_map[s.blockOffset[q] + static_cast<size_t>(r) * s.kfDim[j] + c] =
                  &m_S.coeffRef(s.kfStart[i] + r, s.kfStart[j] + c) - m_S.valuePtr();
      }
    m_diagIndex.resize(s.n);
    for (int i = 0; i < s.n; ++i) m_diagIndex[i] = &m_S.coeffRef(i, i) - m_S.valuePtr();
    m_ldlt.analyzePattern(m_S);
  }

  double evaluate() {
    const auto t0 = Clock::now();
    std::vector<double> partial(m_threads, 0.0);
    const double a = m_pr.huber;
    parallelFor(m_s.obs.size(), m_threads, 1024, [&](size_t b, size_t e, int t) {
      double sum = 0;
      for (size_t i = b; i < e; ++i) {
        double r[kPatternSize];
        evaluateObservation(m_pr, m_s.obs[i], r, nullptr, nullptr);
        double sq = 0;
        for (double v : r) sq += v * v;
        sum += 0.5 * robust(sq, a);
      }
      partial[t] += sum;
    });
    double cost = std::accumulate(partial.begin(), partial.end(), 0.0);
    for (const auto& e : m_pr.odometry) {
      const double* params[] = {(*m_pr.poses)[e.a].data(), (*m_pr.poses)[e.a + 1].data()};
      double r[6];
      e.cost->Evaluate(params, r, nullptr);
      for (double v : r) cost += 0.5 * v * v;
    }
    m_summary.evaluate += since(t0);
    return cost;
  }

  // Normal equations at the current state with the points eliminated; lambda damps the points' inverse depths.
  void linearize(double lambda) {
    const auto t0 = Clock::now();
    const Structure& s = m_s;
    m_acc.resize(m_threads);
    parallelFor(static_cast<size_t>(m_threads), m_threads, 1, [&](size_t t, size_t, int) { m_acc[t].reset(s); });
    const double a = m_pr.huber;
    parallelFor(s.points.size(), m_threads, 64, [&](size_t pb, size_t pe, int t) {
      Accumulator& acc = m_acc[t];
      Eigen::MatrixXd Hl;
      Eigen::VectorXd bl, hpc;
      CompactJacobian J;
      Eigen::Matrix<double, kPatternSize, 1> jr;
      std::vector<int> kfs, kfIndex;  // the point's keyframes, index of each local slot's keyframe in it
      std::vector<size_t> base;       // block offsets of its keyframe pairs, filled on first use
      for (size_t q = pb; q < pe; ++q) {
        const int D = s.localDim[q];
        const bool rhoFree = m_pr.rhoFree[s.points[q]];
        Hl.setZero(D, D), bl.setZero(D), hpc.setZero(D);
        double hpp = 0, bp = 0;
        for (size_t o = s.obsBegin[q]; o < s.obsBegin[q + 1]; ++o) {
          double r[kPatternSize];
          evaluateObservation(m_pr, s.obs[o], r, &J, &jr);
          const Eigen::Map<const Eigen::Matrix<double, kPatternSize, 1>> rv(r);
          const double sq = rv.squaredNorm();
          acc.cost += 0.5 * robust(sq, a);
          const double w = robustWeight(sq, a);
          const auto& lo = s.obsLocal[o];
          for (int g = 0; g < 4; ++g) {
            if (lo[g] < 0) continue;
            const auto Jg = J.middleCols(kGroupColumn[g], kGroupDim[g]);
            for (int h = 0; h < 4; ++h) {
              if (lo[h] < 0) continue;
              Hl.block(lo[g], lo[h], kGroupDim[g], kGroupDim[h]).noalias() +=
                  w * Jg.transpose() * J.middleCols(kGroupColumn[h], kGroupDim[h]);
            }
            bl.segment(lo[g], kGroupDim[g]).noalias() += w * Jg.transpose() * rv;
            if (rhoFree) hpc.segment(lo[g], kGroupDim[g]).noalias() += w * Jg.transpose() * jr;
          }
          if (rhoFree) hpp += w * jr.squaredNorm(), bp += w * jr.dot(rv);
        }
        // Undamped diagonal and gradient of the full system (LM scaling, predicted decrease).
        for (size_t l = s.localBegin[q]; l < s.localBegin[q + 1]; ++l) {
          const int sl = s.localSlot[l], off = s.localOffset[l];
          const int g0 = s.kfStart[s.slotKf[sl]] + s.slotOffset[sl];
          for (int d = 0; d < s.slotDim[sl]; ++d) acc.diag[g0 + d] += Hl(off + d, off + d), acc.gradient[g0 + d] += bl(off + d);
        }
        if (rhoFree) {
          const double hd = hpp + lambda * lmDiagonal(hpp);
          m_hpp[q] = hpp, m_hd[q] = hd, m_bp[q] = bp;
          std::copy_n(hpc.data(), D, m_hpc.data() + s.hpcBegin[q]);
          Hl.noalias() -= (hpc / hd) * hpc.transpose();
          bl.noalias() -= hpc * (bp / hd);
        }
        kfs.clear(), kfIndex.clear();
        for (size_t l = s.localBegin[q]; l < s.localBegin[q + 1]; ++l) {
          const int k = s.slotKf[s.localSlot[l]];
          if (kfs.empty() || kfs.back() != k) kfs.push_back(k);  // slots in keyframe order
          kfIndex.push_back(static_cast<int>(kfs.size()) - 1);
        }
        const size_t nkl = kfs.size();
        base.assign(nkl * nkl, SIZE_MAX);
        for (size_t la = s.localBegin[q]; la < s.localBegin[q + 1]; ++la) {
          const int sa = s.localSlot[la], ka = s.slotKf[sa];
          const int g0 = s.kfStart[ka] + s.slotOffset[sa];
          for (int d = 0; d < s.slotDim[sa]; ++d) acc.b[g0 + d] += bl(s.localOffset[la] + d);
          for (size_t lb = la; lb < s.localBegin[q + 1]; ++lb) {
            const int sb = s.localSlot[lb], kb = s.slotKf[sb];  // ka <= kb
            size_t& bo = base[kfIndex[la - s.localBegin[q]] * nkl + kfIndex[lb - s.localBegin[q]]];
            if (bo == SIZE_MAX) bo = s.block(ka, kb);
            const size_t offset = bo;
            const int ld = s.kfDim[kb];
            auto addBlock = [&](int ra, int rb, int la0, int lb0, int da, int db) {
              for (int x = 0; x < da; ++x)
                for (int y = 0; y < db; ++y)
                  acc.H[offset + static_cast<size_t>(ra + x) * ld + rb + y] += Hl(la0 + x, lb0 + y);
            };
            addBlock(s.slotOffset[sa], s.slotOffset[sb], s.localOffset[la], s.localOffset[lb], s.slotDim[sa],
                     s.slotDim[sb]);
            // Within one keyframe block both triangles are stored.
            if (ka == kb && sa != sb)
              addBlock(s.slotOffset[sb], s.slotOffset[sa], s.localOffset[lb], s.localOffset[la], s.slotDim[sb],
                       s.slotDim[sa]);
          }
        }
      }
    });
    Accumulator& acc = m_acc[0];
    for (const auto& e : m_pr.odometry) {
      const double* params[] = {(*m_pr.poses)[e.a].data(), (*m_pr.poses)[e.a + 1].data()};
      double r[6];
      Eigen::Matrix<double, 6, 7, Eigen::RowMajor> Ja, Jb;
      double* jacobians[] = {Ja.data(), Jb.data()};
      e.cost->Evaluate(params, r, jacobians);
      const Eigen::Map<const Eigen::Matrix<double, 6, 1>> rv(r);
      acc.cost += 0.5 * rv.squaredNorm();
      const int k[2] = {e.a, e.a + 1};
      const Eigen::Matrix<double, 6, 6> Jk[2] = {Ja.leftCols<6>(), Jb.leftCols<6>()};
      for (int x = 0; x < 2; ++x) {
        if (s.poseSlot[k[x]] < 0) continue;
        const int gx = s.kfStart[k[x]] + s.slotOffset[s.poseSlot[k[x]]];
        const Eigen::Matrix<double, 6, 1> g = Jk[x].transpose() * rv;
        for (int d = 0; d < 6; ++d) acc.b[gx + d] += g(d), acc.gradient[gx + d] += g(d);
        for (int y = 0; y < 2; ++y) {
          if (s.poseSlot[k[y]] < 0 || k[y] < k[x]) continue;
          const Eigen::Matrix<double, 6, 6> Hxy = Jk[x].transpose() * Jk[y];
          const size_t base = s.block(k[x], k[y]);
          const int ld = s.kfDim[k[y]], ox = s.slotOffset[s.poseSlot[k[x]]], oy = s.slotOffset[s.poseSlot[k[y]]];
          for (int i = 0; i < 6; ++i)
            for (int j = 0; j < 6; ++j) acc.H[base + static_cast<size_t>(ox + i) * ld + oy + j] += Hxy(i, j);
          if (x == y)
            for (int d = 0; d < 6; ++d) acc.diag[gx + d] += Hxy(d, d);
        }
      }
    }
    // Sum the per-thread copies into the first.
    parallelFor(s.values, m_threads, 1 << 16, [&](size_t b, size_t e, int) {
      for (int t = 1; t < m_threads; ++t)
        for (size_t i = b; i < e; ++i) acc.H[i] += m_acc[t].H[i];
    });
    for (int t = 1; t < m_threads; ++t) {
      for (int i = 0; i < s.n; ++i)
        acc.b[i] += m_acc[t].b[i], acc.diag[i] += m_acc[t].diag[i], acc.gradient[i] += m_acc[t].gradient[i];
      acc.cost += m_acc[t].cost;
    }
    m_summary.linearize += since(t0);
  }

  bool solveReduced(double lambda, Eigen::VectorXd& dc) {
    const auto t0 = Clock::now();
    const Accumulator& acc = m_acc[0];
    double* v = m_S.valuePtr();
    std::fill_n(v, m_S.nonZeros(), 0.0);
    for (size_t i = 0; i < m_map.size(); ++i)
      if (m_map[i] >= 0) v[m_map[i]] += acc.H[i];
    for (int i = 0; i < m_s.n; ++i) v[m_diagIndex[i]] += lambda * lmDiagonal(acc.diag[i]);
    m_ldlt.factorize(m_S);
    bool ok = m_ldlt.info() == Eigen::Success;
    if (ok) {
      dc = m_ldlt.solve(-Eigen::Map<const Eigen::VectorXd>(acc.b.data(), m_s.n));
      ok = m_ldlt.info() == Eigen::Success && dc.allFinite();
    }
    m_summary.factorize += since(t0);
    return ok;
  }

  // Point steps from the camera step; b.delta and delta^T D delta of the full damped system for the predicted
  // decrease.
  void backSubstitute(const Eigen::VectorXd& dc, double& bDelta, double& damp) {
    const Structure& s = m_s;
    const Accumulator& acc = m_acc[0];
    bDelta = 0, damp = 0;
    for (int i = 0; i < s.n; ++i) bDelta += acc.gradient[i] * dc[i], damp += lmDiagonal(acc.diag[i]) * dc[i] * dc[i];
    std::vector<double> pb(m_threads, 0.0), pd(m_threads, 0.0);
    parallelFor(s.points.size(), m_threads, 1024, [&](size_t b, size_t e, int t) {
      for (size_t q = b; q < e; ++q) {
        m_dp[q] = 0;
        if (!m_pr.rhoFree[s.points[q]]) continue;
        double x = m_bp[q];
        const double* hpc = m_hpc.data() + s.hpcBegin[q];
        for (size_t l = s.localBegin[q]; l < s.localBegin[q + 1]; ++l) {
          const int sl = s.localSlot[l];
          const int g0 = s.kfStart[s.slotKf[sl]] + s.slotOffset[sl];
          for (int d = 0; d < s.slotDim[sl]; ++d) x += hpc[s.localOffset[l] + d] * dc[g0 + d];
        }
        m_dp[q] = -x / m_hd[q];
        pb[t] += m_bp[q] * m_dp[q], pd[t] += lmDiagonal(m_hpp[q]) * m_dp[q] * m_dp[q];
      }
    });
    for (int t = 0; t < m_threads; ++t) bDelta += pb[t], damp += pd[t];
  }

  void saveState() {
    m_savedPoses = *m_pr.poses, m_savedAffine = *m_pr.affine;
    m_savedRho.resize(m_s.points.size());
    for (size_t q = 0; q < m_s.points.size(); ++q) m_savedRho[q] = (*m_pr.rho)[m_s.points[q]];
  }

  void restoreState() {
    *m_pr.poses = m_savedPoses, *m_pr.affine = m_savedAffine;
    for (size_t q = 0; q < m_s.points.size(); ++q) (*m_pr.rho)[m_s.points[q]] = m_savedRho[q];
  }

  void apply(const Eigen::VectorXd& dc) {
    const Structure& s = m_s;
    for (int k = 0; k < s.nk; ++k) {
      if (s.poseSlot[k] >= 0) {
        const int g0 = s.kfStart[k] + s.slotOffset[s.poseSlot[k]];
        Eigen::Map<Sophus::SE3d> T((*m_pr.poses)[k].data());
        T = T * Sophus::SE3d::exp(dc.segment<6>(g0));
      }
      for (int c = 0; c < s.nc; ++c) {
        const int sl = s.affineSlot[static_cast<size_t>(k) * s.nc + c];
        if (sl < 0) continue;
        const int g0 = s.kfStart[k] + s.slotOffset[sl];
        auto& a = (*m_pr.affine)[static_cast<size_t>(k) * s.nc + c];
        a[0] += dc[g0], a[1] += dc[g0 + 1];
      }
    }
    for (size_t q = 0; q < s.points.size(); ++q)
      if (m_pr.rhoFree[s.points[q]]) {
        double& r = (*m_pr.rho)[s.points[q]];
        r = std::max(m_pr.minRho, r + m_dp[q]);
      }
  }

  const SolverProblem& m_pr;
  SolverOptions m_options;
  int m_threads = 1;
  SolverSummary m_summary;
  Structure m_s;
  std::vector<Accumulator> m_acc;
  std::vector<double> m_hpc, m_hpp, m_hd, m_bp, m_dp;
  Eigen::SparseMatrix<double> m_S;
  std::vector<std::ptrdiff_t> m_map, m_diagIndex;
  Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>, Eigen::Upper, Eigen::AMDOrdering<int>> m_ldlt;
  std::vector<std::array<double, 7>> m_savedPoses;
  std::vector<std::array<double, 2>> m_savedAffine;
  std::vector<double> m_savedRho;
};

}  // namespace

SolverSummary solve(const SolverProblem& problem, const SolverOptions& options) {
  if (!problem.rig || !problem.points || !problem.images || !problem.poses || !problem.affine || !problem.rho)
    throw std::invalid_argument("photometric BA solver: incomplete problem");
  return Solver(problem, options).run();
}

}  // namespace sdv::pba
