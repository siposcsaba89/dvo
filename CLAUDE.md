# visual_reconstruction

Clean-room implementation of Direct Sparse Odometry (DSO) for commercial use. Target: fast pose + point
initialisation for Gaussian Splatting / neural surface reconstruction on driving data (replacing COLMAP).
Work proceeds step by step along [docs/ROADMAP.md](docs/ROADMAP.md).

## Clean-room rules (strict)
- Never read, consult, paraphrase or take "ideas" from the original DSO source or its GPL forks.
- Sources: papers, Engel's thesis, textbooks, permissively licensed libraries. See [docs/CLEANROOM.md](docs/CLEANROOM.md).
- Constants taken from a paper are cited briefly (e.g. `// DSO §2.3`); otherwise they are our own tuning.

## Code conventions
- C++23, MSVC (VS 2026). Namespace `sdv`, sources in `src/sdv/`, apps in `apps/`, tests in `tests/`.
- Minimal comments: only where the *why* is non-obvious (conventions, derivations, paper references). No
  comments restating code, no section banners, no doc comments on self-explanatory functions.
- Always `#include <...>`, also for project headers (`<sdv/...>`).
- Private/protected data members use an `m_` prefix in camelCase (`m_camera`), never a trailing underscore.
- List headers next to sources in CMake targets.
- Logging: spdlog only. Command-line parsing: Boost.Program_options. PLY output: tinyply.
- Camera model: EUCM (`sdv::Camera`), pinhole is alpha=0, beta=1. Never assume pinhole in algorithms:
  points are bearing vectors + inverse distance, epipolar search walks along the ray.
- Poses: `T_a_b` maps coordinates from frame b to frame a. Optimisation variables are world-to-camera
  with left perturbation `T <- exp(xi) * T`, xi = (translation, rotation) as in Sophus.
- Every new Jacobian gets a numeric-derivative unit test.

## Build & test
- vcpkg (classic mode) at `E:/vcpkg`, triplet x64-windows; toolchain set in CMakePresets.json.
- `cmake --preset vs2026`, `cmake --build --preset release`, `ctest --preset release`.
- Data: KITTI at `E:/records/kitti/sequences/00`, ground truth `E:/records/kitti/poses/00.txt`.
