#pragma once

#include <algorithm>
#include <cstddef>
#include <execution>
#include <numeric>
#include <vector>

namespace sdv {

// A fixed chunk count, independent of the core count, keeps reductions over chunks deterministic.
inline constexpr size_t kParallelChunks = 64;

// Calls body(chunk, begin, end) for kParallelChunks contiguous ranges covering [0, n), in parallel.
template <typename Body>
void parallelChunks(size_t n, Body&& body) {
  std::vector<size_t> chunks(kParallelChunks);
  std::iota(chunks.begin(), chunks.end(), size_t{0});
  auto run = [&](size_t c) { body(c, n * c / kParallelChunks, n * (c + 1) / kParallelChunks); };
  if (n < 2 * kParallelChunks)
    std::for_each(chunks.begin(), chunks.end(), run);
  else
    std::for_each(std::execution::par, chunks.begin(), chunks.end(), run);
}

}  // namespace sdv
