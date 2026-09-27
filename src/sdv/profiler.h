#pragma once

#include <chrono>
#include <map>
#include <string>

namespace sdv {

// Accumulated wall time per named stage.
class StageProfile {
 public:
  struct Entry {
    double ms = 0;
    int calls = 0;
  };

  class Scope {
   public:
    Scope(StageProfile& profile, const char* stage)
        : m_profile(profile), m_stage(stage), m_start(std::chrono::steady_clock::now()) {}
    ~Scope() {
      m_profile.add(m_stage, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_start).count());
    }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;

   private:
    StageProfile& m_profile;
    const char* m_stage;
    std::chrono::steady_clock::time_point m_start;
  };

  Scope scope(const char* stage) { return Scope(*this, stage); }
  void add(const std::string& stage, double ms) {
    Entry& e = m_entries[stage];
    e.ms += ms;
    ++e.calls;
  }
  const std::map<std::string, Entry>& entries() const { return m_entries; }

 private:
  std::map<std::string, Entry> m_entries;
};

}  // namespace sdv
