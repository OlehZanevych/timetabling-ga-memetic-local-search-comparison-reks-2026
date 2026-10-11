// A plain-text instance format.
//
// Deliberately line-oriented and human-readable: an instance can be inspected, diffed and
// version-controlled, and a reader for it can be written in an afternoon in any language. In this
// application the experiments generate their instances in memory, deterministically from (size,
// seed); `experiments instances` writes every instance they use in this format to data/, and
// scripts/check_data.py confirms that the shipped files are exactly what the generator produces.
#pragma once

#include <string>
#include <vector>

#include "model.hpp"

namespace tt {

/// Write an instance, and optionally a schedule alongside it, to `path`.
bool writeInstance(const std::string& path, const Instance& in);
bool readInstance(const std::string& path, Instance& out);

/// A schedule as `n` lines of "day time parity room" (-1 -1 W -1 for unplaced).
bool writeSchedule(const std::string& path, const std::vector<Placement>& sigma);
bool readSchedule(const std::string& path, std::vector<Placement>& out);

}  // namespace tt
