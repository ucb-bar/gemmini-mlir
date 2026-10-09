#ifndef MERLIN_GEMMINI_PRIMITIVE_TELEMETRY_H
#define MERLIN_GEMMINI_PRIMITIVE_TELEMETRY_H

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>

// Observational host-side instrumentation. No functional model state is written.
// A complete process exit and a numeric gate are external receipt obligations.
class gemmini_primitive_telemetry {
  using key_t = std::tuple<uint64_t, unsigned, unsigned, unsigned, unsigned,
                          unsigned, unsigned, unsigned, unsigned, unsigned,
                          uint64_t, unsigned, unsigned, unsigned, unsigned, uint64_t, uint64_t>;
  struct row_t {
    uint64_t count = 0, load_bytes = 0, store_bytes = 0;
    uint64_t unknown_dma = 0, padded_mac_slots = 0, padded_compute_rows = 0;
    uint64_t first_event = 0, last_event = 0;
  };
  const char* path = std::getenv("MERLIN_GEMMINI_TELEMETRY");
  bool aggregate_pc = std::getenv("MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC") &&
                      std::string(std::getenv("MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC")) == "1";
  std::map<key_t, row_t> rows;
  bool ex_valid = false, strides_valid = false, store_valid = false;
  bool load_valid[3] = {false, false, false};
  struct scope_t { uint64_t id, begin, end, entry, invocation = 0; };
  struct entry_t { uint64_t event, scope, invocation, pc; };
  std::vector<scope_t> scopes;
  std::vector<entry_t> entries;
  uint64_t event = 0;
  gemmini_primitive_telemetry() {
    if (!path || !*path) return;
    const char* spec = std::getenv("MERLIN_GEMMINI_TELEMETRY_SCOPES");
    if (!spec || !*spec) return;
    std::ifstream input(spec);
    if (!input) throw std::runtime_error("Cannot open Gemmini telemetry scopes");
    auto integer = [](const std::string& value) {
      size_t used = 0;
      uint64_t result = std::stoull(value, &used, 0);
      if (used != value.size() || value.empty() || value.front() == '-')
        throw std::runtime_error("Invalid Gemmini telemetry scope integer");
      return result;
    };
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty()) continue;
      std::istringstream fields(line);
      std::string id, begin, end, entry, extra;
      if (!(fields >> id >> begin >> end >> entry) || (fields >> extra))
        throw std::runtime_error("Truncated Gemmini telemetry scopes");
      scope_t row{integer(id), integer(begin), integer(end), integer(entry)};
      if (!row.id || row.begin >= row.end || row.entry < row.begin || row.entry >= row.end ||
          row.begin % 2 || row.end % 2 || row.entry % 2)
        throw std::runtime_error("Invalid Gemmini telemetry scope");
      for (const auto& prior : scopes)
        if (row.id == prior.id || (row.begin < prior.end && prior.begin < row.end))
          throw std::runtime_error("Overlapping Gemmini telemetry scopes");
      scopes.push_back(row);
    }
  }
public:
  static gemmini_primitive_telemetry& instance() {
    static gemmini_primitive_telemetry value;
    return value;
  }
  template <class State>
  void record(uint64_t pc, unsigned funct, uint64_t rs1, uint64_t rs2,
              const State& state) {
    if (!path || !*path) return;
    ++event;
    uint64_t scope_id = 0, invocation = 0;
    for (auto& scope : scopes) {
      if (pc < scope.begin || pc >= scope.end) continue;
      scope_id = scope.id;
      if (pc == scope.entry) {
        ++scope.invocation;
        entries.push_back(entry_t{event, scope.id, scope.invocation, pc});
      }
      invocation = scope.invocation;
      break;
    }
    if (funct == 0) {
      unsigned kind = rs1 & 3;
      if (kind == 0) { strides_valid = true; if (!(rs1 & 128)) ex_valid = true; }
      if (kind == 1 && ((rs1 >> 3) & 3) < 3) load_valid[(rs1 >> 3) & 3] = true;
      if (kind == 2) store_valid = true;
    }
    unsigned ar = 0, ac = 0, br = 0, bc = 0, load_id = 3;
    if (funct == 1 || funct == 2 || funct == 14 || funct == 3) {
      ar = (rs2 >> 48) & 0xffff; ac = (rs2 >> 32) & 0xffff;
    } else if (funct == 4 || funct == 5 || funct == 6) {
      ar = (rs1 >> 48) & 0xffff; ac = (rs1 >> 32) & 0xffff;
      br = (rs2 >> 48) & 0xffff; bc = (rs2 >> 32) & 0xffff;
    }
    if (funct == 2) load_id = 0;
    if (funct == 1) load_id = 1;
    if (funct == 14) load_id = 2;
    bool dma_valid = load_id < 3 ? load_valid[load_id] : store_valid;
    uint64_t stride = dma_valid ? (load_id < 3 ? state.load_strides[load_id] : state.store_stride)
                                : std::numeric_limits<uint64_t>::max();
    unsigned mode = ex_valid ? unsigned(state.mode) : 0xffffffff;
    unsigned a_stride = strides_valid ? unsigned(state.a_stride) : 0xffffffff;
    unsigned c_stride = strides_valid ? unsigned(state.c_stride) : 0xffffffff;
    unsigned pool_stride = store_valid ? unsigned(state.pool_stride) : 0xffffffff;
    unsigned pool_rows = store_valid ? unsigned(state.pool_porows) : 0xffffffff;
    unsigned pool_cols = store_valid ? unsigned(state.pool_pocols) : 0xffffffff;
    auto key = std::make_tuple(aggregate_pc ? uint64_t(0) : pc, funct, ar, ac, br, bc,
        mode, a_stride, c_stride, load_id, stride, pool_stride, pool_rows,
        pool_cols, unsigned((rs2 >> 31) & 1), scope_id, invocation);
    auto& row = rows[key];
    if (!row.count) row.first_event = event;
    row.last_event = event; ++row.count;
    if ((load_id < 3 || funct == 3) && !dma_valid) {
      ++row.unknown_dma;
    } else if (load_id < 3) {
      // Zero-fill commands request no external bytes. Broadcast pixels do not
      // multiply instruction-requested bytes; this is not MMU/DRAM traffic.
      unsigned width = ((rs2 >> 31) & 1) && !state.load_shrunks[load_id]
                           ? sizeof(acc_t) : sizeof(elem_t);
      if (rs1 != 0) row.load_bytes += uint64_t(ar) * ac * width;
    } else if (funct == 3) {
      unsigned norm = (rs2 >> 26) & 7;
      if (state.pool_stride) {
        row.store_bytes += uint64_t(state.pool_porows) * state.pool_pocols * ac * sizeof(elem_t);
      } else if (norm != 0) {
        // Nonterminating normalization may suppress writes; refuse a guess.
        ++row.unknown_dma;
      } else {
        unsigned width = ((rs2 >> 31) & 1) && ((rs2 >> 29) & 1)
                             ? sizeof(acc_t) : sizeof(elem_t);
        row.store_bytes += uint64_t(ar) * ac * width;
      }
    }
    if (funct == 4 || funct == 5) {
      // Nominal padded work only. It is neither nonzero MACs nor a cycle floor.
      row.padded_mac_slots += uint64_t(DIM) * DIM * DIM;
      row.padded_compute_rows += DIM;
    }
  }
  ~gemmini_primitive_telemetry() {
    if (!path || !*path) return;
    std::ofstream file(path);
    if (!file) { std::fprintf(stderr, "Gemmini telemetry output open failed\n"); return; }
    file << "{\"schema\":\"gemmini_primitive_operand_telemetry_v1\",\"dim\":" << DIM
         << ",\"elem_bytes\":" << sizeof(elem_t) << ",\"acc_bytes\":" << sizeof(acc_t)
         << ",\"pc_aggregation\":\"" << (aggregate_pc ? "scope_geometry" : "exact_pc_geometry") << "\""
         << ",\"total_commands\":" << event << ",\"rows\":[";
    bool first = true;
    for (const auto& item : rows) {
      if (!first) file << ','; first = false;
      const auto& key = item.first; const auto& row = item.second;
      file << "{\"pc\":" << std::get<0>(key) << ",\"funct\":" << std::get<1>(key)
           << ",\"a_rows\":" << std::get<2>(key) << ",\"a_cols\":" << std::get<3>(key)
           << ",\"b_rows\":" << std::get<4>(key) << ",\"b_cols\":" << std::get<5>(key)
           << ",\"dataflow\":" << std::get<6>(key) << ",\"a_stride\":" << std::get<7>(key)
           << ",\"c_stride\":" << std::get<8>(key) << ",\"load_state\":" << std::get<9>(key)
           << ",\"dram_row_stride\":" << std::get<10>(key)
           << ",\"pool_stride\":" << std::get<11>(key) << ",\"pool_rows\":" << std::get<12>(key)
           << ",\"pool_cols\":" << std::get<13>(key) << ",\"accumulator\":" << std::get<14>(key)
           << ",\"scope_id\":" << std::get<15>(key) << ",\"invocation\":" << std::get<16>(key)
           << ",\"first_event\":" << row.first_event << ",\"last_event\":" << row.last_event
           << ",\"commands\":" << row.count << ",\"requested_load_bytes\":" << row.load_bytes
           << ",\"requested_store_bytes\":" << row.store_bytes << ",\"unknown_dma_commands\":" << row.unknown_dma
           << ",\"padded_mac_slots\":" << row.padded_mac_slots
           << ",\"padded_compute_rows\":" << row.padded_compute_rows << '}';
    }
    file << "],\"entries\":[";
    first = true;
    for (const auto& row : entries) {
      if (!first) file << ','; first = false;
      file << "{\"event\":" << row.event << ",\"scope_id\":" << row.scope
           << ",\"invocation\":" << row.invocation << ",\"pc\":" << row.pc << '}';
    }
    file << "]}\n";
  }
};
#endif
