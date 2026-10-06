"""Independent observer semantics and refusal checks without changing Spike."""

import json
import os
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def observer(tmp_path_factory):
    root = tmp_path_factory.mktemp("operand-observer")
    header = (
        Path(__file__).resolve().parents[1]
        / "support/gemmini_spike_telemetry/telemetry.h"
    )
    source = root / "fixture.cc"
    source.write_text(
        '''#include <cstdint>
#include <string>
#define DIM 16
using elem_t = int8_t;
using acc_t = int32_t;
#include "'''
        + str(header)
        + """"
struct State {
 unsigned mode=1,a_stride=2,c_stride=3,pool_stride=0,pool_porows=0,pool_pocols=0;
 uint64_t load_strides[3]={65,65,65},store_stride=292;
 bool load_shrunks[3]={false,false,false};
};
uint64_t payload(unsigned r,unsigned c,uint32_t addr=0) {
 return (uint64_t(r)<<48) | (uint64_t(c)<<32) | addr;
}
int main(int argc,char**argv) {
 try {
  State s;
  auto& t=gemmini_primitive_telemetry::instance();
  if(argc>1 && std::string(argv[1])=="unconfigured") {
   t.record(0x108,2,0x100000,payload(3,65),s); return 0;
  }
  t.record(0x100,0,4,0,s); // Execution config, exact scope entry marker.
  t.record(0x104,0,1,65,s); // Load-state0 config.
  t.record(0x108,2,0x100000,payload(3,65),s); //195 requested bytes.
  t.record(0x108,2,0,payload(5,65),s); //Zero-fill requests no external bytes.
  s.load_shrunks[0]=true;
  t.record(0x10c,2,0x100000,payload(2,17,0x80000000),s); //34 bytes.
  s.load_shrunks[0]=false;
  t.record(0x10c,2,0x100000,payload(2,17,0x80000000),s); //136 bytes.
  t.record(0x110,0,2,292,s); //Store config.
  t.record(0x114,3,0x200000,payload(3,17,0xa0000000),s); //204 raw-acc bytes.
  s.pool_stride=2;s.pool_porows=3;s.pool_pocols=2;
  t.record(0x114,3,0x200000,payload(5,17,0x80000000),s); //102 pooled bytes.
  s.pool_stride=0;
  t.record(0x114,3,0x200000,payload(3,17,0x84000000),s); //Unknown normalization write.
  t.record(0x118,4,payload(2,16),payload(16,16,0xffffffff),s);
  t.record(0x100,0,4,0,s); //Actual second scope invocation, no presumed division.
  t.record(0x108,2,0x100000,payload(1,1),s);
 } catch (const std::exception&) { return 2; }
 return 0;
}
"""
    )
    exe = root / "fixture"
    subprocess.run(
        ["g++", "-std=c++17", "-O2", str(source), "-o", str(exe)],
        check=True,
        capture_output=True,
    )
    return exe


def run(observer, tmp_path, *, scopes=None, enabled=True, mode=None, aggregate=False):
    output = tmp_path / "telemetry.json"
    env = dict(os.environ)
    for key in (
        "MERLIN_GEMMINI_TELEMETRY",
        "MERLIN_GEMMINI_TELEMETRY_SCOPES",
        "MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC",
    ):
        env.pop(key, None)
    if enabled:
        env["MERLIN_GEMMINI_TELEMETRY"] = str(output)
    if aggregate:
        env["MERLIN_GEMMINI_TELEMETRY_AGGREGATE_PC"] = "1"
    if scopes is not None:
        spec = tmp_path / "scopes.txt"
        spec.write_text(scopes)
        env["MERLIN_GEMMINI_TELEMETRY_SCOPES"] = str(spec)
    result = subprocess.run(
        [str(observer), *([mode] if mode else [])],
        env=env,
        capture_output=True,
        check=False,
    )
    return result, json.loads(output.read_text()) if output.exists() else None


def test_default_disabled_observer_produces_no_file(observer, tmp_path):
    result, data = run(observer, tmp_path, enabled=False, scopes="malformed")
    assert result.returncode == 0 and data is None


def test_requested_payload_width_zero_pool_and_unknown_normalization(
    observer, tmp_path
):
    result, data = run(observer, tmp_path)
    assert result.returncode == 0
    rows = data["rows"]
    assert sum(r["requested_load_bytes"] for r in rows) == 366
    assert sum(r["requested_store_bytes"] for r in rows) == 306
    assert sum(r["unknown_dma_commands"] for r in rows) == 1
    assert sum(r["padded_mac_slots"] for r in rows) == 4096
    assert sum(r["padded_compute_rows"] for r in rows) == 16
    assert data["total_commands"] == 13


def test_unconfigured_lane_is_unknown_without_reading_state(observer, tmp_path):
    result, data = run(observer, tmp_path, mode="unconfigured")
    assert result.returncode == 0
    assert data["rows"][0]["unknown_dma_commands"] == 1
    assert data["rows"][0]["requested_load_bytes"] == 0


def test_ordered_markers_assign_actual_invocations_and_compact_summary(
    observer, tmp_path
):
    result, data = run(
        observer, tmp_path, scopes="7 0x100 0x200 0x100\n", aggregate=True
    )
    assert result.returncode == 0
    assert data["pc_aggregation"] == "scope_geometry"
    assert data["entries"] == [
        {"event": 1, "scope_id": 7, "invocation": 1, "pc": 0x100},
        {"event": 12, "scope_id": 7, "invocation": 2, "pc": 0x100},
    ]
    assert {r["scope_id"] for r in data["rows"]} == {7}
    assert {r["invocation"] for r in data["rows"]} == {1, 2}
    assert {r["pc"] for r in data["rows"]} == {0}
    assert sum(r["commands"] for r in data["rows"]) == 13
    assert sum(r["commands"] for r in data["rows"] if r["invocation"] == 2) == 2


@pytest.mark.parametrize(
    "scope",
    [
        "1 0x100 0x200\n",
        "1foo 0x100 0x200 0x100\n",
        "-1 0x100 0x200 0x100\n",
        "0 0x100 0x200 0x100\n",
        "1 0x100 0x200 0x201\n",
        "1 0x100 0x200 0x101\n",
        "1 0x100 0x200 0x100 extra\n",
        "1 0x100 0x200 0x100\n2 0x150 0x250 0x150\n",
        "1 0x100 0x200 0x100\n1 0x200 0x300 0x200\n",
    ],
)
def test_malformed_duplicate_or_overlapping_scopes_refused(observer, tmp_path, scope):
    result, _ = run(observer, tmp_path, scopes=scope)
    assert result.returncode == 2
