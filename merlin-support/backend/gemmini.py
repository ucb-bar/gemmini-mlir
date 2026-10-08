"""Run Merlin command buffers on Gemmini, via the chipyard bare-metal flow.

Pipeline: command buffer -> :mod:`gemmini_codegen` C driver (low-level libgemmini intrinsics)
-> compile with the chipyard ``riscv64-unknown-elf-gcc`` against the gemmini-rocc-tests
bare-metal harness -> run on an oracle:

  - ``spike --extension=gemmini``   : functional model, **bootstrap only** (derived_from_rtl=False)
  - the prebuilt Verilator RTL sim  : **certification** (derived_from_rtl=True)
  - the prebuilt GSIM emulator      : **certification** (derived_from_rtl=True) — the same elaborated
    design compiled FIRRTL->C++ instead of Verilog->C++, so it answers at the same fidelity and is
    chosen over Verilator by :mod:`merlin.targetgen.rtl_engine_policy` purely on cost.

-> parse OUT/METRIC/DONE -> gate the outputs against
:func:`merlin.runtime.reference.reference_outputs` (the same oracle the Python simulator
backend is held to). All three oracles run the *exact same ELF*.

Toolchain resolution mirrors ``build_tools/scripts/probe_gemmini_oracle.py``:
``MERLIN_CHIPYARD`` (default ``/path/to/chipyard``), plus optional
``MERLIN_RISCV_GCC`` / ``MERLIN_GEMMINI_SPIKE`` / ``MERLIN_GEMMINI_VERILATOR`` /
``MERLIN_GEMMINI_GSIM_EMU`` / ``MERLIN_GEMMINI_HARNESS_DIR`` overrides.
"""

from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from merlin.runtime.backends.base import BackendInfo, BackendKind, TargetClass, register
from merlin.runtime.metrics import COMMON_METRIC_NAMES

from .gemmini_codegen import DIM, CodegenError, generate_driver  # sibling — moves with this backend package

# Self-register this reference NPU backend with the class registry (base._REGISTRY). Discovery in
# base._ensure_discovered imports this module to run the call, so the core carries no name -> module
# map for the accelerator; the identity lives with the backend that owns it.
register(BackendInfo("gemmini", TargetClass.NPU, BackendKind.KERNEL, __name__))

# Software/harness capabilities, not hardware traits.  The generic performance corpus reads these
# through runtime.backends.base.execution_capability_facts; it never branches on this module's target
# name.  Each statement is implemented below: ``render_harness`` selects ``_whole_program_harness_c``
# from an explicit ABI discriminator, and that renderer emits one unmeasured call followed by one
# counter-bracketed call of the complete submitted kernel while printing only the cycle metric/results.
EXECUTION_CAPABILITIES = {
    "whole_program_kernel_abi": "render_harness accepts kernel_abi.kind=whole_program and calls declared tensor arguments "
    "without performing model arithmetic in the runner-owned harness",
    "warm_single_counter_region_cycles": "the whole-program harness executes one unmeasured warm invocation, then brackets exactly one "
    "complete kernel invocation with the target cycle counter",
}

DEFAULT_CHIPYARD = "/path/to/chipyard"
VERILATOR_CONFIG = "GemminiRocketConfig"

#: Wall-cycle budget for the GSIM emulator: the cap IS the hang bound, since a kernel that never reaches
#: its stop condition otherwise runs until the caller's wall timeout. NOT derived from a measurement on
#: this harness — it is a starting bound, raised or lowered per run with MERLIN_GEMMINI_GSIM_MAXCYCLES
#: rather than edited here (for scale, the SIMT target's GSIM oracle defaults to 2M).
GSIM_MAX_CYCLES = 20_000_000

ORACLE = {
    "spike": {"kind": "spike_gemmini_functional", "derived_from_rtl": False},
    "verilator": {"kind": "rtl_verilator", "derived_from_rtl": True},
    # Same elaborated design, different lowering of it (FIRRTL->C++). `derived_from_rtl` is the claim
    # the grade cites, and it is true of GSIM for exactly the reason it is true of Verilator; a missing
    # entry here is not "no oracle metadata" but a KeyError in `contract.compile.run_on_oracle`, i.e. the
    # engine would be selectable and then crash at grade time.
    "gsim": {"kind": "rtl_gsim", "derived_from_rtl": True},
}


class GemminiError(RuntimeError):
    pass


def counter_partition_inputs() -> dict[str, Any]:
    """Target boundary for the generic CIRCT occupancy-partition verifier.

    Module identities select structures in this target's elaborated artifact; no event predicate,
    engine name, code, or numeric parameter is copied here.  The generic verifier must still prove
    the boolean partition from the artifact before an overlap value can be called measured.
    """
    from merlin.targetgen.rtl import mlc_bridge

    path = mlc_bridge.core_hw_mlir("gemmini")
    if path is None or not Path(path).is_file():
        return {"status": "unknown", "why": "elaborated CIRCT core HW is unavailable"}
    return {
        "status": "available",
        "hw_text": Path(path).read_text(encoding="utf-8", errors="replace"),
        "module": "Gemmini",
        "counter_module": "CounterController",
        "source": str(path),
    }


# --- per-layer library bench (merlin.perf.layer_bench) ------------------------------------------------
# What the generic layer bench needs from this target, each DERIVED from the curated harness headers the
# bench programs are compiled against, so the reference and the facts can never come from two headers.
# The sealable N-window frame (WINDOW_UART, WINDOW_CHECKSUM_LINE, render_window_program,
# window_marker) is re-exported here too: a caller that reached past the backend for the frame while
# going through it for the renderers could pair a program with a policy describing a different
# spelling of the same lines.
from .gemmini_layer_bench import CHECKSUM_LINE as WINDOW_CHECKSUM_LINE  # noqa: E402
from .gemmini_layer_bench import HARNESS_VERSION as LIBRARY_LAYER_HARNESS_VERSION  # noqa: E402
from .gemmini_layer_bench import OPERAND_BLOB as LIBRARY_LAYER_OPERAND_BLOB  # noqa: E402
from .gemmini_layer_bench import (  # noqa: E402,F401
    WINDOW_UART,
    autocomp_window_call,
    package_window_call,
    render_autocomp_kernel_unit,
    render_window_program,
    window_marker,
)


def _params_header() -> Path:
    return rocc_tests_dir() / "include" / "gemmini_params.h"


def render_library_layer(spec: dict[str, Any], *, offsets: dict[str, int]) -> str:
    """C source measuring one layer with this target's own library kernel (see gemmini_layer_bench).
    The build directory must hold the packed operand blob as ``LIBRARY_LAYER_OPERAND_BLOB``."""
    from .gemmini_layer_bench import render_library_layer as _render

    return _render(spec, offsets=offsets)


def render_external_layer(
    spec: dict[str, Any],
    *,
    offsets: dict[str, int],
    symbol: str,
    header: str,
    declared_shape: dict[str, Any],
) -> str:
    """C source measuring a THIRD-PARTY kernel on the same operands as the library and the package.

    ``declared_shape`` is the shape the external kernel's own descriptor states; the renderer refuses
    a spec that differs, because a kernel searched for one shape is not that kernel on another."""
    from .gemmini_layer_bench import render_external_layer as _render

    return _render(spec, offsets=offsets, symbol=symbol, header=header, declared_shape=declared_shape)


def render_package_layer(
    cb: dict[str, Any], *, offsets: dict[str, int], label: str, output: str, protocol: str = "warm_then_measured"
) -> str:
    """C source measuring one layer compiled by an mlir_oot package (see gemmini_layer_bench); the
    package kernel object is linked beside it and called through this target's harness entry symbol."""
    from merlin.targetgen.contract.harness_abi import for_target

    from .gemmini_layer_bench import render_package_layer as _render

    return _render(
        cb,
        offsets=offsets,
        label=label,
        output=output,
        entry_symbol=for_target(_backend_name()).entry_symbol,
        protocol=protocol,
    )


def _backend_name() -> str:
    from merlin.runtime.backends import base as _base

    try:
        return _base.name_of_module(__package__)
    except KeyError:
        return _base.name_of_module(__name__)


def sched_instruction_set():
    """This target's schedule instruction set (``merlin.sched.isa``), derived from the curated harness
    header the layer programs compile against (see gemmini_sched)."""
    from .gemmini_sched import instruction_set

    return instruction_set(
        str(rocc_tests_dir() / "include" / "gemmini.h"), str(_params_header()), target=_backend_name()
    )


def sched_matmul_reference(**kwargs):
    """The reference LOOP_WS schedule of one matmul (see gemmini_sched.matmul_reference)."""
    from .gemmini_sched import matmul_reference, schedule_facts

    facts = schedule_facts(
        str(rocc_tests_dir() / "include" / "gemmini.h"), str(_params_header()), target=_backend_name()
    )
    return matmul_reference(facts=facts, **kwargs)


def render_schedule_layer(
    spec: dict[str, Any], *, offsets: dict[str, int], kernel_c: str, symbol: str, arg_names: list[str]
) -> str:
    """C source measuring one layer through a scheduled kernel (see gemmini_layer_bench)."""
    from .gemmini_layer_bench import render_schedule_layer as _render

    return _render(spec, offsets=offsets, kernel_c=kernel_c, symbol=symbol, arg_names=arg_names)


def readout_facts() -> dict[str, Any]:
    """The header-verified scalar readout contract of the harness the bench programs compile against."""
    from .gemmini_readout_semantics import narrow_readout_contract

    return narrow_readout_contract(_params_header())


def library_layer_emitter_digest() -> str:
    """sha256 over everything that decides a library-layer program's instructions: the renderer and
    the library/params headers it includes. A change to any of them is a different emitter."""
    import hashlib

    from . import gemmini_layer_bench

    h = hashlib.sha256()
    for part in (
        Path(gemmini_layer_bench.__file__),
        rocc_tests_dir() / "include" / "gemmini.h",
        rocc_tests_dir() / "include" / "gemmini_nn.h",
        _params_header(),
    ):
        h.update(part.name.encode() + b"\0" + part.read_bytes() + b"\0")
    return h.hexdigest()


def gsim_backdoor_env() -> dict[str, str]:
    """Environment that makes the GSIM harness load the ELF by BACKDOOR -- fesvr's load_mem_write memcpys
    each segment into the SimDRAM backing store -- instead of pushing every byte (and every .bss zero)
    over the TSI serial link at under ~0.47 B/cycle.

    MEASURED 2026-09-14 on the certified serial-clock emulator: the ResNet-50 conv_7 library program runs
    527,711 cycles by backdoor vs 527,409 over TSI (0.06%), and the whole run drops 601,947 -> 534,667
    cycles -- the difference was loading. The harness source's comment says the backdoor writes a store
    nothing reads on this design; the core in fact fetches its program from SimDRAM ([gsim-ar] reads at
    0x80000000), which is exactly that store. The comment predates the serial-TL clock fix.
    """
    return {"MERLIN_GSIM_LOADMEM": "1"}


def mac_per_cycle_peak() -> int:
    """Peak MACs per cycle of the mesh, from the harness header's DIM (a DIM x DIM array)."""
    from merlin.targetgen.capability_discovery import parse_c_header

    dim = parse_c_header(_params_header()).macro("DIM")
    if dim is None or not dim.body.strip().isdigit():
        raise GemminiError("params header declares no integer DIM")
    return int(dim.body.strip()) ** 2


def readout_epilogue_capability() -> list[dict[str, Any]]:
    """Which epilogue stages each of this target's readouts APPLIES -- declared, not inferred.

    Consumed by :func:`merlin.verify.epilogue_applicability.assess`, which states the general rule
    (a declared stage must be applied by the readout the program selected) and knows nothing about
    any target's readouts. This is the declaration for THIS one.

    ⚠️ THE FULL-WIDTH READOUT APPLIES NOTHING, and that is the whole reason this exists. The RoCC
    model computes the scaled and activated value and then, on the full-width path, writes the raw
    accumulator instead::

        auto shifted     = acc_scale(acc_value, gemmini_state.acc_shift);
        elem_t activated = apply_activation_acc(shifted);
        if (full) { write_to_dram<acc_t>(addr, acc_value);  }   // raw: pre-scale, pre-activation
        else      { write_to_dram<elem_t>(addr, activated); }   // scale AND activation applied

    Measured consequence: a capsule declaring ``output_dtype='i32' epilogue=['relu']`` returned 126
    of 256 outputs negative (min -85) -- the raw accumulator -- while its command-buffer numeric floor
    and trace both passed. Its sibling with the same declaration passes only because the default
    stimulus is non-negative, so the activation is the identity and cannot be observed.

    The stage names are the command-buffer ABI's own vocabulary
    (``runtime.commandbuffer.EPILOGUE_STAGES``), and the narrowing readout's set is the one recorded
    RTL-certified bit-exact by decision A in the requant reconciliation: float ``acc_scale``
    (round-to-nearest-even) plus the activation, emitted as the accumulator address without the
    full-width bit. ``requant`` is deliberately ABSENT from both: merlin's integer round-half-up
    shift is not what this hardware's float scale computes, so it stays a host-side op rather than
    being declared as something this readout applies.
    """
    return [
        {
            "selector": "i8",
            "applies": ["acc_scale", "relu", "bias_add", "bias", "maxpool"],
            "evidence": "the narrowing readout applies the accumulator scale and the activation; "
            "RTL-certified bit-exact against Tensor.requant_acc_scale (decision A)",
        },
        {
            "selector": "i32",
            # THE BIAS IS NOT A READOUT STAGE ON THIS TARGET: it is preloaded into the accumulator (the
            # D operand) before the contraction accumulates onto it, so the raw accumulator the
            # full-width readout writes already carries it. Declaring it unapplied refused every
            # biased int32 commit -- ResNet-50's classifier among them -- for every package, and left
            # the group to a library host path. MEASURED: the vendor library's full-width matmul with
            # a bias (D) on the GSIM emulator, job 6169f722 (vendor_reference_gsimdump), classifier g71
            # correct and argmax 21 = oracle.
            "applies": ["bias_add", "bias"],
            "evidence": "the full-width readout writes the raw accumulator -- the preloaded bias plus "
            "the products -- discarding the scaled and activated value it computed (RoCC model, "
            "accumulator mvout path); measured correct with a bias on the GSIM emulator (vendor "
            "reference 6169f722, classifier g71)",
        },
    ]


def readout_scalar_abi() -> dict[str, Any] | None:
    """The scalar readout contract of the parameter header this backend builds programs against.

    Consumed by :func:`merlin.targetgen.readout_facet.for_target` as its ``scalar_abi`` rung. The
    header is the one the harness compiles with, so the dtypes, clamp bounds and rounding recorded
    here are the ones a built program actually gets. ``None`` when the header is absent; a header
    whose scale or rounding source is not the verified construction raises in the reader and is
    reported by the caller as an absent rung.
    """
    from .gemmini_readout_semantics import narrow_readout_contract

    header = rocc_tests_dir() / "include" / "gemmini_params.h"
    return narrow_readout_contract(header) if header.is_file() else None


def readout_operand_sum() -> dict[str, Any] | None:
    """What a LOAD does to an operand on its way into the accumulator, from the same header.

    Consumed by :func:`merlin.targetgen.readout_facet.for_target` as its ``operand_sum`` rung: a
    design whose load multiplies can add two separately scaled tensors without the host.
    """
    from .gemmini_readout_semantics import operand_sum_contract

    header = rocc_tests_dir() / "include" / "gemmini_params.h"
    return operand_sum_contract(header) if header.is_file() else None


def counter_engine_kinds() -> dict[str, Any]:
    """Which RESOURCE KIND each counted engine is -- declared here, never read off a counter name.

    ``merlin.perf.decompose.activity_from_busy`` refuses a unit whose kind is not declared, and it is
    right to: a kind inferred from a spelling is exactly how a local register load once got mapped
    onto "DMA".  These three are this target's own controllers -- EX executes on the mesh, LD and ST
    move operands between DRAM and the scratchpad/accumulator -- so calling LD/ST movement is a
    statement about this hardware, not an inference from two-letter tokens.

    A STRONGER binding exists and should be preferred where it has been run:
    ``gemmini_roofline_auxiliary`` derives the same roles by PROBING -- DMA read/write/copy plus a
    compute probe, proved against the elaborated CIRCT artifact -- and records them under
    ``resource_role_binding``. This declaration is the cheap always-available form, and it is what the
    graded path asks for (``capsule_grade._activity_source`` requires the PRODUCER to state a kind and
    refuses to infer one); a run that has the probed binding should use that instead.

    Unlocks ``overlap_cycles.across_kinds``, which counts only cycles spanning two DIFFERENT kinds.
    That is the quantity a compute/movement roofline needs: LD and ST busy together is not
    movement/compute overlap, and reporting ``overlap_cycles.observed`` in its place overstates what
    a compute/movement pairing achieved.
    """
    return {"EX": "compute", "LD": "movement", "ST": "movement"}


def chipyard_root() -> Path:
    """Chipyard root, honoring ``.env`` (not just the process env). ``os.environ.get`` alone missed
    ``MERLIN_CHIPYARD`` when it lives in the repo ``.env`` (the repo-wide contract), leaving the
    ``/path/to/chipyard`` placeholder — which made ``available('spike'/'verilator')`` False even with
    a real toolchain, so every oracle reported NOT_RUN_IS_NOT_PASS. Resolve through
    ``merlin.common.paths`` (env/.env → ext_path) with the placeholder only as a last resort."""
    from merlin.common.paths import env as _env
    from merlin.common.paths import ext_path as _ext_path

    root = os.environ.get("MERLIN_CHIPYARD") or _env("MERLIN_CHIPYARD")
    if root:
        return Path(root)
    cy = _ext_path("chipyard")
    return Path(cy) if cy else Path(DEFAULT_CHIPYARD)


def gcc_path() -> Path:
    env = os.environ.get("MERLIN_RISCV_GCC")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/riscv64-unknown-elf-gcc"


def spike_path() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_SPIKE")
    if env:
        return Path(env)
    return chipyard_root() / ".conda-env/riscv-tools/bin/spike"


def libgemmini_dir() -> Path:
    return chipyard_root() / ".conda-env/riscv-tools/lib"


#: This backend's registered target name and the Spike extension CLASS its functional model registers.
#: Two different kinds of fact that happen to spell the same, kept apart because only the first is a
#: merlin target: a sibling elaboration of this generator registers the same extension class from a
#: DIFFERENT ``.so``, which is exactly the confusion :mod:`merlin.targetgen.spike_extension` refuses.
TARGET_NAME = "gemmini"
SPIKE_EXTENSION_NAME = "gemmini"


def spike_extension() -> "tuple[tuple[str, ...], Path]":
    """``(spike argv flags, LD_LIBRARY_PATH directory)`` for THIS target's L2 functional model.

    Target-parameterized on purpose: the extension a functional model must load is a property of the
    ELABORATION, not of whichever chipyard ``$MERLIN_CHIPYARD`` names, and a model built from a
    different generated ``gemmini_params.h`` returns wrong numbers rather than failing
    (``libgemmini-driver-allzeros-header-skew``). A target that declares
    ``runner.spike_extension`` in its own contract gets that ``.so``, digest-verified, or an
    exception — never a fallback to a sibling's build.

    THIS target declares none, so the resolution is byte-identical to what it has always been:
    ``--extension=gemmini`` with ``libgemmini_dir()`` on the library path. Asserted, not assumed, by
    ``merlin/tests/gemmini/test_spike_extension.py``.
    """
    from merlin.targetgen.spike_extension import spike_invocation

    return spike_invocation(
        TARGET_NAME, default_library_dir=libgemmini_dir(), default_extension_name=SPIKE_EXTENSION_NAME
    )


def simulator_identity(simulator: str) -> "dict | None":
    """Which build of ``simulator`` runs a program here, by content: the binary and, for spike, the
    extension library it loads -- ``None`` for a simulator this backend states no identity for. A result
    cached against a simulator (a whole model's reference arm) is keyed on this, so a rebuilt simulator
    is a different key rather than a silently reused answer."""
    import hashlib

    def digest(path: Path) -> "str | None":
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    if simulator != "spike":
        return None
    flags, libdir = spike_extension()
    names = [flag.partition("=")[2] for flag in flags if flag.startswith("--extension=")]
    libraries = {name: digest(Path(libdir) / f"lib{name}.so") for name in names}
    parts = {"binary": digest(spike_path()), "flags": list(flags), "extensions": libraries}
    if parts["binary"] is None or not libraries or None in libraries.values():
        return {**parts, "digest": None}
    return {**parts, "digest": hashlib.sha256(repr(sorted(parts.items())).encode()).hexdigest()}


def platform_dram_base() -> int:
    """The bare-metal linker load address, DERIVED from this target's chipyard RTL build memory map
    (``runtime_build.platform_dram_base`` reads the ``memory@`` region of the sim config's ``memmap.json``)
    rather than baked in the linker script. Falls back to the platform default if the build is absent."""
    from merlin.targetgen import runtime_build as _rb

    return _rb.platform_dram_base("gemmini", "chipyard")


def _rtl_sim_config() -> str:
    """The verilator harness config that realizes gemmini — a DECLARED target fact (capability manifest
    ``runtime.rtl_sim_config``), read via the target registry rather than a hardcoded backend constant.
    Env override wins; then the manifest; then the module fallback (kept coherent with the facts config)."""
    env = os.environ.get("MERLIN_GEMMINI_VERILATOR_CONFIG")
    if env:
        return env
    try:
        from merlin.targetgen.target_experiment import load_capability_manifest

        cfg = (load_capability_manifest("gemmini").contract.get("runtime") or {}).get("rtl_sim_config")
        if cfg:
            return str(cfg)
    except Exception:  # noqa: BLE001 — manifest unavailable ⇒ fall back, never crash the backend
        pass
    return VERILATOR_CONFIG


def verilator_path() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_VERILATOR")
    if env:
        return Path(env)
    return chipyard_root() / "sims/verilator" / f"simulator-chipyard.harness-{_rtl_sim_config()}"


GSIM_EMU_ENV = "MERLIN_GEMMINI_GSIM_EMU"

_PINNED_RUNTIME_LOCK = threading.RLock()


def runtime_environment(
    *, binaries: Mapping[str, Path], gsim_max_cycles: int | None, environment: Mapping[str, str]
) -> dict[str, str]:
    """Copy an environment with explicit engine choices; never mutate process state."""
    if not isinstance(environment, Mapping) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in environment.items()
    ):
        raise ValueError("runtime environment must map strings to strings")
    if not isinstance(binaries, Mapping) or set(binaries) != {"gsim", "verilator"}:
        raise ValueError("pinned runtime requires exactly GSIM and Verilator binaries")
    if gsim_max_cycles is not None and (type(gsim_max_cycles) is not int or gsim_max_cycles <= 0):
        raise ValueError("GSIM max cycles must be a positive integer or null")
    paths = {engine: Path(path).resolve(strict=True) for engine, path in binaries.items()}
    if any(not path.is_file() for path in paths.values()):
        raise ValueError("pinned runtime binaries must be ordinary files")
    result = dict(environment)
    result[GSIM_EMU_ENV] = str(paths["gsim"])
    result["MERLIN_GEMMINI_VERILATOR"] = str(paths["verilator"])
    if gsim_max_cycles is None:
        result.pop("MERLIN_GEMMINI_GSIM_MAXCYCLES", None)
    else:
        result["MERLIN_GEMMINI_GSIM_MAXCYCLES"] = str(gsim_max_cycles)
    return result


@contextmanager
def pinned_runtime(*, binaries: Mapping[str, Path], gsim_max_cycles: int | None):
    """Apply the pure policy under a nested lock, restoring only owned keys.

    Callers independently check certificate hashes. Unrelated environment writers
    are not synchronized; no simulator runs during configuration.
    """
    keys = (GSIM_EMU_ENV, "MERLIN_GEMMINI_VERILATOR", "MERLIN_GEMMINI_GSIM_MAXCYCLES")
    with _PINNED_RUNTIME_LOCK:
        configured = runtime_environment(binaries=binaries, gsim_max_cycles=gsim_max_cycles, environment=os.environ)
        previous = {key: os.environ.get(key) for key in keys}
        try:
            for key in keys:
                value = configured.get(key)
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
            yield sys.modules[__package__]
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def gsim_path() -> Path:
    """The prebuilt GSIM emulator binary for this target's RTL, or where one would be installed.

    Resolved by :mod:`merlin.targetgen.gsim_emulator`, which owns the layout for every target: the env
    override first, then the DERIVED home under the build root (``out/build/rtl_engines/<target>/gsim/emulator``).
    The old fallback pointed inside the chipyard checkout at a path chipyard never produces -- GSIM emits
    a standalone C++ model built OUT of tree, so there is no ``sims/gsim`` rule whose output could be
    derived there -- which made the derived branch decorative and the env var mandatory in practice.

    Existence is not checked here (see :func:`available`), so this never raises.
    """
    from merlin.targetgen import gsim_emulator as _gsim

    return _gsim.emulator_path("gemmini", env_var=GSIM_EMU_ENV)


def gsim_status() -> tuple[bool, str]:
    """``(available, reason)`` for the GSIM engine -- the reason is the point.

    ``available()`` must stay a bool (its callers and its contract are boolean), but a bare False is
    exactly what made the Verilator fallback silent: "gsim reports unavailable" says nothing about
    whether the binary is absent, unexecutable, or REFUSED because its build receipt describes different
    bytes. This carries that sentence to whoever records the selection.
    """
    from merlin.targetgen import gsim_emulator as _gsim

    return _gsim.probe("gemmini", env_var=GSIM_EMU_ENV)


def rocc_tests_dir() -> Path:
    env = os.environ.get("MERLIN_GEMMINI_HARNESS_DIR")
    if env:
        return Path(env)
    # This provider owns its curated int8 harness. A missing resource is not
    # permission to borrow mutable Chipyard headers for another configuration.
    return Path(__file__).resolve().parents[1] / "resources/gemmini-rocc-tests"


def _common_dir() -> Path:
    return rocc_tests_dir() / "riscv-tests/benchmarks/common"


def _test_ld() -> Path:
    return _common_dir() / "test.ld"


def available(simulator: str = "verilator") -> bool:
    """True when gcc + the harness + the requested simulator are all present.

    An engine this backend KNOWS but cannot find answers False; an engine it does not know still RAISES.
    The distinction is load-bearing for :func:`merlin.targetgen.capsule_runner.chipyard_l3_selection`,
    which probes every engine in the priority list: a raise is recorded as "this target has no such
    engine", a False as "it has one and the binary is absent" — two different things to fix.
    """
    base = gcc_path().is_file() and _test_ld().is_file() and _common_dir().is_dir()
    if simulator == "spike":
        return base and spike_path().is_file()
    if simulator == "verilator":
        return base and verilator_path().is_file()
    if simulator == "gsim":
        # Executability is checked as well as existence, unlike the Verilator path: that binary is a
        # build product of a chipyard rule that leaves it executable, whereas the GSIM emu arrives via an
        # env var a caller points anywhere — at a copied-without-mode artifact, or at the emitted .cpp.
        emu = gsim_path()
        return base and emu.is_file() and os.access(emu, os.X_OK)
    raise GemminiError(f"unknown simulator {simulator!r}")


def compile_command_buffer(cb: dict[str, Any], workdir: str | Path, driver_src: str | None = None) -> Path:
    """Compile a Gemmini command buffer to a bare-metal ELF and return its path.

    ``driver_src`` overrides the in-tree codegen with externally-provided C (used to certify
    an agent-generated kernel). Direct rank-N contractions use the production LLVM/RoCC emitter and
    runner-owned harness/link path; legacy command forms retain the C-driver path.
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    # BATCHED_MATMUL is production-owned by the LLVM/RoCC emitter.  Sending it through the legacy C
    # generator changes the operation into the resident-matmul grammar (and currently refuses it for
    # lacking RES_PACK), so use the existing runner-owned MLIR -> object -> harness -> link path.  The
    # externally supplied C override remains authoritative, and every other command buffer keeps the
    # byte-identical legacy build below.
    if driver_src is None and any(
        isinstance(command, dict) and command.get("opcode") == "BATCHED_MATMUL"
        for command in (cb.get("commands") or [])
    ):
        from merlin.runtime.backends import base as _backend_registry
        from merlin.targetgen.contract.compile import compile_lowered_to_elf

        from .gemmini_codegen_mlir import emit_kernel_mlir

        lowered, _arguments = emit_kernel_mlir(cb)
        # OOT discovery registers the backend package while this function lives in its ``.gemmini``
        # implementation submodule; an in-tree direct import registers the implementation module.
        # Resolve either packaging form from module identity instead of copying the target name.
        try:
            target = _backend_registry.name_of_module(__package__)
        except KeyError:
            target = _backend_registry.name_of_module(__name__)
        return compile_lowered_to_elf(cb, lowered, work, target=target)
    main_c = work / "main.c"
    main_c.write_text(driver_src if driver_src is not None else generate_driver(cb), encoding="utf-8")
    elf = work / "merlin_gemmini_c0.elf"
    rt = rocc_tests_dir()
    common = _common_dir()
    # Mirror gemmini-rocc-tests/bareMetalC/Makefile (CFLAGS_BAREMETAL) EXACTLY — both the
    # flag set and the include ORDER matter: a wrong order shadows the riscv-tests/env
    # syscall headers and corrupts the tohost protocol ("bad syscall" on spike).
    cmd = [
        str(gcc_path()),
        "-DPREALLOCATE=1",
        "-DMULTITHREAD=1",
        "-mcmodel=medany",
        "-std=gnu99",
        "-O2",
        "-ffast-math",
        "-fno-common",
        "-fno-builtin-printf",
        "-fno-tree-loop-distribute-patterns",
        "-march=rv64gc",
        "-Wa,-march=rv64gc",
        "-lm",
        "-lgcc",
        "-I",
        str(rt / "riscv-tests"),
        "-I",
        str(rt / "riscv-tests/env"),
        "-I",
        str(rt),
        "-I",
        str(common),
        "-DID_STRING=",
        "-DPRINT_TILE=0",
        "-nostdlib",
        "-nostartfiles",
        "-static",
        "-T",
        str(_test_ld()),
        "-DBAREMETAL=1",
        str(main_c),
        "-o",
        str(elf),
        *(str(p) for p in sorted(common.glob("*.c"))),
        *(str(p) for p in sorted(common.glob("*.S"))),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise GemminiError(f"riscv gcc failed:\n{' '.join(cmd)}\n{proc.stderr}")
    return elf


def _unlimited_stack() -> None:  # pragma: no cover - runs in the child process
    """Lift RLIMIT_STACK for a simulator child.

    The Verilator model needs a large stack; the default (e.g. 12500 kb) makes it warn ("%Warning: System
    has stack size ...") onto the console, corrupting output capture. The GSIM-emitted model holds the
    whole design state in one C++ object and is driven from the same harness, so it is raised the same way.
    """
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        pass


def gsim_max_cycles() -> str:
    """The GSIM wall-cycle cap, as the plusarg value. Env override wins; see :data:`GSIM_MAX_CYCLES`."""
    return os.environ.get("MERLIN_GEMMINI_GSIM_MAXCYCLES", "").strip() or str(GSIM_MAX_CYCLES)


def _gsim_argv(elf: str | Path, *, max_cycles: int | None = None) -> list[str]:
    """One flag spelling shared by legacy execution and bounded command preparation."""
    cycles = gsim_max_cycles() if max_cycles is None else str(max_cycles)
    return [str(gsim_path()), str(elf), f"+max-cycles={cycles}", f"+loadmem={elf}"]


def prepare_gsim_command(elf, **kwargs):
    """Describe a pinned GSIM invocation; caller owns deadline, sandbox and execution."""
    from .gemmini_execution_command import prepare_gsim_command as prepare

    return prepare(elf, **kwargs)


def prepare_short_program_build(**kwargs):
    """Prepare the host-owned complete-short-program build worker; never execute it here."""
    from .gemmini_program_build import prepare_short_program_build as prepare

    return prepare(**kwargs)


def short_program_environment(sandbox):
    """Resolve exact already-granted build paths and the host-selected engine."""
    from .gemmini_program_environment import short_program_environment as prepare

    return prepare(sandbox)


def prepare_short_program_execution(elf, **kwargs):
    """Complete-short-program command; admission and deadlines remain host-owned."""
    return prepare_gsim_command(elf, **kwargs)


def run_elf(elf: str | Path, simulator: str = "verilator", timeout: int = 600) -> str:
    """Run the ELF on the chosen oracle; return raw console output."""
    preexec = None
    if simulator == "spike":
        env = dict(os.environ)
        # WHICH functional model, resolved from the target's own contract rather than from the ambient
        # chipyard. For this target nothing is declared, so `flags` is `("--extension=gemmini",)` and
        # `libdir` is `libgemmini_dir()` — the exact strings this line has always produced.
        flags, libdir = spike_extension()
        env["LD_LIBRARY_PATH"] = str(libdir) + ":" + env.get("LD_LIBRARY_PATH", "")
        # The DRAM span an image states it was laid out for (an open whole model's arena lies past
        # spike's default span); an image that states none keeps the command it always had.
        from merlin.runtime.backends.spike_model import declared_harts, declared_isa, declared_memory

        span = declared_memory(elf)
        memory = [f"-m{hex(span[0])}:{hex(span[1])}"] if span else []
        # Likewise the harts and the ISA a two-hart image states (its host code on a vector hart).
        harts, isa = declared_harts(elf), declared_isa(elf)
        machine = [*([f"-p{harts}"] if harts else []), *([f"--isa={isa}"] if isa else [])]
        cmd = [str(spike_path()), *flags, *machine, *memory, str(elf)]
    elif simulator == "verilator":
        env = dict(os.environ)
        cmd = [str(verilator_path()), str(elf)]
        preexec = _unlimited_stack
    elif simulator == "gsim":
        env = dict(os.environ)
        # The SAME ELF the Verilator path runs, so the console it prints is the same OUT/METRIC/DONE text
        # and `parse_output` is unchanged. GSIM re-roots the circuit at ChipTop and so has no SimTSI to
        # load the image: `+loadmem` is the backdoor that writes it into the backing store, and it is
        # passed BESIDE the positional argument rather than instead of it (the emitted harness reads the
        # symbol table from the positional path). `+max-cycles` is the hang bound, not a perf knob.
        cmd = _gsim_argv(elf)
        preexec = _unlimited_stack
        # NB the console is buffered in this process, like every other oracle here. That is fine for the
        # OUT/METRIC/DONE protocol but NOT for a model built with a per-instruction commit trace: on the
        # SIMT target such a console reached 72 GB in one run and had to be spooled to disk. If a gemmini
        # GSIM model is ever emitted with tracing on, this branch needs that same treatment.
    else:
        raise GemminiError(f"unknown simulator {simulator!r}")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env, preexec_fn=preexec)
    # The Verilator harness exits 0 on $finish; spike exits 0 on htif_exit(0); the GSIM-emitted model
    # exits 0 when the design's own stop condition fires before +max-cycles.
    if proc.returncode != 0:
        raise GemminiError(f"{simulator} exited {proc.returncode}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    _refuse_on_rtl_assertion(simulator, proc.stdout, proc.stderr)
    return proc.stdout


#: What an elaborated-RTL model prints when the design's own `assert` fails. Both engines emit it; only
#: one of them STOPS. It is a property of the RTL, not of a simulator, so it is matched on the design's
#: text rather than on any engine's exit convention.
_RTL_ASSERTION_MARKER = "Assertion failed"


def _refuse_on_rtl_assertion(simulator: str, stdout: str, stderr: str) -> None:
    """Fail a run whose design asserted, whatever the engine did about it afterwards.

    The exit code alone is not enough, and the gap is in the dangerous direction. Verilator turns a
    failed assertion into `$stop` and exits non-zero, so the check above catches it. The GSIM model
    REPORTS the same assertion and keeps going: measured on a real submission that issued a zero-byte
    mvin, GSIM printed three assertion failures (`LoadController.scala:192`, `DMACommandTracker.scala:88`
    and `:89`), then finished `done=1 exit_code=0` and printed a complete OUT/DONE console. Nothing
    downstream reads the console for assertions, so the capsule would have been judged on its output
    bytes alone -- and a kernel the hardware refuses would PASS whenever those bytes happen to match.

    An assertion is the DESIGN saying the program did something it does not support. Certifying past it
    is the wrong-device hazard in miniature: the numbers would describe a machine that would not have
    run this.
    """
    for stream in (stdout, stderr):
        if not stream or _RTL_ASSERTION_MARKER not in stream:
            continue
        # The marker line carries the message; the SITE is on the next line (`at <file>:<line> ...`).
        # Reporting only the marker gives a reader "Assertion failed" with nowhere to look, and the
        # design asserts in several places at once -- three fired on the measured submission.
        rows = [ln.rstrip() for ln in stream.splitlines()]
        quoted: list[str] = []
        for index, line in enumerate(rows):
            if _RTL_ASSERTION_MARKER not in line:
                continue
            quoted.append(line.strip())
            nxt = rows[index + 1].strip() if index + 1 < len(rows) else ""
            if nxt.startswith("at "):
                quoted.append(nxt)
        raise GemminiError(
            f"{simulator}: the DESIGN asserted, so this run certifies nothing — the program did "
            f"something the hardware does not support:\n  " + "\n  ".join(quoted[:8])
        )


def parse_output(text: str) -> tuple[dict[str, list], dict[str, int]]:
    """Parse the OUT/METRIC/DONE console into (outputs, raw metrics) — shared protocol parser, with
    the gemmini-specific robustness: strip stray Verilator ``%Warning:`` fragments + tolerate a
    malformed METRIC line instead of raising."""
    from merlin.runtime.backends.base import _strip_warning_fragments, parse_console

    cleaned = _strip_warning_fragments(text)
    outputs, raw = parse_console(cleaned, error_cls=GemminiError, tolerant_metric=True)

    def reshape(values: list[int], dimensions: tuple[int, ...]):
        if len(dimensions) == 1:
            return values
        stride = 1
        for extent in dimensions[1:]:
            stride *= extent
        return [reshape(values[index : index + stride], dimensions[1:]) for index in range(0, len(values), stride)]

    # ``OUT`` remains byte-for-byte compatible. ``OUT_ND`` is the target-owned extension needed by a
    # rank-N whole-op ABI; flattening it to an OUT matrix would discard source-visible batch rank.
    for line in cleaned.splitlines():
        parts = line.split()
        if not parts or parts[0] != "OUT_ND":
            continue
        try:
            name = parts[1]
            rank = int(parts[2])
            if rank <= 0:
                raise ValueError("rank must be positive")
            if len(parts) < 3 + rank:
                raise ValueError("dimension list is truncated")
            dimensions = tuple(int(value) for value in parts[3 : 3 + rank])
            if any(extent <= 0 for extent in dimensions):
                raise ValueError("dimensions must be positive")
            values = [int(value) for value in parts[3 + rank :]]
        except (IndexError, ValueError) as exc:
            raise GemminiError(f"malformed OUT_ND line: {line!r}: {exc}") from exc
        expected = 1
        for extent in dimensions:
            expected *= extent
        if len(values) != expected:
            raise GemminiError(f"OUT_ND {name}: expected {expected} values, got {len(values)}")
        if name in outputs:
            raise GemminiError(f"output {name!r} was printed more than once")
        outputs[name] = reshape(values, dimensions)
    return outputs, raw


def _metrics(raw: dict[str, int], simulator: str) -> dict[str, Any]:
    metrics = {name: int(raw.get(name, 0)) for name in COMMON_METRIC_NAMES}
    metrics["cycles"] = int(raw.get("cycles", 0))
    metrics["cycle_source"] = "rdcycle" if "cycles" in raw else "unknown"
    metrics["cycle_window"] = "gemmini_region" if raw.get("cycle_window_gemmini_region") else "unknown"
    metrics["memory_model"] = "functional_model" if simulator == "spike" else "unknown"
    return metrics


def run_command_buffer(
    cb: dict[str, Any],
    *,
    workdir: str | Path | None = None,
    simulator: str = "verilator",
    timeout: int = 600,
    driver_src: str | None = None,
) -> dict[str, Any]:
    """Compile + run a command buffer on Gemmini and gate on reference equality.

    ``driver_src`` certifies an externally-provided (e.g. agent-generated) kernel instead of
    the in-tree codegen. Returns {outputs, metrics, raw_metrics, correct, oracle, elf, console}.
    """
    # Build-only workers run with reference answers masked. Only this evaluator
    # consumes the reference; resolving the target renderer must not import it.
    from merlin.runtime.reference import outputs_match, reference_outputs

    if not available(simulator):
        raise GemminiError(f"gemmini {simulator} oracle not available (set MERLIN_CHIPYARD)")
    own_tmp = workdir is None
    work = Path(tempfile.mkdtemp(prefix="merlin_gemmini_")) if own_tmp else Path(workdir)
    elf = compile_command_buffer(cb, work, driver_src=driver_src)
    console = run_elf(elf, simulator=simulator, timeout=timeout)
    outputs, raw = parse_output(console)
    ref = reference_outputs(cb)
    return {
        "outputs": outputs,
        "metrics": _metrics(raw, simulator),
        "raw_metrics": raw,
        "correct": outputs_match(outputs, ref),
        "oracle": dict(ORACLE[simulator]),
        "elf": str(elf),
        "console": console,
    }


def preflight_codegen_smoke(*, target: str) -> tuple[bool, str]:
    """Compile the production command-buffer emitter and run it bit-exact on RTL.

    This is the target-owned implementation of the generic pre-spend codegen-smoke hook.  It exercises
    the same ``generate_driver -> riscv gcc -> selected L3 engine -> parse -> reference equality`` path
    used by a real grade.  The engine is resolved through the shared RTL policy, including
    ``MERLIN_REQUIRED_RTL_ENGINE``; otherwise a GSIM-pinned run would silently pay for an unrelated
    Verilator pass before every resume.  Merely finding the simulator or compiling an empty file is
    insufficient: both have been true while the emitted kernel itself was wrong.
    """
    try:
        # Import at call time: capsule_runner discovers this backend while it is itself importing, so a
        # module-level import would create a cycle.  This is the SAME selector used to bind the L3 grade;
        # the preflight must not maintain a second engine policy.
        from merlin.targetgen.capsule_runner import chipyard_l3_selection

        selection = chipyard_l3_selection(target)
        rtl_engine = str(selection["engine"])
    except Exception as e:  # noqa: BLE001 — no policy-selected L3 means the smoke cannot certify codegen
        return False, (
            f"Gemmini production codegen smoke cannot select its L3 RTL engine: {type(e).__name__}: {str(e)[-200:]}"
        )
    if not available(rtl_engine):
        return False, (
            f"Gemmini production codegen smoke cannot run: the selected {rtl_engine} RTL "
            "oracle, RISC-V compiler, or curated harness is unavailable"
        )
    tile = int(DIM)
    cb = {
        "abi_version": "0.1",
        "target": target,
        "tensors": {
            "probe_w": {"shape": [tile, tile], "dtype": "i8", "role": "weight"},
            "probe_a": {"shape": [tile, tile], "dtype": "i8", "role": "input"},
            "probe_y": {"shape": [tile, tile], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {
                "opcode": "RES_PACK",
                "operands": {"src": "probe_w", "dst": "probe_w_res"},
                "attributes": {"layout": "packed_rhs"},
            },
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "probe_a", "rhs": "probe_w_res", "dst": "probe_acc"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "probe_acc", "dst": "probe_y"},
                "attributes": {"epilogue": [], "output_dtype": "i32"},
            },
            {"opcode": "EVICT", "operands": {"handle": "probe_w_res"}},
        ],
    }
    try:
        with tempfile.TemporaryDirectory(prefix="merlin_gemmini_codegen_smoke_") as td:
            result = run_command_buffer(cb, workdir=td, simulator=rtl_engine, timeout=600)
            elf_present = Path(str(result.get("elf") or "")).is_file()
    except Exception as e:  # noqa: BLE001 — this is the failure the launch gate exists to surface
        return False, f"Gemmini production codegen smoke failed: {type(e).__name__}: {str(e)[-240:]}"
    oracle = result.get("oracle") or {}
    output = (result.get("outputs") or {}).get("probe_y")
    if result.get("correct") is not True or not output or not elf_present or oracle.get("derived_from_rtl") is not True:
        return False, (
            "Gemmini production codegen smoke ran but lacked a bit-exact RTL proof "
            f"(correct={result.get('correct')!r}, output={bool(output)}, "
            f"elf={elf_present}, oracle={oracle!r})"
        )
    return True, (
        f"production command-buffer codegen compiled and ran a {tile}x{tile} kernel bit-exact on {rtl_engine} RTL"
    )


def harness_build_recipe():
    """How to compile + link a runner-owned harness against this target's bare-metal environment.

    Declared here, where the target is owned, so the GENERIC contract-compile path can orchestrate the
    build without importing this module. Every value is resolved the same way the backend's own build
    resolves it — the curated harness tree (env-overridable), the riscv-tests include layout, the
    toolchain gcc, and a link script whose ORIGIN is derived from the RTL memory map rather than baked
    into the vendored script. Nothing new is hardcoded: this is the existing recipe, named.
    """
    # Absolute, like the other two imports of this module: the package registers OUT-OF-TREE as
    # `merlin._oot_backends.gemmini`, so a relative `.base` resolves to a sibling that does not
    # exist there and the spike/verilator invocation dies with ModuleNotFoundError at grade time.
    from merlin.runtime.backends import base as backend_base
    from merlin.targetgen.contract.build_recipe import KernelStackFramePolicy
    from merlin.targetgen.contract.harness_abi import for_target

    rt, common = rocc_tests_dir(), _common_dir()
    # OOT discovery registers the backend package while this implementation lives in its submodule;
    # a direct in-tree import registers this module.  Resolve either packaging form structurally.
    try:
        target = backend_base.name_of_module(__package__)
    except KeyError:
        target = backend_base.name_of_module(__name__)
    # The curated CRT reserves 128 KiB for the runtime stack.  Cap the package entry frame at half of
    # that reservation, leaving the other half for the active harness/call chain.  This is a target
    # software-ABI policy, not a mesh-size or model-shape heuristic.
    kernel_stack_frame = KernelStackFramePolicy(
        entry_symbol=for_target(target).entry_symbol, max_static_bytes=64 * 1024
    )
    return backend_base.HarnessBuildRecipe(
        compiler=gcc_path(),
        include_roots=(rt / "riscv-tests", rt / "riscv-tests/env", rt, common),
        support_sources=tuple(sorted(common.glob("*.c"))) + tuple(sorted(common.glob("*.S"))),
        link_script=_test_ld(),
        load_address=platform_dram_base(),
        cflags=(
            "-DPREALLOCATE=1",
            "-DMULTITHREAD=1",
            "-mcmodel=medany",
            "-std=gnu99",
            "-O2",
            "-ffast-math",
            "-fno-common",
            "-fno-builtin-printf",
            "-fno-tree-loop-distribute-patterns",
            "-march=rv64gc",
            "-Wa,-march=rv64gc",
            "-DID_STRING=",
            "-DPRINT_TILE=0",
            "-nostdlib",
            "-nostartfiles",
            "-static",
            "-DBAREMETAL=1",
        ),
        ldflags=("-lm", "-lgcc"),
        error_cls=GemminiError,
        kernel_stack_frame=kernel_stack_frame,
    )


def whole_model_driver():
    """Where this target's WHOLE-MODEL program driver lives (the `whole_model_driver` capability).

    A whole model runs as one bare-metal program with one call per compute group: the package's own
    kernel where it answered the group, and this target's vendor library where it did not. The
    library calls, the timing brackets and the UART protocol that program prints are this target's
    software environment, so the target owns them -- the generic ``merlin.perf.whole_model_build``
    orchestrates and never writes a line of that C itself.

    ``program`` extracts the model's steps, renders and links the C; ``kernels`` binds the package's
    per-group kernel objects into it; ``dispatch`` answers an OPEN model's device dispatches, whose
    host code is the model's own lowered IR. All three live beside this backend in the selected
    support package, so the driver a build loads is the one this provider ships.
    """
    root = Path(__file__).resolve().parent.parent / "whole_model"
    return {
        "program": root / "group_model_program.py",
        "kernels": root / "group_model_submission_kernels.py",
        "dispatch": root / "group_model_dispatch.py",
    }


# --- runner-owned harness rendering (the `harness_renderer` backend capability) ---------------------
# Moved here from the GENERIC contract-compile path, which had to import this module to render a
# harness at all. Both renderers are target-owned for the same underlying reason: they pad to this
# accelerator's tile edge and lay out its accumulator readout. That is codegen, not four declarable
# strings, so it belongs with the backend rather than behind a contract key no second target could
# implement. What the CONTRACT still supplies is the harness ABI (entry symbol, fence, includes,
# metric), read through `harness_abi.for_target` below.
from .gemmini_codegen_mlir import (
    _batched_matmul_harness_c,
    _harness_c,
    _measurement_c_fragments,
    container_for,
    container_words,
    kernel_abi_from_commands,
)


def _is_movement_cb(cb: dict) -> bool:
    cmds = cb.get("commands", [])
    return not any(c.get("opcode") == "RES_PACK" for c in cmds) and any(
        c.get("opcode") == "MOVEMENT"
        or (c.get("opcode") == "VECTOR_MAP" and c.get("attributes", {}).get("combine") == "identity")
        for c in cmds
    )


_NATIVE_INTERFACE_OPS = frozenset({"ATTENTION_QK", "ATTENTION_PV", "BATCHED_MATMUL", "CONV2D"})


def _native_interface_command(cb: dict) -> dict | None:
    """Return the one whole interface op whose LLVM ABI is the interface tensor list.

    The resident-matmul backend has an intentionally different internal ABI
    (weights ++ materialized lhs matrices ++ outputs).  A whole-op artifact does not: its pointers are
    the tensors declared by the interface, including an original NHWC activation rather than a
    runner-materialized im2col matrix.  Keep the distinction explicit so the harness cannot silently
    call one ABI with the other's argument list.
    """
    native = [command for command in cb.get("commands", []) if command.get("opcode") in _NATIVE_INTERFACE_OPS]
    if not native:
        return None
    if len(native) != 1:
        raise CodegenError(f"native interface harness supports exactly one whole op, got {len(native)}")
    return native[0]


def _declared_output_dtype(cb: dict, cmd: dict, dst: str) -> str:
    """The dtype the destination of ``cmd`` is DECLARED to land in: the command's own ``output_dtype``
    attribute, else the declared dtype of the ``dst`` tensor. Raises when neither states one — a movement
    capsule's whole point is the container widening, so an unstated output dtype is not a thing to guess.
    """
    dtype = (cmd.get("attributes") or {}).get("output_dtype") or ((cb.get("tensors") or {}).get(dst) or {}).get("dtype")
    if not dtype:
        raise CodegenError(
            f"movement destination {dst!r} declares no output dtype (neither the command's "
            f"output_dtype attribute nor the tensor's dtype), so its buffer cannot be sized"
        )
    return str(dtype)


def _movement_harness_c(cb: dict, *, target: str, inputs: dict | None = None) -> str:
    """Harness for a pure-movement kernel ``<entry>(src*, dst*)``: embed src, print dst.

    The entry symbol, the fence, the includes and the cycle-window metric are read from ``target``'s
    declared harness ABI rather than written here; see :mod:`.harness_abi` for why they cannot be
    derived. ``target`` is required: a default would be one target's name, which is the weld this is
    removing.
    """
    # Absolute: this package registers OUT-OF-TREE as `merlin._oot_backends.gemmini`, where `..` is the
    # synthetic namespace rather than `merlin.runtime` — the layout these relative imports were written
    # against before the eviction. Left relative they raise ModuleNotFoundError at GRADE time, inside the
    # harness renderer, which the runner reports as an opaque "spike invocation failed".
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.targetgen.contract.harness_abi import for_target

    from .gemmini_codegen import _ceil_dim, _pad_rowmajor

    mv = next(
        c
        for c in cb["commands"]
        if c.get("opcode") == "MOVEMENT"
        or (c.get("opcode") == "VECTOR_MAP" and c.get("attributes", {}).get("combine") == "identity")
    )
    src = mv["operands"].get("src") or mv["operands"].get("lhs")
    dst = mv["operands"]["dst"]
    m, n = cb["tensors"][src]["shape"]
    mp, np_ = _ceil_dim(m), _ceil_dim(n)
    leaves = materialize_inputs(cb, inputs)
    sp = _pad_rowmajor(list(leaves[src].data), m, n, mp, np_)
    # The destination is allocated from the DECLARED OUTPUT dtype, not from the operand dtype. Movement
    # is a container widening (operand dtype in, accumulate dtype out), so pinning the destination to
    # `elem_t` under-allocated it by the width ratio: two shipped capsules declare an i32 output, and a
    # CORRECT 4-byte store into a 1-byte-per-element buffer ran ~700 bytes off the end of .bss and
    # trapped AFTER printing DONE — a harness overrun reported as a failure of the submission.
    odt = _declared_output_dtype(cb, mv, dst)
    # Both containers come from the DECLARED dtype's own storage width (`container_for`), so the
    # source is embedded and the destination read at the width the kernel actually moves. A float
    # operand is carried as its stored bit pattern in an unsigned container of the same width — the
    # readback decodes it back to a value from this same declared dtype — rather than being refused
    # for having no "integer" spelling.
    src_container = container_for(str(cb["tensors"][src].get("dtype") or ""))
    dst_container = container_for(odt)
    decls = [
        src_container.decl(
            f"T_{src}",
            mp * np_,
            const=True,
            initializer=",".join(str(w) for w in container_words(sp, str(cb["tensors"][src].get("dtype") or ""))),
        ),
        dst_container.decl(f"T_{dst}", mp * np_),
    ]
    prints = [
        f'  printf("OUT {dst} {m} {n}");',
        f"  for (long i = 0; i < {m}; i++) for (long j = 0; j < {n}; j++)"
        f" {dst_container.printf_element(f'T_{dst}[i * {np_} + j]')}",
        '  printf("\\n");',
    ]
    # Print METRIC cycles BEFORE the (possibly huge) OUT tensor dump: large-output kernels flood the
    # UART and the per-ELF capture truncates mid-dump, so a trailing METRIC line would be lost. Emitting
    # the (tiny) cycle metric first guarantees it is always captured; the OUT dump follows for correctness.
    abi = for_target(target)
    window = abi.cycle_window_line()
    measured_call = abi.call(f"(void*)T_{src}, (void*)T_{dst}")
    fragments = _measurement_c_fragments(measured_call)
    return (
        "#include <stdint.h>\n#include <stdio.h>\n"
        + abi.declarations()
        + "\n"
        + fragments["include"]
        + "\n".join(decls)
        + "\nint main() {\n"
        + fragments["warmup"]
        + fragments["prologue"]
        + "  uint64_t c0 = read_cycles();\n"
        + measured_call
        + "\n"
        + "  uint64_t c1 = read_cycles();\n"
        + fragments["epilogue"]
        + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
        + (window + "\n" if window else "")
        + "\n".join(prints)
        + "\n"
        '  printf("DONE\\n");\n  return 0;\n}\n'
    )


def _flat_matrix_shape(spec: dict, *, name: str) -> tuple[int, int]:
    """Flatten leading logical dimensions into rows, preserving the final dimension as columns.

    Gemmini's DRAM-facing matrix layout pads both dimensions to its tile edge.  For NHWC this means
    one physical row per pixel and, critically, a padded physical C stride; compact NHWC bytes do not
    match the addresses emitted by a tiled whole-convolution artifact.
    """
    shape = spec.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) < 2
        or any(not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0 for dim in shape)
    ):
        raise CodegenError(f"native interface tensor {name!r} needs a positive rank >= 2 shape, got {shape!r}")
    rows = 1
    for dim in shape[:-1]:
        rows *= dim
    return rows, shape[-1]


def _native_interface_harness_c(cb: dict, command: dict, *, inputs: dict | None = None) -> str:
    """Harness for a schema-native whole op with the interface's pointer ABI.

    The tensor table is emitted by the interface parser in declaration order.  Preserve that order
    exactly: unlike the in-tree resident-matmul emitter, a package artifact receives neither reordered
    weights nor codegen-only derived buffers.  Every physical buffer is a zero-padded matrix whose last
    logical dimension is its padded row stride.
    """
    from merlin.runtime.commandbuffer import materialize_inputs

    from .gemmini_codegen import _ceil_dim, _pad_rowmajor

    tensors = cb.get("tensors") or {}
    opcode = command.get("opcode")
    if opcode == "BATCHED_MATMUL":
        # This helper validates the exact a/w/dst interface, preserves rank-N output, and uses the
        # identical target-padded buffer geometry as the direct MLIR emitter.
        return _batched_matmul_harness_c(cb, inputs=inputs)
    operands = command.get("operands") or {}
    packs = {
        item.get("operands", {}).get("dst"): item.get("operands", {}).get("src")
        for item in cb.get("commands", [])
        if item.get("opcode") == "RES_PACK"
    }
    if opcode == "ATTENTION_QK":
        required = [operands.get("q"), operands.get("k"), operands.get("dst")]
    elif opcode == "ATTENTION_PV":
        required = [operands.get("p"), operands.get("v"), operands.get("dst")]
    elif opcode == "CONV2D":
        weight = operands.get("weight")
        required = [operands.get("ifm"), packs.get(weight, weight), operands.get("dst")]
    else:  # pragma: no cover - caller selects only _NATIVE_INTERFACE_OPS
        raise CodegenError(f"unsupported native interface opcode {opcode!r}")
    if any(not isinstance(name, str) or not name for name in required):
        raise CodegenError(f"native interface {opcode} has incomplete operands {operands!r}")
    missing = [name for name in required if name not in tensors]
    if missing:
        raise CodegenError(f"native interface {opcode} operand(s) {missing} have no declared tensor buffer")

    # Dict insertion order is the only representation of interface declaration order retained by the
    # command-buffer JSON.  Restrict it to external buffers, then prove every whole-op operand is present;
    # do not reconstruct a plausible order from roles (CONV commonly declares IFM before its weight).
    external_roles = {"input", "weight", "bias", "output"}
    args = [name for name, spec in tensors.items() if spec.get("role") in external_roles]
    if any(name not in args for name in required):
        raise CodegenError(f"native interface {opcode} operands are not all declared external buffers: {required!r}")

    leaves = materialize_inputs(cb, inputs)
    output_names = [name for name in args if tensors[name].get("role") == "output"]
    if operands["dst"] not in output_names:
        raise CodegenError(f"native interface destination {operands['dst']!r} is not a declared output tensor")

    decls: list[str] = []
    layouts: dict[str, tuple[int, int, int, int]] = {}
    for name in args:
        spec = tensors[name]
        rows, cols = _flat_matrix_shape(spec, name=name)
        prows, pcols = _ceil_dim(rows), _ceil_dim(cols)
        layouts[name] = (rows, cols, prows, pcols)
        if spec.get("role") == "output":
            dtype = str(spec.get("dtype") or "")
            decls.append(container_for(dtype).decl(f"T_{name}", prows * pcols))
            continue
        if spec.get("dtype") != "i8":
            raise CodegenError(
                f"native interface input {name!r} must use the target operand dtype i8, got {spec.get('dtype')!r}"
            )
        if name not in leaves:
            raise CodegenError(f"native interface input {name!r} was not materialized")
        padded = _pad_rowmajor(list(leaves[name].data), rows, cols, prows, pcols)
        decls.append(
            f"static const elem_t T_{name}[{prows * pcols}] row_align(1) = "
            f"{{{','.join(str(int(value)) for value in padded)}}};"
        )

    call = ", ".join(f"(void*)T_{name}" for name in args)
    prints: list[str] = []
    for name in output_names:
        shape = tensors[name].get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            raise CodegenError(f"native interface output {name!r} must be a rank-2 flattened tensor, got {shape!r}")
        rows, cols, _, pcols = layouts[name]
        container = container_for(str(tensors[name].get("dtype") or ""))
        prints.extend(
            [
                f'  printf("OUT {name} {rows} {cols}");',
                f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
                f" {container.printf_element(f'T_{name}[i * {pcols} + j]')}",
                '  printf("\\n");',
            ]
        )

    measured_call = f"  gemmini_kernel({call});\n  gemmini_fence();"
    fragments = _measurement_c_fragments(measured_call)
    return (
        '#include <stdint.h>\n#include <stdio.h>\n#include "include/gemmini_testutils.h"\n'
        + fragments["include"]
        + "extern void gemmini_kernel();\n"
        + "\n".join(decls)
        + "\nint main() {\n"
        + fragments["warmup"]
        + fragments["prologue"]
        + "  uint64_t c0 = read_cycles();\n"
        + measured_call
        + "\n"
        + "  uint64_t c1 = read_cycles();\n"
        + fragments["epilogue"]
        + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
        '  printf("METRIC cycle_window_gemmini_region 1\\n");\n' + "\n".join(prints) + "\n"
        '  printf("DONE\\n");\n  return 0;\n}\n'
    )


#: External buffer roles — the tensors a harness allocates and passes to the kernel. An intermediate
#: (a scratchpad handle, an accumulator) is produced by a command and never appears in the ABI.
_EXTERNAL_ROLES = ("input", "weight", "bias", "output")


def _buffer_extent(spec: dict, *, name: str) -> tuple[int, int]:
    """``(rows, cols)`` for a host-lane buffer of ANY rank: leading extents are rows, the last is
    columns, and a rank-1 tensor is one row.

    Deliberately NOT :func:`_flat_matrix_shape`, whose rank >= 2 requirement is a real constraint of
    the whole-op path (an NHWC activation with no channel dimension is a malformed conv operand) and
    is no constraint at all here. A host-lane program routinely carries a rank-1 leaf -- a layernorm's
    per-channel weight and bias are exactly that -- and refusing one made two capsules fail with
    "needs a positive rank >= 2 shape", which reads as a defect in the submission that declared a
    perfectly ordinary vector.

    The split is the same one the emitting side makes (leading dims multiply into rows, the last is
    the row pitch), so the harness and the kernel agree on the layout for every rank.
    """
    from .gemmini_codegen import _build_support

    return _build_support.format.buffer_extent(spec, name=name)


def _is_host_lane_cb(cb: dict) -> bool:
    """Does this buffer describe a program with NO accelerator command?

    Structural, not a claim about intent and not keyed on any ``params`` spelling: a buffer that
    declares tensors including at least one output, and declares no command at all, is a program the
    compiler routed entirely onto the host lane. There is nothing for the accelerator to do and
    nothing for the harness to schedule — only buffers in, buffers out.

    This is deliberately distinct from the zero-tensor, zero-command CALIBRATION buffer (which has no
    output to read back and goes on to the tiled path), and from a declined buffer (which never
    reaches a harness at all: the runner reads ``declined`` before it compiles anything).
    """
    if cb.get("commands"):
        return False
    tensors = cb.get("tensors") or {}
    return any((spec or {}).get("role") == "output" for spec in tensors.values())


def _strict_warm_profile_renderer(*, target: str, warm_profile):
    """Bind the generic final-profile ordering to this target's declared ABI hooks."""
    from merlin.perf.warm_profile_harness import (
        render_warm_then_measure_main,
        require_strict_final_warm_profile,
    )
    from merlin.targetgen.contract.harness_abi import for_target

    contract = require_strict_final_warm_profile(warm_profile)
    abi = for_target(target)

    def render(*, arguments: str, readback: str) -> dict[str, str]:
        success = readback + ("\n" if readback else "") + 'printf("DONE\\n");'
        return {
            "declarations": abi.declarations(),
            "main": render_warm_then_measure_main(
                prepare_input="/* ABI inputs are statically initialized before main. */",
                invocation=abi.warm_profile_invocation(arguments),
                validate_outputs="0",  # OUT parsing + golden validation are runner-owned.
                contract=contract,
                cycle_reader="read_cycles",
                success_body=success,
            ),
        }

    return render


def _whole_program_harness_c(
    cb: dict, *, target: str, inputs: dict | None = None, prepack_authorizations=None, warm_profile=None
) -> str:
    """Legacy entrypoint: delegate to the one pure renderer with unchanged policy."""
    from merlin.runtime.commandbuffer import materialize_inputs

    from .gemmini_codegen import _build_support, _ceil_dim, _pad_rowmajor

    return _build_support.render_whole_program(
        cb,
        inputs=inputs,
        prepack_authorizations=prepack_authorizations,
        legacy_helpers=(_ceil_dim, _pad_rowmajor, _buffer_extent, materialize_inputs),
        measurement_fragments=_measurement_c_fragments,
        strict_profile_renderer=(
            _strict_warm_profile_renderer(target=target, warm_profile=warm_profile)
            if warm_profile is not None
            else None
        ),
    )


def build_source_paths():
    """Exact pure target renderer closure, in addition to the legacy backend source."""
    from .gemmini_codegen import _build_support

    return _build_support.build_source_paths()


def _host_lane_harness_c(cb: dict, *, target: str, inputs: dict | None = None) -> str:
    """Harness for a program that runs entirely on the host lane: embed the leaves, print the outputs.

    A region this datapath admits no lowering for belongs on the CPU lane, and a compiler that says so
    and generates the CPU-lane program has done the right thing. Until this existed the right thing was
    not deliverable: the only harnesses here were the tiled resident-matmul one (which REFUSES a buffer
    with no ``RES_PACK``/matmul/commit) and the movement one (which needs a movement command), so a
    correct host-lane program had no shape to be handed back in at all. A compiler had to fabricate a
    matmul it never ran just to get a buffer allocated — which is a false claim about the mesh sitting
    in the command buffer, and it also breaks the operand attachment (the fabricated carrier tensors
    outnumber the interface's real leaves, so the canonical stimulus no longer binds by position).

    Everything is derived from the buffer's own declarations: the pointer ABI is the tensor
    DECLARATION order restricted to external buffers (the same rule the whole-op harness uses, and
    what a positional binder on the emitting side produces), each buffer is sized and printed through
    :func:`container_for` at its declared dtype, and rows are padded to the accelerator's tile edge so
    the layout is the one every other harness here uses.
    """
    from merlin.runtime.commandbuffer import materialize_inputs
    from merlin.targetgen.contract.harness_abi import for_target

    from .gemmini_codegen import _ceil_dim, _pad_rowmajor

    tensors = cb.get("tensors") or {}
    args = [name for name, spec in tensors.items() if (spec or {}).get("role") in _EXTERNAL_ROLES]
    outputs = [name for name in args if tensors[name].get("role") == "output"]
    if not outputs:
        raise CodegenError("a host-lane buffer declares no output tensor to read back")
    # Extents FIRST, before anything materializes. `materialize_inputs` builds a Tensor per leaf and
    # raises its own shape error, so a buffer with a malformed extent was refused by the wrong voice
    # ("shape (0, 4) needs 0 elements") and this function's own check could never fire -- a guard that
    # cannot fail is not a guard.
    extents = {name: _buffer_extent(tensors[name], name=name) for name in args}
    leaves = materialize_inputs(cb, inputs)

    decls: list[str] = []
    prints: list[str] = []
    for name in args:
        spec = tensors[name]
        dtype = str(spec.get("dtype") or "")
        rows, cols = extents[name]
        prows, pcols = _ceil_dim(rows), _ceil_dim(cols)
        container = container_for(dtype)
        if spec.get("role") == "output":
            decls.append(container.decl(f"T_{name}", prows * pcols))
            prints.extend(
                [
                    f'  printf("OUT {name} {rows} {cols}");',
                    f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
                    f" {container.printf_element(f'T_{name}[i * {pcols} + j]')}",
                    '  printf("\\n");',
                ]
            )
            continue
        if name not in leaves:
            raise CodegenError(f"host-lane input {name!r} was not materialized")
        padded = _pad_rowmajor(list(leaves[name].data), rows, cols, prows, pcols)
        decls.append(
            container.decl(
                f"T_{name}",
                prows * pcols,
                const=True,
                initializer=",".join(str(w) for w in container_words(padded, dtype)),
            )
        )

    abi = for_target(target)
    window = abi.cycle_window_line()
    measured_call = abi.call(", ".join(f"(void*)T_{name}" for name in args))
    fragments = _measurement_c_fragments(measured_call)
    # METRIC before the OUT dump, for the same reason as every other harness here: a large output
    # floods the console and a trailing metric line is what gets truncated away.
    return (
        "#include <stdint.h>\n#include <stdio.h>\n"
        + abi.declarations()
        + "\n"
        + fragments["include"]
        + "\n".join(decls)
        + "\nint main() {\n"
        + fragments["warmup"]
        + fragments["prologue"]
        + "  uint64_t c0 = read_cycles();\n"
        + measured_call
        + "\n"
        + "  uint64_t c1 = read_cycles();\n"
        + fragments["epilogue"]
        + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
        + (window + "\n" if window else "")
        + "\n".join(prints)
        + "\n"
        '  printf("DONE\\n");\n  return 0;\n}\n'
    )


def prepare_compact_caller(cb, compact_contract, logical_payloads, **kwargs):
    """Optional trusted-host compact allocation preparation; no target execution."""
    from .gemmini_compact_caller import prepare_compact_caller as prepare

    return prepare(cb, compact_contract, logical_payloads, **kwargs)


def verify_compact_caller_link(cb, prepared, **kwargs):
    """Check the linked target allocations, without asserting kernel numerics."""
    from .gemmini_compact_caller import verify_compact_caller_link as verify

    return verify(cb, prepared, **kwargs)


def render_harness(
    cb: dict,
    *,
    target: str,
    inputs: dict | None = None,
    prepack_authorizations=None,
    compact_caller=None,
    warm_profile=None,
) -> str:
    """Render the runner-owned harness for ``cb`` — the `harness_renderer` capability.

    Chooses between the pure-movement and tiled forms itself, because which one applies is a property
    of this target's command vocabulary rather than something the generic path can decide.

    ``inputs`` (name -> nested-list) INJECTS explicit operand values so the DEVICE computes on the same
    data the reference and the simulator were given. Without it the harness materializes each leaf from
    its NAME, so a caller injecting real activations got a three-way gate in which the reference and the
    simulator saw the injected operands and the device saw different ones -- guaranteed to mismatch, and
    reported as a functional failure of the target.
    """
    whole_program = (cb.get("kernel_abi") or {}).get("kind") == "whole_program"
    if warm_profile is not None and not whole_program:
        raise CodegenError("strict warm profiling is available only for an explicit whole-program kernel ABI")
    if compact_caller is not None:
        if inputs is not None or prepack_authorizations is not None:
            raise CodegenError("compact caller accepts only its already validated explicit byte inputs")
        from .gemmini_compact_caller import render_compact_caller

        return render_compact_caller(cb, compact_caller, target=target, warm_profile=warm_profile)
    if whole_program:
        return _whole_program_harness_c(
            cb, target=target, inputs=inputs, prepack_authorizations=prepack_authorizations, warm_profile=warm_profile
        )
    if prepack_authorizations is not None:
        raise CodegenError("host prepack authorization requires the explicit whole-program caller")
    if _is_host_lane_cb(cb):
        return _host_lane_harness_c(cb, target=target, inputs=inputs)
    if _is_movement_cb(cb):
        return _movement_harness_c(cb, target=target, inputs=inputs)
    native = _native_interface_command(cb)
    if native is not None:
        return _native_interface_harness_c(cb, native, inputs=inputs)
    return _harness_c(cb, inputs)
