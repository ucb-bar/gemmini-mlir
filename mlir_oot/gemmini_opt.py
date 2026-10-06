"""`gemmini-opt` — the interface pipeline and explicit golden compiler exports.

    gemmini-opt --verify-diagnostics <in.mlir>
    gemmini-opt --convert-iface-to-gemmini <in.mlir>
    gemmini-opt --convert-iface-to-gemmini --emit-command-buffer=<out.json> <in.mlir>
    gemmini-opt --convert-iface-to-gemmini --emit-target-artifact <in.mlir>
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from xdsl.printer import Printer                                       # noqa: E402

from mlir_oot import cmdbuf                                            # noqa: E402
from mlir_oot.codegen import gemmini_module, llvm_emit                 # noqa: E402
from mlir_oot.frontend import linalg_reader, parse as _parse, reader   # noqa: E402
from mlir_oot.lowering.plan import Builder, LoweringDeclined           # noqa: E402
from mlir_oot.lowering import host_lane, model_lane                     # noqa: E402
from mlir_oot.lowering.schedule import schedule                        # noqa: E402


def _print(op) -> str:
    out = io.StringIO()
    Printer(stream=out).print_op(op)
    return out.getvalue() + "\n"


class Pipeline:
    """One run of the backend over one interface module."""

    def __init__(self, text: str):
        self.text = text
        self.module = None
        self.plan = None
        self.instrs = None
        self.staging = None
        self.declined = None
        self.linalg = None
        self.artifact = None
        self.mixed_declined = None

    def parse(self):
        self.module = _parse.parse_module(self.text)
        return self.module

    def lower(self):
        if self.module is None:
            self.parse()
        if not _parse.is_merlin_iface(self.module):
            wl = linalg_reader.read(self.module)
            self.linalg = wl
            if wl.mesh_regions:
                # Some region of this module IS work the mesh admits.  Lower it as a mixed-lane
                # program; leaving admitted work on the host would be a placement defect.
                try:
                    plan = model_lane.build(self.module, wl)
                    instrs, staging = schedule(plan)
                    # Prove the program EMITS before committing to it.  Every entrypoint has to
                    # answer the same way about the same capsule: a command buffer that says a
                    # program exists next to an artifact that could not be built is the one
                    # inconsistency the runner reads as a protocol failure rather than a decline.
                    self.artifact = llvm_emit.emit(plan, instrs, staging)
                    self.plan, self.instrs, self.staging = plan, instrs, staging
                    return self.plan
                except LoweringDeclined as exc:
                    # Not a program this backend can build after all -- fall through to the
                    # host-lane form, which STATES why rather than emitting nothing.
                    self.mixed_declined = exc.reason
            self.plan = host_lane.build(wl)
            self.staging = {}
            try:
                self.instrs = host_lane.host_instrs(self.plan, self.module, wl)
                # Same rule as the mixed path: prove the artifact EMITS before the command buffer
                # claims a program exists.  A buffer with commands beside an empty artifact is the
                # one inconsistency the runner cannot read as a decline.  A buffer that ALREADY
                # declines claims nothing, so it costs nothing to prove -- and for a host-lane
                # region the artifact is megabytes, which is time the command-buffer entrypoint
                # should not spend twice.
                if self.plan.command_buffer.get("commands"):
                    self.artifact = llvm_emit.emit(self.plan, self.instrs, self.staging)
            except LoweringDeclined as exc:
                cb = self.plan.command_buffer
                cb["commands"] = []
                cb.setdefault("params", {})["host_lane_program_emitted"] = False
                cb["declined"] = {
                    "reason": exc.reason,
                    "op": exc.op or (wl.regions[0].op if wl.regions else "linalg_on_tensors"),
                    "shape": list(exc.shape) or (list(wl.results[0][0]) if wl.results else [])}
                self.instrs, self.artifact = [], None
            if self.mixed_declined:
                cb = self.plan.command_buffer
                cb.setdefault("params", {})["mesh_lowering_declined"] = self.mixed_declined
                if not cb.get("commands") and "declined" not in cb:
                    cb["declined"] = {
                        "reason": self.mixed_declined,
                        "op": wl.regions[0].op or "linalg_on_tensors",
                        "shape": list(wl.results[0][0]) if wl.results else []}
            return self.plan
        self.plan = Builder(reader.read(self.module)).build()
        self.instrs, self.staging = schedule(self.plan)
        return self.plan

    def run(self):
        try:
            self.lower()
        except LoweringDeclined as exc:
            target = "gemmini"
            if self.module is not None:
                attr = self.module.attributes.get("merlin_iface.target")
                target = getattr(attr, "data", target)
            self.declined = cmdbuf.declined(target, exc.reason, op=exc.op, shape=exc.shape)
        return self


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gemmini-opt", add_help=True)
    ap.add_argument("--verify-diagnostics", action="store_true")
    ap.add_argument("--convert-iface-to-gemmini", action="store_true")
    ap.add_argument("--emit-target-artifact", action="store_true")
    ap.add_argument("--emit-command-buffer", default=None, metavar="PATH")
    ap.add_argument("--export-golden-contraction", action="store_true")
    ap.add_argument("--optimize-golden-contraction", action="store_true")
    ap.add_argument("--calibration", type=Path)
    ap.add_argument("--export-golden-capture", action="store_true")
    ap.add_argument("--emit-golden-inventory", metavar="PATH")
    ap.add_argument("--region")
    ap.add_argument("--llvm-bin", type=Path)
    ap.add_argument("--workdir", type=Path)
    ap.add_argument("--large-n", action="store_true")
    ap.add_argument("--prefetch-b", action="store_true")
    ap.add_argument("--dense-input-policy", choices=("banked_command_cost","resident_a_command_cost","transfer_command_cost","resident_a_prefetch"))
    ap.add_argument("--dense-b-slot-policy", choices=("remaining_rows",))
    ap.add_argument("--flat-spatial", action="store_true")
    ap.add_argument("--virtual-padding", action="store_true")
    ap.add_argument("--exact-integer-readout", action="store_true")
    ap.add_argument("--banked-prefetch", action="store_true")
    ap.add_argument("--grouped-b", action="store_true")
    ap.add_argument("--separate-b-bank", action="store_true")
    ap.add_argument("--resident-input-policy", choices=("compact_channel_planes","compact_channel_planes_prefetch_b"))
    ap.add_argument("--resident-stripes", action="store_true")
    ap.add_argument("--source-stride-resident", action="store_true")
    ap.add_argument("--source-stride-row-residue", action="store_true")
    ap.add_argument("--resident-a-load-coalescing", action="store_true")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("input", nargs="?", default="-")
    args = ap.parse_args(argv)

    exports=(args.export_golden_contraction,args.optimize_golden_contraction,
             args.export_golden_capture,bool(args.emit_golden_inventory))
    if sum(exports)>1:
        ap.error('select one golden export command')
    compilation_options=(args.region,args.llvm_bin,args.workdir,args.large_n,args.prefetch_b,
                         args.dense_input_policy,args.dense_b_slot_policy,args.calibration)
    capture_options=(args.flat_spatial,args.virtual_padding,args.exact_integer_readout,
                     args.banked_prefetch,args.grouped_b,args.separate_b_bank,
                     args.resident_input_policy,args.resident_stripes,args.source_stride_resident,args.source_stride_row_residue,args.resident_a_load_coalescing)
    if not any(exports) and any((*compilation_options,*capture_options)):
        ap.error('golden options require an explicit golden export command')
    if any(exports) and any((args.verify_diagnostics,args.convert_iface_to_gemmini,
                             args.emit_target_artifact,args.emit_command_buffer,args.output)):
        ap.error('golden export commands cannot mix with interface pipeline options')
    if args.emit_golden_inventory and (args.input!='-' or any((*compilation_options,*capture_options))):
        ap.error('inventory export accepts no compilation options or input')
    if (args.export_golden_contraction or args.optimize_golden_contraction) and any(capture_options):
        ap.error('capture schedule options require --export-golden-capture')
    if args.calibration is not None and not args.optimize_golden_contraction:
        ap.error('calibration requires --optimize-golden-contraction')
    if args.optimize_golden_contraction and any((args.large_n,args.prefetch_b,
                                                args.dense_input_policy,args.dense_b_slot_policy)):
        ap.error('optimized alternatives derive options from their pinned exports')
    if args.export_golden_capture and any((args.region,args.large_n,args.prefetch_b)):
        ap.error('contraction identity and schedule options require --export-golden-contraction')
    if any(exports):
        import json
        from .golden_compiler_export import export_inventory,export_contraction,export_capture
        try:
            if args.emit_golden_inventory:
                result=export_inventory(_HERE.parent)
                Path(args.emit_golden_inventory).write_text(json.dumps(result,indent=2)+'\n')
                print(json.dumps(dict(surfaces=len(result['package_inventory']['surfaces']),
                                      inventory=args.emit_golden_inventory)))
            else:
                if args.input=='-' or args.llvm_bin is None or args.workdir is None:
                    raise ValueError('golden compilation requires an input file, --llvm-bin and --workdir')
                if args.export_golden_contraction or args.optimize_golden_contraction:
                    if not args.region:raise ValueError('contraction export requires --region identity')
                    if args.optimize_golden_contraction:
                        from .golden_calibrated_plan import optimize_contraction
                        if args.calibration is None:raise ValueError('optimized contraction requires --calibration')
                        result=optimize_contraction(Path(args.input),args.region,args.llvm_bin,
                                                   args.workdir,args.calibration)
                    else:
                        result=export_contraction(Path(args.input),args.region,args.llvm_bin,args.workdir,
                            large_n=args.large_n,prefetch_b=args.prefetch_b,
                            dense_input_policy=args.dense_input_policy,dense_b_slot_policy=args.dense_b_slot_policy)
                    print(json.dumps(dict(kernel_symbol=result['kernel_symbol'],
                        object_sha256=result['compilation']['object_sha256'],
                        selected_plan_controls_emitted_code=result['selected_plan_controls_emitted_code'],
                        receipt=str(args.workdir/('golden_optimized_plan.json'
                            if args.optimize_golden_contraction else 'golden_export.json')))))
                else:
                    result=export_capture(Path(args.input),args.llvm_bin,args.workdir,
                        flat_spatial=args.flat_spatial,virtual_padding=args.virtual_padding,
                        exact_integer_readout=args.exact_integer_readout,
                        banked_prefetch=args.banked_prefetch,grouped_b=args.grouped_b,
                        separate_b_bank=args.separate_b_bank,resident_input_policy=args.resident_input_policy,
                        resident_stripes=args.resident_stripes,source_stride_resident=args.source_stride_resident,source_stride_row_residue=args.source_stride_row_residue,dense_input_policy=args.dense_input_policy,
                        dense_b_slot_policy=args.dense_b_slot_policy,resident_a_load_coalescing=args.resident_a_load_coalescing)
                    print(json.dumps(dict(routes=len(result['bundle']['routes']),
                        object_sha256=result['bundle']['object_sha256'],
                        shared_solver_selected=False,receipt=str(args.workdir/'requant.json'))))
            return 0
        except Exception as exc:
            print(f'golden export: {type(exc).__name__}: {exc}',file=sys.stderr)
            return 1

    text = sys.stdin.read() if args.input == "-" else Path(args.input).read_text()
    pipe = Pipeline(text)

    # ---- parse + verify ------------------------------------------------------------------
    try:
        pipe.parse()
    except Exception as exc:                                          # noqa: BLE001
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.verify_diagnostics and not (args.convert_iface_to_gemmini
                                        or args.emit_command_buffer
                                        or args.emit_target_artifact):
        return 0

    pipe.run()

    # ---- command buffer ------------------------------------------------------------------
    if args.emit_command_buffer:
        cb = pipe.declined if pipe.declined is not None else pipe.plan.command_buffer
        problems = cmdbuf.write(cb, args.emit_command_buffer)
        for p in problems:
            print(f"command-buffer: {p}", file=sys.stderr)
        if problems:
            return 1
        if not args.emit_target_artifact and not args.convert_iface_to_gemmini:
            return 0

    if pipe.declined is not None:
        entry = pipe.declined["declined"]
        print(f"declined: {entry['reason']}", file=sys.stderr)
        if not (args.convert_iface_to_gemmini or args.emit_target_artifact):
            return 0 if args.emit_command_buffer else 1
        # A DECLINE still has to be answered in the entrypoint's own language.  An empty stdout
        # here is read as a crashed tool; a module that carries the reason and no command is the
        # same refusal, stated where the runner can read it.
        module = (llvm_emit.declined_artifact(entry["reason"]) if args.emit_target_artifact
                  else gemmini_module.declined_module(entry["reason"], entry.get("op", ""),
                                                      entry.get("shape")))
        text_out = _print(module)
        if args.output:
            Path(args.output).write_text(text_out)
        else:
            sys.stdout.write(text_out)
        return 0

    # ---- target artifact -----------------------------------------------------------------
    try:
        if args.emit_target_artifact:
            module = (pipe.artifact if pipe.artifact is not None
                      else llvm_emit.emit(pipe.plan, pipe.instrs, pipe.staging))
        elif args.convert_iface_to_gemmini:
            module = gemmini_module.build(pipe.plan, pipe.instrs, pipe.staging)
        else:
            return 0
    except LoweringDeclined as exc:
        # A stage that only CODEGEN can refuse (a buffer with no address, say) still has to
        # arrive as a stated decline -- and it has to arrive the SAME WAY at every entrypoint.
        # Printing nothing here used to leave the command buffer already written as a program
        # with real commands while stdout stayed empty and the exit code was 1: three
        # entrypoints claiming the capsule lowers and the fourth reading as a crashed tool.
        # The refusal is restated on the command buffer (the place the contract puts it) and
        # answered in this entrypoint's own language.
        print(f"declined: {exc.reason}", file=sys.stderr)
        entry = cmdbuf.declined("gemmini", exc.reason, op=exc.op, shape=exc.shape)
        if args.emit_command_buffer:
            cmdbuf.write(entry, args.emit_command_buffer)
        module = (llvm_emit.declined_artifact(exc.reason) if args.emit_target_artifact
                  else gemmini_module.declined_module(exc.reason, exc.op, exc.shape))
        text_out = _print(module)
        if args.output:
            Path(args.output).write_text(text_out)
        else:
            sys.stdout.write(text_out)
        return 0

    text_out = _print(module)
    if args.output:
        Path(args.output).write_text(text_out)
    elif not args.emit_command_buffer or args.emit_target_artifact or True:
        sys.stdout.write(text_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
