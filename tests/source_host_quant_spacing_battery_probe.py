"""Cross consumer spacing while keeping typed quantizer packet geometry fixed.

Calibration driver, not a production rewrite. Merlin proves and emits the
source map; the owned RV64 experiment permutes only the emitter's independent
three-instruction clamp/conversion chains. All original forecasts stay frozen.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from current_host_quant_fixture_probe import pin
from merlin.perf.layer_bench import build_program
from source_host_quant_model_battery_probe import compile_map, c_source, evaluate, rebind_extent
from source_host_quant_transfer_battery_probe import contents

from mlir_oot.frontend.parse import parse_module
from mlir_oot.no_fsm_audit import audit_elf


def reschedule_packet(source, *, group):
    """Accept only the independently proved eight-lane RV64 emission."""
    if type(group) is not int or group not in (1,8):
        raise ValueError("predeclared spacing groups are one/eight")
    stages = [
        [f"fmax.s ft{i}, ${8+i}, $16" for i in range(8)],
        [f"fmin.s ft{i}, ft{i}, $17" for i in range(8)],
        [f"fcvt.w.s ${i}, ft{i}, rne" for i in range(8)],
    ]
    original = r"\0A".join(instruction for stage in stages for instruction in stage)
    replacement = r"\0A".join(instruction for first in range(0,8,group) for stage in stages for instruction in stage[first:first+group])
    pattern = 'asm "'+original+'", "'+','.join(["=r"]*8+["f"]*10+[f"~{{ft{i}}}" for i in range(8)])+'"'
    if source.count(pattern) != 1:
        raise ValueError("complete authoritative packet/constraints absent or ambiguous")
    # No opcode/operand/rounding/clobber or within-lane order changes.
    old_instructions, new_instructions = original.split(r"\0A"), replacement.split(r"\0A")
    if sorted(old_instructions) != sorted(new_instructions):
        raise ValueError("instruction multiset changed")
    for lane in range(8):
        positions = [new_instructions.index(stage[lane]) for stage in stages]
        if positions != sorted(positions):
            raise ValueError("source lane dependency order changed")
    changed = source.replace(pattern,pattern.replace(original,replacement),1)
    # All surrounding LLVM, constraints and arithmetic producers stay exact.
    if changed.replace(replacement,original,1) != source:
        raise ValueError("nonpacket source change")
    return changed, {"group":group,"lanes":8,"instructions":24,"opcodes":{"fmax.s":8,"fmin.s":8,"fcvt.w.s":8},"within_lane_order_preserved":True,"all_other_source_bytes_identical":True,"same_constraints":True,"source_numeric_policy":"Merlin complete typed bounded-RNE proof; nontrapping and unobserved floating flags; ambient FMUL producers unchanged"}


def generate(args):
    work=args.workdir.resolve()
    work.mkdir(parents=True,exist_ok=False)
    original=parse_module((args.fixture/"source.mlir").read_text())
    binding=json.loads((args.fixture/"receipt.json").read_text())
    shape=binding["input_shape"]
    if shape != [1,3,224,224]:
        raise ValueError("original bound source fixture required")
    source_input=np.fromfile(args.fixture/"input.bin",dtype=np.float32).reshape(shape)
    shutil.copyfile(args.fixture/"input.bin",work/"original_input.bin")
    cases, objects=[],[]
    for extent in (112,224,448):
        module, proof=rebind_extent(original,input_axis=2,extent=extent)
        selected=source_input[:,:,np.arange(extent)%shape[2],:].copy()
        height=work/f"height{extent}"
        compile_map(module,height/"oracle",args.llvm_bin)
        oracle=evaluate(height/"oracle",selected,proof["output_shape"])
        if extent==224 and oracle.tobytes()!=(args.fixture/"expected.bin").read_bytes():
            raise ValueError("original current source oracle changed")
        (height/"input.bin").write_bytes(selected.tobytes())
        (height/"expected.bin").write_bytes(oracle.tobytes())
        for spacing in (1,8):
            arm=height/f"spacing{spacing}"
            implementation=compile_map(module,arm,args.llvm_bin,8)
            if evaluate(arm,selected,proof["output_shape"]).tobytes()!=oracle.tobytes():
                raise ValueError("complete native scheduled source differs")
            target=(arm/"target.ll").read_text()
            (arm/"original_staged_target.ll").write_text(target)
            shutil.copyfile(arm/"kernel.o",arm/"original_staged_kernel.o")
            changed, schedule=reschedule_packet(target,group=spacing)
            (arm/"target.ll").write_text(changed)
            subprocess.run([str(args.llvm_bin/"clang"),"--target=riscv64-unknown-elf","-march=rv64gc","-mabi=lp64d","-mcmodel=medany","-O2","-ffreestanding","-fno-builtin","-c",str(arm/"target.ll"),"-o",str(arm/"kernel.o")],check=True)
            if spacing==8 and (arm/"kernel.o").read_bytes()!=(arm/"original_staged_kernel.o").read_bytes():
                raise ValueError("unchanged control object did not reproduce")
            implementation["experimental_packet_schedule"]=schedule
            tag=f"quant_h{extent}_s{spacing}"
            renamed=arm/"kernel_link.o"
            command=[str(args.objcopy)]
            for symbol in ("forward","_mlir_ciface_forward","dealloc_helper",*implementation["routes"][0]["helpers"].values()):
                new="mapped_"+tag if symbol=="forward" else "_mlir_ciface_"+tag if symbol.startswith("_mlir") else symbol+"_"+tag
                command += ["--redefine-sym",symbol+"="+new]
            subprocess.run(command+[str(arm/"kernel.o"),str(renamed)],check=True)
            if contents(renamed)!=contents(arm/"kernel.o"):
                raise ValueError("namespace changed executable bytes")
            objects.append(renamed)
            cases.append({"id":len(cases),"family":"host_quant","kernel":tag,"source_function":"mapped_"+tag,"size":selected.size,"extent":extent,"independent_lanes":8,"consumer_spacing_group":spacing,"partition":"heldout" if extent==224 else "training","expected_fflags":1,"input_shape":proof["input_shape"],"output_shape":proof["output_shape"],"extent_proof":proof,"source_operations":{"fmul":selected.size,"minimum":selected.size,"maximum":selected.size,"fixed_rne_convert":selected.size},"requested_cpu_load_bytes":selected.nbytes,"requested_cpu_store_bytes":selected.size,"scalar_graph_sha256":binding["source_operation_sha256"],"implementation":implementation,"original_object":pin(arm/"kernel.o"),"renamed_object":pin(renamed),"renaming_executable_bytes_identical":True,"source_oracle":pin(height/"expected.bin"),"input":pin(height/"input.bin")})
    checksum=14695981039346656037
    for case in cases:
        for _ in range(2):
            for byte in Path(case["source_oracle"]["path"]).read_bytes():
                checksum=((checksum^byte)*1099511628211)&((1<<64)-1)
    hypotheses=[{"name":"work_only","features":["source_elements"],"include_fixed":False},{"name":"work_plus_ordered_fp_spacing","features":["source_elements","source_fp_inverse_distance"],"include_fixed":False}]
    manifest={"schema":"source_host_quant_spacing_crossed_v1","cases":cases,"repetitions":2,"empty_windows":3,"expected_checksum":f"{checksum:016x}","predeclared_hypotheses":hypotheses,"partition_rule":"112/448 extents train both spacing arms; complete224 middle pair and all repeats held. No previous held label becomes training.","source_binding":pin(args.fixture/"receipt.json"),"memory_regime":"Same private aligned source owners/reset arena protocol as original2057/2059; only inline-ASM instruction order changes within each fixed eight-lane packet. Complete linked physical traffic/warmth unknown.","timing":"same fenced common ranked map; descriptors/allocations/copy included; setup/verification/UART excluded","causal_scope":"Equal source arithmetic, eight-lane full packet counts and branch-loop geometry, no tails. Emitted consumer distance varies. Actual final FP/GPR/stack/opcode counters must verify compiled equivalence; no guessed FPU latency or whole prediction.","new_unpriced_dimensions":["different linked footprint/instruction order","backend register allocation/GPR dependencies may differ"],"producer":pin(__file__)}
    (work/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    (work/"battery.c").write_text(c_source(cases,shape,checksum))
    assembly='.section .rodata.source,"a",@progbits\n.balign 1048576\n.global original_source\noriginal_source:\n.incbin "original_input.bin"\n'
    for extent in (112,224,448):
        assembly+=f'.balign64\n.global expected_h{extent}\nexpected_h{extent}:\n.incbin "height{extent}/expected.bin"\n'.replace('.balign64','.balign 64')
    (work/"operands.S").write_text(assembly)
    for path in (args.benchmark_header,args.allocator,args.allocator.with_name("htif.h")):
        shutil.copyfile(path,work/path.name)
    program=build_program([work/"battery.c",work/"operands.S",*objects,work/"merlin_malloc.c"],work,target="gemmini",max_loaded_bytes=None,extra_cflags=["-O2","-fno-fast-math","-ffp-contract=off","-I",str(work),f"-DMERLIN_ARENA_BASE_ADDR={args.arena_base}ULL","-DMERLIN_ARENA_SIZE_BYTES=0x10000000ULL"],support_first=True)
    audit=audit_elf(program.elf.read_bytes())
    if audit["status"]!="pass" or audit["custom_funct_counts"]:
        raise ValueError("no-custom/FSM executable gate failed")
    (work/"built.json").write_text(json.dumps({"schema":"source_host_quant_spacing_built_v1","elf":pin(program.elf),"manifest":pin(work/"manifest.json"),"audit":audit,"arena_base":args.arena_base,"load_bytes":program.loaded_bytes,"sources":[pin(p) for p in (work/"battery.c",work/"operands.S",work/"benchmark_buffer.h",work/"merlin_malloc.c",Path(__file__))]},indent=2)+"\n")
    print(json.dumps({"cases":6,"same_packet_width":8,"spacing_groups":[1,8],"native_source_bytes_checked":sum(c["size"] for c in cases),"elf":pin(program.elf),"checksum":manifest["expected_checksum"]}))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("fixture","llvm-bin","objcopy","benchmark-header","allocator","workdir"):
        parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--arena-base",type=lambda value:int(value,0),required=True)
    generate(parser.parse_args())
