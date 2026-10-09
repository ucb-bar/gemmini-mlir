"""Independent actual tensor lowering, scale-load bounds and refusal behavior."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import replace

import numpy as np
import pytest
from test_source_expression_interval import source

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.scaled_integer_finite_llvm import (
    bind_finite_scale_helpers,
    emit_finite_scale_helper,
    validate_finite_scale_helper,
)
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    build_source_interval_table,
    close_scalar_i8_observer,
    emit_source_interval_i8_lookup,
)
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_i8_lookup
from merlin.llvmlower.toolchain import clang

EFFECTS = IntervalEffectContract(True, True, True, True, True)


def tensor_source(rows, width):
    body = source().split("->i8 {", 1)[1].split("return %r:i8", 1)[0]
    return f"""module {{func.func @unrelated(%inputa:tensor<{rows}x{width}xi32>,%sa:tensor<{width}xf32>,
      %inputb:tensor<{rows}x{width}xi32>,%sb:tensor<{width}xf32>)->tensor<{rows}x{width}xi8> {{
      %e=tensor.empty():tensor<{rows}x{width}xi8>
      %v=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d1)>,
        affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d1)>,affine_map<(d0,d1)->(d0,d1)>],
        iterator_types=["parallel","parallel"]}}
      ins(%inputa,%sa,%inputb,%sb:tensor<{rows}x{width}xi32>,tensor<{width}xf32>,tensor<{rows}x{width}xi32>,tensor<{width}xf32>)
      outs(%e:tensor<{rows}x{width}xi8>) {{^bb0(%ia:i32,%as:f32,%ib:i32,%bs:f32,%old:i8):
        %af=arith.sitofp %ia:i32 to f32
        %bf=arith.sitofp %ib:i32 to f32
        %factor=arith.constant 0.03125:f32
        %ac=arith.mulf %af,%factor:f32
        %bc=arith.mulf %bf,%factor:f32
        %x=arith.mulf %ac,%as:f32
        %up=arith.mulf %bc,%bs:f32
        {body}
        linalg.yield %r:i8
      }}->tensor<{rows}x{width}xi8>
      return %v:tensor<{rows}x{width}xi8>
    }} }}"""


def analyze(rows, width):
    text = tensor_source(rows, width)
    module = parse_mlir_text(text)
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    cut = next(
        op.results[0]
        for op in generic.body.block.ops
        if op.name == "arith.mulf" and op.operands[1] == generic.body.block.args[1]
    )
    activation = next(
        op.results[0] for op in generic.body.block.ops if op.name == "arith.mulf" and op.operands[0] == cut
    )
    proof = close_scalar_i8_observer(cut, activation, effects=EFFECTS)
    return text, proof


def llvm_source(rows, width):
    # Ordinary upstream scalar-loop shape. The independent source tensor proof
    # binds arithmetic/types/maps; this fixture also checks tail index coverage.
    return f"""define void @helper(ptr %a,ptr %sa,ptr %b,ptr %sb,ptr %out) {{
entry:
 br label %row
row:
 %i=phi i64 [0,%entry],[%next,%done]
 %ic=icmp slt i64 %i,{rows}
 br i1 %ic,label %inner,label %exit
inner:
 br label %col
col:
 %j=phi i64 [0,%inner],[%advance,%body]
 %jc=icmp slt i64 %j,{width}
 br i1 %jc,label %body,label %done
body:
 %base=mul i64 %i,{width}
 %idx=add i64 %base,%j
 %ap=getelementptr i32,ptr %a,i64 %idx
 %bp=getelementptr i32,ptr %b,i64 %idx
 %sap=getelementptr float,ptr %sa,i64 %j
 %sbp=getelementptr float,ptr %sb,i64 %j
 %av=load i32,ptr %ap,align 4
 %bv=load i32,ptr %bp,align 4
 %sav=load float,ptr %sap,align 4
 %sbv=load float,ptr %sbp,align 4
 %af=sitofp i32 %av to float
 %bf=sitofp i32 %bv to float
 %ac=fmul float %af,3.125000e-02
 %bc=fmul float %bf,3.125000e-02
 %cut=fmul float %ac,%sav
 %up=fmul float %bc,%sbv
 %result=fmul float %cut,2.000000e+00
 %prod=fmul float %result,%up
 %scaled=fmul float %prod,3.000000e+00
 %op=getelementptr i8,ptr %out,i64 %idx
 store i8 0,ptr %op,align 1
 %advance=add i64 %j,1
 br label %col
done:
 %next=add i64 %i,1
 br label %row
exit:
 ret void
}}"""


def binding(rows=3, width=17, compiled=None):
    _, proof = analyze(rows, width)
    llvm = llvm_source(rows, width) if compiled is None else compiled
    routes = (
        {
            "function_body_index": 0,
            "input": "%cut",
            "up": "%up",
            "quant_factor_bits": proof.quant_factor_bits,
            "source_expression_sha256": proof.expression.canonical_sha256,
        },
    )
    result = bind_finite_scale_helpers(
        llvm, routes=routes, observers=(proof,), effects=EFFECTS, immutable_inputs=True, fresh_disjoint_output=True
    )
    assert len(result) == 1
    return result[0]


@pytest.mark.parametrize("rows,width", [(3, 17), (9, 7), (2, 1)])
def test_source_arithmetic_and_bounded_physical_projection(rows, width):
    bound = binding(rows, width)
    assert [(s.argument, s.count) for s in bound.scans] == [(1, width), (3, width)]
    validate_finite_scale_helper(bound)
    guarded = emit_finite_scale_helper(
        bound, wrapper_symbol="guard", finite_symbol="finite", fallback_symbol="original"
    )
    assert "else original" in guarded
    ordinary = emit_source_interval_i8_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=12
    )
    prepared = emit_source_interval_i8_lookup(
        table_name="table",
        activation_name="activation",
        quantizer_name="quant",
        lookup_name="lookup",
        leading_bits=12,
        finite_inputs=(bound,),
    )
    assert "if((w&0x7f800000u)" in ordinary and "if((w&0x7f800000u)" not in prepared
    assert "if(!(lo<=hi))goto source_fallback;" in prepared


@pytest.mark.parametrize(
    "change",
    [
        "constant",
        "cast",
        "projection",
        "out_of_range",
        "negative_seed",
        "step",
        "comparison",
        "unknown_call",
        "input_store",
        "escape",
        "strict",
        "fast",
        "volatile",
    ],
)
def test_compiled_mutations_refuse_transactionally(change):
    src = llvm_source(3, 17)
    if change == "constant":
        src = src.replace("3.125000e-02", "6.250000e-02", 1)
    elif change == "cast":
        src = src.replace("sitofp i32 %av", "uitofp i32 %av")
    elif change == "projection":
        src = src.replace("float,ptr %sa,i64 %j", "float,ptr %sa,i64 %idx")
    elif change == "out_of_range":
        src = src.replace("%j,17", "%j,18")
    elif change == "negative_seed":
        src = src.replace("[0,%inner]", "[-1,%inner]")
    elif change == "step":
        src = src.replace("%j,1", "%j,-1")
    elif change == "comparison":
        src = src.replace("icmp slt i64 %j", "icmp sle i64 %j")
    elif change == "unknown_call":
        src = src.replace(" %advance=", " call void @unknown()\n %advance=")
    elif change == "input_store":
        src = src.replace("store i8 0,ptr %op", "store i32 0,ptr %ap")
    elif change == "escape":
        src = src.replace(" ret void", " ret ptr %sa")
    elif change == "strict":
        src = src.replace("define void", "define strictfp void")
    elif change == "fast":
        src = src.replace("fmul float", "fmul fast float", 1)
    elif change == "volatile":
        src = src.replace("load float", "load volatile float", 1)
    old = src
    with pytest.raises(ValueError):
        binding(compiled=src)
    assert src == old


def test_explicit_ownership_empty_default_and_forged_span_refusal():
    bound = binding()
    validate_finite_scale_helper(bound)
    with pytest.raises(ValueError):
        validate_finite_scale_helper(replace(bound, scans=(replace(bound.scans[0], count=18), *bound.scans[1:])))
    with pytest.raises(ValueError):
        bind_finite_scale_helpers(bound._source, routes=bound._routes, observers=bound._observers, effects=EFFECTS)
    assert bind_finite_scale_helpers("not LLVM", routes=(), observers=(), effects=None) == ()


def test_native_guard_dirty_outputs_tails_immutable_scales_and_refusal(tmp_path):
    bound = binding(3, 17)
    code = emit_finite_scale_helper(bound, wrapper_symbol="guard", finite_symbol="finite", fallback_symbol="original")
    code += """\nint selected=0;
void finite(void*a,void*sa,void*b,void*sb,void*out){selected=1;for(int i=0;i<51;i++)((signed char*)out)[i]=7;}
void original(void*a,void*sa,void*b,void*sb,void*out){selected=2;for(int i=0;i<51;i++)((signed char*)out)[i]=9;}
"""
    c, so = tmp_path / "guard.c", tmp_path / "guard.so"
    c.write_text(code)
    subprocess.run([str(clang()), "-O3", "-fPIC", "-shared", str(c), "-o", str(so)], check=True, capture_output=True)
    lib = ctypes.CDLL(str(so))
    fn = lib.guard
    fn.argtypes = [ctypes.c_void_p] * 5
    fn.restype = None
    selected = ctypes.c_int.in_dll(lib, "selected")
    environment = ctypes.CDLL(None)
    oldmode = environment.fegetround()
    oldflags = environment.fetestexcept(61)
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            environment.fesetround(mode)
            for word, expected in (
                (0, 1),
                (0x80000000, 1),
                (1, 1),
                (bound.scans[0].proof.domain.scale_limit_word(), 1),
                (0x7F800000, 2),
                (0x7F800001, 2),
                (0xFF800001, 2),
            ):
                scale = np.ones(17, np.float32)
                scale.view(np.uint32)[-1] = word
                before = scale.tobytes()
                dst = np.full(59, 73, np.int8)
                environment.feclearexcept(61)
                environment.feraiseexcept(61)
                flags = environment.fetestexcept(61)
                fn(None, scale.ctypes.data, None, scale.ctypes.data, dst.ctypes.data + 4)
                assert selected.value == expected and environment.fetestexcept(61) == flags
                assert np.all(dst[:4] == 73) and np.all(dst[-4:] == 73) and scale.tobytes() == before
    finally:
        environment.fesetround(oldmode)
        environment.feclearexcept(61)
        environment.feraiseexcept(oldflags)
