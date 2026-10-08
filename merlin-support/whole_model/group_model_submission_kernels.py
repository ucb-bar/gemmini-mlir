"""The per-group kernels a SUBMISSION's own backend emitted, as objects this program links.

`group_model_sched_kernels` renders kernels from this repo's schedule IR. This one renders none:
each kernel is a relocatable the submission's package produced through its declared
`lower_target_to_llvm` entrypoint, and all this does is declare it, call it with the arguments the
OOT backend ABI contract says it takes, and say which groups fell back to the library.

WHY THE KERNELS ARE NOT REGENERATED HERE. A transcription of the backend's output, written in this
repo, would put a cycle number on instructions no oracle ever graded -- the same reason
`merlin.llvmlower.device_shim` declares the kernel `extern` and refuses to emit it.

THE IM2COL GATHER IS THIS HARNESS'S CODE, AND THAT MATTERS TO THE NUMBER. A package that lowers a
convolution to `resident_matmul` over a derived im2col matrix requires its CALLER to materialize
that matrix; the contract says so, and the vendor's `tiled_conv_auto` requires no such thing. The
cost is real and belongs to the submission's own lowering choice, so it is counted inside the
group's bracket. But the loop performing it is written here, so the published `im2col` line is a
FLOOR on that cost -- a better gather would lower it -- and not a verdict on the compiler.

WHICH BUFFER EACH ARGUMENT IS, IS DECIDED ONCE, UPSTREAM. The whole-program statement splices each
group's reply and records which program tensor every one of the package's tensors was bound to
(`merlin.llvmlower.whole_program`); `merlin.perf.whole_model_build` reads that binding and the
contract's argument order. This module used to re-derive both -- roles mapped to step fields, and
each operand's extents recomputed from the step -- which was a second statement of the same fact that
nothing held to the first.

AND IT IS A COST NO CAPSULE GRADE CAN SEE. The capsule harness hands a kernel an operand that is
already materialized, so a package passing every convolution capsule has never once paid for the
gather its own lowering requires. The model pays it 19 times. That is the same shape as the
coverage gap: the capsule passes, the model pays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

#: This driver links a MERGED group's kernel in place of every group it absorbed (read by
#: `merlin.perf.whole_model_build.build`; without it no group-formation offer is made).
LINKS_MERGED_GROUPS = True

#: This driver links a FUSED REGION's one kernel in place of every member's call and grades the region
#: at its boundary (``group_model_program.region_steps``). Read by ``merlin.perf.whole_model_build``:
#: without it no region is ever offered to a package.
LINKS_FUSED_REGIONS = True

#: The STATEMENT's ops (``merlin.llvmlower.whole_program``'s vocabulary) a fused region may hold as an
#: INTERNAL member: the ones this driver restates on the core for the region's check
#: (``group_model_program.REGION_INTERNAL_KINDS``; a window mean is stated as a matmul). A region is
#: offered only across boundaries whose producer is one of these.
FUSED_REGION_INTERNAL_OPS = ("conv2d", "matmul")


def _product(values) -> int:
    total = 1
    for value in values:
        total *= int(value)
    return total


def _gather(recipe: dict[str, Any], out_dim: int, in_dim: int, symbol: str, pitch: int | None = None) -> str:
    """The im2col matrix the package's OWN RECIPE describes, as C.

    THE GEOMETRY COMES FROM THE PACKAGE, NOT FROM THE DRIVER'S STEP. The command buffer declares
    `params.im2col_recipes` -- source, target, kh, kw, ci, stride, padding, dilation, layout -- which
    is the package stating exactly what matrix its kernel will read. Deriving the same numbers again
    from the model's own step would be one fact with two homes, free to disagree, and a gather built
    to a different geometry than the kernel expects is a program that runs and is numerically wrong.

    The column order is the weight's row order, `(krow * Kw + kcol) * Ci + kch`, which the CONV2D
    contract states as "column order kh, kw, ci". Padding contributes zero because the capture's
    scales are symmetric: a group carrying a zero point is refused upstream, so zero here is the
    zero of the operand's own domain.

    THE INNERMOST RUN IS CONTIGUOUS AND MOVES AS ONE. In nhwc a tap's Ci channels are adjacent in
    both source and destination, so the copy is a single run rather than Ci separate load/stores.
    Measured on the FPGA before this: the gather was 82.6% of arm B's convolution time and 73.8M of
    its 80.1M whole-model excess, while its kernels ran at 1.03x the vendor's.
    """
    k_h, k_w, ci = int(recipe["kh"]), int(recipe["kw"]), int(recipe["ci"])
    stride_h, stride_w = (int(v) for v in recipe["stride"])
    pad = [int(v) for v in recipe["padding"]]
    row_elements = k_h * k_w * ci
    # ``pitch`` is the row pitch the kernel reads the matrix at (the contract's padded row); the columns
    # past the row are zero, written every call -- the scratch is shared with other groups' gathers.
    pitch = int(pitch or row_elements)
    tail = (
        f"\n            __builtin_memset(row + {row_elements}, 0, {pitch - row_elements} * sizeof(elem_t));"
        if pitch > row_elements
        else ""
    )
    return f"""
static void {symbol}(const elem_t *src, elem_t *dst) {{
    for (int oh = 0; oh < {out_dim}; oh++) {{
        for (int ow = 0; ow < {out_dim}; ow++) {{
            elem_t *row = dst + (size_t)(oh * {out_dim} + ow) * {pitch};{tail}
            int iw0 = ow * {stride_w} - {pad[1]};
            for (int kr = 0; kr < {k_h}; kr++) {{
                int ih = oh * {stride_h} - {pad[0]} + kr;
                elem_t *seg = row + kr * {k_w * ci};
                if (ih < 0 || ih >= {in_dim}) {{
                    __builtin_memset(seg, 0, {k_w * ci} * sizeof(elem_t));
                }} else if (iw0 >= 0 && iw0 + {k_w} <= {in_dim}) {{
                    /* A KERNEL ROW IS ONE CONTIGUOUS RUN IN THE SOURCE. In nhwc the tap index
                       advances by one column whatever the stride, and a column's Ci channels are
                       adjacent, so all Kw taps of this row are Kw*Ci adjacent elements. The whole
                       row moves in a single copy instead of Kw of them -- the same bytes, a third
                       of the calls and none of the per-tap bounds tests. */
                    __builtin_memcpy(seg, src + (size_t)(ih * {in_dim} + iw0) * {ci},
                                     {k_w * ci} * sizeof(elem_t));
                }} else {{
                    /* The row straddles an edge, so each tap is tested on its own. */
                    for (int kc = 0; kc < {k_w}; kc++) {{
                        int iw = iw0 + kc;
                        elem_t *col = seg + kc * {ci};
                        if (iw < 0 || iw >= {in_dim}) {{
                            __builtin_memset(col, 0, {ci} * sizeof(elem_t));
                        }} else {{
                            __builtin_memcpy(col, src + (size_t)(ih * {in_dim} + iw) * {ci},
                                             {ci} * sizeof(elem_t));
                        }}
                    }}
                }}
            }}
        }}
    }}
}}"""


#: Which buffer of THIS DRIVER'S step a program operand of the whole-program statement is, per step
#: kind. Both sides are statements of the same group -- ``llvmlower.whole_program`` names a group's
#: operands ``lhs`` / ``rhs`` / ``bias`` / ``dst`` in the program's namespace, and ``extract`` names
#: the C buffers the step reads -- so this is a join between two vocabularies of one fact, not a
#: derivation of it: it says nothing about roles, positions or extents. Every join is CHECKED below
#: (by name where both sides name the same buffer, by element count, and by content digest where the
#: statement carries one), so a wrong entry here is a refusal, never a mis-bound pointer.
_JOIN = {
    "conv2d": {"lhs": "in", "rhs": "weight", "bias": "bias", "dst": "out"},
    "matmul": {"lhs": "in", "rhs": "weight", "bias": "bias", "dst": "out"},
    "sum": {"lhs": "lhs", "rhs": "rhs", "dst": "out"},
    # A merged group (a contraction and the sum it absorbed, see `group_model_program.merge_steps`):
    # the contraction's operands, the skip tensor the sum's other side held, and the sum's output.
    "fused": {"lhs": "in", "rhs": "weight", "bias": "bias", "skip": "skip", "dst": "out"},
    # A window mean is stated with its constant ones on the LEFT (see `whole_program`), and the driver
    # materializes that constant once as `ONES`, sized to the widest window any mean needs.
    "mean": {"lhs": "@ONES", "rhs": "in", "dst": "out"},
}


def _symbols(model: dict[str, Any]) -> dict[str, int]:
    """Every C buffer this driver declares or embeds, with its element count."""
    sizes = {str(b["name"]): int(b["elements"]) for b in model.get("buffers") or ()}
    for name, array in (model.get("arrays") or {}).items():
        sizes[str(name)] = int(getattr(array, "size", 0))
    if "IMAGE_DATA" in sizes:
        sizes["IMAGE"] = sizes["IMAGE_DATA"]
    steps = [m for s in model.get("steps") or () for m in (s["members"] if s.get("kind") == "region" else [s])]
    windows = [int(s["window"]) for s in steps if s.get("kind") == "mean"]
    sizes["ONES"] = max(windows or [1])
    return sizes


def _symbol_of(step: dict[str, Any], operands: dict[str, str], program: str, entry: str | None) -> str | None:
    """The C buffer this driver holds program tensor ``program`` in, for the group ``step`` states.

    A FUSED REGION's kernel reads its members' operands: ``operands`` is then ``{group: that member's
    operands}`` and each member is joined by its own kind, in order -- the same join, member by member."""
    if entry and program == entry:
        return "IMAGE"
    if step.get("kind") == "region":
        for member in step["members"]:
            found = _symbol_of(member, dict(operands.get(str(member["group"])) or {}), program, entry)
            if found is not None:
                return found
        return None
    for key, name in operands.items():
        if name != program:
            continue
        field = _JOIN.get(str(step.get("kind")), {}).get(key)
        if field is None:
            return None
        if field.startswith("@"):
            return field[1:]
        found = step.get(field)
        return str(found) if found else None
    return None


#: The model input, laid out as the backend contract declares a kernel's pointee: its rows (all
#: leading axes flattened) zero-padded to the row padding. The dense ``IMAGE`` stays what the library
#: and every host-side reference read.
PADDED_IMAGE = "IMAGE_ROWS_PADDED"


def _rows_and_width(declared: Sequence[int]) -> tuple[int, int]:
    extents = [int(v) for v in declared]
    width = extents[-1] if extents else 1
    return _product(extents[:-1]) if len(extents) > 1 else 1, width


def _needs_row_padding(declared: Sequence[int], multiple: int | None) -> bool | None:
    """Whether a pointee of ``declared`` extents differs between its dense and its contract layout:
    only when it has more than one row and its row is not a multiple of the padding. None when the
    padding is unknown and it would matter."""
    rows, width = _rows_and_width(declared)
    if rows <= 1:
        return False
    if not multiple:
        return None if width else False
    return width % int(multiple) != 0


def padded_rows(array: Any, width: int, multiple: int) -> Any:
    """``array`` viewed as rows of ``width`` elements, each zero-padded to a multiple of ``multiple``."""
    import numpy as np

    flat = np.asarray(array).reshape(-1, int(width))
    pitch = -(-int(width) // int(multiple)) * int(multiple)
    out = np.zeros((flat.shape[0], pitch), dtype=flat.dtype)
    out[:, : int(width)] = flat
    return np.ascontiguousarray(out.reshape(-1))


def _content_digest(model: dict[str, Any], symbol: str) -> str | None:
    import hashlib

    array = (model.get("arrays") or {}).get(symbol)
    return None if array is None else hashlib.sha256(array.tobytes()).hexdigest()


def render_kernels(
    model: dict[str, Any],
    groups: Sequence[Mapping[str, Any]],
    *,
    entry: str | None = None,
    row_padding: int | None = None,
) -> dict[str, Any]:
    """``{"definitions", "calls", "census", "objects"}`` for the groups the submission answered.

    ``groups`` is one row per group from :func:`merlin.perf.whole_model_build.bind_groups`: either a
    named refusal (the group keeps its library call) or the package's kernel -- its object, its
    symbol, and its arguments IN THE ORDER THE ABI CONTRACT DECLARES, each naming the PROGRAM tensor
    the splice bound it to, or the package's own gather recipe for a buffer private to the kernel.
    Nothing here decides which buffer an argument is; it only finds the C buffer that holds that
    program tensor and checks that the two are the same size (and, for a laid-out weight, the same
    bytes). ``entry`` names the program's model-input tensor, which this driver holds as ``IMAGE``.

    THE CONTRACT'S POINTEE LAYOUT, NOT THE DRIVER'S. The backend contract declares every pointer
    argument row-major with its rows zero-padded to ``row_padding`` (the device's tile edge). Every
    buffer a kernel produces already has that layout when its rows are a multiple of it; the model
    INPUT does not when its channel count is not (a 3-channel image), and handed densely a package
    reads it at the padded pitch and gets plausible, wrong numbers. So a kernel reading the input is
    handed ``IMAGE_ROWS_PADDED`` -- the same values, laid out as declared -- and any other argument whose
    rows would need padding is refused by name rather than handed dense.
    """
    steps = {int(s["group"]): s for s in model["steps"]}
    rows = {int(r["group"]): r for r in groups}
    sizes = _symbols(model)
    definitions: list[str] = []
    calls: dict[int, str] = {}
    census: list[dict[str, Any]] = []
    objects: list[str] = []
    scratch = 0

    for group, step in sorted(steps.items()):
        row = rows.get(group)
        if row is None or row.get("on") != "package":
            why = (row or {}).get("why") or "the package was not asked for this group"
            census.append(
                {"group": group, "kind": step["kind"], "on": "vendor", "why": why, "cause": (row or {}).get("cause")}
            )
            continue
        if step["kind"] == "region":
            # One kernel for the whole region: its arguments are any member's operands, each member
            # joined by its own kind (`_symbol_of`), stated by the splice for every member.
            operands = {
                str(g): {str(k): str(v) for k, v in (ops or {}).items()}
                for g, ops in (row.get("member_operands") or {}).items()
            }
        else:
            operands = {str(k): str(v) for k, v in (row.get("operands") or {}).items()}
        args: list[str] = []
        laid_out: dict[str, Any] = {}
        gather_pitch_used: int | None = None
        gathered = gather_recipe = gather_source = None
        gather_dims: tuple[int, int] = (0, 0)
        refusal = None
        for arg in row["args"]:
            gather = arg.get("gather")
            if gather:
                if gathered is not None:
                    refusal = f"the kernel reads a second gathered operand {arg['tensor']!r}; one gather per call"
                    break
                if step["kind"] != "conv2d" and not (
                    step["kind"] == "region" and any(m["kind"] == "conv2d" for m in step["members"])
                ):
                    refusal = f"a {step['kind']!r} group's kernel reads a gathered operand {arg['tensor']!r}"
                    break
                recipe = dict(gather["recipe"])
                source = _symbol_of(step, operands, str(gather["source"]), entry)
                held = [int(e) for e in gather.get("source_declared") or ()]
                if source is None or len(held) != 4 or held[1] != held[2]:
                    refusal = (
                        f"the recipe for {arg['tensor']!r} gathers from {gather['source']!r} declared "
                        f"{held}, which is not a square nhwc image this driver holds"
                    )
                    break
                from merlin.runtime.commandbuffer import conv_out_dims

                out_h, out_w = conv_out_dims(
                    held[1],
                    held[2],
                    int(recipe["kh"]),
                    int(recipe["kw"]),
                    recipe["stride"],
                    recipe["padding"],
                    recipe.get("dilation") or [1, 1],
                )
                declared = [int(e) for e in arg.get("declared") or ()]
                if out_h != out_w or _product(declared) != out_h * out_w * int(recipe["kh"]) * int(recipe["kw"]) * int(
                    recipe["ci"]
                ):
                    refusal = (
                        f"the package declares {arg['tensor']!r} as {declared}, which its own recipe over "
                        f"a {held} image does not produce"
                    )
                    break
                if str(recipe.get("layout")) != "nhwc" or [int(v) for v in recipe.get("dilation", [1, 1])] != [1, 1]:
                    refusal = (
                        f"the recipe for {arg['tensor']!r} is {recipe.get('layout')!r} with dilation "
                        f"{recipe.get('dilation')}, which this gather does not state"
                    )
                    break
                rows_g, width_g = out_h * out_h, int(recipe["kh"]) * int(recipe["kw"]) * int(recipe["ci"])
                padding_g = _needs_row_padding([rows_g, width_g], row_padding)
                if padding_g is None:
                    refusal = (
                        f"the gathered {arg['tensor']!r} has rows the contract pads to the tile edge, and the "
                        "edge is not derivable; it is not handed over dense"
                    )
                    break
                gather_pitch = -(-width_g // int(row_padding)) * int(row_padding) if padding_g else width_g
                gathered, gather_recipe, gather_source = f"im2col_g{group}", recipe, source
                gather_pitch_used = gather_pitch
                gather_dims = (out_h, held[1])
                args.append("IM2COL")
                scratch = max(scratch, rows_g * gather_pitch)
                if padding_g:
                    laid_out[str(arg["tensor"])] = {
                        "symbol": "IM2COL",
                        "declared": declared,
                        "row_padding": int(row_padding),
                        "pitch": gather_pitch,
                    }
                continue
            program = str(arg.get("program") or "")
            symbol = _symbol_of(step, operands, program, entry)
            declared = [int(v) for v in arg.get("declared") or ()]
            padding = _needs_row_padding(declared, row_padding) if symbol is not None else False
            if padding is None:
                refusal = (
                    f"the kernel's {arg['tensor']!r} argument {declared} has rows the contract pads to the tile "
                    "edge, and the edge is not derivable; it is not handed over dense"
                )
                break
            if padding and symbol == "IMAGE":
                _rows, width = _rows_and_width(declared)
                if _product(declared) != sizes["IMAGE"]:
                    refusal = (
                        f"the package reads {arg['tensor']!r} as {_product(declared)} element(s) where the model "
                        f"input holds {sizes['IMAGE']}"
                    )
                    break
                if PADDED_IMAGE not in (model.get("arrays") or {}):
                    model["arrays"][PADDED_IMAGE] = padded_rows(model["arrays"]["IMAGE_DATA"], width, row_padding)
                    sizes[PADDED_IMAGE] = int(model["arrays"][PADDED_IMAGE].size)
                args.append(PADDED_IMAGE)
                laid_out[str(arg["tensor"])] = {
                    "symbol": PADDED_IMAGE,
                    "declared": declared,
                    "row_padding": int(row_padding),
                    "elements": sizes[PADDED_IMAGE],
                }
                continue
            if padding:
                refusal = (
                    f"the kernel's {arg['tensor']!r} argument {declared} needs its rows padded to {row_padding} "
                    f"and {symbol!r} is a buffer this driver lays out densely"
                )
                break
            if symbol is None or symbol not in sizes:
                refusal = (
                    f"the kernel's {arg.get('role')!r} argument {arg['tensor']!r} is program tensor "
                    f"{program!r}, which this driver holds in no buffer for a {step['kind']!r} group"
                )
                break
            if step["kind"] == "fused" and program == operands.get("skip") and program != step.get("skip"):
                # THE SKIP IS NAMED BY BOTH SIDES, and they must agree. Joined by operand key alone, a
                # statement whose skip names any other buffer would still resolve to this step's skip
                # symbol, and the kernel would sum a tensor neither side meant.
                refusal = (
                    f"the program binds the skip {arg['tensor']!r} to {program!r} where this driver's merged "
                    f"step reads {step.get('skip')!r}"
                )
                break
            if program in sizes and program != symbol:
                # Both statements name a committed buffer, and they must name the SAME one. A sum whose
                # operands arrive swapped is exactly this case: the counts agree and the scales do not.
                refusal = (
                    f"the program binds {arg['tensor']!r} to {program!r} where this driver's step reads {symbol!r}"
                )
                break
            want = _product(arg.get("declared") or ())
            have = sizes[symbol]
            if (want > have) if symbol == "ONES" else (want != have):
                refusal = f"the package reads {arg['tensor']!r} as {want} element(s) where {symbol!r} holds {have}"
                break
            digest = arg.get("sha256")
            if digest and _content_digest(model, symbol) not in (None, digest):
                refusal = f"{symbol!r} does not hold the bytes the program laid out for {program!r}"
                break
            args.append(symbol)
        if refusal:
            census.append(
                {
                    "group": group,
                    "kind": step["kind"],
                    "on": "vendor",
                    "why": refusal,
                    "cause": "driver_binding_refused",
                }
            )
            continue
        symbol = str(row["symbol"])
        definitions.append(f"extern void {symbol}({', '.join('void *' for _ in args)});")
        body = ""
        if gathered:
            assert gather_recipe is not None
            definitions.append(
                _gather(
                    gather_recipe,
                    gather_dims[0],
                    gather_dims[1],
                    gathered,
                    pitch=gather_pitch_used,
                )
            )
            # THE GATHER IS TIMED ON ITS OWN, INSIDE THE SAME BRACKET. Both halves stay in the
            # window -- both have to happen for the model to compute its answer -- but a single
            # number for the pair cannot say whether a slow convolution is a slow kernel or a slow
            # materialization, and the materialization is THIS harness's code while the kernel is
            # the submission's. `fm_g` is the gather's own cycles, which the caller stores beside
            # the group's total so the two are reported apart.
            body = (
                f"{{ uint64_t i0 = read_cycles(); {gathered}((const elem_t *){gather_source}, IM2COL);"
                f" fm_g = read_cycles() - i0; fm_im2col += fm_g; fm_im2col_groups++; "
            )
        call = f"{symbol}({', '.join(f'(void *){a}' for a in args)});"
        calls[group] = (body + call + " }") if gathered else call
        objects.append(str(row["object"]))
        census.append(
            {
                "group": group,
                "kind": step["kind"],
                "on": "submission",
                "shape": row.get("shape"),
                "arg_order": [str(a["tensor"]) for a in row["args"]],
                "arguments": args,
                "row_padded": laid_out or None,
                "gather": bool(gathered),
                "object": str(row["object"]),
            }
        )
    if scratch:
        definitions.insert(0, f"static elem_t IM2COL[{scratch}] row_align(1);")
    return {
        "definitions": "\n".join(definitions),
        "calls": calls,
        "census": census,
        "objects": objects,
        "im2col_scratch_elements": scratch,
    }
