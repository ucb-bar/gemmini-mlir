"""Pure whole-program caller source renderer, not an execution or admission service."""

from __future__ import annotations

from math import prod

from .format import (
    CodegenError,
    buffer_extent,
    ceil_dim,
    container_for,
    container_words,
    pad_rowmajor,
)
from .measurement import assemble_measurement_fragments


_COHERENT_OUTPUT_DTYPES = frozenset({"i8", "i16", "i32", "i64", "f32"})
COHERENT_DUMP_V1 = "coherent_dump_v1"
COHERENT_PACKET_V1 = "coherent_packet_v1"


def _caller_layouts(cb, storage, *, ceil_extent=None, extent=None):
    """The exact buffer geometry consumed by both C rendering and public inspection."""
    abi, tensors = cb.get("kernel_abi") or {}, cb.get("tensors") or {}
    args = abi.get("args") or []
    if abi.get("kind") != "whole_program" or not isinstance(args, list) or not isinstance(tensors, dict):
        raise CodegenError("caller layout requires a declared whole-program pointer ABI")
    if len({arg.get("tensor") for arg in args if isinstance(arg, dict)}) != len(args):
        raise CodegenError("caller layout requires distinct typed pointer arguments")
    rows = []
    for arg in args:
        if (
            not isinstance(arg, dict)
            or arg.get("tensor") not in tensors
            or arg.get("access") not in ("read", "write", "readwrite")
        ):
            raise CodegenError("caller layout requires complete typed pointer arguments")
        name, spec = arg["tensor"], tensors[arg["tensor"]]
        if not isinstance(spec, dict):
            raise CodegenError("caller layout requires a typed tensor descriptor")
        container_for(str(spec.get("dtype") or ""))
        if storage is not None:
            binding = storage[name]
            encoding = binding.encoding
            rows.append(
                {
                    "tensor": name,
                    "dtype": encoding.dtype,
                    "logical_shape": list(encoding.logical_shape),
                    "physical_extents": list(encoding.physical_shape),
                    "logical_strides_elements": list(encoding.logical_strides_elements),
                    "storage_elements": encoding.storage_elements,
                    "offset_elements": encoding.offset_elements,
                }
            )
            continue
        if ceil_extent is None or extent is None:
            raise CodegenError("legacy caller layout requires the selected target's exact layout helpers")
        logical_shape = spec.get("shape")
        logical_rows, logical_cols = extent(spec, name=name)
        prows, pcols = ceil_extent(logical_rows), ceil_extent(logical_cols)
        if type(prows) is not int or type(pcols) is not int or prows < logical_rows or pcols < logical_cols:
            raise CodegenError("selected caller layout returned invalid physical extents")
        strides = [prod(logical_shape[axis + 1 : -1]) * pcols for axis in range(len(logical_shape) - 1)]
        rows.append(
            {
                "tensor": name,
                "dtype": spec["dtype"],
                "logical_shape": list(logical_shape),
                "physical_extents": [prows, pcols],
                "logical_strides_elements": strides + [1],
                "storage_elements": prows * pcols,
                "offset_elements": 0,
            }
        )
    if set(tensors) != {row["tensor"] for row in rows}:
        raise CodegenError("caller layout must cover exactly every declared tensor")
    return rows


def _contiguous_output_pointer(name: str, binding) -> str | None:
    """A closed row-major proof for the optional typed readback fast path.

    An output's dtype alone does not prove contiguity: explicit storage can pad
    or permute logical axes.  The same selected encoding that supplied the
    scalar readback offsets must prove every linear logical element has stride
    one in its physical buffer and lies inside its allocation.
    """
    encoding = binding.encoding
    shape = tuple(encoding.logical_shape)
    strides = tuple(encoding.logical_strides_elements)
    count = prod(shape)
    expected = tuple(prod(shape[axis + 1 :]) for axis in range(len(shape)))
    offset = encoding.offset_elements
    if (
        any(type(dim) is not int or dim <= 0 for dim in shape)
        or strides != expected
        or type(offset) is not int
        or offset < 0
        or type(encoding.storage_elements) is not int
        or count <= 0
        or count > encoding.storage_elements - offset
    ):
        return None
    return f"T_{name} + {offset}"


def describe_whole_program_layout(cb, *, legacy_helpers=None, legacy_dim=None):
    """Answer-free layout of the same caller buffers the renderer allocates."""
    from merlin.runtime.storage_binding import resolve_storage_bindings

    storage = resolve_storage_bindings(cb, max_storage_bytes=256 * 1024 * 1024, describe_only=True)
    if storage is not None:
        return {
            "schema": "caller_storage_layout_v1",
            "policy": {"mode": "declared_grouped_axes_storage_v1"},
            "tensors": _caller_layouts(cb, storage),
        }
    if legacy_dim is not None:
        if legacy_helpers is not None or type(legacy_dim) is not int or legacy_dim <= 0:
            raise CodegenError("legacy caller layout requires one explicit positive dimension")
        ceil_extent, extent = lambda value: ceil_dim(value, legacy_dim), buffer_extent
    elif legacy_helpers is not None:
        ceil_extent, extent = legacy_helpers
    else:
        raise CodegenError("legacy caller layout requires the selected target's layout helpers")
    return {
        "schema": "caller_storage_layout_v1",
        "policy": {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": ceil_extent(1)},
        "tensors": _caller_layouts(cb, None, ceil_extent=ceil_extent, extent=extent),
    }


def render_whole_program(
    cb: dict,
    *,
    inputs: dict | None = None,
    prepack_authorizations=None,
    legacy_helpers=None,
    measurement_fragments=None,
    legacy_dim=None,
    strict_profile_renderer=None,
    source_owned_mutables=None,
    readback_policy=None,
) -> str:
    """Call a submitted whole-program kernel through its explicitly declared pointer ABI.

    The harness allocates buffers, performs one unmeasured warm call, measures one call, and prints the
    declared result buffers. It does not interpret or reconstruct any operation between those calls;
    mesh and scalar-host work must both be present in the submitted ``gemmini_kernel`` artifact.
    """

    tensors = cb.get("tensors") or {}
    abi = cb.get("kernel_abi") or {}
    args = abi.get("args") or []
    outputs = abi.get("outputs") or []
    scratch: frozenset[str] = frozenset()
    if source_owned_mutables is not None:
        if (
            type(source_owned_mutables) is not tuple
            or any(type(name) is not str or not name for name in source_owned_mutables)
            or len(set(source_owned_mutables)) != len(source_owned_mutables)
            or (abi.get("kind") != "whole_program")
            or not isinstance(inputs, dict)
        ):
            raise CodegenError("source-owned scratch requires a distinct whole-program binding roster")
        scratch = frozenset(source_owned_mutables)
        if (
            len({arg.get("tensor") for arg in args}) != len(args)
            or set(inputs) != {arg["tensor"] for arg in args if arg.get("access") == "read"}
            or scratch
            != {
                arg["tensor"]
                for arg in args
                if arg.get("access") in ("write", "readwrite")
                and tensors.get(arg.get("tensor"), {}).get("role") == "intermediate"
            }
            or any(arg.get("access") == "readwrite" and arg.get("tensor") not in scratch for arg in args)
            or scratch.intersection(outputs)
        ):
            raise CodegenError("source-owned scratch differs from the exact pointer and entry ABI")
    # Opt-in storage is a format contract, not permission to compute derived operands here.
    # This is a host allocation safety cap, not a discovered hardware memory capacity.
    from merlin.runtime.storage_binding import resolve_storage_bindings

    try:
        storage = resolve_storage_bindings(
            cb, inputs, max_storage_bytes=256 * 1024 * 1024, prepack_authorizations=prepack_authorizations
        )
    except ValueError as exc:
        error = CodegenError(str(exc))
        if hasattr(exc, "storage_obligations"):
            error.storage_obligations = exc.storage_obligations
        raise error from exc
    leaves = None
    if storage is None:
        if legacy_dim is not None:
            if legacy_helpers is not None or type(legacy_dim) is not int or legacy_dim <= 0:
                raise CodegenError("legacy exact inputs require one explicit positive host ABI dimension")
            if any(arg.get("access") not in {"read", "write"} for arg in args):
                raise CodegenError("legacy exact input renderer refuses readwrite warm inputs")
            reads = {arg["tensor"] for arg in args if arg["access"] == "read"}
            if not isinstance(inputs, dict) or set(inputs) != reads:
                raise CodegenError("legacy caller requires all and only exact read input tensors")
            for name in reads:
                leaf, spec = inputs[name], tensors[name]
                if (
                    tuple(getattr(leaf, "shape", ())) != tuple(spec["shape"])
                    or getattr(leaf, "dtype", None) != spec["dtype"]
                ):
                    raise CodegenError("legacy exact input dtype/shape differs from caller tensor")

            def _ceil_dim(value):
                return ceil_dim(value, legacy_dim)

            _pad_rowmajor, _buffer_extent, leaves = pad_rowmajor, buffer_extent, inputs
        elif legacy_helpers is None:
            raise CodegenError("legacy whole-program storage requires explicit host layout helpers")
        else:
            _ceil_dim, _pad_rowmajor, _buffer_extent, materialize_inputs = legacy_helpers
            leaves = materialize_inputs(cb, inputs)

    decls: list[str] = []
    layouts: dict[str, tuple[int, int, int, int]] = {}
    if storage is None:
        for row in _caller_layouts(cb, None, ceil_extent=_ceil_dim, extent=_buffer_extent):
            shape = row["logical_shape"]
            prows, pcols = row["physical_extents"]
            rows = prod(shape[:-1]) if len(shape) > 1 else 1
            layouts[row["tensor"]] = (rows, shape[-1], prows, pcols)
    for arg in args:
        name, access = arg["tensor"], arg["access"]
        spec = tensors[name]
        if storage is not None:
            binding = storage[name]
            container = container_for(binding.encoding.dtype)
            initializer = (
                None
                if name in scratch or binding.logical_values is None
                else ",".join(
                    str(word)
                    for word in binding.pack_words(container_words(binding.logical_values, binding.encoding.dtype))
                )
            )
            decls.append(
                container.decl(
                    f"T_{name}", binding.encoding.storage_elements, const=access == "read", initializer=initializer
                )
            )
            continue
        rows, cols, prows, pcols = layouts[name]
        container = container_for(str(spec.get("dtype") or ""))
        if access == "write" or name in scratch:
            decls.append(container.decl(f"T_{name}", prows * pcols))
        else:
            padded = _pad_rowmajor(list(leaves[name].data), rows, cols, prows, pcols)
            decls.append(
                container.decl(
                    f"T_{name}",
                    prows * pcols,
                    const=access == "read",
                    initializer=",".join(str(word) for word in container_words(padded, str(spec.get("dtype") or ""))),
                )
            )

    call = ", ".join(f"(void*)T_{arg['tensor']}" for arg in args)
    prints: list[str] = []
    # A declared cap on how many values an output may print. Above it the output is printed as one
    # digest of exactly the text its values would have had (see runtime.backends.base): a simulated
    # UART cannot drain a million values inside any time limit, and a program that cannot finish
    # cannot be sealed.
    from merlin.runtime.commandbuffer import CONSOLE_VALUE_CAP_PARAM, OUTPUT_DIGEST_LINE
    from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, FULL_VALUES_BIN, selected

    value_cap = (cb.get("params") or {}).get(CONSOLE_VALUE_CAP_PARAM)
    if value_cap is not None and (isinstance(value_cap, bool) or not isinstance(value_cap, int) or value_cap < 1):
        raise CodegenError(f"params.{CONSOLE_VALUE_CAP_PARAM} must be a positive integer, got {value_cap!r}")
    policy = selected(readback_policy)
    full_values = policy is not None and policy.transport == FULL_VALUES_B64
    binary_values = policy is not None and policy.transport == FULL_VALUES_BIN
    packet_values = policy is not None and policy.transport == COHERENT_PACKET_V1
    coherent_values = policy is not None and policy.transport in (COHERENT_DUMP_V1, COHERENT_PACKET_V1)
    if packet_values:
        from merlin.runtime.out_packet import ExpectedOutput, out_bin_packet_capacity

    packet_expected = []
    if coherent_values:
        # The host dump names the same static caller allocations by their ELF
        # symbols. It never authorizes a new ABI, a dynamic output, or a dtype
        # whose physical word interpretation has not been reviewed.
        if (
            abi.get("kind") != "whole_program"
            or type(outputs) is not list
            or not outputs
            or len(outputs) > 1024
            or any(
                type(name) is not str
                or not name
                or not name.isascii()
                or not all(character.isalnum() or character == "_" for character in name)
                for name in outputs
            )
            or len(set(outputs)) != len(outputs)
        ):
            raise CodegenError("coherent dump requires a closed whole-program output roster")
        writable = {
            arg["tensor"]
            for arg in args
            if isinstance(arg, dict) and arg.get("access") in ("write", "readwrite")
        }
        if any(
            name not in writable
            or not isinstance(tensors.get(name), dict)
            or tensors[name].get("role") != "output"
            or tensors[name].get("dtype") not in _COHERENT_OUTPUT_DTYPES
            or container_for(tensors[name]["dtype"]).word_bytes not in (1, 2, 4, 8)
            for name in outputs
        ):
            raise CodegenError("coherent dump requires declared writable outputs of supported physical dtype")
        if set(outputs) != {
            name for name, spec in tensors.items()
            if isinstance(spec, dict) and spec.get("role") == "output"
        }:
            raise CodegenError("coherent dump output roster differs from declared output tensors")
        total_output_bytes = 0
        for name in outputs:
            if storage is not None:
                encoding = storage[name].encoding
                shape, allocated = encoding.logical_shape, encoding.storage_elements
                if encoding.dtype != tensors[name]["dtype"]:
                    raise CodegenError("coherent dump output storage dtype differs from declared dtype")
            else:
                rows, cols, prows, pcols = layouts[name]
                shape, allocated = (rows, cols), prows * pcols
            if (
                any(type(dim) is not int or dim <= 0 for dim in shape)
                or type(allocated) is not int
                or allocated <= 0
                or allocated > (256 * 1024 * 1024) // container_for(tensors[name]["dtype"]).word_bytes
            ):
                raise CodegenError("coherent dump requires positive static output storage")
            total_output_bytes += allocated * container_for(tensors[name]["dtype"]).word_bytes
            if total_output_bytes > 256 * 1024 * 1024:
                raise CodegenError("coherent dump exceeds the bounded output region total")
    digested = False

    def readback(name: str, rows: int, cols: int, container, element: str,
                 contiguous_pointer: str | None = None) -> list[str]:
        nonlocal digested
        if coherent_values and not packet_values:
            # The selected simulator captures these exact T_* allocations only
            # after a clean terminal barrier. No serial-value shortcut is used.
            return []
        loop = f"  for (long i = 0; i < {rows}; i++) for (long j = 0; j < {cols}; j++)"
        if packet_values:
            count = rows * cols
            if count <= 0 or count > ((1 << 64) - 1) // 8:
                raise CodegenError("coherent packet word count exceeds the versioned wire bound")
            shape = storage[name].encoding.logical_shape if storage is not None else tensors[name]["shape"]
            packet_expected.append(ExpectedOutput(
                name=name, rows=rows, cols=cols, logical_dtype=tensors[name]["dtype"],
                container_signed=container.signed, container_max_bytes=container.word_bytes,
                logical_shape=tuple(shape),
            ))
            sign = 1 if container.signed else 0
            scan = "signed" if container.signed else "unsigned"
            scan_type = "int64_t" if container.signed else "uint64_t"
            pack = (
                f"  if (!merlin_out_bin_words_i32(&merlin_out, {contiguous_pointer}, "
                f"{count}ULL)) return 2;"
                if container.ctype == "int32_t" and contiguous_pointer is not None
                else f"{loop} if (!merlin_out_bin_word(&merlin_out, (uint64_t)({element}))) return 2;"
            )
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_BIN_BEGIN v1 {name} ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {rows}ULL)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, " ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, {cols}ULL)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, " ")) return 2;',
                "  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, merlin_wire_width)) return 2;",
                "  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "
                'merlin_wire_signed ? " s " : " u ")) return 2;',
                f"  if (!merlin_out_bin_memory_decimal(&merlin_packet_sink, "
                f"{count}ULL * merlin_wire_width)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                f"  merlin_out_bin_init(&merlin_out, {count}ULL, merlin_wire_width, "
                + "merlin_wire_signed, merlin_packet_write);",
                pack,
                "  if (!merlin_out_bin_finish(&merlin_out)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "OUT_BIN_END v1 ")) return 2;',
                "  if (!merlin_out_bin_memory_hex16(&merlin_packet_sink, merlin_out.checksum)) return 2;",
                '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "\\n")) return 2;',
                "  }",
            ]
        if full_values:
            if not name.isascii() or not name or not all(character.isalnum() or character == "_" for character in name):
                raise CodegenError("packed output name is not a closed ASCII identifier")
            sign = 1 if container.signed else 0
            literal = f"OUT_B64_BEGIN v1 {name} {rows} {cols} 8 s\\n"
            header = f"OUT_B64_BEGIN v1 {name} {rows} {cols} %u %c\\n"
            scan = "signed" if container.signed else "unsigned"
            scan_type = "int64_t" if container.signed else "uint64_t"
            pack = (
                f"  if (!merlin_out_b64_words_i32(&merlin_out, {contiguous_pointer}, "
                f"{rows * cols}ULL)) return 2;"
                if container.ctype == "int32_t" and contiguous_pointer is not None
                else f"{loop} if (!merlin_out_b64_word(&merlin_out, (uint64_t)({element}))) return 2;"
            )
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {rows * cols}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  char merlin_begin[sizeof("{literal}")];',
                f'  sprintf(merlin_begin, "{header}", merlin_wire_width, merlin_wire_signed ? \'s\' : \'u\');',
                "  printstr(merlin_begin);",
                f"  merlin_out_b64_init(&merlin_out, {rows * cols}ULL, "
                + "merlin_wire_width, merlin_wire_signed, printstr);",
                pack,
                "  if (!merlin_out_b64_finish(&merlin_out)) return 2;",
                '  printstr("OUT_B64_END\\n");',
                "  }",
            ]
        if binary_values:
            if not name.isascii() or not name or not all(character.isalnum() or character == "_" for character in name):
                raise CodegenError("binary output name is not a closed ASCII identifier")
            count = rows * cols
            if count <= 0 or count > ((1 << 64) - 1) // 8:
                raise CodegenError("binary output byte count exceeds the versioned wire bound")
            sign = 1 if container.signed else 0
            scan = "signed" if container.signed else "unsigned"
            scan_type = "int64_t" if container.signed else "uint64_t"
            biggest = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} 8 s {count * 8}\\n"
            header = f"OUT_BIN_BEGIN v1 {name} {rows} {cols} %u %c %llu\\n"
            return [
                "  {",
                f"  merlin_out_b64_range_init(&merlin_out_range, {count}ULL, {container.word_bytes}u, {sign});",
                f"{loop} if (!merlin_out_b64_range_{scan}(&merlin_out_range, ({scan_type})({element}))) return 2;",
                "  unsigned merlin_wire_width = merlin_out_b64_range_width(&merlin_out_range);",
                "  int merlin_wire_signed = merlin_out_b64_range_wire_signed(&merlin_out_range);",
                "  if (!merlin_wire_width || merlin_wire_signed < 0) return 2;",
                f'  char merlin_begin[sizeof("{biggest}")];',
                f'  sprintf(merlin_begin, "{header}", merlin_wire_width, '
                f"merlin_wire_signed ? 's' : 'u', (unsigned long long)({count}ULL * merlin_wire_width));",
                "  printstr(merlin_begin);",
                f"  merlin_out_bin_init(&merlin_out, {count}ULL, "
                + "merlin_wire_width, merlin_wire_signed, printbuf);",
                f"{loop} if (!merlin_out_bin_word(&merlin_out, (uint64_t)({element}))) return 2;",
                "  if (!merlin_out_bin_finish(&merlin_out)) return 2;",
                "  char merlin_end[sizeof(\"OUT_BIN_END v1 0123456789abcdef\\n\")];",
                '  sprintf(merlin_end, "OUT_BIN_END v1 %016llx\\n", '
                "(unsigned long long)merlin_out.checksum);",
                "  printstr(merlin_end);",
                "  }",
            ]
        print_element = container.printf_element(element)
        if value_cap is None or rows * cols <= value_cap:
            return [f'  printf("OUT {name} {rows} {cols}");', f"{loop} {print_element}", '  printf("\\n");']
        if not print_element.startswith("printf("):
            raise CodegenError("an output's element printer is not a printf call; it cannot be digested")
        digested = True
        return [
            "  merlin_outsum = 1469598103934665603ULL;",
            f"{loop} MERLIN_OUTSUM_ADD({print_element[len('printf(') :]}",
            f'  printf("{OUTPUT_DIGEST_LINE} {name} {rows} {cols} %016llx\\n", merlin_outsum);',
        ]

    for name in outputs:
        if storage is not None:
            binding = storage[name]
            shape = binding.encoding.logical_shape
            rows, cols = (prod(shape[:-1]), shape[-1]) if shape else (1, 1)
            terms = [str(binding.encoding.offset_elements)]
            for divisor, extent, stride in binding.logical_offset_terms():
                terms.append(f"(((i * {cols} + j) / {divisor}) % {extent}) * {stride}")
            element = f"T_{name}[{' + '.join(terms)}]"
            prints.extend(readback(name, rows, cols, container_for(binding.encoding.dtype), element,
                                   _contiguous_output_pointer(name, binding)))
            continue
        rows, cols, prows, pcols = layouts[name]
        container = container_for(str(tensors[name].get("dtype") or ""))
        prints.extend(readback(name, rows, cols, container, f"T_{name}[i * {pcols} + j]",
                               f"T_{name}" if pcols == cols and prows >= rows else None))
    if packet_values:
        try:
            packet_capacity = out_bin_packet_capacity(packet_expected)
        except ValueError as exc:
            raise CodegenError(str(exc)) from exc
    if coherent_values:
        # The selected Spike/FESVR host may export this *existing* static
        # allocation after normal guest exit. GSim obtains the same bytes from
        # its separately selected coherent terminal dump. The aliases add no
        # guest reads, stores, branches, or changes to the caller ABI.
        # Selected GCC currently lays out these distinct static declarations
        # in reverse source order. These aliases only propose a span: the
        # trusted post-link reader must prove that every unique sized writable
        # output symbol exactly tiles it, with no gap or overlap. Another
        # linker layout is a refusal, not a silently inferred signature.
        output_names = set(outputs)
        declared_outputs = [arg["tensor"] for arg in args if arg["tensor"] in output_names]
        if len(declared_outputs) != len(outputs) or set(declared_outputs) != output_names:
            raise CodegenError("coherent dump output declarations differ from the exact output roster")
        first, last = declared_outputs[-1], declared_outputs[0]
        allocated = storage[last].encoding.storage_elements if storage is not None else layouts[last][2] * layouts[last][3]
        byte_count = allocated * container_for(tensors[last]["dtype"]).word_bytes
        decls.append(
            '__asm__(".globl begin_signature\\n"\n'
            f'        ".set begin_signature, T_{first}\\n"\n'
            '        ".globl end_signature\\n"\n'
            f'        ".set end_signature, T_{last}+{byte_count}\\n");'
        )
    if full_values:
        decls.append("static merlin_out_b64 merlin_out;")
        decls.append("static merlin_out_b64_range merlin_out_range;")
    if binary_values:
        decls.append("static merlin_out_bin merlin_out;")
        decls.append("static merlin_out_b64_range merlin_out_range;")
    if packet_values:
        decls.extend([
            f"unsigned char merlin_readback_packet[{packet_capacity}] __attribute__((used));",
            "volatile uint64_t merlin_readback_packet_used __attribute__((used));",
            "static merlin_out_bin_memory merlin_packet_sink;",
            "static merlin_out_bin merlin_out;",
            "static merlin_out_b64_range merlin_out_range;",
            "static int merlin_packet_write(const void *bytes, size_t count) {",
            "  return merlin_out_bin_memory_append(&merlin_packet_sink, bytes, count);",
            "}",
        ])
    if digested:
        # Same text the OUT line would have carried, through the runtime's own formatter, so the
        # digest is a function of the values as printed and of nothing else.
        decls.append(
            "static unsigned long long merlin_outsum;\n"
            "#define MERLIN_OUTSUM_ADD(...) do { char b_[48]; int n_ = sprintf(b_, __VA_ARGS__); "
            "for (int k_ = 0; k_ < n_; k_++) merlin_outsum = (merlin_outsum ^ (unsigned char)b_[k_]) "
            "* 1099511628211ULL; } while (0)"
        )

    if strict_profile_renderer is not None:
        if packet_values:
            raise CodegenError("coherent packet requires the declared standard whole-program measurement")
        if not callable(strict_profile_renderer):
            raise CodegenError("strict warm profile renderer must be target-owned callable code")
        rendered = strict_profile_renderer(arguments=call, readback="\n".join(prints))
        if (
            not isinstance(rendered, dict)
            or set(rendered) != {"declarations", "main"}
            or not all(isinstance(rendered[name], str) and rendered[name].strip() for name in rendered)
        ):
            raise CodegenError("strict warm profile renderer returned an invalid source assembly")
        return (
            "#include <stdint.h>\n#include <stdio.h>\n"
            + ('#include "out_b64.h"\nextern void printstr(const char *);\n' if full_values else "")
            + ('#include "out_b64.h"\n#include "out_bin.h"\n'
               'extern void printstr(const char *);\nextern int printbuf(const void *, size_t);\n' if binary_values else "")
            + rendered["declarations"]
            + "\n"
            + "\n".join(decls)
            + "\n"
            + rendered["main"]
        )

    measured_call = f"  gemmini_kernel({call});\n  gemmini_fence();"
    fragments = (measurement_fragments or assemble_measurement_fragments)(measured_call)
    packet_setup = (
        f"  merlin_out_bin_memory_init(&merlin_packet_sink, merlin_readback_packet, {packet_capacity}u, "
        "&merlin_readback_packet_used);\n"
        "  if (!merlin_packet_sink.valid) return 2;\n"
    ) if packet_values else ""
    packet_finish = (
        '  if (!merlin_out_bin_memory_cstr(&merlin_packet_sink, "DONE\\n")) return 2;\n'
        '  __asm__ __volatile__("fence rw,rw" ::: "memory");\n'
        "  if (!merlin_out_bin_memory_finish(&merlin_packet_sink)) return 2;\n"
        '  __asm__ __volatile__("fence rw,rw" ::: "memory");\n'
    ) if packet_values else ""
    return (
        '#include <stdint.h>\n#include <stdio.h>\n#include "include/gemmini_testutils.h"\n'
        + ('#include "out_b64.h"\nextern void printstr(const char *);\n' if full_values else "")
        + ('#include "out_b64.h"\n#include "out_bin.h"\n'
           'extern void printstr(const char *);\nextern int printbuf(const void *, size_t);\n' if binary_values else "")
        + ('#include "out_b64.h"\n#include "out_bin.h"\n#include "out_bin_memory.h"\n' if packet_values else "")
        + fragments["include"]
        + "extern void gemmini_kernel();\n"
        + "\n".join(decls)
        + "\nint main() {\n"
        + packet_setup
        + fragments["warmup"]
        + fragments["prologue"]
        + "  uint64_t c0 = read_cycles();\n"
        + measured_call
        + "\n"
        + "  uint64_t c1 = read_cycles();\n"
        + fragments["epilogue"]
        + '  printf("METRIC cycles %lu\\n", (unsigned long)(c1 - c0));\n'
        '  printf("METRIC cycle_window_gemmini_region 1\\n");\n' + "\n".join(prints) + "\n"
        + packet_finish
        + '  printf("DONE\\n");\n  return 0;\n}\n'
    )
