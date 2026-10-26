"""Gemmini-owned source catalog and Spike extension selection for Core ATen."""

import hashlib
from pathlib import Path

from merlin.llvmlower import toolchain
from merlin.llvmlower.device_build import DeviceRouting
from merlin.targetgen.plugins import load_module

TARGET = "gemmini"


def _selected_numeric_policy(directory=None):
    import json
    import os
    from merlin.targetgen.software_spec import load_software_spec

    admission = Path(directory) / 'spec-int8-admission.json' if directory is not None else None
    if admission is not None and admission.exists():
        return json.loads(admission.read_text())['selection']['spec']
    selected = os.environ.get('MERLIN_CORE_ATEN_SOFTWARE_SPEC')
    return load_software_spec(Path(selected), target=TARGET) if selected else None


def _readout_options(numeric_policy):
    """An explicitly selected f32 scheme may request the derived store path."""
    strategy = ((numeric_policy or {}).get('numerical_semantics', {}).get('readout') or {}).get('strategy')
    if strategy != 'f32_product_rne':
        return None
    from merlin.targetgen.rtl.facts import load_facts
    from merlin.targetgen.rtl.output_epilogue import derive
    from merlin.targetgen.target_experiment import load_capability_manifest

    facts = load_facts(TARGET)
    contract = load_capability_manifest(TARGET).contract
    path = (contract.get('output_epilogue') or derive(facts, contract=contract)).get('output_path')
    if path is None or not any(path.get('exits', {}).values()):
        return None
    options = dict(path=path, scheme=dict(readout_strategy=strategy), facts=facts)
    if numeric_policy.get('compositions'):
        # Declared compositions are planned only from the selected policy.
        options.update(compositions=numeric_policy['compositions'],
                       numerical_semantics=numeric_policy['numerical_semantics'])
    return options


def prepare_bundle(directory, *, target, facts):
    from merlin.llvmlower.modular_contractions import prepare_bundle as prepare
    from merlin.system.offload import device_dtype_triples

    return prepare(Path(directory), device_dtype_triples(target))


def routing(directory, *, target, package, facts, eligible, readout_options=None):
    from merlin.system.offload import device_dtype_triples

    # A closed seeded MAC + byte readout needs the region ABI: its seed and
    # result types differ from the old two-input/raw-i32 contraction ABI.
    regions = _core_aten_module(package, "integer_regions")
    limbs = _core_aten_module(package, "limbs")
    parsing = _core_aten_module(package, "catalog")
    source = (Path(directory) / "model.mlir").read_text()
    parsed = parsing.parse_module(source)
    numeric_policy = _selected_numeric_policy(directory)
    readout_options = readout_options or _readout_options(numeric_policy)
    selected = []
    if regions.has_integer_readout(parsed) or limbs.has_limb_plan(parsed) or readout_options is not None:
        _, inventory, matches = parsing.build(source, readout_options=readout_options)
        selected = [m for m in matches if m.kernel.meta.get("integer_producer") or m.kernel.meta.get("integer_readout") or m.kernel.meta.get('limb_plan')
                    or m.kernel.meta.get('scaled_operand_sum') or m.kernel.meta.get('composition_kernel')]
    if selected:
        # Logical centered operands are i32, but their proved physical
        # products are signed byte x byte -> i32. Admission derives that
        # datapath from the closed lowering, never by narrowing the source.
        if ('i8','i8','i32') not in set(device_dtype_triples(target)):
            return None
        operand, weight, accum = 'i8','i8','i32'
        def region_builder(prepared, workdir):
            # Re-prove the actual prepared bytes, not a preceding capture.
            if (Path(directory) / "spec-int8-admission.json").exists():
                import json
                policy = json.loads((Path(directory) / "spec-int8-admission.json").read_text())["selection"]["spec"]
                proof = admit_numeric_schedule(Path(prepared).read_text(), package=package, numeric_policy=policy,
                                                readout_options=readout_options)
                if proof.get('status') != 'proven':
                    workdir = Path(workdir);workdir.mkdir(parents=True,exist_ok=True)
                    manifest = workdir/'core_aten_catalog.json'
                    manifest.write_text(json.dumps(dict(status='declined',reason=proof.get('reason'),
                        kernels=[],bindings=[],numeric_schedule_proof=proof),indent=2)+'\n')
                    return None,None,None,manifest
            return core_aten_offload(prepared, workdir, package=package, numeric_policy=numeric_policy,
                                     readout_options=readout_options)
        audit = load_module(package, "mlir_oot.golden_device_catalog", package_name="core_aten_region_audit")
        return DeviceRouting(device=target, package_dir=package, operand_dtype=operand,
                             accum_dtype=accum, region_offload=region_builder,
                             final_elf_audit=audit.final_elf_audit)
    triples = {tuple(shape.dtypes) for _, shape in eligible}
    if len(triples) != 1 or not triples.issubset(set(device_dtype_triples(target))):
        return None
    operand, weight, accum = next(iter(triples))
    if operand != weight:
        return None
    # Immutable snapshots at different roots must not share cached Python modules.
    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
    backend = load_module(package, "mlir_oot.golden_device_catalog", package_name=f"core_aten_backend_{identity}")
    # The provider owns semantic matching. Unsupported operations remain host work.
    source = (Path(directory) / "model.mlir").read_text()
    _, inventory = backend.build_catalog(source)
    if not inventory["covered_contractions"]:
        return None
    builder = backend.merlin_builder(toolchain.llvm_install() / "bin")
    admission_path = Path(directory) / "spec-int8-admission.json"
    if admission_path.exists():
        import json

        admission = json.loads(admission_path.read_text())
        numeric_policy = admission["selection"]["spec"]

        def checked_builder(prepared, workdir):
            proof = admit_numeric_schedule(Path(prepared).read_text(), package=package, numeric_policy=numeric_policy)
            manifest, obj = builder(prepared, workdir)
            actual = json.loads(Path(manifest).read_text())
            if actual["source_sha256"] != proof["source_sha256"] or any(
                kernel["schedule"] != next(p["schedule"] for p in proof["kernels"] if p["symbol"] == kernel["symbol"])
                for kernel in actual["kernels"]
            ):
                raise ValueError("compiled catalog differs from the proved numerical schedule")
            actual["numeric_schedule_proof"] = proof
            Path(manifest).write_text(json.dumps(actual, indent=2) + "\n")
            return manifest, obj

        selected_builder = checked_builder
    else:
        selected_builder = builder
    return DeviceRouting(
        device=target,
        package_dir=package,
        operand_dtype=operand,
        accum_dtype=accum,
        catalog_builder=selected_builder,
        final_elf_audit=backend.final_elf_audit,
    )


def _core_aten_module(package, name):
    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
    return load_module(package, f"mlir_oot.core_aten.{name}", package_name=f"core_aten_backend_{identity}")


def core_aten_catalog(prepared, workdir, *, package, llvm_bin=None, numeric_policy=None, readout_options=None):
    """Bind and compile the exactly implementable Core ATen regions of ``prepared``.

    Returns (manifest path, object path), or (manifest path, None) when declined.
    The manifest (schema gemmini_core_aten_catalog_v1)
    lists each kernel's argument table and each binding's root ordinal, replaced ordinals and
    operand sources; regions it cannot implement exactly are listed under ``declines``.  This is
    opt-in beside ``routing``: the integer GEMM catalog and its bindings are unchanged.
    ``readout_options`` selects Merlin's path/scheme/facts composition contract;
    a missing or refused proof retains the source host boundary.
    """
    catalog = _core_aten_module(package, "catalog")
    if readout_options is None:
        readout_options = _readout_options(numeric_policy or _selected_numeric_policy())
    workdir = Path(workdir)
    manifest = catalog.compile_catalog(Path(prepared), workdir,
                            llvm_bin=Path(llvm_bin) if llvm_bin else toolchain.llvm_install() / "bin",
                            readout_options=readout_options)
    return workdir / "core_aten_catalog.json", (workdir / "core_aten.o" if manifest['kernels'] else None)


def core_aten_offload(prepared, workdir, *, package, llvm_bin=None, numeric_policy=None, readout_options=None):
    """Rewrite the bound regions of ``prepared`` into C-interface calls.

    Returns (rewritten MLIR, shim C source, kernel object, manifest).  After any lowering with
    function-boundary bufferization and ``llvm.emit_c_interface`` lowering, the rewritten model calls
    ``_mlir_ciface_ca_<symbol>``, which the shim defines over the kernel object.  The host image
    must link both (and provide malloc for the call results' destinations).
    With no kernel, returns (None, None, None, refusal manifest) without emitting
    rewritten/shim files. Callers must inspect this refusal before linking.
    """
    import json

    manifest_path, obj = core_aten_catalog(prepared, workdir, package=package, llvm_bin=llvm_bin,
                                         numeric_policy=numeric_policy, readout_options=readout_options)
    if obj is None:
        return None, None, None, manifest_path
    offload = _core_aten_module(package, "offload")
    manifest = json.loads(Path(manifest_path).read_text())
    workdir = Path(workdir)
    rewritten = workdir / "core_aten_offloaded.mlir"
    rewritten.write_text(offload.rewrite(Path(prepared).read_text(), manifest) + "\n")
    shim = workdir / "core_aten_shim.c"
    shim.write_text(offload.shim_source(manifest))
    return rewritten, shim, obj, manifest_path


def runner_options():
    from merlin.runtime.backends.spike import spike_path

    binary = Path(spike_path()).resolve()
    # Extension and tool search roots are properties of the selected tool installation.
    prefix = binary.parent.parent
    return {
        "spike_binary": binary,
        "extension": "gemmini",
        "extlib": prefix / "lib" / "libgemmini.so",
        "path_prepend": [prefix.parent / "bin"],
    }


def full_call_oracle(target):
    """Use submitted source-bound device catalogs at the full-call L2 boundary."""
    from merlin.targetgen.rtl.facts import ensure_facts
    from merlin.targetgen.spike_full_call import full_call_adapter
    from merlin.targetgen.target_experiment import load_capability_manifest

    isa = load_capability_manifest(target).contract["isa"]["march"]
    return full_call_adapter(
        target,
        isa=isa,
        runner_options=runner_options(),
        execution_provider=Path(__file__),
        rtl_facts=ensure_facts(target),
    )


def admit_numeric_schedule(source, *, package, numeric_policy, readout_options=None):
    """Prove the selected integer stream and its complete source-bound region."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    if numeric_policy is None:
        # Standalone callers may use the same explicitly selected declaration
        # as the real lane. Never infer internal MAC arithmetic from a name.
        import os
        from merlin.targetgen.software_spec import load_software_spec
        selected = os.environ.get('MERLIN_CORE_ATEN_SOFTWARE_SPEC')
        if not selected:
            return dict(status='declined', reason='no selected numerical policy; pass numeric_policy or set MERLIN_CORE_ATEN_SOFTWARE_SPEC',
                        source_sha256=hashlib.sha256(source.encode()).hexdigest(), kernels=[], bindings=[])
        numeric_policy = load_software_spec(Path(selected), target=TARGET)

    regions = _core_aten_module(package, "integer_regions")
    catalog_module = _core_aten_module(package, "catalog")
    limbs = _core_aten_module(package, "limbs")
    parsed = catalog_module.parse_module(source)
    if limbs.has_limb_plan(parsed):
        return _admit_limb_regions(source, package=package, numeric_policy=numeric_policy)
    readout_options = readout_options or _readout_options(numeric_policy)
    integer_matches = []
    if readout_options is not None:
        _, _, matches = catalog_module.build(source, readout_options=readout_options)
        # The same region kinds routing selects; a scaled operand sum has no
        # contraction, so the legacy GEMM admission would find nothing to cover.
        integer_matches = [m for m in matches if m.kernel.meta.get('integer_producer') or
                           m.kernel.meta.get('integer_readout') or m.kernel.meta.get('scaled_operand_sum') or
                           m.kernel.meta.get('composition_kernel')]
    if regions.has_integer_readout(parsed) or integer_matches:
        return _admit_integer_regions(source, package=package, numeric_policy=numeric_policy,
                                      readout_options=readout_options)
    linear_refusal = None
    if any(getattr(getattr(o, 'op_name', None), 'data', o.name).startswith('quant_ext.') for o in parsed.walk()):
        proof = _admit_quantized_linear_regions(source, package=package, numeric_policy=numeric_policy)
        if proof is not None:
            if proof.get('reason') != 'provider did not cover every live quantized arithmetic producer':
                return proof
            # A layout prelude is not a complete linear island. The legacy
            # contraction ABI deliberately retains source layout/host work;
            # authenticate its actual GEMMs independently before refusing.
            linear_refusal = proof
    identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
    backend = load_module(package, "mlir_oot.golden_device_catalog", package_name=f"core_aten_backend_{identity}")
    _, catalog = backend.build_catalog(source)
    # These are the very Shape/GoldenGemm classes used by build_catalog.
    from dataclasses import asdict
    from importlib import import_module

    kernel_module = import_module(backend.__package__ + ".golden_gemm")
    isa = kernel_module.isa
    geometry = kernel_module.F.DIM
    proofs = []
    for kernel in catalog["kernels"]:
        shape = kernel_module.Shape(**kernel["schedule"])
        bound = integer_split_k_limit(numeric_policy, tile_dim=geometry, full_k=shape.k)
        if bound is None or geometry > bound["max_padded_k"]:
            raise ValueError("provider padded mesh K exceeds selected internal MAC bound")
        emitted = kernel_module.GoldenGemm(shape).build()
        commands = list(emitted.walk())
        configs = [op for op in commands if isinstance(op, kernel_module.G.ConfigExOp)]
        if not configs or any(op.attributes["dataflow"].value.data != isa.WEIGHT_STATIONARY for op in configs):
            raise ValueError("numeric proof requires the emitted weight-stationary dataflow")
        if shape.bias or shape.output_dtype != "i32" or shape.scale != 1.0 or shape.relu:
            raise ValueError("numeric proof requires from-zero raw-i32 readout")
        computes = [op for op in commands if isinstance(op, kernel_module.G.ComputeOp)]
        if not computes:
            raise ValueError("schedule emitted no primitive computes")
        extents = []
        reset_rows, accumulation_rows = set(), set()
        for compute in computes:
            k = int(compute.attributes["a_cols"].value.data)
            if not 1 <= k <= geometry <= bound["max_padded_k"]:
                raise ValueError("emitted compute exceeds proved padded primitive K")
            siblings = list(compute.parent_block().ops)
            prior = siblings[: siblings.index(compute)]
            preload = next((op for op in reversed(prior) if isinstance(op, kernel_module.G.PreloadOp)), None)
            if preload is None or preload.attributes["bd_rows"].value.data != k:
                raise ValueError("compute lacks matching stationary weight preload")
            if compute.a("bd", isa.GARBAGE_ADDR) != isa.GARBAGE_ADDR:
                raise ValueError("WS mesh D must be garbage, not prior accumulator feedback")
            c = int(preload.attributes["c"].value.data)
            if not c & isa.ACC_ADDR_BIT:
                raise ValueError("WS partial dot product does not target accumulator SRAM")
            row = c & ~(isa.ACC_ADDR_BIT | isa.ACC_ACCUMULATE_BIT | isa.ACC_FULL_ROW_BIT)
            if c & isa.ACC_ACCUMULATE_BIT:
                if row not in reset_rows:
                    raise ValueError("i32 accumulation lacks prior zero/overwrite initialization")
                accumulation_rows.add(row)
            else:
                reset_rows.add(row)
            # In WS, PRELOAD's BD is B, D is garbage (no accumulator feedback
            # into the 20-bit mesh). C's accumulation bit addresses 32-bit SRAM.
            extents.append(k)
        if shape.k > geometry and reset_rows != accumulation_rows:
            raise ValueError("not all initialized output rows receive later K tiles")
        stores = [op for op in commands if isinstance(op, kernel_module.G.MvoutOp)]
        if not stores or any(not op.a("local") & isa.ACC_FULL_ROW_BIT for op in stores):
            raise ValueError("accumulator DMA readout is not raw full-i32")
        maximum = geometry * bound["maximum_product"]
        proofs.append(
            dict(
                symbol=kernel["symbol"],
                schedule=asdict(shape),
                status="proven",
                primitive_k=sorted(set(extents)),
                initialized_accumulator_rows=sorted(reset_rows),
                accumulated_rows=sorted(accumulation_rows),
                padded_primitive_k=geometry,
                k_split=[min(geometry, shape.k - start) for start in range(0, shape.k, geometry)],
                max_partial_sum_bound=maximum,
                mac_positive_limit=(1 << (bound["mac_result_bits"] - 1)) - 1,
                headroom=((1 << (bound["mac_result_bits"] - 1)) - 1) - maximum,
                full_result_bound=bound["full_result_bound"],
                bound=bound,
                aggregation="WS mesh independent dot products; accumulator SRAM i32 overwrite first K tile, add subsequent K tiles",
                mesh_initialization="WS D is garbage; compute accumulate preserves stationary B, not a preceding dot product",
                emitted_schedule_sha256=hashlib.sha256(str(emitted).encode()).hexdigest(),
            )
        )
    if not proofs or not catalog["coverage_complete"]:
        if linear_refusal is not None:
            return linear_refusal
        return dict(status='declined', reason='provider did not cover every proposed integer contraction',
                    source_sha256=catalog['source_sha256'], kernels=[], bindings=[])
    return dict(status="proven", source_sha256=catalog["source_sha256"], kernels=proofs, bindings=catalog["bindings"])


def _admit_quantized_linear_regions(source, *, package, numeric_policy):
    """Authenticate closed rational linear islands, not their Q/DQ boundaries alone."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit
    catalog_module = _core_aten_module(package, 'catalog')
    _, catalog, matches = catalog_module.build(source)
    if not any(m.kernel.meta.get('form') in ('exact_linear_i32','exact_movement_dma') or 'reduced' in m.kernel.meta
               or m.numeric.get('readout') == 'storage_codes' for m in matches):
        return None
    parsed = catalog_module.parse_module(source)
    operations = list(parsed.walk())
    live = _core_aten_module(package, 'liveness').live_source_ops(parsed)
    covered = {ordinal for b in catalog['bindings'] for ordinal in b['replaced_ordinals']}
    def decline(reason):
        return dict(status='declined', reason=reason, source_sha256=catalog['source_sha256'],
                    kernels=[], bindings=[], declines=catalog['declines'])
    # A partial linear match must not hide another live arithmetic producer.
    if any(o in live and o.name.startswith('linalg.') and o.name not in ('linalg.yield', 'linalg.index')
           and i not in covered for i, o in enumerate(operations)):
        return decline('provider did not cover every live quantized arithmetic producer')
    proofs = {}
    kernels = {k['symbol']: k for k in catalog['kernels']}
    for binding, match in zip(catalog['bindings'], matches):
        meta, stream = match.kernel.meta, match.kernel.stream
        instructions = list(stream.ins.templates()) if hasattr(stream.ins,'templates') else stream.ins
        if meta.get('readout') == 'storage_codes':
            transfers = [i for i in instructions if i.kind in ('mvin', 'mvout')]
            if (stream.readouts or stream.packs or not transfers
                    or any(i.a('local') & (1 << 31) for i in transfers)
                    or any(i.kind not in ('fence', 'flush', 'config_ld', 'config_st', 'mvin', 'mvout')
                           for i in instructions)
                    or any(i.a('scale') != 1 for i in instructions if i.kind == 'config_ld')
                    or any(i.a('acc_scale') != 1 for i in instructions if i.kind == 'config_st')
                    or meta.get('identical_qparam_proof') != binding['numeric'].get('identical_qparam_proof')):
                return decline('storage-code movement is not a proved scratchpad byte stream')
            proofs[binding['symbol']] = dict(symbol=binding['symbol'], status='proven', exact_arithmetic=True,
                device_readout='storage_codes', schedule=meta, requantization=[],
                emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest())
            continue
        if meta.get('form')=='exact_movement_dma':
            if (not stream.readouts
                    or any(i.kind not in ('fence','flush','config_ld','config_st','mvin','mvout') for i in instructions)
                    or any(i.kind=='mvout' and i.a('local') & (1<<31) and not i.a('local') & (1<<29) for i in instructions)
                    or any(i.a('scale')!=1 for i in instructions if i.kind=='config_ld')
                    or any(i.a('acc_scale')!=1 for i in instructions if i.kind=='config_st')):
                return decline('exact movement must use unit DMA and full-width widening before host requantization')
            proofs[binding['symbol']]=dict(symbol=binding['symbol'],status='proven',exact_arithmetic=True,
                device_readout='full_i32',schedule=meta,requantization=kernels[binding['symbol']]['requantization'],
                emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest())
            continue
        if not (meta.get('form') == 'exact_linear_i32' or 'reduced' in meta):
            return decline('closed linear proof cannot authenticate a different numeric schedule')
        try:
            bound = integer_split_k_limit(numeric_policy, tile_dim=16, full_k=1)
        except ValueError as exc:
            return decline(str(exc))
        if bound is None or bound['signed_operand_bits'] != 8:
            return decline('no reviewed signed integer MAC bound for closed linear schedule')
        intervals = meta.get('accumulator_bounds') or [dict(accumulator_min=meta.get('accumulator_min'),
                                                        accumulator_max=meta.get('accumulator_max'))]
        computes = [i for i in stream.ins if i.kind == 'compute']
        stores = [i for i in stream.ins if i.kind == 'mvout']
        if (not stream.readouts or not stores
                or any(not i.a('local') & (1 << 29) for i in stores)
                or any(i.a('scale') != 1 for i in stream.ins if i.kind == 'config_ld')
                or any(i.a('acc_scale') != 1 for i in stream.ins if i.kind == 'config_st')
                or any(i.a('dataflow') != 1 for i in stream.ins if i.kind == 'config_ex')
                or any(i.a('a_cols') > 16 or i.a('bd') != 0xffffffff
                       or i.a('a_cols') * bound['maximum_product'] > (1 << (bound['mac_result_bits']-1))-1
                       for i in computes)
                or any(type(row.get('accumulator_min')) is not int or type(row.get('accumulator_max')) is not int
                       or not -(1 << 31) <= row['accumulator_min'] <= row['accumulator_max'] < (1 << 31)
                       for row in intervals)):
            return decline('closed linear schedule exceeds selected exact MAC/readout contract')
        proofs[binding['symbol']] = dict(symbol=binding['symbol'], status='proven', exact_arithmetic=True,
            device_readout='full_i32', schedule={**meta, 'readout':'full_i32',
                'requantization':'exact_rational_host_epilogue'}, bound=bound,
            requantization=kernels[binding['symbol']]['requantization'],
            emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest())
    return dict(status='proven', source_sha256=catalog['source_sha256'], kernels=list(proofs.values()),
                bindings=catalog['bindings'], declines=catalog['declines'], abi='gemmini_core_aten_catalog_v1')


def _admit_limb_regions(source, *, package, numeric_policy):
    """Authenticate the actual F2/F4 stream against the selected MAC facts."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit
    catalog_module = _core_aten_module(package, 'catalog')
    _, catalog, matches = catalog_module.build(source)
    parsed = catalog_module.parse_module(source)
    live = _core_aten_module(package, 'liveness').live_source_ops(parsed)
    proposed = sum('gemmini.limb' in o.attributes for o in live)
    proofs, covered = {}, 0
    for binding, match in zip(catalog['bindings'], matches):
        meta = match.kernel.meta
        if 'limb_plan' not in meta:
            continue
        try:
            bound = integer_split_k_limit(numeric_policy, tile_dim=16,
                                          full_k=meta.get('dimensions', {}).get('k', 1))
        except ValueError as exc:
            return dict(status='declined', reason=str(exc), kernels=[], bindings=[],
                        source_sha256=catalog['source_sha256'])
        if bound is None:
            return dict(status='declined', reason='no reviewed signed integer MAC bound for limb schedule',
                        kernels=[], bindings=[], declines=catalog['declines'], source_sha256=catalog['source_sha256'])
        stream = match.kernel.stream
        computes = [i for i in stream.ins if i.kind == 'compute']
        proof = meta['bound_proof']
        valid = (bound['signed_operand_bits'] == 8 and
                 proof['required_signed_mac_bits'] <= bound['mac_result_bits'] and
                 all(i.a('a_cols') <= proof['primitive_k'] and i.a('bd') == 0xffffffff for i in computes) and
                 all(i.a('dataflow') == 1 for i in stream.ins if i.kind == 'config_ex') and
                 all(i.a('local') & (1 << 29) for i in stream.ins if i.kind == 'mvout') and
                 all(i.a('scale') == 1 for i in stream.ins if i.kind == 'config_ld') and
                 all(i.a('acc_scale') == 1 for i in stream.ins if i.kind == 'config_st'))
        if not valid:
            return dict(status='declined', reason='limb schedule exceeds selected exact MAC/readout contract',
                        kernels=[], bindings=[], source_sha256=catalog['source_sha256'])
        covered += 1
        proofs[binding['symbol']] = dict(symbol=binding['symbol'], status='proven', schedule=meta,
                                        exact_arithmetic=True, device_readout='full_i32', bound=bound,
                                        emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest())
    if not proofs or covered != proposed:
        return dict(status='declined', reason='provider did not cover every exact limb producer: ' + str(catalog['declines']),
                    kernels=[], bindings=[], declines=catalog['declines'], source_sha256=catalog['source_sha256'])
    return dict(status='proven', source_sha256=catalog['source_sha256'], kernels=list(proofs.values()),
                bindings=catalog['bindings'], declines=catalog['declines'], abi='gemmini_core_aten_catalog_v1')


def _admit_shared_input(match, binding, *, package, numeric_policy, readout_options):
    """Prove one fused sibling kernel from its decoded stream; see ``core_aten.shared_input``."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit
    shared = _core_aten_module(package, 'shared_input')
    tile_dim = _core_aten_module(package, 'stream').DIM
    dims = match.kernel.meta['dimensions']
    try:
        bound = integer_split_k_limit(numeric_policy, tile_dim=tile_dim, full_k=dims['k'])
    except ValueError as exc:
        return dict(status='declined', reason=str(exc))
    if bound is None or bound['max_padded_k'] < tile_dim:
        return dict(status='declined', reason='provider padded mesh K exceeds selected internal MAC bound')
    if match.kernel.meta['readout'] == 'f32_product_rne' and numeric_policy.get(
            'numerical_semantics', {}).get('readout', {}).get('strategy') != 'f32_product_rne':
        return dict(status='declined', reason='composition boundary differs from selected numerical policy')
    computes = [i for i in match.kernel.stream.ins if i.kind == 'compute']
    if not computes or any(i.a('a_cols') > tile_dim for i in computes):
        return dict(status='declined', reason='sibling primitive exceeds the proved mesh depth')
    slices = [(row['offset'], tuple(row['shape'])) for row in binding.get('result_slices', ())]
    if slices != [(offset, tuple(extent)) for _, offset, extent, _ in match.outputs]:
        return dict(status='declined', reason='bound result slices differ from the emitted sibling results')
    try:
        proof = shared.admit(match, readout_options=readout_options)
    except ValueError as exc:
        return dict(status='declined', reason=str(exc))
    return dict(symbol=binding['symbol'], status='proven', schedule=match.kernel.meta, shared_input=proof,
                padded_primitive_k=tile_dim, max_partial_sum_bound=tile_dim*bound['maximum_product'],
                bound=bound, exact_arithmetic=True, device_readout=match.kernel.meta['device_readout'],
                quantization='device' if match.kernel.meta['device_readout'] == 'f32_product_rne' else 'none',
                seed='zero', emitted_schedule_sha256=hashlib.sha256(repr(match.kernel.stream.ins).encode()).hexdigest())


def _admit_scaled_operand_sum(binding, match, *, package, numeric_policy):
    """Authenticate the emitted scaled loads and readout against the source-derived proof."""
    operand_sum = _core_aten_module(package, 'operand_sum')
    proof = match.kernel.meta['scaled_operand_sum']
    try:
        device_readout = operand_sum.validate(match.kernel.stream, match.kernel.meta)
    except ValueError as exc:
        return dict(status='declined', reason=str(exc))
    strategy = numeric_policy.get('numerical_semantics', {}).get('readout', {}).get('strategy')
    if proof['reference'] == 'source_f32_operations' and strategy != 'f32_product_rne':
        return dict(status='declined', reason='scaled operand sum differs from selected numerical policy')
    return dict(symbol=binding['symbol'], status='proven', schedule=match.kernel.meta, exact_arithmetic=True,
                device_readout=device_readout, quantization='device', operand_sum=proof,
                emitted_schedule_sha256=hashlib.sha256(repr(match.kernel.stream.ins).encode()).hexdigest())


def _admit_integer_regions(source, *, package, numeric_policy, readout_options=None):
    """Prove the emitted seeded MAC/readout stream, including its source binding."""
    from merlin.targetgen.operation_numerics import integer_split_k_limit
    catalog_module = _core_aten_module(package, "catalog")
    stream_module = _core_aten_module(package, 'stream')
    requant = _core_aten_module(package, 'requant')
    from importlib import import_module
    isa = import_module(stream_module.__package__.rsplit('.', 1)[0] + '.tables.isa')
    tile_dim = stream_module.DIM
    _, catalog, matches = catalog_module.build(source, readout_options=readout_options)
    proofs = []
    def refuse(reason):
        return dict(status='declined',reason=reason,source_sha256=catalog['source_sha256'],
                    kernels=[],bindings=[],declines=catalog['declines'])
    for binding, match in zip(catalog["bindings"], matches):
        if match.kernel.meta.get('composition_kernel'):
            try:
                proofs.append(dict(symbol=binding['symbol'], **_core_aten_module(package, 'pooled_store').admit(
                    match, numeric_policy, tile_dim=tile_dim)))
            except ValueError as exc:
                return refuse(str(exc))
            continue
        if match.kernel.meta.get('scaled_operand_sum'):
            proof = _admit_scaled_operand_sum(binding, match, package=package, numeric_policy=numeric_policy)
            if proof.get('status') != 'proven':
                return refuse(proof['reason'])
            proofs.append(proof)
            continue
        if not (match.kernel.meta.get("integer_producer") or match.kernel.meta.get("integer_readout")):
            continue
        if match.kernel.meta.get('shared_input'):
            row = _admit_shared_input(match, binding, package=package, numeric_policy=numeric_policy,
                                      readout_options=readout_options)
            if row.get('status') != 'proven':
                return refuse(row['reason'])
            proofs.append(row)
            continue
        dims = match.kernel.meta["dimensions"]
        try:
            bound = integer_split_k_limit(numeric_policy, tile_dim=tile_dim, full_k=dims["k"])
        except ValueError as exc:
            return refuse(str(exc))
        if bound is None or bound["max_padded_k"] < tile_dim:
            return refuse("provider padded mesh K exceeds selected internal MAC bound")
        stream = match.kernel.stream
        computes = [i for i in stream.ins if i.kind == "compute"]
        composed = (readout_options or {}).get('numerical_semantics')
        if composed is not None and composed != numeric_policy.get('numerical_semantics'):
            return refuse('composition facts differ from the selected numerical policy')
        if not computes or any(i.a("a_cols") > tile_dim for i in computes):
            return refuse("seeded primitive exceeds the proved mesh depth")
        # The replayed dataflow (``contraction_proof``) admits a non-WS command
        # only where the source-derived composition plan licenses it.
        if not match.kernel.meta.get('dataflow_plan') and (
                any(i.a("bd") != isa.GARBAGE_ADDR for i in computes) or
                any(i.a("dataflow") != isa.WEIGHT_STATIONARY for i in stream.ins if i.kind == "config_ex")):
            return refuse("seeded proof requires weight-stationary execution")
        try:
            device_readout = requant.validate_device_readout(stream, match.kernel.meta, numeric=match.numeric,
                                                             match=match, readout_options=readout_options)
        except ValueError as exc:
            return refuse(str(exc))
        plan = match.kernel.meta.get('composition_plan')
        if plan is not None and plan['decision']['proof'].get('definitional') == 'f32_product_rne':
            strategy = numeric_policy.get('numerical_semantics', {}).get('readout', {}).get('strategy')
            if strategy != 'f32_product_rne':
                return refuse('composition boundary differs from selected numerical policy')
        proofs.append(dict(symbol=binding["symbol"], status="proven", schedule=match.kernel.meta,
            primitive_k=sorted({i.a("a_cols") for i in computes}), padded_primitive_k=tile_dim,
            k_split=[min(tile_dim, dims["k"]-k) for k in range(0,dims["k"],tile_dim)],
            max_partial_sum_bound=tile_dim*bound["maximum_product"],
            mac_positive_limit=(1 << (bound["mac_result_bits"]-1))-1,
            headroom=((1 << (bound["mac_result_bits"]-1))-1)-tile_dim*bound["maximum_product"],
            bound=bound, exact_arithmetic=True, device_readout=device_readout,
            quantization=('device' if device_readout == 'f32_product_rne' else
                          'none' if match.kernel.meta.get('raw_integer') else 'retained_source_exact_integer_host_readout'),
            seed=match.kernel.meta["seed"],
            accumulator_min=match.kernel.meta["accumulator_min"], accumulator_max=match.kernel.meta["accumulator_max"],
            full_result_bound=max(abs(match.kernel.meta["accumulator_min"]),abs(match.kernel.meta["accumulator_max"])),
            emitted_schedule_sha256=hashlib.sha256(repr(stream.ins).encode()).hexdigest()))
    parsed = catalog_module.parse_module(source)
    from merlin.kernels.shapes import observe_contractions
    live = _core_aten_module(package, 'liveness').live_source_ops(parsed)
    ordinals = {op:i for i,op in enumerate(parsed.walk())}
    proposed = {ordinals[op] for op,shape in observe_contractions(parsed)
                if op in live and shape.dtypes[-1] == 'i32'}
    covered = [ordinal for binding,match in zip(catalog['bindings'],matches)
               if match.kernel.meta.get('integer_producer') or match.kernel.meta.get('integer_readout')
               for ordinal in binding.get('source_operation_ordinals', [binding['source_operation_ordinal']])]
    if proofs and (set(covered) != proposed or len(covered) != len(set(covered))):
        return refuse('provider did not cover every live integer producer')
    proofs = list({row['symbol']: row for row in proofs}.values())
    if not proofs:
        return dict(status='declined', reason='provider did not cover any exact seeded readout region',
                    source_sha256=catalog['source_sha256'], kernels=[], bindings=[], declines=catalog['declines'])
    return dict(status="proven", source_sha256=catalog["source_sha256"], kernels=proofs,
                bindings=catalog["bindings"], declines=catalog["declines"], abi="gemmini_core_aten_catalog_v1")
