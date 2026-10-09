"""Reproduce the geometry pairing of two separately qualified hardware profiles.

This is a performance study, not compiler selection logic. Read only the named
owned reference extraction and our receipts; never search external directories.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def pin(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def reconcile(current_root, reference_root):
    current_root, reference_root = Path(current_root), Path(reference_root)
    files = {
        "current_profile": current_root / "docs/perf_records/resnet1849_current_profile_firesim.json",
        "current_geometry": current_root / "docs/perf_records/resnet1849_current_issue_geometry.json",
        "reference_profile": current_root / "docs/perf_records/q1013_diagnostic_reference_firesim.json",
        "reference_shapes": reference_root / "work/layers.json",
        "reference_dynamic_commands": reference_root / "work/dyn.json",
        "reference_group_commands": reference_root / "work/groups.json",
        "driver": Path(__file__),
    }
    pins = {name: pin(path) for name, path in files.items()}
    profile = json.loads(files["current_profile"].read_text())
    geometry = json.loads(files["current_geometry"].read_text())
    reference = json.loads(files["reference_profile"].read_text())
    shapes = json.loads(files["reference_shapes"].read_text())
    dynamic = json.loads(files["reference_dynamic_commands"].read_text())
    groups = json.loads(files["reference_group_commands"].read_text())
    assert profile["job_id"] == geometry["profile_job"] == 1850
    assert geometry["unprofiled_control_job"] == 1849 and reference["job_id"] == 1876
    assert geometry["actual_profile_sha256"] == pins["current_profile"]["sha256"]
    for receipt_name, receipt in (("current", profile), ("reference", reference)):
        assert receipt["state"] == receipt["phase"] == "DONE" and receipt["exit_code"] == 0
        for field in ("elf", "uart"):
            pins[receipt_name + "_" + field] = pin(receipt[field])
            assert pins[receipt_name + "_" + field]["sha256"] == receipt[field + "_sha256"]
    for field in ("hw_config", "hwdb_sha256", "bitstream_archive_sha256", "bitstream_sha256"):
        assert profile[field] == reference[field], field
    pins["current_catalog"] = pin(geometry["catalog"])
    assert pins["current_catalog"]["sha256"] == geometry["catalog_sha256"]
    catalog = json.loads(Path(geometry["catalog"]).read_text())
    pins["current_prepared_source"] = pin(catalog["source_snapshot"])
    assert pins["current_prepared_source"]["sha256"] == catalog["source_sha256"]
    current = {event["region"]: event for event in geometry["events"]}
    source_calls = {call["region"]: call for call in profile["source_bound_calls"]}
    timing = {record["index"]: record for record in reference["layer_cycles"]}
    assert len(shapes) == 53 and set(timing) == set(range(1, 55))
    routes = catalog["fused_requantizations"]
    assert len(routes) == 52 and len(current) == len(source_calls) == 70
    function_calls = Counter(row[7] for row in shapes)
    rows = []
    reference_classes, current_classes = Counter(), Counter()
    for index, ref in enumerate(shapes, 1):
        assert ref[0] == "conv_" + str(index)
        if index == 1:
            region, category = "stem", "pooled_stem"
            dims = (12544, 147, 64)
            schedule = catalog["pooled_stem"]["shape"]
            kind = "source-proved pooled stem"
            readout = None
        else:
            route = routes[index - 2]
            region, schedule, kind = route["region"], route["schedule"], route["schedule_kind"]
            readout = route["integer_readout"]
            category = "direct" if ref[1].startswith("3x3_") else "pointwise"
            if route["direct_conv"]:
                assert category == "direct"
                dims = (((schedule["h"] - 1) // schedule["stride"] + 1)
                        * ((schedule["w"] - 1) // schedule["stride"] + 1),
                        9 * schedule["cin"], schedule["cout"])
            else:
                assert category == "pointwise"
                dims = (schedule["m"], schedule["k"], schedule["n"])
        assert dims == tuple(ref[2:5]), (region, dims, ref)
        event, call = current[region], source_calls[region]
        assert event["category"] == call["category"] == category
        assert event["actual_device_cycles"] == call["device_cycles"]
        assert event["call_index"] == call["call_index"]
        commands = dynamic[ref[7]]
        divisor = function_calls[ref[7]]
        assert all(value % divisor == 0 for name, value in commands.items() if name != "INSN")
        commands = {name: value // divisor for name, value in commands.items() if name != "INSN"}
        ref_cycles, cycles = timing[index]["cycles"], event["actual_device_cycles"]
        reference_classes[category] += ref_cycles
        current_classes[category] += cycles
        rows.append(dict(reference_index=index, reference_layer=ref[0],
            reference_function=ref[7], reference_geometry_class=ref[1],
            reference_printed_timer_kind=timing[index]["kind"],
            current_region=region, current_call_index=call["call_index"],
            current_source_operation_ordinal=call["operation_ordinal"],
            geometry=dict(m=dims[0], k=dims[1], n=dims[2]),
            reference_cycles=ref_cycles, current1850_device_cycles=cycles,
            difference_cycles=cycles-ref_cycles,
            current1850_preceding_host_interval=call["preceding_host_gap_cycles"],
            current1849_schedule=kind, current1849_shape=schedule,
            current_integer_readout=readout,
            reference_executed_commands=commands,
            current1849_analytical_commands=event["commands"],
            current_device_object_sha256=event["device_object_sha256"],
            semantics_equal=False,
            timing_boundary_note="Separate numerical/epilogue contracts; preceding host interval includes all intervening CPU operations"))
    classifier = current["matmul_53"]
    call = source_calls["matmul_53"]
    assert classifier["category"] == "classifier"
    assert classifier["commands"]["m"] == 1 and classifier["commands"]["n"] == 1000
    assert classifier["commands"]["k"] == 2048
    rows.append(dict(reference_index=54, reference_layer="classifier", current_region="matmul_53",
        current_call_index=call["call_index"], current_source_operation_ordinal=call["operation_ordinal"],
        geometry=dict(m=1, k=2048, n=1000), reference_cycles=timing[54]["cycles"],
        current1850_device_cycles=classifier["actual_device_cycles"],
        difference_cycles=classifier["actual_device_cycles"]-timing[54]["cycles"],
        current1850_preceding_host_interval=call["preceding_host_gap_cycles"],
        semantics_equal=False, timing_boundary_note="Reference narrowed output versus source i32 device accumulation/f32 output path"))
    reference_classes["classifier"] = timing[54]["cycles"]
    current_classes["classifier"] = classifier["actual_device_cycles"]
    residual = sum(event["actual_device_cycles"] for event in current.values() if event["category"] == "residual")
    aggregate = {}
    for line in reference["aggregate_lines"]:
        label, value = line.split(": ", 1)
        aggregate[label] = int(value.split()[0])
    reference_classes["residual"] = aggregate["Res add cycles"]
    current_classes["residual"] = residual
    assert sum(timing[index]["cycles"] for index in timing) == aggregate["Matmul cycles"] + aggregate["Conv cycles"]
    assert sum(reference_classes.values()) + aggregate["Other cycles"] == aggregate["Total cycles"]
    assert sum(current_classes.values()) == profile["boundary_profile"]["device_counter"]
    assert aggregate["FM full model cycles"] == reference["kernel_cycles"]
    assert aggregate["Total cycles"] + aggregate["FM uncounted delta"] == reference["kernel_cycles"]
    current_classes["host_intervals"] = profile["boundary_profile"]["host_gap_counter"]
    reference_classes["host_intervals"] = aggregate["Other cycles"] + aggregate["FM uncounted delta"]
    comparison = [dict(category=name, reference1876_cycles=reference_classes[name],
        current1850_cycles=current_classes[name], difference_cycles=current_classes[name]-reference_classes[name])
        for name in ("pointwise", "direct", "residual", "pooled_stem", "classifier", "host_intervals")]
    total_difference = sum(row["difference_cycles"] for row in comparison)
    assert total_difference == profile["boundary_profile"]["forward_counter"] - reference["kernel_cycles"]
    host = sorted(profile["source_bound_calls"], key=lambda call: call["preceding_host_gap_cycles"], reverse=True)
    return dict(schema="q1013_1849_paired_hardware_intervals_v1",
        timestamp_utc=datetime.now(timezone.utc).isoformat(), owner="reference_parity", pins=pins,
        hardware={field: reference[field] for field in ("hw_config", "hwdb_sha256", "bitstream_archive_sha256", "bitstream_sha256")},
        reference_scope="Job1876 modified reference diagnostic; active path matches owned archive; own numeric contract only",
        current_scope="Job1850 profiled exact1849 control; neither1853 nor1874 per-layer timing",
        reference1876_whole_cycles=reference["kernel_cycles"],
        current1850_forward_cycles=profile["boundary_profile"]["forward_counter"],
        current1850_whole_cycles=profile["kernel_cycles"],
        instrumentation_overhead_vs1849=profile["profiled_minus_control_cycles"],
        causal_savings_claim=False, source_numeric_equivalence=False,
        reference_printed_class_reclassified_by_geometry=True,
        matching_convolution_geometries=53, class_intervals=comparison, paired_layers=rows,
        residual_scope="16 current source-bound calls versus reference aggregate only; no reference per-residual timers",
        residual_commands=dict(reference_executed=groups["resadd"]["PRE"],
            current1849=sum(event["selected_padded_issue_cycles"] // 16 for event in current.values()
                            if event["category"] == "residual"),
            explanation="Source quantization scales differ;39chunks for first residual are minimal only within current coefficient arithmetic family"),
        host_intervals=host, current_host_tail_cycles=profile["boundary_profile"]["tail_counter"],
        top7_host_intervals_cycles=sum(call["preceding_host_gap_cycles"] for call in host[:7]),
        intrinsic_differences=["Current f32 NCHW entry versus reference int8 entry",
            "Current immutable quantization scales/weights differ from reference",
            "Current full1000f32 output versus reference narrowed classifier output",
            "Current integer-readout epilogues can reside outside device timer",
            "Stem/pool and timer ABI scopes must be retained when interpreting interval differences"],
        token_usage_available=False,
        token_note="Parent retains owned-thread counters; child goal tracker unavailable, no exclusive optimization attribution")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-root", required=True, type=Path)
    parser.add_argument("--reference-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    receipt = reconcile(args.current_root, args.reference_root)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["class_intervals"], indent=2))
