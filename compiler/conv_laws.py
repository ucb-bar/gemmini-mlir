"""RTL ordering admission checks adapted from the public conv_epoch recipe."""

class Unavailable(ValueError):
    pass

def roles(record):
    """Command roles from the facts' local accesses and the shared ISA class vocabulary: a transfer
    in (one local write), a readout (one local read), a descriptor (a contraction class that reads
    one local operand and writes another) and a compute (a contraction class with one local read)."""
    from merlin.targetgen.semantic_families import from_isa_class, movement_classes

    if not isinstance(record, dict) or record.get("status") not in ("derived", "partial"):
        raise Unavailable("command_ordering facts are not derived")
    extents = record.get("extents") or {}
    by_class = {}
    for key, access in (record.get("accesses") or {}).items():
        by_class.setdefault(access["class"], []).append(dict(access, key=key))

    def shape(cls):
        rows = by_class[cls]
        derived = all((extents.get(a["key"]) or {}).get("status") == "derived" for a in rows)
        reads = [a for a in rows if a["direction"] == "read"]
        writes = [a for a in rows if a["direction"] == "write"]
        return derived, len(reads), len(writes)

    def role(cls):
        rows = by_class[cls]
        return dict(
            cls=cls,
            funct=rows[0]["funct"],
            queue=rows[0]["queue"],
            read=next((a for a in rows if a["direction"] == "read"), None),
            write=next((a for a in rows if a["direction"] == "write"), None),
        )

    def pick(test, label):
        found = sorted((by_class[c][0]["funct"], c) for c in by_class if shape(c)[0] and test(c, *shape(c)[1:]))
        if not found:
            raise Unavailable(f"no {label} class with derived local accesses")
        # the lowest funct encodes the stream; every alternative of the role is checked for the flags
        return dict(role(found[0][1]), alternatives=[role(c) for _, c in found])

    inbound, outbound = movement_classes("in"), movement_classes("out")
    contraction = {c for c in by_class if from_isa_class(c) == "contraction"}
    r = dict(
        transfer=pick(lambda c, n_r, n_w: c in inbound and (n_r, n_w) == (0, 1), "transfer-in"),
        readout=pick(lambda c, n_r, n_w: c in outbound and (n_r, n_w) == (1, 0), "readout"),
        descriptor=pick(lambda c, n_r, n_w: c in contraction and (n_r, n_w) == (1, 1), "descriptor"),
        compute=pick(lambda c, n_r, n_w: c in contraction and (n_r, n_w) == (1, 0), "compute"),
    )
    if r["descriptor"]["queue"] != r["compute"]["queue"]:
        raise Unavailable("descriptor and compute classes enter different queues")
    return r

def law_flags(record, structure, r):
    """Each law flag -> {status: fact | program | UNKNOWN, basis}. ``fact`` cites the ordering record;
    ``program`` is a property of this program's allocation or completion callback, not of hardware."""
    cross = {(c["older"], c["younger"], c["space"]): c for c in record.get("cross_queue") or ()}
    same = record.get("same_queue") or {}
    order = record.get("completion_order") or {}
    by_code = {}
    for row in record.get("funct_queue") or ():
        for code, queue in (row.get("by_code") or {}).items():
            by_code[int(code)] = queue
    classes = ((structure.get("command_classes") or {}).get("classes")) or {}
    T, R, D, C = (r[k] for k in ("transfer", "readout", "descriptor", "compute"))

    def waits(older, younger, space):
        row = cross.get((older["key"], younger["key"], space))
        if row is None or row.get("overlap") != "completion":
            return None
        return (
            f"cross_queue[{older['key']} -> {younger['key']}, {space}]: younger waits for completion "
            f"when overlapping (disjoint: {row.get('disjoint')}, tracked: {row.get('tracked')})"
        )

    def every(pairs):
        cites = [waits(*p) for p in pairs]
        return cites if all(cites) else None

    def flag(cites, missing):
        return {"status": "fact", "basis": cites} if cites else {"status": "UNKNOWN", "basis": [missing]}

    def in_order(queue):
        o, rel = (order.get(queue) or {}).get("order"), (same.get(queue) or {}).get("release")
        if o == "in_order" or rel == "completion":
            return f"queue {queue}: completion_order {o}, same_queue release {rel}"
        return None

    def entries(queue, need):
        e = (classes.get(queue) or {}).get("entries") or {}
        if e.get("status") == "derived" and type(e.get("value")) is int and e["value"] >= need:
            return f"timing_structure.command_classes.{queue}.entries = {e['value']} (derived, finite, >= {need})"
        return None

    def config(accesses):
        cites = []
        for a in accesses:
            ext = (record.get("extents") or {}).get(a["key"]) or {}
            for code in sorted({int(s["code"]) for s in ([ext["scale"]] if ext.get("scale") else [])}):
                rel = (same.get(a["queue"]) or {}).get("release")
                if by_code.get(code) != a["queue"] or rel not in ("issue", "completion"):
                    return None
                cites.append(f"configuration code {code} enters queue {a['queue']} (funct_queue), issued in order")
        return cites or [f"{', '.join(a['key'] for a in accesses)}: no scaled extent; nothing latched"]

    loads = [t["write"] for t in T["alternatives"]]
    readers = [c["read"] for c in C["alternatives"]] + [D["read"]]
    raw = every([(w, x, "scratchpad") for w in loads for x in readers])
    consumed = every([(x, w, "scratchpad") for w in loads for x in readers])
    readout_war = every([(R["read"], D["write"], "accumulator")])
    war = consumed + readout_war if consumed and readout_war else None
    acc = every([(D["write"], R["read"], "accumulator")])
    waw = None
    initialized = every([(w, D["write"], "accumulator") for w in loads])  # a loaded bias, then products
    if in_order(D["queue"]) and raw and war and initialized:
        same_t = in_order(T["queue"])
        waw = [in_order(D["queue"]), *initialized] + (
            [same_t]
            if same_t
            else [
                f"queue {T['queue']} writes may land out of order: a rewrite of a row is ordered only through "
                "the compute that consumed it (the read-after-write then write-after-read waits above); "
                "the row-reuse and completion-order checks below verify every rewrite has that consumer"
            ]
        )
    pair = None
    if (same.get(C["queue"]) or {}).get("release") in ("issue", "completion"):
        pair = [f"{D['cls']} and {C['cls']} both enter queue {C['queue']}, which issues in order"]
    progress = entries(C["queue"], 2)
    backpressure = [x for x in (entries(T["queue"], 1), entries(C["queue"], 2)) if x]
    drain = {"status": "program", "basis": ["the completion callback is the same full drain the drain route uses"]}
    epoch = {
        "local_read_after_write": flag(raw, "a load write -> compute read wait is not derived"),
        "local_write_after_read": flag(war, "a consumer read -> later write wait is not derived"),
        "local_write_after_write": flag(waw, "same-queue write order and its consumer chain are not derived"),
        "accumulator_result_visibility": flag(acc, "descriptor write -> readout read wait is not derived"),
        "operand_consumption_retirement": flag(consumed, "a load does not wait for the consumer's completion"),
        "ordered_configuration_capture": flag(
            config([C["read"], D["read"], D["write"]]), "a latched configuration code enters another queue"
        ),
        "adjacent_descriptor_compute": flag(pair, "descriptor/compute queue issue order is not derived"),
        "finite_queue_pair_progress": flag([progress] if progress else None, "dispatch entries not derived"),
        "complete_device_drain": drain,
        "exclusive_private_state": {
            "status": "program",
            "basis": ["the plan owns its LHS/RHS/accumulator rows inside the declared stores for the invocation"],
        },
    }
    issue = {
        "local_range_dependencies": flag(raw and war and raw + war, "range waits are not derived"),
        "ordered_load_configuration": flag(config(loads), "a transfer's configuration enters another queue"),
        "paired_descriptor_compute": flag(pair, "descriptor/compute queue issue order is not derived"),
        "finite_queue_backpressure": flag(backpressure if len(backpressure) == 2 else None, "entries not derived"),
        "complete_device_drain": drain,
        "immutable_mapped_disjoint_buffers": {
            "status": "program",
            "basis": ["inputs and weights stay mapped and unchanged; outputs are disjoint from them"],
        },
    }
    return {"epoch": epoch, "issue": issue}

def admitted(table):
    return all(row["status"] != "UNKNOWN" for row in table.values())


def selected_laws(source_id, load_bound, pair_bound):
    """Consume frozen local ordering selection; unknown facts keep seed route."""
    import hashlib
    import json
    from pathlib import Path
    from merlin.compile.scheduling.window_transfer import ConvolutionLoadIssueLaw, ConvolutionComputeEpochLaw
    if not (0 < load_bound < 2**31 and 1 < pair_bound < 2**31):
        return None, None, {'refusal': 'software issue work exceeds declared bounded law domain'}
    here = Path(__file__).resolve().parent
    selected = here / 'rtl_facts/command_ordering_selection.json'
    if not selected.is_file():
        return None, None, {'refusal': 'ordering selection unavailable'}
    data = selected.read_bytes()
    if hashlib.sha256(data).hexdigest() != 'f941f5e2199b59f1d6649739682b66081958cbb0e3b35dabc839913c86a80970':
        return None, None, {'refusal': 'ordering selection bytes changed'}
    doc = json.loads(data)
    # Configuration persistence/capture is tied to the same selected controller
    # bytes used by this emitter's fixed-stream configuration proof.
    controller_pins = {
        'LoadController.scala': '780ee0a292d8653fa8a822aee28e2fb2cffbc25301431b4332182008d34fcf6a',
        'StoreController.scala': 'a0b556d109196c3c9d9ac5ed4dffc1bcad334c0f1f87780d948b84bde7402c6d',
    }
    for name, digest in controller_pins.items():
        path = here / 'rtl_facts' / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            return None, None, {'refusal': 'selected configuration controller bytes unavailable or changed'}
    facts_data = (here / 'rtl_facts/facts.json').read_bytes()
    if hashlib.sha256(facts_data).hexdigest() != doc['base_facts']['sha256']:
        return None, None, {'refusal': 'ordering selection targets different facts'}
    facts = json.loads(facts_data)['facts']
    flags = law_flags(doc['ordering'], facts['timing_structure'], roles(doc['ordering']))
    ids = ('gemmini-selected-ordering-' + doc['ordering_sha256'],
           'signed-i8-product-modular-i32-v1', source_id,
           'source-pinned-dispatch-ordering-private-rows-v1')
    issue = ConvolutionLoadIssueLaw(*ids, load_bound,
        **{k: v['status'] != 'UNKNOWN' for k, v in flags['issue'].items()}) if admitted(flags['issue']) else None
    epoch = ConvolutionComputeEpochLaw(*ids, pair_bound,
        **{k: v['status'] != 'UNKNOWN' for k, v in flags['epoch'].items()},
        pending_compute_pairs=facts['timing_structure']['command_classes']['classes']['execute']['entries']['value']//2
        ) if admitted(flags['epoch']) else None
    return issue, epoch, flags


def selected_readout_law(issue, epoch, source_id, maximum):
    """Independent ACC RAW and ordered store configuration admission."""
    import json
    from pathlib import Path
    from merlin.compile.scheduling.window_transfer import ConvolutionReadoutEpochLaw
    if issue is None or epoch is None:
        return None
    here = Path(__file__).resolve().parent
    record = json.loads((here/'rtl_facts/command_ordering_selection.json').read_text())['ordering']
    config_store = any(r.get('funct') == 0 and r.get('by_code',{}).get('2') == 'store'
                       for r in record['funct_queue'])
    ordered = record['same_queue'].get('store',{}).get('release') == 'completion'
    facts = json.loads((here/'rtl_facts/facts.json').read_text())['facts']
    entries = facts['timing_structure']['command_classes']['classes']['store']['entries']
    if not config_store or not ordered or entries.get('status') != 'derived':
        return None
    return ConvolutionReadoutEpochLaw(issue.hardware_identity, issue.numerical_policy_identity,
        source_id, 'selected-ACC-RAW-store-queue-configuration-v1', maximum, entries['value'],
        local_read_after_write=epoch.accumulator_result_visibility,
        accumulator_result_visibility=epoch.accumulator_result_visibility,
        ordered_configuration_capture=True, finite_queue_progress=True,
        complete_device_drain=True, exclusive_private_state=True)
