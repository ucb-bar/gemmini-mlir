"""Complete decoding and symbolic replay of an emitted primitive stream.

Admission must not trust the emitter. ``decode`` turns every command into the
RoCC words the kernel will actually issue and splits each word into every
field of its RTL-derived register bundle. A bit outside all declared fields
refuses. ``Replay`` then walks the decoded commands with the effective
CONFIG_EX/CONFIG_LD/CONFIG_ST state. A field whose value the replay does not
model (an allowlist, not a list of checked fields) refuses at the first
command that depends on it, so a configuration changed between two dependent
commands is seen at the command it affects.

The replay records, for every local row, which source elements it holds: a
DRAM row tag ``(argument, first flat element, valid columns)`` for loads, and
for an accumulator row its seed plus every mesh contribution ``(A row tag,
stationary row tags, depth, columns)``. Stores write symbolic DRAM contents,
so a later load of a stored element (a scratch gather, or an accumulator
re-seeded from an earlier partial result) resolves to what was stored, after a
fence. ``execute`` runs the same decoded commands on concrete values: the
differential model admission compares against an independent reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np


class ProofError(ValueError):
    pass


def require(condition, reason):
    if not condition:
        raise ProofError(reason)


@dataclass(frozen=True)
class Tag:
    name: str
    flat: int  # flat element index of the row's first column
    width: int  # valid columns
    # Per column ``(store index, stored row, stored column)`` when the bytes were
    # produced earlier in this stream; ``None`` for host-defined arguments.
    content: tuple | None = None


ZERO = Tag("", 0, 0)


def is_zero(tag):
    return tag is not None and tag.name == ""


@dataclass
class AccRow:
    seed: Tag | None
    terms: list = field(default_factory=list)  # (A tag, stationary tags, depth, columns)
    stored: bool = False

    def snapshot(self):
        return AccRow(self.seed, list(self.terms), self.stored)


@dataclass(frozen=True)
class Store:
    """One decoded MVOUT: its source rows as they were when it issued."""
    index: int
    ins: object
    name: str
    accumulator: bool
    full: bool
    config: dict
    rows: tuple  # AccRow snapshots or scratchpad Tags, one per stored row
    columns: int


# ---------------------------------------------------------------------------------------------
# Decoding: encoded words -> every field of the RTL-derived bundles
# ---------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Command:
    kind: str  # semantic class (config subtypes resolved), or FENCE
    fields: dict
    arg: int | None
    off: int  # DRAM byte offset relative to ``arg``; the raw rs1 for a zero fill


@lru_cache(maxsize=None)
def _layout(bundle):
    from .readout_encoding import field, slots
    layout = slots().get(bundle)
    require(isinstance(layout, dict) and layout.get("fields"), f"UNKNOWN register bundle {bundle}")
    return {name: field(bundle, name) for name in layout["fields"]}


def _mask(offset, width):
    return ((1 << width) - 1) << offset


def _bundle(word, bundle, *, prefix="", covered=0):
    fields, mask = {}, covered
    for name, (offset, width) in _layout(bundle).items():
        fields[prefix + name] = (word >> offset) & ((1 << width) - 1)
        mask |= _mask(offset, width)
    require(not word & ~mask, f"{bundle} sets bits outside every modelled field")
    return fields


@lru_cache(maxsize=None)
def _names(table):
    from .readout_encoding import contract
    return {int(k): v for k, v in contract()["encoding"][table].items()}


@lru_cache(maxsize=None)
def _selector():
    from .readout_encoding import contract
    spec = contract()["rocc_operand_roles"]["config"]["selector"]
    return spec["bundle"], spec["field"]


@lru_cache(maxsize=None)
def _load_states():
    """MVIN class -> load state, from the provider's own command table."""
    from ..tables import isa
    names = _names("semantic_class")
    return {names[isa.mvin(local_addr=0, cols=1, rows=1, load_id=state)[0]]: state for state in range(3)}


def decode(ins) -> Command:
    from .emit import encode
    record = encode(ins)
    if record.key == ("fence",):
        return Command("FENCE", {}, None, 0)
    require(len(record.key) in (2, 3), "unknown encoded operand key")
    funct, arg = record.key[:2]
    token = record.key[2] if len(record.key) == 3 else None
    if len(record.key) == 3:
        from .runtime_scale import RuntimeScaleRef
        require(isinstance(token, RuntimeScaleRef) and arg is None,
                "runtime scale key has an invalid token or DMA argument")
    rs1, rs2 = record.values
    kind = _names("semantic_class").get(funct)
    require(kind is not None, f"funct {funct} has no declared semantic class")
    if kind == "CONFIG":
        bundle, name = _selector()
        offset, width = _layout(bundle)[name]
        kind = _names("config_subtype").get((rs1 >> offset) & ((1 << width) - 1))
        require(kind is not None, "configuration subtype is undeclared")
        selector = _mask(offset, width)
        if kind == "CONFIG_EX":
            fields = _bundle(rs1, "ConfigExRs1", covered=selector)
            fields.update(_bundle(rs2, "ConfigExRs2"))
        elif kind == "CONFIG_LD":
            fields = _bundle(rs1, "ConfigMvinRs1", covered=selector)
            stride_width = _layout("ConfigMvoutRs2")["stride"][1]
            require(rs2 < 1 << stride_width, "CONFIG_LD DRAM stride exceeds its register field")
            fields["dram_stride"] = rs2
        else:
            fields = _bundle(rs1, "ConfigMvoutRs1", covered=selector)
            fields.update(_bundle(rs2, "ConfigMvoutRs2"))
        if token is not None:
            require(kind == "CONFIG_ST" and fields["acc_scale"] == 0,
                    "runtime token must fill only the zeroed CONFIG_ST scale field")
            fields["acc_scale"] = token
        return Command(kind, fields, None, 0)
    require(token is None, "runtime token is only legal on CONFIG_ST")
    if kind in _load_states():
        fields = _bundle(rs2, "MvinRs2")
        fields["state"] = _load_states()[kind]
        return Command(kind, fields, arg, rs1)
    if kind == "MVOUT":
        return Command(kind, _bundle(rs2, "MvoutRs2"), arg, rs1)
    if kind == "PRELOAD":
        fields = _bundle(rs1, "PreloadRs", prefix="bd_")
        fields.update(_bundle(rs2, "PreloadRs", prefix="c_"))
        return Command(kind, fields, None, 0)
    if kind in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
        fields = _bundle(rs1, "ComputeRs", prefix="a_")
        fields.update(_bundle(rs2, "ComputeRs", prefix="bd_"))
        return Command(kind, fields, None, 0)
    if kind == "FLUSH":
        require(rs1 == 0 and rs2 == 0, "FLUSH carries an unmodelled operand")
        return Command(kind, {}, None, 0)
    raise ProofError(f"{kind} has no modelled semantics")


@dataclass(frozen=True)
class Local:
    garbage: bool
    accumulator: bool = False
    accumulate: bool = False
    full: bool = False
    row: int = 0


def local(address) -> Local:
    """A local address split by the contract's flags; any other set bit refuses."""
    from .readout_encoding import flags
    from .stream import ACC_ROWS, SPAD_ROWS
    width = _layout("MvinRs2")["local_addr"][1]
    if address == (1 << width) - 1:
        return Local(True)
    bits = flags()
    accumulator, accumulate, full = (bool(address & bits[k]) for k in ("acc_i8", "acc_accum", "full_c_bit"))
    row = address & ~(bits["acc_i8"] | bits["acc_accum"] | bits["full_c_bit"])
    require(row < (ACC_ROWS if accumulator else SPAD_ROWS), "local address sets unmodelled bits")
    require(accumulator or not (accumulate or full), "scratchpad address sets accumulator flags")
    return Local(False, accumulator, accumulate, full, row)


@lru_cache(maxsize=None)
def _f1():
    from .readout_encoding import flags
    return flags()["f1"]


def execute_model():
    """The modelled CONFIG_EX: untransposed unit-stride MACs without activation.

    ``dataflow`` is the weight-stationary code here; a replay that also models
    output-stationary accepts that code instead (``Replay(dataflows=...)``).
    """
    from ..tables import isa
    from .readout_encoding import code
    model = dict(cmd_type=code("config_subtype", "CONFIG_EX"), dataflow=isa.WEIGHT_STATIONARY,
                 activation=isa.NO_ACTIVATION, set_only_strides=0, a_transpose=0, b_transpose=0, a_stride=1,
                 acc_scale=_f1(), in_shift=0, relu6_shift=0, c_stride=1)
    return model


def load_model():
    """Fixed CONFIG_LD fields; ``state_id`` and ``dram_stride`` are modelled for every value."""
    from .stream import DIM
    return dict(shrink=0, pixel_repeats=1, stride=DIM, scale=_f1())


def _matches(fields, fixed, variable, reason):
    require(set(fields) <= set(fixed) | set(variable), reason + " (unmodelled field)")
    require(all(fields.get(k) == v for k, v in fixed.items()), reason)


# ---------------------------------------------------------------------------------------------
# Symbolic replay
# ---------------------------------------------------------------------------------------------

class Replay:
    """Symbolic replay of decoded commands; ``on_store(replay, ins, config)`` checks each MVOUT.

    ``store_fields`` names the CONFIG_ST fields beyond activation, stride and
    scale that the caller's store check models; any other non-zero field refuses.
    ``dataflows`` names the CONFIG_EX dataflow codes the caller's proof covers
    (weight-stationary by default). Under output-stationary the mesh holds a
    per-row term list: a COMPUTE_PRELOADED restarts it, every compute appends
    ``(A row tag, B row tags, depth, columns)`` from its rs1/rs2 operands, and
    the whole list is written (or accumulated) into the preload's target.
    """

    def __init__(self, stream, on_store=None, *, store_fields=(), dataflows=None, byte_widening=False):
        from ..tables import isa
        from .stream import DIM

        self.stream, self.on_store, self.isa, self.dim = stream, on_store, isa, DIM
        self.store_fields = frozenset(store_fields)
        self.byte_widening = byte_widening
        self.dataflows = frozenset((isa.WEIGHT_STATIONARY,) if dataflows is None else dataflows)
        self.pe = None  # output-stationary per-row terms held in the mesh
        self.ld, self.used_ld = {}, {}
        self.st = None
        self.ex = None
        self.spad: dict[int, Tag] = {}
        self.acc: dict[int, AccRow] = {}
        self.weights = None
        self.target = None
        self.pending = None  # the preload a compute must consume
        self.loads_since_fence: set[int] = set()
        self.pooled_since_fence: set[int] = set()
        self.stores: list[Store] = []
        self.content: dict[str, dict[int, tuple]] = {}
        self.unfenced: set[tuple[str, int]] = set()
        self.operand_reads: set[tuple[str, int]] = set()
        self.command = None
        self.host = {a.name for a in stream.args if a.role in ("input", "const")}
        self.host |= {dest for _, dest in stream.packs}

    # -- helpers -----------------------------------------------------------------------------
    def _arg(self, command):
        require(command.arg is not None and 0 <= command.arg < len(self.stream.args), "DMA without a kernel argument")
        return self.stream.args[command.arg]

    def _elements(self, arg, byte, width_bytes, cols):
        from .storage import itemsize
        require(itemsize(arg.dtype) == width_bytes, "DMA element width differs from its argument type")
        require(byte % width_bytes == 0, "DMA row is not element aligned")
        flat = byte // width_bytes
        count = int(np.prod(arg.shape)) if arg.shape else 1
        require(0 <= flat and flat + cols <= count, "DMA row leaves its argument")
        return flat

    def _row_tag(self, arg, byte, width_bytes, cols, *, operand):
        flat = self._elements(arg, byte, width_bytes, cols)
        if arg.role == "const":
            from .storage import numpy_storage
            data = self.stream.consts.get(arg.name)
            require(data is not None and data.dtype == numpy_storage(arg.dtype) and
                    tuple(data.shape) == tuple(arg.shape), "constant storage differs from its declared argument")
        if arg.name in self.host:
            return Tag(arg.name, flat, cols)
        stored = self.content.get(arg.name, {})
        content = tuple(stored.get(flat + c) for c in range(cols))
        require(all(entry is not None for entry in content), "load reads an element no command produced")
        require(not any((arg.name, flat + c) in self.unfenced for c in range(cols)),
                "load reads a store that no fence ordered before it")
        if operand:
            self.operand_reads.update((arg.name, flat + c) for c in range(cols))
        return Tag(arg.name, flat, cols, content)

    def _execute_state(self):
        require(self.ex is not None, "effective CONFIG_EX is unset at an execute command")
        model = execute_model()
        model.pop("dataflow")
        _matches(self.ex, model, ("dataflow",), "effective CONFIG_EX differs from the proved mesh configuration")
        require(self.ex["dataflow"] in self.dataflows, "effective CONFIG_EX differs from the proved mesh dataflow")
        # Consumed load states stay live through compute and readout: rebinding
        # one without a new proved load is outside every replayed proof.
        require(all(self.ld.get(state) == config for state, config in self.used_ld.items()),
                "effective CONFIG_LD changed after a proved load")

    def _live(self, row):
        state = self.acc.get(row)
        return state is not None and not state.stored and (state.seed is not None or state.terms)

    # -- commands ----------------------------------------------------------------------------
    def run(self):
        for index, ins in enumerate(self.stream.ins):
            try:
                self.command = decode(ins)
                getattr(self, "_" + self.command.kind.lower())(self.command, ins)
            except ProofError as error:
                raise ProofError(f"#{index} {ins.kind}: {error}") from None
        require(self.pending is None, "preload has no consuming compute")
        return self

    def _flush(self, command, ins):
        pass

    def _fence(self, command, ins):
        self.loads_since_fence.clear()
        self.pooled_since_fence.clear()
        self.unfenced.clear()

    def _config_ex(self, command, ins):
        self.ex = dict(command.fields)

    def _config_ld(self, command, ins):
        self.ld[command.fields["state_id"]] = dict(command.fields)

    def _config_st(self, command, ins):
        require(command.fields["acc_scale"] != 0xffffffff,
                "CONFIG_ST retain sentinel is outside the replay model")
        from .runtime_scale import RuntimeScaleRef, validate_scale_ref
        if isinstance(command.fields["acc_scale"], RuntimeScaleRef):
            validate_scale_ref(command.fields["acc_scale"], self.stream.args)
        self.st = dict(command.fields)

    def _load(self, command, ins):
        from .stream import DIM
        fields = command.fields
        cfg = self.ld.get(fields["state"])
        place = local(fields["local_addr"])
        rows, cols = fields["num_rows"], fields["num_cols"]
        what = "accumulator seed load" if place.accumulator else "operand load"
        require(cfg is not None, f"{what} before its CONFIG_LD state")
        widening = self.byte_widening and place.accumulator and cfg.get('shrink') == 1
        model = dict(load_model(), shrink=1) if widening else load_model()
        _matches(cfg, model, ("state_id", "dram_stride"), f"effective CONFIG_LD differs at an {what}")
        require(not place.garbage and not place.accumulate and not place.full,
                f"{what} uses an unmodelled destination (accumulating or garbage)")
        require(1 <= rows <= DIM and 1 <= cols <= (DIM if place.accumulator else 4 * DIM),
                f"{what} extent is outside the modelled primitive")
        self.used_ld[fields["state"]] = cfg
        zero = command.arg is None
        require(not zero or command.off == 0, f"{what} has no source argument")
        arg = None if zero else self._arg(command)
        width = 4 if place.accumulator and not widening else 1
        for b0 in range(0, cols, DIM):
            nc = min(DIM, cols - b0)
            base = place.row + (b0 // DIM) * cfg["stride"]
            for r in range(rows):
                if zero:
                    tag = Tag("", 0, nc)
                else:
                    byte = command.off + r * cfg["dram_stride"] + b0 * width
                    try:
                        tag = self._row_tag(arg, byte, width, nc, operand=not place.accumulator)
                    except ProofError as error:
                        raise ProofError(f"{what}: {error}") from None
                if place.accumulator:
                    require(not self._live(base + r), "accumulator seed overwrites a live unstored accumulator row")
                    self.acc[base + r] = AccRow(tag)
                else:
                    require(base + r not in self.pooled_since_fence,
                            "load overwrites rows of an unfenced scratchpad-source pooled store")
                    self.spad[base + r] = tag
                    self.loads_since_fence.add(base + r)

    _mvin = _mvin2 = _mvin3 = _load

    def _preload(self, command, ins):
        self._execute_state()
        require(self.pending is None, "preload replaces an unconsumed preload")
        fields = command.fields
        bd, c = local(fields["bd_local_addr"]), local(fields["c_local_addr"])
        if self._output_stationary():
            require(bd.garbage, "output-stationary D preload is outside the replay")
        elif not bd.garbage:
            require(not bd.accumulator, "stationary operand from the accumulator")
            rows, cols = fields["bd_num_rows"], fields["bd_num_cols"]
            require(1 <= rows <= self.dim and 1 <= cols <= self.dim, "stationary extent is outside the mesh")
            tags = tuple(self.spad.get(bd.row + i) for i in range(rows))
            require(all(t is not None for t in tags), "stationary rows were never loaded")
            self.weights = (tags, cols)
        require(c.garbage or (c.accumulator and not c.full), "mesh result does not target the accumulator")
        require(1 <= fields["c_num_rows"] <= self.dim and 1 <= fields["c_num_cols"] <= self.dim,
                "mesh result extent is outside the mesh")
        self.target = (c, fields["c_num_rows"], fields["c_num_cols"])
        self.pending = not bd.garbage

    def _output_stationary(self):
        return self.ex is not None and self.ex["dataflow"] != self.isa.WEIGHT_STATIONARY

    def _rows_of(self, row, rows, what):
        tags = tuple(self.spad.get(row + i) for i in range(rows))
        require(all(t is not None for t in tags), f"{what} rows were never loaded")
        return tags

    def _compute_os(self, command, fresh):
        fields = command.fields
        a, b = local(fields["a_local_addr"]), local(fields["bd_local_addr"])
        require(not a.garbage and not a.accumulator and not b.garbage and not b.accumulator,
                "output-stationary operands are not scratchpad rows")
        rows, depth = fields["a_num_rows"], fields["a_num_cols"]
        require(1 <= rows <= self.dim and 1 <= depth <= self.dim and fields["bd_num_rows"] == depth and
                1 <= fields["bd_num_cols"] <= self.dim, "output-stationary operand extents differ")
        lhs = self._rows_of(a.row, rows, "mesh input")
        rhs = self._rows_of(b.row, depth, "mesh B operand")
        c, c_rows, c_cols = self.target
        if fresh:
            self.pe = [[] for _ in range(rows)]
        require(self.pe is not None and len(self.pe) == rows, "output-stationary accumulation changes its tile")
        for i in range(rows):
            self.pe[i].append((lhs[i], rhs, depth, min(fields["bd_num_cols"], c_cols)))
        if c.garbage:
            return
        require(c_rows <= rows, "output-stationary result rows exceed the mesh tile")
        for i in range(c_rows):
            row = c.row + i
            if c.accumulate:
                require(self._live(row), "accumulation onto an unseeded or already stored accumulator row")
                self.acc[row].terms.extend(self.pe[i])
            else:
                require(not self._live(row), "mesh result overwrites a live unstored accumulator row")
                self.acc[row] = AccRow(None, list(self.pe[i]))

    def _compute(self, command, ins, fresh):
        self._execute_state()
        require(self.pending is not None, "compute without a preloaded operand")
        if self._output_stationary():
            self.pending = None
            return self._compute_os(command, fresh)
        require(self.pending == fresh, "compute reuses or replaces weights unlike its preload")
        self.pending = None
        fields = command.fields
        a, bd = local(fields["a_local_addr"]), local(fields["bd_local_addr"])
        require(bd.garbage, "WS mesh D operand is not garbage")
        require(not a.garbage and not a.accumulator, "mesh input is not a scratchpad row")
        require(self.weights is not None, "compute without stationary weights")
        rows, depth = fields["a_num_rows"], fields["a_num_cols"]
        tags, weight_cols = self.weights
        require(1 <= rows <= self.dim and depth == len(tags), "mesh depth differs from the stationary rows")
        c, c_rows, c_cols = self.target
        if c.garbage:
            return
        for i in range(c_rows):
            row = c.row + i
            term = None
            if i < rows:
                lhs = self.spad.get(a.row + i)
                require(lhs is not None, "mesh input row was never loaded")
                term = (lhs, tags, depth, min(weight_cols, c_cols))
            if c.accumulate:
                require(self._live(row), "accumulation onto an unseeded or already stored accumulator row")
                if term is not None:
                    self.acc[row].terms.append(term)
            else:
                require(not self._live(row), "mesh result overwrites a live unstored accumulator row")
                self.acc[row] = AccRow(None, [term] if term is not None else [])

    def _compute_preloaded(self, command, ins):
        self._compute(command, ins, True)

    def _compute_accumulate(self, command, ins):
        self._compute(command, ins, False)

    def _mvout(self, command, ins):
        from .readout_encoding import code
        from .stream import DIM
        require(self.st is not None, "store before CONFIG_ST")
        st = self.st
        modelled = {"cmd_type", "activation", "stride", "acc_scale"} | self.store_fields
        require(st["cmd_type"] == code("config_subtype", "CONFIG_ST") and
                all(v == 0 for k, v in st.items() if k not in modelled),
                "effective CONFIG_ST sets an unmodelled store field")
        fields = command.fields
        place = local(fields["local_addr"])
        rows, cols = fields["num_rows"], fields["num_cols"]
        require(not place.garbage and not place.accumulate, "accumulator store width or address flags differ")
        require(1 <= rows <= DIM and 1 <= cols <= DIM, "store extent is outside the modelled primitive")
        if place.accumulator and not (self.byte_widening and place.full):
            self._execute_state()
        if self.on_store is not None:
            self.on_store(self, ins, dict(st))
        if st.get("pool_stride") and not place.accumulator:
            image = st["orows"] * st["ocols"]
            self.pooled_since_fence.update(range(place.row, place.row + image))
        if st.get("pool_stride"):
            return  # Pooled stores are checked entirely by the caller's model.
        arg = self._arg(command)
        require(arg.role in ("output", "scratch", "inout") and arg.name not in self.host,
                "store writes a host-defined argument")
        width = 4 if place.full else 1
        index = len(self.stores)
        sources = []
        for r in range(rows):
            if place.accumulator:
                state = self.acc.get(place.row + r)
                require(state is not None, "store drains an unwritten accumulator row")
                sources.append(state.snapshot())
                state.stored = True
            else:
                tag = self.spad.get(place.row + r)
                require(tag is not None, "store drains an unwritten scratchpad row")
                sources.append(tag)
            try:
                flat = self._elements(arg, command.off + r * st["stride"], width, cols)
            except ProofError as error:
                what = "accumulator store width or layout" if place.accumulator else "scratchpad store layout"
                raise ProofError(f"{what}: {error}") from None
            written = self.content.setdefault(arg.name, {})
            for c in range(cols):
                require((arg.name, flat + c) not in self.operand_reads,
                        "store overwrites an element an operand load already consumed")
                written[flat + c] = (index, r, c)
                self.unfenced.add((arg.name, flat + c))
        self.stores.append(Store(index, ins, arg.name, place.accumulator, place.full, dict(st), tuple(sources), cols))


# -- source-semantics checks ------------------------------------------------------------------

def contraction_row(state: AccRow, *, lhs: str, row: int, depth: int, weight_ok, seed_ok, cols: int):
    """``state`` holds seed + sum_k lhs[row, k] * W[k] over exactly k in [0, depth)."""
    require(seed_ok(state.seed), "accumulator seed differs from the proved seed")
    spans = []
    for a_tag, w_tags, kr, width in state.terms:
        require(a_tag.name == lhs and a_tag.flat // depth == row, "mesh row reads a different source row")
        k0 = a_tag.flat - row * depth
        require(k0 + kr <= depth and a_tag.width >= kr and width >= cols, "mesh tile leaves its source extent")
        for j, tag in enumerate(w_tags):
            require(weight_ok(tag, k0 + j), "stationary operand differs from the source contraction")
        spans.append((k0, k0 + kr))
    spans.sort()
    position = 0
    for lo, hi in spans:
        require(lo == position, "reduction tiles do not cover K exactly once")
        position = hi
    require(position == depth, "reduction tiles do not cover K exactly once")


def written_once(counts, reason="stores do not write every output element exactly once"):
    require(bool((counts == 1).all()), reason)


def decoded_store_config(ins):
    """CONFIG_ST fields decoded from the encoded command words."""
    command = decode(ins)
    require(command.kind == "CONFIG_ST", "command is not a CONFIG_ST")
    return dict(command.fields)


# ---------------------------------------------------------------------------------------------
# Concrete execution of the same decoded commands (the differential model)
# ---------------------------------------------------------------------------------------------

def _f32(bits):
    return np.array([bits], dtype=np.uint32).view(np.float32)[0]


def _pooled_store(command, place, cols, st, acc, spad, write, relu):
    """Store-path max pooling: garbage taps (outside orows x ocols) read zero data."""
    from ..tables import isa
    require(not place.full, "pooled store of full-width accumulator rows")
    if place.accumulator and st["activation"] not in (relu, isa.NO_ACTIVATION):
        raise ProofError("store activation has no concrete model")
    for porow in range(st["porows"]):
        for pocol in range(st["pocols"]):
            value = None
            for wrow in range(st["pool_size"]):
                for wcol in range(st["pool_size"]):
                    orow = porow * st["pool_stride"] + wrow - st["upad"]
                    ocol = pocol * st["pool_stride"] + wcol - st["lpad"]
                    if not (0 <= orow < st["orows"] and 0 <= ocol < st["ocols"]):
                        elem = np.zeros(cols, np.int64)
                    else:
                        row = place.row + orow * st["ocols"] + ocol
                        elem = (acc if place.accumulator else spad)[row, :cols]
                        if place.accumulator:
                            if st["activation"] == relu:
                                elem = np.maximum(elem, 0)
                            elem = np.clip(np.rint(elem.astype(np.float32) * _f32(st["acc_scale"])), -128, 127)
                    value = elem if value is None else np.maximum(value, elem)
            offset = command.off + (porow * st["pool_out_dim"] + pocol) * st["stride"]
            write(command.arg, offset, np.asarray(value, np.int64).reshape(1, -1), 0, 1)


def execute(stream, buffers, *, host_epilogues=False):
    """Run ``stream`` on ``buffers`` ({arg index: ndarray}, modified in place).

    Only the semantics ``Replay`` admits are implemented; anything else raises.
    Narrow readout is RNE(f32(f32(acc) * f32(scale))) after the activation,
    saturated to i8, and the accumulator wraps at i32.
    """
    from .stream import ACC_ROWS, DIM, SPAD_ROWS

    dram = {k: v.reshape(-1).view(np.uint8) for k, v in buffers.items()}
    spad = np.zeros((SPAD_ROWS, DIM), np.int64)
    acc = np.zeros((ACC_ROWS, DIM), np.int64)
    weights = np.zeros((DIM, DIM), np.int64)
    ld, st, target, ex = {}, None, None, None
    pe = np.zeros((DIM, DIM), np.int64)

    def read(arg, off, rows, cols, stride, width):
        out = np.zeros((rows, cols), np.int64)
        dt = np.int32 if width == 4 else np.int8
        for r in range(rows):
            start = off + r * stride
            require(arg in dram and 0 <= start and start % width == 0
                    and start + cols * width <= dram[arg].size, 'concrete DMA read leaves its argument')
            out[r] = np.frombuffer(dram[arg][start:start + cols * width].tobytes(), dt)
        return out

    def write(arg, off, block, stride, width):
        dt = np.int32 if width == 4 else np.int8
        for r in range(block.shape[0]):
            start = off + r * stride
            require(arg in dram and 0 <= start and start % width == 0
                    and start + block.shape[1] * width <= dram[arg].size, 'concrete DMA write leaves its argument')
            dram[arg][start:start + block.shape[1] * width] = np.frombuffer(block[r].astype(dt).tobytes(), np.uint8)

    from ..tables import isa
    from .readout_encoding import activation
    relu = activation(True)
    boundaries = {}
    if host_epilogues:
        from .requant import apply
        names = {a.name: i for i, a in enumerate(stream.args)}
        for source, dest in stream.packs:
            buffers[names[dest]].view(np.uint8).flat[:] = buffers[names[source]].view(np.uint8).reshape(-1) ^ 128
        for readout in stream.readouts:
            boundary = len(stream.ins) if readout.after is None else readout.after
            require(type(boundary) is int and 0 <= boundary <= len(stream.ins), 'host readout boundary leaves stream')
            boundaries.setdefault(boundary, []).append(readout)
    for index, ins in enumerate(stream.ins):
        if index in boundaries:
            apply(stream, buffers, boundaries[index])
        # Resolve on a temporary instruction. The authenticated stream stays symbolic.
        from .runtime_scale_replay import resolve_config_instruction
        command = decode(resolve_config_instruction(ins, stream.args, buffers))
        kind, f = command.kind, command.fields
        if kind in ("FENCE", "FLUSH"):
            continue
        if kind == "CONFIG_EX":
            fixed = execute_model()
            for name in ('dataflow', 'activation', 'c_stride'):
                fixed.pop(name)
            _matches(f, fixed, ('dataflow', 'activation', 'c_stride'), 'execute fields have no concrete model')
            require(f['dataflow'] in (isa.WEIGHT_STATIONARY, isa.OUTPUT_STATIONARY)
                    and f['activation'] in (isa.NO_ACTIVATION, relu), 'execute mode has no concrete model')
            ex = f
        elif kind == "CONFIG_LD":
            require(set(f) <= {'cmd_type', 'state_id', 'dram_stride', 'shrink', 'pixel_repeats', 'stride', 'scale'}
                    and f['pixel_repeats'] == 1 and f['state_id'] in (0, 1, 2)
                    and np.isfinite(_f32(f['scale'])), 'load fields have no concrete model')
            ld[f["state_id"]] = f
        elif kind == "CONFIG_ST":
            from .pool_plan import POOL_FIELDS
            require(not any(value for name, value in f.items()
                            if name not in {'cmd_type', 'stride', 'activation', 'acc_scale', *POOL_FIELDS})
                    and np.isfinite(_f32(f['acc_scale'])), 'store fields have no concrete model')
            st = f
        elif kind in ("MVIN", "MVIN2", "MVIN3"):
            cfg, place = ld[f["state"]], local(f["local_addr"])
            rows, cols = f["num_rows"], f["num_cols"]
            width = 4 if place.accumulator and not cfg['shrink'] else 1
            data = (np.zeros((rows, cols), np.int64) if command.arg is None else
                    read(command.arg, command.off, rows, cols, cfg["dram_stride"], width))
            if width == 4:
                require(cfg['scale'] == _f1(), 'scaled full-width accumulator load has no concrete model')
            elif cfg['scale'] != _f1():
                data = np.clip(np.rint(data.astype(np.float32) * _f32(cfg['scale'])), -128, 127).astype(np.int64)
            for b0 in range(0, cols, DIM):
                block = data[:, b0:b0 + DIM]
                base = place.row + (b0 // DIM) * cfg["stride"]
                target_rows = (acc if place.accumulator else spad)[base:base + rows, :block.shape[1]]
                target_rows[:] = target_rows + block if place.accumulate else block
                if place.accumulator:
                    target_rows[:] = ((target_rows + (1 << 31)) % (1 << 32)) - (1 << 31)
        elif kind == "PRELOAD":
            bd, c = local(f["bd_local_addr"]), local(f["c_local_addr"])
            if not bd.garbage and ex["dataflow"] == isa.WEIGHT_STATIONARY:
                weights = np.zeros((DIM, DIM), np.int64)
                weights[:f["bd_num_rows"], :f["bd_num_cols"]] = spad[bd.row:bd.row + f["bd_num_rows"],
                                                                     :f["bd_num_cols"]]
            target = (c, f["c_num_rows"], f["c_num_cols"])
        elif kind in ("COMPUTE_PRELOADED", "COMPUTE_ACCUMULATE"):
            a = local(f["a_local_addr"])
            lhs = np.zeros((DIM, DIM), np.int64)
            lhs[:f["a_num_rows"], :f["a_num_cols"]] = spad[a.row:a.row + f["a_num_rows"], :f["a_num_cols"]]
            if ex["dataflow"] == isa.WEIGHT_STATIONARY:
                out = lhs @ weights
            else:
                b = local(f["bd_local_addr"])
                rhs = np.zeros((DIM, DIM), np.int64)
                rhs[:f["bd_num_rows"], :f["bd_num_cols"]] = spad[b.row:b.row + f["bd_num_rows"], :f["bd_num_cols"]]
                pe = (0 if kind == "COMPUTE_PRELOADED" else pe) + lhs @ rhs
                out = pe
            c, c_rows, c_cols = target
            if c.garbage:
                continue
            for row in range(c_rows):
                dest = c.row + row * ex['c_stride']
                if c.accumulator:
                    block = acc[dest, :c_cols]
                    block[:] = (block if c.accumulate else 0) + out[row, :c_cols]
                    block[:] = ((block + (1 << 31)) % (1 << 32)) - (1 << 31)
                else:
                    block = np.clip(out[row, :c_cols], -128, 127)
                    require(ex['activation'] in (0, relu), 'execute activation has no concrete model')
                    spad[dest, :c_cols] = np.maximum(block, 0) if ex['activation'] == relu else block
        elif kind == "MVOUT":
            place = local(f["local_addr"])
            rows, cols = f["num_rows"], f["num_cols"]
            if st.get("pool_stride"):
                _pooled_store(command, place, cols, st, acc, spad, write, relu)
                continue
            if not place.accumulator:
                write(command.arg, command.off, spad[place.row:place.row + rows, :cols], st["stride"], 1)
                continue
            data = acc[place.row:place.row + rows, :cols]
            if place.full:
                write(command.arg, command.off, data, st["stride"], 4)
                continue
            if st["activation"] not in (relu, isa.NO_ACTIVATION):
                raise ProofError("store activation has no concrete model")
            if st["activation"] == relu:
                data = np.maximum(data, 0)
            scaled = np.rint(data.astype(np.float32) * _f32(st["acc_scale"]))
            write(command.arg, command.off, np.clip(scaled, -128, 127).astype(np.int64), st["stride"], 1)
        else:
            raise ProofError(f"{kind} has no concrete model")
    if len(stream.ins) in boundaries:
        apply(stream, buffers, boundaries[len(stream.ins)])
    return buffers
