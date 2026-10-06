# Gather service calibration

Owner: `reference_parity`. Same immutable ELF on strict RV64GC Spike and GSIM;
stock admission is pending root review through `firesim_recovery`.

Each window performs 4,096 indexed uint64 loads and a rotating XOR accumulator.
The same deterministic raw index stream is masked by the typed 64 KiB, 512 KiB,
or 8 MiB word extent. The middle working set and both repetitions are held out.
All initialization, complete table/index checks, guards and checksum are outside
the common fenced kernel-call window. All nine counter rows pass the unchanged
exact parser. Spike and GSIM report the same 45,071 instructions per gather.

| Working set | Partition | GSIM repeat 0 | GSIM repeat 1 |
| --- | --- | ---: | ---: |
| 64 KiB | training | 93,374 | 79,808 |
| 512 KiB | held out | 130,235 | 104,802 |
| 8 MiB | training | 148,069 | 115,830 |

This measures an operational gather stream. The requested payload is 49,152
load bytes per window, including the index stream. The recorded unique 64-byte
address groups are not a cache-miss count or a pinned cache-line geometry.
No pure bandwidth or numerical-alternative ranking is claimed. Repetitions
retain their actual state; there is no cold-cache assertion.

The full GSIM program takes 18,770,703 cycles including initialization, checks,
UART and startup. That total is separate from the windows above. The final
checksum is `01a0e1c0a2239af5`, supplementing complete numeric checks.

The source/expected values regenerate byte exactly. Actual compile and link
flags, compiler stages, headers, libraries, helper source, ELF, engine,
histogram, console and parser are pinned in
`rv64gc_gather_service_battery_qualified.json`. CPU21's released source and
parser files remain unchanged. No timing fit or Jack label is used here.
