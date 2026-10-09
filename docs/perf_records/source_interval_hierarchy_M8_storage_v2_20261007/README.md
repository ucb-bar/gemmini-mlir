# Named-storage measurement successor

This packet retains the first 300-pin hierarchy packet and its source, candidate,
control, table and all-mode qualification objects. It changes only the timing
harness: full-table hashing is performed after all four measured windows, with
source-derived expected hashes. No full-table sweep runs before or between
windows. Every window still reports all input, expected-output, guard-arena,
output, fine-table and coarse-table addresses.

The initial actual candidate source-output gate remains outside timing, so the
pair follows those first-M8 requests; it is not an independent cold-cache test.
Both scale scans, source preparation, table traffic, certificates, fallback,
finishing, frames and stores remain inside the unchanged helper ROIs. Additional
storage is 64 KiB; both arms share the same fine-table object, input data and
destination arena. No dynamic private scratch arena is used.

Strict source output, dirty guards, immutable table/input bytes and final noFSM
checks pass. Control executes 1,726,434 retired instructions; candidate executes
1,764,719 (+2.217%). Actual hardware cost and whole cost remain unknown. The
first harness was cancelled during startup as job 2096 after its interwindow
table sweep was identified; no validated timing or profitability is assigned
to that attempt.

The first v2 qualification retained an earlier ELF's symbol list despite
checking new runtime pointers against new symbol output. This packet's separate
qualification closes every actual new address; it preserves the earlier
observation. A fresh final link reproduces the v2 ELF byte for byte. No source
arithmetic or numerical policy changes, defaults or whole-model promotion are
introduced.
