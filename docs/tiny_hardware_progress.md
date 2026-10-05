# TinyLlama whole-model hardware validation

Stock FireSimGemminiRocketConfig job1747 completed at **1,800,267,524 forward cycles** for the optimized prepacked TinyLlama artifact. This is an absolute measurement; no hardware speedup versus an unoptimized whole-model baseline is claimed.

All256,000 first-output float32 values (1,024,000 canonical little-endian bytes) match the independently validated native and actual Gemmini Spike reference by SHA256 `ebf524607c3254286fc5eda393436b607ace81866cb28b80fda8c4f62f435fe3`. Hashing runs after the timed forward interval. The collector also rechecked reference-file and Spike-console hashes, completion, zero rank mismatches, and the captured Torch oracle with recorded tolerances. Observed Torch relative L2 is2.1918433e-7 and maximum absolute difference9.536743e-6.

Actual staged ELF and actual FPGA bitstream hashes match the pinned identities; final ELF no-FSM audit passed. Exact identity, job-bound UART hash and reference-validation receipt hash are in `docs/perf_records/tiny_prepacked_firesim1747.json`. This result does not establish maximum achievable performance.
