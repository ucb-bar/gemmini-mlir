# Original-input attention certificate capsule: hardware decision

Both complete head0 runs preserve all65,536 original BF16 outputs (losslessly
reported as f32), with packing, device work, readback, certificates, exact replay,
softmax and output processing inside the timed region. The stock configuration
is FireSimGemminiRocketConfig; final ELF audits contain no FSM instructions.

| Case | Stock job | Whole-head cycles | Decision |
| --- | --- | ---: | --- |
|18-plane fair control|1894|2,618,580,085|Exact reference for this experiment|
|9-plane gamma certificate|1895|3,077,601,494|Rejected:17.53% slower|

Fewer planes and readback bytes did not compensate for host certificate work.
No refined certificate variant is promoted or queued merely because its Spike
instruction count decreases. These are complete-head measurements, not whole
SmolVLA timings or a5B-cycle result. The exact CPU baseline remains separate.

[Control receipt](perf_records/firesim1894_original_attention_head_control_verified.json)
and [candidate receipt](perf_records/firesim1895_original_attention_head_gamma_verified.json)
pin actual staged ELF, stock bitstream, UART and original output digest.
Token usage is unavailable to this agent.
