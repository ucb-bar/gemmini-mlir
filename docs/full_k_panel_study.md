# Full-K A panels for the exact dense kernel

The fresh ResNet1801 profile places source `matmul_11` at611,025 hardware
cycles, against401,408 selected padded issue cycles. Its current schedule already
has cached B, separate B banks, stationary-weight reuse and wide stores. Owned
ZIP-derived notes motivate efficient wide transfers and bank separation; they do
not establish this candidate's gain.

The remaining A path loaded16 columns per command. The explicit `wide_a` option
now permits a resource-checked full K panel, loaded in groups of at most64columns.
For M3136/N128/K256, bm8/bn8, this reduces3,136 A-MVIN commands to784, preserving
25,088 compute/preload commands,32 cached-B loads and392 stores. A occupies rows
[0,2048), B [8192,10240), and the output accumulator [0,1024). The cached-B control
already unrolls K, so this change adds no K-unroll expansion. The automatic
selector and defaults are unchanged.

Actual GSIM at the captured scalar scale/ReLU gives582,086→555,498 kernel cycles
(4.57% fewer). Both complete401,408-value comparisons and2048-byte guards pass;
actual Gemmini Spike and final-ELF zero-FSM gates also pass. An additional
M17/N65/K80 Spike case covers partial M/N tiles and a final64+16-column K panel.
No whole-model or FireSim promotion is implied. The saving is modest, so this
arm should be composed with another qualified change instead of occupying a
standalone queue job.

The analytical model counts each64-column command separately; retaining a full
K panel does not mean one arbitrarily wide DMA command. Resource refusals remain
mandatory before generation. Future automatic selection needs the live panel's
full storage size and lifetime, legal command width, and measured overlap costs.

## Banked M follow-up: positive screen

The same matmul11 capsule completes in **407,566 GSIM kernel cycles**, versus
582,086 for the control and 555,498 for the full-K bm8 panel. This is 29.98%
below the control, and 1.53% above the unchanged 401,408-cycle padded issue
floor. All 401,408 outputs and the 2,048-byte guard pass independently in GSIM
and actual Gemmini Spike; the final ELF has zero FSM instructions. These are
synthetic operands at the exact captured source scale/ReLU, not a full-model
performance result.

The explicit schedule uses bm1/bn8, full-K wide A, cached wide B, wide stores,
M prefetch, and alternating physical A/accumulator banks. A slots occupy
scratch rows [0,256) and [4096,4352); cached B occupies [8192,10240). The
accumulator slots occupy [0,128) and [512,640). Each output group reads
resident B again, losing bm8 stationary-B reuse, but overlap removes most of
the exposed load/store stalls. Compute work remains 25,088 commands; A/B
MVIN counts are 784/32 and MVOUT is 392. No default selector changed.

The receipt `resnet_matmul11_banked_m_gsim.json` pins the source route,
complete transition proof, final ELF/IR, engine and Spike log. A full-model
composition must retain the source proof and pass the original full golden.
