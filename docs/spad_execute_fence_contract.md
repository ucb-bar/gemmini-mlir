# Same-tile execute dependency contract

This default-off transform removes only marked fences inside a straight-line
SPAD execute chain. It validates the actual xDSL primitive operands before any
mutation. It retains external-memory, setup, panel-completion and final fences.
It does not grant ACC execute-read or STORE_SPAD support.

## Pinned hardware semantics

`OrderingContract` rehashes seven exact implementation/configuration files.
Source identity is a compiler legality witness; engine and stock-bitstream
support still require independent complete execution qualification.

* ExecuteController lines 210–228 compare every in-flight output tag against
  upcoming preload B and compute A/D bases. Enabled SPAD execute writes make
  RAW checking necessary. Lines 585–607 stall a preload or compute/preload pair
  with a matching pending tile. Single-compute fallback completes the already
  admitted operation; its input admission was checked with its preload.
* The comparison uses LocalAddr address space and row-base equality, rather
  than interval overlap. Therefore this pass requires aligned full 16-row
  tiles; an overlapping producer and consumer must be exactly the same tile.
  Shifted/partial/dynamic tiles and transpose/non-unit strides refuse.
* ExecuteController lines 638–680 pop commands only once their input rows have
  fired. The ordered execute stream therefore consumes an old SPAD tile before
  a later product overwrites it. MeshWithDelays/TagQueue retain output tags
  through their last response row; same-shaped wave outputs follow that queue.
* ReservationStation lines 221–310 derives local ranges from real dimensions,
  strides and address fields. Lines 327–378 establish load/execute/store local
  RAW/WAR/WAW dependencies. Completion clears those dependencies at lines
  493–530. ConfigEx drains the mesh and pending completion IDs at
  ExecuteController line 541 before changing execution state.

These rules cover full, non-in-place products in a chain with disjoint reserved
full storage extents. All reads of a prior product are either an exact tile
match or disjoint. The pass rejects unrecognized side effects and semantic
attributes inside the chain. These facts do not establish external DRAM alias
ordering: the predictor DMA store followed by DMA reload keeps its fence.

## Finite rectifier use

The 14-product residual experiment has four offset products and four indicator
products between two ConfigEx boundaries. Its eight internal fences become
unnecessary under the restricted contract. Source arithmetic, tables, A/B
immutability, output ownership, K/product order, shapes and all non-fence
primitive operands remain unchanged. The first predictor-store/reload fence,
the final panel fence and the three setup/function boundaries remain.

For 784 panels the emitted kernel has 1,571 fences instead of 7,843. This is a
command-count consequence, not a cycle forecast. Complete ranked ABI cost and
all 65,536 signed-byte pairs must qualify independently. No whole-model route
or automatic profitability selection is enabled by this option.
