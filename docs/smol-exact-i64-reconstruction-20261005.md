# Exact i64 reconstruction: qualified selected head candidate

Merlin's canonical radix proof bounds every i32 group and the sum of absolute
weighted integer terms. When the latter is at most 2 to the power 53, every term
and prefix fits signed i64 and converts exactly to binary64. Under the original
RNE and positive-zero seed, exact cancellation also preserves positive zero.
The host consumer can use defined signed multiplication/addition and one final
conversion. The device, source floating replay, scales, norms and certificates
stay unchanged.

## Actual implementation and measurements

The first generic emitter selected a runtime weight before its loop. Ordinary
GCC emitted a 64-bit multiply for each result. Moving each proved constant into
its own group case lets ordinary code generation emit shifts 7, 14, 21 and 28.
The C still expresses multiplication, so negative signed left-shift UB is absent.
There is no host assembly or device policy in the shared numeric helper.

Both complete small capsules use the same original first 32 Q rows and first
64 K rows, K64, and byte-pinned dense device/source objects. Each timed interval
includes original encoding, all ten signed/absolute group calls and readbacks,
fresh integer scratch initialization, updates and final conversion. After each
of four AB/BA calls, all 4,096 reconstruction values, 192 i32 dirty guards and
two i64 dirty guards pass. The entire ELF contains no forbidden FSM instructions.

| Independent pair | Control cold / warm cycles | Candidate cold / warm cycles | Mean reduction |
| --- | --- | --- | --- |
| Runtime variable weight | 1,431,307 / 1,420,021 | 1,320,424 / 1,320,718 | 7.3715% |
| Constant inside group case | 1,422,443 / 1,412,890 | 1,212,047 / 1,213,640 | 14.4479% |

These are actual GSIM cycles from complete matched pairs. The variable and
constant cases have separate immutable sources, ELFs and receipts. Their own
controls have different layout contexts and are compared within each pair.
The earlier zero/support composition remains a separate rejected 1.7625% loss.

## Complete original-head gates

The selected constant-case head uses a separate 8 MiB integer buffer, reused
only after each signed or absolute reconstruction's final conversion. This
extra allocation, initialization, traffic and conversion remain inside the
complete attention ROI. There is no pointer type punning. All original dense
device kernels, dispatch, input data, startup and explicit runtime objects are
unchanged; only the host driver object is substituted. The saved original link
recipe reproduces the control ELF byte for byte before the controlled link.

Native and strict target both preserve all 65,536 original BF16 output values,
their lossless f32 digest and the original elementwise gate. Native replay
counts stay 58,369 QK and 4,971 PV. Target replay counts stay 58,353 QK and 4,971
PV. Both retain 30 group calls and 47,185,920 readback bytes. Native and target
QK counts are deliberately recorded separately. The final target audit covers
all 57 executable sections and passes with no FSM instructions.

The selected ELF is bfc94675241f9cdd27771a36521e9328c742873b7aea6a12199073b1f4b8fa84;
its object-bound marker is 3a17a0dd0c62. Strict Spike retires 1,022,873,401
instructions; this is not hardware cycles. Recovery owns one stock comparison
against complete grouped control 1917. Variable-weight and zero-support arms
are held. No full-head or full-model speedup is predicted from the local result.

## Generalization and qualification boundary

The shared API consumes a rederived numeric plan and explicit ownership/RNE
contract. No workload name, source ordinal, golden value or previous input
selects it. OOT owns unchanged target execution and measured ISA/cost evidence.
Independent compiled tests cover multiple radix widths, one/two/three digits,
positive and negative limits, exact cancellation and tails; UBSan checks defined
arithmetic. Test shared libraries use distinct filenames per numeric plan:
pytest can remove and reuse successful temporary paths while ctypes retains
an older mapping. The corrected twelve-test regression closes that hazard.

Automatic source routing is not implemented by this helper. This is one exact
original attention head; the complete 1,600-value Smol model accuracy and total
hardware performance gates remain separate. Extra live memory and full-program
cache context make the complete hardware comparison necessary.

Receipts are perf_records/smol_i64_radix_*.json. Exact token allocation per
optimization is unavailable; the root ledger owns shared campaign checkpoints.
