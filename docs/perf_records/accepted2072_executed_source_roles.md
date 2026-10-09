# Accepted 2072: executed provider cost, not whole histogram

Frozen original first-group ROI: **1,734,429,991 retired instructions**. Stock group cycles remain 3,918,275,805; no instruction-to-cycle or whole-model projection.

| Exclusive source role | Retired instructions |
|---|---:|
| canonical encoding | 283,588,481 |
| softmax control and observation | 250,777,666 |
| polynomial endpoint propagation | 233,438,574 |
| provider other or unresolved | 198,643,939 |
| interval and observer helpers | 196,552,652 |
| dot bounds and norms | 190,335,516 |
| source gather and mask validation | 106,909,920 |
| fused integer reconstruction | 95,160,288 |
| final quantizer observer | 45,290,407 |
| source dot replay inline | 44,194,584 |
| row scale finish | 39,003,072 |
| products unresolved line zero | 6,721,960 |
| primitive callback instruction body | 3,740,040 |
| source polynomial replay inline | 2,453,508 |
| product callback loop and admission | 11,904 |

Unassigned ROI residual: 37,607,480, including shared callees, allocation/runtime and any unresolved external bodies. Provider line-zero work is retained in its explicit role; no forced allocation.

## Post-ROI correction

All 15,738,880 `__truncsfbf2` calls originate in the two compiled source-consumer validation calls **after** elapsed timing. Their 236,079,104 instructions and the consumer body’s 145,133,788 instructions are outside provider ROI. `rintf` is mixed: 1,572,864 calls from outside `roundevenf`, 732,672 from provider `nearbyintf`; its input-dependent body split remains UNKNOWN.

## Sparse successors

Matched fresh normal-owner control is 1,747,436,465 instructions; it is not byte-identical to historical 2072. Initial sparse: 2,820,424,503. Cached dyadic powers: 2,116,532,012. Checked coefficient reuse: 2,023,442,125 (+15.7949%). All preserve the original consumer, eight statistics, three carrier diagnostics, 404 callback count and default exact fallback. No candidate full48 or hardware admission. Recode preparation still costs 194,937,309 exclusive instructions, plus 164,401,471 for sparse finishing; the original encoder remains paid.

## Next mechanism supported by these counts

Sparse representation alone adds too much preparation/finish work. Prioritize complete softmax probability-observer construction: 233.44M polynomial propagation + 196.55M interval helpers + 250.78M surrounding softmax source/control, with per-coordinate ownership/domain proofs already present. These role boundaries are exclusive instructions, not fully independent algorithm costs. Source replay is only 44.19M inline dot + 2.45M inline polynomial here. Any next change must preserve exact source words needed for ordered denominator and escaping scale observations, or carry an explicit approximate policy and pass the unchanged full1600 gate. The earlier monotonicity proof is useful but its measured small gain is not an answer to the whole5B objective.
