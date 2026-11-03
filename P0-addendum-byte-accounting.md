# P0 addendum: byte accounting resolved

Date: 2026-11-03
Status: the earlier "12x discrepancy" in `P0-report.md` section 7 was a
measurement-harness error, not an EdgeVisor defect. The reported bytes were
correct for what actually ran.

## What was wrong

The P0 harness injected `layerCount=1`, but the dynamic TPOT scheduler was
running in the same process and issued its own PP commands shortly after. Both
commands share the plan command cache, so the applied route and the resulting
transfer belonged to whichever command landed last — in that run, a six-layer
move emitted by the scheduler:

```
apply pp command cacheSeq=4 layerCount=1 route=0->1 layers=[6]
apply pp command cacheSeq=6 layerCount=6 route=0->1 layers=[1,2,3,4,5,6]
```

The 4.05 MB was therefore six layers, not one, and `binding_update_count=6`
was accurate.

## Why the arithmetic always reconciled

Per row, one position of one layer carries K and V at `KvDim=1024`, float32:

```
1024 * 2 * 4 B  = 8192 B payload
              +  32 B LlmKvTransferHeader
                = 8224 B per row
```

Six layers over 41 positions (0–40), counted once per direction:

```
6 * 41 * 8224 * 2 = 4,046,208 B  (reported 4,048,208; +2000 B framing)
```

No accounting bug existed on either side.

## Fix

`inject_migration.py` now suppresses the scheduler by default:

```python
env["DLLAMA_TPOT_MIN_PP_GAIN_MS"] = "1000000000"
```

An unreachable gain threshold stops the scheduler from issuing PP commands while
leaving the plan UDS available for injection. `--allow-scheduler` restores the
old behaviour for experiments that want both.

## Verified single-layer migration

```
[plan]  cacheSeq=4 layerCount=1 route=0->1 layers=[6]
[kv]    exported=41 queued=41 layers=1 posRange=[0,40]
        sourceBytes=337348 + targetBytes=337380 = 674728
[event] stall_time_ms=8.396459  state_transfer_bytes=674728
        recompute_tokens_or_layers=0  apply_success=true
```

Predicted payload for that row: `41 * 8224 = 337,184 B`. Reported per direction
is 337,348 / 337,380 — a 164 / 196 byte framing difference. The total matches.

## Corrected numbers for the report

| Quantity | P0 original | Corrected (single layer) |
| --- | --- | --- |
| Layers moved | 6 | 1 |
| Positions | 41 | 41 |
| Bytes on the wire | 4.05 MB | 674.7 KB total |
| Stall | 43.36 ms | 8.40 ms |
| Recompute | 0 | 0 |

## Effect on rho

Recomputing with the isolated single-layer figure:

| Quantity | Value |
| --- | --- |
| Rows/sec | 41 / 0.0084 s ≈ 4880 rows/s |
| Tokens generated during the move at ~17 tok/s | 0.14 |
| **rho = r / B** | **≈ 0.0035** |

Even lower than the earlier estimate. The conclusion is unchanged and now rests
on a clean measurement: on this testbed one patching round converges well under
tau=50, so bandwidth shaping — not convergence — is what later comparisons must
exercise.

## Rule for later experiments

Never let the scheduler and an injected command run together. Either suppress
the scheduler, or attach `layers=` / `exported=` / `posRange=` from the
`shadow forward transfer` line to every byte figure so the layer count a number
belongs to is unambiguous. Bytes without an accompanying layer count are not
comparable.
