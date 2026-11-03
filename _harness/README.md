# Harness

Measurement tooling shared by every baseline reproduction. Everything here runs
as a normal user: no sudo, no host network changes. Bandwidth shaping for later
sweeps runs inside a container.

## Scripts

| Script | Purpose |
| --- | --- |
| `measure_migration.py` | Runs a 4-stage pipeline on local GPUs and records the ablation events EdgeVisor emits. Produces `summary.json`. |
| `inject_migration.py` | Drives one PP migration at a fixed token position through the root's plan UDS. Used because the dynamic scheduler never fires on four identical GPUs. |

## Testbed assumptions

Both scripts assume the P0 layout: four Tesla T4s on one host, one worker
process per device, root on GPU 0.

| Stage | Device | Port |
| --- | --- | --- |
| root | GPU 0 | — |
| worker 1 | GPU 1 | 9992 |
| worker 2 | GPU 2 | 9993 |
| worker 3 | GPU 3 | 9994 |

## Metrics

Every run records the same three quantities, because a migration mechanism can
trade any of them against the others:

- `stall_time_ms` — how long the main loop was blocked
- `state_transfer_bytes` — how much actually crossed the wire
- `recompute_tokens_or_layers` — how much work was redone instead of moved

These come from EdgeVisor's ablation event log (`EdgeVisorAblationEvent`), not
from log scraping, so they stay comparable across mechanisms.

## Two settings that look optional but are not

**`--auto` is required to inject a command.** The plan controller that listens
on `/tmp/dllama_plan.sock` only starts in dynamic mode. In static mode the
socket never appears and injection times out.

**`EDGEVISOR_JIT_MODE=greedy` is required for injected commands.** The default
`enabled` gate requires online estimates (`predictedBenefitMs`,
`survivalProbability`, ...) that an injected command cannot supply, so it
rejects every one with `jit_missing_or_invalid_online_estimates`. `greedy` is
the documented experiment-only bypass.

## Usage

```bash
# Static baseline plus one dynamic pass
python3 measure_migration.py --out results/

# One migration at a known position
python3 inject_migration.py \
  --steps 200 --trigger-pos 40 --trigger-layer 24 --layer-count 1 \
  --prompt "Explain in detail the history and construction of the Eiffel Tower" \
  --out results/inject
```

## Known open item

The byte accounting does not reconcile yet: migrating one layer over 41
positions reports 4.05 MB where the model geometry implies about 328 KiB, and
`binding_update_count` reports 6 for a `layerCount` of 1. Until this is
resolved, treat `state_transfer_bytes` as a relative indicator, not an absolute
quantity. Tracked in `../P0-report.md` section 7.
