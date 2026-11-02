# EdgeVisor Baseline Reproduction

Reproduction of four pipeline-parallel / multi-device LLM serving baselines, used as comparison
points for EdgeVisor. This repository lives outside the EdgeVisor repo and does not contribute code
back to it.

## Baselines

| Directory | Baseline | Focus |
| --- | --- | --- |
| `PipeLive/` | PipeLive | |
| `ReMP/` | ReMP | |
| `DynaPipe/` | DynaPipe | |
| `Llumnix/` | Llumnix | |

Each baseline directory is self-contained: upstream code (as a submodule or vendored copy),
reproduction scripts, configuration, and measured results live inside its own folder.

## Layout convention

```
<Baseline>/
  README.md        # what was reproduced, how to run, results
  upstream/        # upstream source at a pinned commit (submodule preferred)
  scripts/         # build / run / sweep scripts
  configs/         # per-experiment configuration
  results/         # measured output (git-ignored by default)
```

## Status

Directories are scaffolded. Each baseline's upstream source, build instructions, and reproduction
runs are added per-baseline as work proceeds.

## Repository boundary

This repo is independent of `wansongcc/EdgeVisor`. Nothing here is merged into EdgeVisor, and
EdgeVisor source is not vendored here.
