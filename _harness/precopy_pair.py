#!/usr/bin/env python3
"""Compare migration cost with and without background pre-copy.

EdgeVisor already ships an incremental pre-copy path (`runBackgroundPreCopy`).
This script measures what it buys, under a controlled pair of runs that differ
only in `DLLAMA_BACKGROUND_PRECOPY`.

Three settings are load-bearing and each fails silently if wrong:

  DLLAMA_EPISODE_SCRIPT        pre-copy only runs inside an agentic episode,
                               because that is where idle tool windows exist
  EDGEVISOR_SHADOW_KV_MODE     must be `disabled_transfer`; the default
                               `enabled` path ignores bgCopiedPos and moves
                               everything regardless
  trigger position             must be later than the first pre-copy, or no
                               pass has happened and the delta is the whole range

Inherited environment is scrubbed first: a leftover export in the calling shell
otherwise makes both arms of the comparison pre-copy runs.

Usage:
    python3 precopy_pair.py --trigger-pos 100 --layer-count 1 --out results/precopy
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket
import subprocess
import sys
import time

DLLAMA = pathlib.Path("/home/byh/B01/EdgeVisor-wansongcc/EdgeVisor/dllama")
MODEL = "/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
TOKENIZER = "/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
PLAN_SOCK = "/tmp/dllama_plan.sock"
BASE_PORT = 9992
N_STAGES = 4

# Variables that must not leak in from the calling shell.
SCRUB = (
    "DLLAMA_BACKGROUND_PRECOPY",
    "DLLAMA_BACKGROUND_PRECOPY_MAX_PASSES",
    "EDGEVISOR_SHADOW_KV_MODE",
    "DLLAMA_EPISODE_SCRIPT",
    "DLLAMA_TPOT_MIN_PP_GAIN_MS",
    "EDGEVISOR_JIT_MODE",
    "EDGEVISOR_ABLATION_LOG_PATH",
)


def wait_for_port(port: int, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def wait_for_socket(path: str, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


def write_episode(path: pathlib.Path, rounds: int, wait_ms: int, gen_tokens: int) -> None:
    lines = ["PROMPT\tExplain in detail the history and construction of the Eiffel Tower."]
    for i in range(rounds):
        lines.append(f"ROUND\t{wait_ms}\t{gen_tokens}\ttool result {i + 1}")
    path.write_text("\n".join(lines) + "\n")


def run_arm(label: str, precopy: bool, trigger_pos: int, layer_count: int,
            episode: pathlib.Path, args, out_dir: pathlib.Path) -> dict:
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    ablation = run_dir / "ablation.jsonl"
    if ablation.exists():
        ablation.unlink()
    if os.path.exists(PLAN_SOCK):
        os.unlink(PLAN_SOCK)

    env = os.environ.copy()
    for key in SCRUB:
        env.pop(key, None)
    env.update({
        "EDGEVISOR_ABLATION_LOG_PATH": str(ablation),
        "EDGEVISOR_JIT_MODE": "greedy",
        "DLLAMA_TPOT_MIN_PP_GAIN_MS": "1000000000",
        "DLLAMA_EPISODE_SCRIPT": str(episode),
        # Without this the default path transfers the full range and the
        # pre-copy work is discarded.
        "EDGEVISOR_SHADOW_KV_MODE": "disabled_transfer",
    })
    if precopy:
        env["DLLAMA_BACKGROUND_PRECOPY"] = "1"

    workers = []
    for stage in range(1, N_STAGES):
        port = BASE_PORT + (stage - 1)
        workers.append(subprocess.Popen(
            [str(DLLAMA), "worker", "--port", str(port), "--model", args.model,
             "--gpu-index", str(stage), "--nthreads", str(args.worker_threads)],
            stdout=(run_dir / f"worker_gpu{stage}.log").open("w"),
            stderr=subprocess.STDOUT,
        ))
    for stage in range(1, N_STAGES):
        if not wait_for_port(BASE_PORT + (stage - 1)):
            for w in workers:
                w.terminate()
            raise RuntimeError(f"worker {stage} never listened")

    ratios = "*".join(f"1@{args.stage_size}" for _ in range(N_STAGES))
    root = subprocess.Popen(
        [str(DLLAMA), "inference", "--model", args.model, "--tokenizer", args.tokenizer,
         "--steps", str(args.steps), "--max-seq-len", "2048",
         "--nthreads", str(args.root_threads), "--gpu-index", "0", "--auto",
         "--ratios", ratios,
         "--workers", *[f"127.0.0.1:{BASE_PORT + i}" for i in range(N_STAGES - 1)]],
        env=env, stdout=(run_dir / "root.log").open("w"), stderr=subprocess.STDOUT,
    )

    injection = None
    try:
        if wait_for_socket(PLAN_SOCK, 90):
            req = {"op": "set_pp_migration", "cmd": {
                "seq": 1, "mode": "exact", "stageIndex": 0,
                "fromNodeIndex": 0, "toNodeIndex": 1,
                "layerCount": layer_count, "triggerPos": trigger_pos,
            }}
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(5.0)
                s.connect(PLAN_SOCK)
                s.sendall((json.dumps(req) + "\n").encode())
                buf = b""
                while b"\n" not in buf:
                    chunk = s.recv(8192)
                    if not chunk:
                        break
                    buf += chunk
            injection = buf.decode(errors="replace").strip()
        root.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        root.kill()
    finally:
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for w in workers:
            try:
                w.wait(timeout=10)
            except subprocess.TimeoutExpired:
                w.kill()

    events = []
    if ablation.exists():
        for line in ablation.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            ev = json.loads(line)
            if ev.get("event_id") == "pp_migration_recover":
                events.append(ev)

    log = (run_dir / "root.log").read_text()
    precopy_passes = [l for l in log.splitlines() if "bg-precopy" in l]
    transfers = [l for l in log.splitlines() if "real transfer prepared" in l]
    delta_lines = [l for l in log.splitlines() if "trigger delta" in l]

    return {
        "label": label,
        "precopy": precopy,
        "injection": injection,
        "precopy_passes": len([l for l in precopy_passes if "copied_to=" in l]),
        "delta_lines": delta_lines,
        "transfers": transfers,
        "events": events,
    }


def summarize(arm: dict) -> dict:
    evs = arm["events"]
    stall = sum(float(e.get("stall_time_ms") or 0.0) for e in evs)
    moved = sum(int(e.get("state_transfer_bytes") or 0) for e in evs)
    exported = 0
    for line in arm["transfers"]:
        for token in line.split():
            if token.startswith("exported="):
                exported = max(exported, int(token.split("=")[1]))
    return {
        "label": arm["label"],
        "precopy": arm["precopy"],
        "precopy_passes": arm["precopy_passes"],
        "migration_rows": exported,
        "stall_ms": round(stall, 3),
        "transfer_bytes": moved,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--stage-size", type=int, default=7)
    ap.add_argument("--root-threads", type=int, default=8)
    ap.add_argument("--worker-threads", type=int, default=4)
    ap.add_argument("--trigger-pos", type=int, default=100,
                    help="must exceed the first pre-copy position (about 47)")
    ap.add_argument("--layer-count", type=int, default=1)
    ap.add_argument("--episode-rounds", type=int, default=5)
    ap.add_argument("--tool-wait-ms", type=int, default=200)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--out", default="results/precopy")
    args = ap.parse_args()

    if not DLLAMA.exists():
        print(f"missing binary: {DLLAMA}", file=sys.stderr)
        return 1

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    episode = out_dir / "episode.script"
    write_episode(episode, args.episode_rounds, args.tool_wait_ms, args.gen_tokens)

    arms = []
    for label, precopy in (("no_precopy", False), ("precopy", True)):
        print(f"\n=== {label} (trigger={args.trigger_pos}) ===", flush=True)
        arm = run_arm(label, precopy, args.trigger_pos, args.layer_count,
                      episode, args, out_dir)
        arms.append(arm)
        s = summarize(arm)
        print(f"  precopy passes: {s['precopy_passes']}  rows: {s['migration_rows']}  "
              f"stall: {s['stall_ms']} ms  bytes: {s['transfer_bytes']}", flush=True)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "trigger_pos": args.trigger_pos,
        "layer_count": args.layer_count,
        "summary": [summarize(a) for a in arms],
        "arms": arms,
    }
    (out_dir / "result.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    a, b = payload["summary"]
    print("\n=== comparison ===")
    if a["stall_ms"] and b["stall_ms"]:
        print(f"  stall   : {a['stall_ms']} ms -> {b['stall_ms']} ms "
              f"({1 - b['stall_ms'] / a['stall_ms']:+.1%})")
    if a["transfer_bytes"] and b["transfer_bytes"]:
        print(f"  bytes   : {a['transfer_bytes']:,} -> {b['transfer_bytes']:,} "
              f"({1 - b['transfer_bytes'] / a['transfer_bytes']:+.1%})")
    if a["migration_rows"] and b["migration_rows"]:
        print(f"  rows    : {a['migration_rows']} -> {b['migration_rows']}")
    print(f"\nwritten: {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
