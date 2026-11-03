#!/usr/bin/env python3
"""Drive one PP migration on demand through the root's plan UDS.

The dynamic scheduler only issues a migration when it predicts a gain, which
never happens on four identical T4s. For mechanism experiments we therefore
inject the command ourselves, at a known token position, so every run moves the
same layers at the same point in the sequence.

Usage:
    python3 inject_migration.py --trigger-pos 24 --trigger-layer 24 \
        --from-node 0 --to-node 1 --out results/inject/
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
PLAN_SOCK = "/tmp/dllama_plan.sock"
BASE_PORT = 9992
N_STAGES = 4


def wait_for_socket(path: str, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
    return False


def wait_for_port(port: int, timeout_s: float = 90.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def send_plan(req: dict, timeout_s: float = 5.0) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        s.connect(PLAN_SOCK)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(8192)
            if not chunk:
                break
            buf += chunk
    text = buf.decode(errors="replace").strip()
    try:
        return json.loads(text.splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return {"raw": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m")
    ap.add_argument("--tokenizer", default="/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=192)
    ap.add_argument("--stage-size", type=int, default=7)
    ap.add_argument("--trigger-pos", type=int, default=24)
    ap.add_argument("--trigger-layer", type=int, default=24)
    ap.add_argument("--from-node", type=int, default=0)
    ap.add_argument("--to-node", type=int, default=1)
    ap.add_argument("--layer-count", type=int, default=1,
                    help="how many contiguous boundary layers to move")
    ap.add_argument("--mode", default="exact", choices=["exact", "next_barrier"])
    ap.add_argument("--jit-mode", default="greedy",
                    choices=["enabled", "static", "greedy", "oracle"],
                    help="greedy accepts every injected command; enabled needs "
                         "online estimates that an injected command cannot supply")
    ap.add_argument("--allow-scheduler", action="store_true",
                    help="let the dynamic TPOT scheduler also emit PP commands; "
                         "off by default so the measured bytes belong to our "
                         "injection alone")
    ap.add_argument("--out", default="results/inject")
    ap.add_argument("--timeout", type=float, default=240.0)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ablation = out_dir / "ablation.jsonl"
    if ablation.exists():
        ablation.unlink()
    if os.path.exists(PLAN_SOCK):
        os.unlink(PLAN_SOCK)

    env = os.environ.copy()
    env["EDGEVISOR_ABLATION_LOG_PATH"] = str(ablation)
    # The root only arms a collector at a known position; the command below is
    # what actually requests the move.
    env["DLLAMA_MIGRATION_VERBOSE_LOG"] = "1"
    # Mechanism experiments issue the command themselves at a fixed position, so
    # the conservative JIT gate (which needs online estimates we do not have at
    # injection time) would reject every command. "greedy" accepts what is asked
    # for and is the documented experiment-only bypass.
    env["EDGEVISOR_JIT_MODE"] = args.jit_mode
    # Keep the dynamic TPOT scheduler from issuing its own PP commands. Without
    # this it races our injection: the first apply honours layerCount, then the
    # scheduler emits a six-layer command of its own, and the measured bytes
    # belong to whichever finished last. An unreachable gain threshold silences
    # it while leaving the plan UDS available for our command.
    if not args.allow_scheduler:
        env["DLLAMA_TPOT_MIN_PP_GAIN_MS"] = "1000000000"

    print("starting workers…", flush=True)
    workers = []
    for stage in range(1, N_STAGES):
        port = BASE_PORT + (stage - 1)
        log = (out_dir / f"worker_gpu{stage}.log").open("w")
        workers.append(subprocess.Popen(
            [str(DLLAMA), "worker", "--port", str(port), "--model", args.model,
             "--gpu-index", str(stage), "--nthreads", "4"],
            stdout=log, stderr=subprocess.STDOUT,
        ))
    for stage in range(1, N_STAGES):
        if not wait_for_port(BASE_PORT + (stage - 1)):
            print(f"worker {stage} never listened", file=sys.stderr)
            for w in workers:
                w.terminate()
            return 1

    ratios = "*".join(f"1@{args.stage_size}" for _ in range(N_STAGES))
    cmd = [
        str(DLLAMA), "inference",
        "--model", args.model, "--tokenizer", args.tokenizer,
        "--prompt", args.prompt, "--steps", str(args.steps),
        "--max-seq-len", "2048", "--nthreads", "8", "--gpu-index", "0",
        "--workers", *[f"127.0.0.1:{BASE_PORT + i}" for i in range(N_STAGES - 1)],
        "--ratios", ratios, "--verbose",
        # The plan controller that owns the UDS only starts in dynamic mode.
        "--auto",
    ]
    root_log = (out_dir / "root.log").open("w")
    print(f"starting root, waiting for plan socket at {PLAN_SOCK}…", flush=True)
    root = subprocess.Popen(cmd, env=env, stdout=root_log, stderr=subprocess.STDOUT)

    injected: dict = {}
    try:
        if wait_for_socket(PLAN_SOCK, timeout_s=90):
            print("plan socket is up", flush=True)
            # PP layer migration is a distinct operation from a head/FFN plan
            # command: it carries layerCount and no head units.
            req = {
                "op": "set_pp_migration",
                "cmd": {
                    "seq": 1,
                    "mode": args.mode,
                    "stageIndex": 0,
                    "fromNodeIndex": args.from_node,
                    "toNodeIndex": args.to_node,
                    "layerCount": args.layer_count,
                },
            }
            if args.mode == "exact":
                req["cmd"]["triggerPos"] = args.trigger_pos
                req["cmd"]["triggerLayer"] = args.trigger_layer
            injected = send_plan(req)
            print(f"injection response: {json.dumps(injected)[:400]}", flush=True)
        else:
            print("plan socket never appeared", file=sys.stderr)

        try:
            root.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print("root timed out", file=sys.stderr)
            root.kill()
    finally:
        root_log.close()
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
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("event_id") != "runtime_config":
                events.append(ev)

    result = {
        "injection_request": {
            "trigger_pos": args.trigger_pos,
            "trigger_layer": args.trigger_layer,
            "from_node": args.from_node,
            "to_node": args.to_node,
            "layer_count": args.layer_count,
            "mode": args.mode,
        },
        "injection_response": injected,
        "migration_event_count": len(events),
        "events": events,
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nmigration events recorded: {len(events)}")
    for ev in events:
        print("  " + json.dumps({k: ev.get(k) for k in (
            "event_id", "trigger_pos", "from_node", "to_node",
            "stall_time_ms", "state_transfer_bytes",
            "recompute_tokens_or_layers", "apply_success")}))
    print(f"written: {out_dir / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
