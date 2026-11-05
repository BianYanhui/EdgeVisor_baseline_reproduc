#!/usr/bin/env python3
"""ReMP's pipeline-dimension layer move, on EdgeVisor's existing kv-migrate.

ReMP reconfigures pipeline parallelism by changing which stage owns a layer
and moving that layer's KV. This script does only that: one injected
set_pp_migration, scheduler suppressed, no pre-copy and no PipeLive patching.

Two routes matter:

  forward   stage 0 -> stage 1. Shadow KV still transfers the whole range.
  backward  stage 2 -> stage 1. The enabled shadow path switches on the
            precomputed redundant state and transfers nothing.

Usage:
    python3 remp_pp.py --model ... --tokenizer ... --stage-size 9 \
        --steps 160 --trigger-pos 80 --out /tmp/ev_p4/remp_8b
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import socket
import subprocess
import sys
import time

DLLAMA = pathlib.Path("/home/byh/B01/EdgeVisor-wansongcc/EdgeVisor/dllama")
PLAN_SOCK = "/tmp/dllama_plan.sock"
BASE_PORT = 9992
N_STAGES = 4

SCRUB = (
    "DLLAMA_BACKGROUND_PRECOPY",
    "DLLAMA_PIPELIVE_PATCH",
    "DLLAMA_EPISODE_SCRIPT",
    "DLLAMA_TPOT_MIN_PP_GAIN_MS",
    "EDGEVISOR_JIT_MODE",
    "EDGEVISOR_ABLATION_LOG_PATH",
    "EDGEVISOR_SHADOW_KV_MODE",
    "DLLAMA_IGNORE_EOS",
)


def wait_for_port(port: int, timeout_s: float = 120.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.4)
    return False


def wait_for_socket(path: str, timeout_s: float = 180.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.2)
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


def parse_root(log_path: pathlib.Path) -> dict:
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    recover = None
    for line in text.splitlines():
        if "recover mode=" in line and "stallMs=" in line:
            recover = line
    tok = re.findall(r"Prediction \(root wall-clock\)\s+nTokens: (\d+)\s+tokens/s:\s*([0-9.]+)", text)
    reason = None
    for key in (
        "shadow_kv_forward_real_transfer",
        "shadow_kv_precomputed_redundant_state",
        "shadow_kv_disabled_real_transfer",
        "background_precopy_delta_transfer",
    ):
        if key in text:
            reason = key
    return {"recover_line": recover, "reason": reason, "pred": tok[-1] if tok else None}


def run_arm(label: str, shadow_mode: str, src: int, dst: int, args, out_dir: pathlib.Path) -> dict:
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
        "EDGEVISOR_SHADOW_KV_MODE": shadow_mode,
        "DLLAMA_IGNORE_EOS": "1",
    })

    print(f"[{label}] workers", flush=True)
    workers = []
    logs = []
    for stage in range(1, N_STAGES):
        port = BASE_PORT + (stage - 1)
        log = (run_dir / f"worker_gpu{stage}.log").open("w")
        logs.append(log)
        workers.append(subprocess.Popen(
            [str(DLLAMA), "worker", "--port", str(port), "--model", args.model,
             "--gpu-index", str(stage), "--nthreads", "4"],
            stdout=log, stderr=subprocess.STDOUT,
        ))
    for stage in range(1, N_STAGES):
        if not wait_for_port(BASE_PORT + (stage - 1)):
            for w in workers:
                w.terminate()
            return {"label": label, "error": "worker_listen"}

    ratios = "*".join(f"1@{args.stage_size}" for _ in range(N_STAGES))
    cmd = [
        str(DLLAMA), "inference",
        "--model", args.model, "--tokenizer", args.tokenizer,
        "--prompt", args.prompt, "--steps", str(args.steps),
        "--max-seq-len", str(args.max_seq_len), "--nthreads", "8", "--gpu-index", "0",
        "--workers", *[f"127.0.0.1:{BASE_PORT + i}" for i in range(N_STAGES - 1)],
        "--ratios", ratios, "--verbose", "--auto",
    ]
    root_log_path = run_dir / "root.log"
    root_log = root_log_path.open("w")
    root = subprocess.Popen(cmd, env=env, stdout=root_log, stderr=subprocess.STDOUT)
    injected: dict = {}
    try:
        if wait_for_socket(PLAN_SOCK):
            req = {
                "op": "set_pp_migration",
                "cmd": {
                    "seq": 1,
                    "mode": "exact",
                    "stageIndex": 0,
                    "fromNodeIndex": src,
                    "toNodeIndex": dst,
                    "layerCount": 1,
                    "triggerPos": args.trigger_pos,
                    "triggerLayer": args.trigger_layer,
                },
            }
            injected = send_plan(req)
            print(f"[{label}] injected {json.dumps(injected)[:180]}", flush=True)
        else:
            print(f"[{label}] plan socket missing", file=sys.stderr)
        try:
            root.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            root.kill()
    finally:
        root_log.close()
        for w in workers:
            if w.poll() is None:
                w.terminate()
        for w in workers:
            try:
                w.wait(timeout=15)
            except subprocess.TimeoutExpired:
                w.kill()
        for log in logs:
            log.close()

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
            if ev.get("event_id") == "pp_migration_recover":
                events.append(ev)
    parsed = parse_root(root_log_path)
    last = events[-1] if events else {}
    result = {
        "label": label,
        "shadow_mode": shadow_mode,
        "from_node": src,
        "to_node": dst,
        "trigger_pos": args.trigger_pos,
        "reason": parsed["reason"],
        "stall_time_ms": last.get("stall_time_ms"),
        "state_transfer_bytes": last.get("state_transfer_bytes"),
        "recompute_tokens_or_layers": last.get("recompute_tokens_or_layers"),
        "apply_success": last.get("apply_success"),
        "injection": injected,
        "events": events,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"[{label}] reason={parsed['reason']} stall_ms={result['stall_time_ms']} "
        f"bytes={result['state_transfer_bytes']} recompute={result['recompute_tokens_or_layers']}",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="Explain the history and construction of the Eiffel Tower.")
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--stage-size", type=int, default=9)
    ap.add_argument("--trigger-pos", type=int, default=80)
    ap.add_argument("--trigger-layer", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        run_arm("forward_shadow", "enabled", 0, 1, args, out_dir),
        run_arm("forward_transfer", "disabled_transfer", 0, 1, args, out_dir),
        run_arm("backward_shadow", "enabled", 2, 1, args, out_dir),
    ]
    (out_dir / "summary.json").write_text(json.dumps(arms, indent=2))
    print(json.dumps([{k: a.get(k) for k in (
        "label", "reason", "stall_time_ms", "state_transfer_bytes",
        "recompute_tokens_or_layers", "apply_success",
    )} for a in arms], indent=2))
    return 0 if all(a.get("stall_time_ms") is not None for a in arms) else 1


if __name__ == "__main__":
    sys.exit(main())
