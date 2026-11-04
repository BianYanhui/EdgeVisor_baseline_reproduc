#!/usr/bin/env python3
"""Compare three ways to move the same layer at the same token position.

  stop       the trigger copies the whole KV range
  idle       tool-wait pre-copy, then the trigger copies the residual
  pipelive   decode copies a chunk once a full chunk has built up, and the
             trigger copies the residual. Chunk size and tau come from
             DLLAMA_PIPELIVE_CHUNK and DLLAMA_PIPELIVE_TAU

EDGEVISOR_SHADOW_KV_MODE=disabled_transfer is required. The default enabled
path ignores the copied cursor and transfers the full range again.

Usage:
    python3 pipelive_pair.py --model ... --tokenizer ... --stage-size 9 \
        --steps 120 --trigger-pos 80 --out /tmp/ev_p3/pipelive_8b
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
    "DLLAMA_BACKGROUND_PRECOPY_MAX_PASSES",
    "DLLAMA_PIPELIVE_PATCH",
    "DLLAMA_PIPELIVE_CHUNK",
    "DLLAMA_PIPELIVE_TAU",
    "EDGEVISOR_SHADOW_KV_MODE",
    "DLLAMA_EPISODE_SCRIPT",
    "DLLAMA_TPOT_MIN_PP_GAIN_MS",
    "EDGEVISOR_JIT_MODE",
    "EDGEVISOR_ABLATION_LOG_PATH",
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
    chunks = []
    for match in re.finditer(
        r"\[pipelive\] pos=(\d+) copied_through=(\d+) residual_before=(\d+) residual_after=(\d+) elapsed_us=(\d+)",
        text,
    ):
        chunks.append({
            "pos": int(match.group(1)),
            "copied_through": int(match.group(2)),
            "residual_before": int(match.group(3)),
            "residual_after": int(match.group(4)),
            "elapsed_us": int(match.group(5)),
        })
    copies = []
    for match in re.finditer(
        r"\[bg-precopy\] copied_to=(\d+) exported=(\d+) queued=(\d+) bytes=(\d+) total_bytes=(\d+)",
        text,
    ):
        copies.append({
            "copied_to": int(match.group(1)),
            "exported": int(match.group(2)),
            "bytes": int(match.group(4)),
            "total_bytes": int(match.group(5)),
        })
    delta = re.search(
        r"trigger delta posRange=\[(\d+),(\d+)\] trigger_bytes=(\d+) bg_total_bytes=(\d+) bg_passes=(\d+)",
        text,
    )
    tok_matches = re.findall(r"tokens/s:\s*([0-9]+(?:\.[0-9]+)?)", text)
    # Later episode rounds keep copying after the switch. The bytes that
    # belong to this move are the total recorded on the trigger line.
    bg_at_trigger = int(delta.group(4)) if delta else (copies[-1]["total_bytes"] if copies else 0)
    return {
        "tokens_per_s": float(tok_matches[-1]) if tok_matches else None,
        "patch_chunks": len(chunks),
        "last_residual_after": chunks[-1]["residual_after"] if chunks else None,
        "patch_elapsed_us": sum(c["elapsed_us"] for c in chunks),
        "precopy_bytes": bg_at_trigger,
        "precopy_passes": int(delta.group(5)) if delta else len(copies),
        "delta_pos_range": [int(delta.group(1)), int(delta.group(2))] if delta else None,
        "chunks": chunks,
    }


def write_episode(path: pathlib.Path, prompt: str) -> None:
    # Prefill lands at position 20. Tool replay inserts extra tokens, so the
    # trigger at 80 falls during the second tool replay, after the wait that
    # copied through position 77.
    lines = [f"PROMPT\t{prompt}"]
    rounds = [(20, 20), (20, 20), (20, 30), (20, 10), (20, 10)]
    for index, (wait_ms, gen_tokens) in enumerate(rounds, start=1):
        lines.append(f"ROUND\t{wait_ms}\t{gen_tokens}\ttool result {index}")
    path.write_text("\n".join(lines) + "\n")


def run_arm(label: str, mode: str, args, out_dir: pathlib.Path) -> dict:
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
        "EDGEVISOR_SHADOW_KV_MODE": "disabled_transfer",
        "DLLAMA_PIPELIVE_CHUNK": str(args.chunk),
        "DLLAMA_PIPELIVE_TAU": str(args.tau),
    })
    if mode == "pipelive":
        env["DLLAMA_PIPELIVE_PATCH"] = "1"
    elif mode == "idle":
        episode = run_dir / "episode.txt"
        write_episode(episode, args.prompt)
        env["DLLAMA_BACKGROUND_PRECOPY"] = "1"
        env["DLLAMA_EPISODE_SCRIPT"] = str(episode)
    elif mode != "stop":
        raise ValueError(mode)

    print(f"[{label}] starting workers", flush=True)
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
            print(f"[{label}] worker {stage} never listened", file=sys.stderr)
            for w in workers:
                w.terminate()
            return {"label": label, "error": "worker_listen"}

    ratios = "*".join(f"1@{args.stage_size}" for _ in range(N_STAGES))
    cmd = [
        str(DLLAMA), "inference",
        "--model", args.model, "--tokenizer", args.tokenizer,
        "--prompt", args.prompt, "--steps", str(args.steps),
        "--max-seq-len", "2048", "--nthreads", "8", "--gpu-index", "0",
        "--workers", *[f"127.0.0.1:{BASE_PORT + i}" for i in range(N_STAGES - 1)],
        "--ratios", ratios, "--verbose", "--auto",
    ]
    root_log_path = run_dir / "root.log"
    root_log = root_log_path.open("w")
    print(f"[{label}] starting root", flush=True)
    root = subprocess.Popen(cmd, env=env, stdout=root_log, stderr=subprocess.STDOUT)
    injected: dict = {}
    try:
        if not wait_for_socket(PLAN_SOCK, timeout_s=180):
            print(f"[{label}] plan socket never appeared", file=sys.stderr)
        else:
            req = {
                "op": "set_pp_migration",
                "cmd": {
                    "seq": 1,
                    "mode": "exact",
                    "stageIndex": 0,
                    "fromNodeIndex": 0,
                    "toNodeIndex": 1,
                    "layerCount": args.layer_count,
                    "triggerPos": args.trigger_pos,
                    "triggerLayer": args.trigger_layer,
                },
            }
            injected = send_plan(req)
            print(f"[{label}] injected {json.dumps(injected)[:240]}", flush=True)
        try:
            root.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"[{label}] root timed out", file=sys.stderr)
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
    stall = events[-1].get("stall_time_ms") if events else None
    transfer = events[-1].get("state_transfer_bytes") if events else None
    result = {
        "label": label,
        "mode": mode,
        "chunk": args.chunk,
        "tau": args.tau,
        "trigger_pos": args.trigger_pos,
        "injection": injected,
        "stall_time_ms": stall,
        "state_transfer_bytes": transfer,
        "precopy_bytes": parsed["precopy_bytes"],
        "precopy_passes": parsed["precopy_passes"],
        "delta_pos_range": parsed["delta_pos_range"],
        "patch_chunks": parsed["patch_chunks"],
        "patch_elapsed_ms": parsed["patch_elapsed_us"] / 1000.0,
        "last_residual_after": parsed["last_residual_after"],
        "tokens_per_s": parsed["tokens_per_s"],
        "events": events,
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(
        f"[{label}] stall_ms={stall} final_bytes={transfer} "
        f"precopy_bytes={parsed['precopy_bytes']} chunks={parsed['patch_chunks']} "
        f"residual={parsed['last_residual_after']} tok/s={parsed['tokens_per_s']}",
        flush=True,
    )
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", default="Explain the history and construction of the Eiffel Tower.")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--stage-size", type=int, default=9)
    ap.add_argument("--trigger-pos", type=int, default=80)
    ap.add_argument("--trigger-layer", type=int, default=8)
    ap.add_argument("--layer-count", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--tau", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = [
        run_arm("stop_and_copy", "stop", args, out_dir),
        run_arm("idle_precopy", "idle", args, out_dir),
        run_arm("pipelive", "pipelive", args, out_dir),
    ]
    (out_dir / "summary.json").write_text(json.dumps(arms, indent=2))
    print(json.dumps([{k: a.get(k) for k in (
        "label", "stall_time_ms", "state_transfer_bytes", "precopy_bytes",
        "patch_chunks", "last_residual_after", "patch_elapsed_ms", "tokens_per_s",
    )} for a in arms], indent=2))
    return 0 if all("error" not in a and a.get("stall_time_ms") is not None for a in arms) else 1


if __name__ == "__main__":
    sys.exit(main())
