#!/usr/bin/env python3
"""P0: local 4xT4 pipeline + KV migration measurement harness.

Runs one pipeline-parallel inference on local GPUs (one process per device),
optionally injects a PP migration command over the plan UDS, and collects the
ablation events that EdgeVisor already emits (stall, transferred bytes,
recomputed tokens) into a single JSON record per run.

No sudo is required: everything here runs as the invoking user. Network
shaping for later bandwidth sweeps is expected to run inside a container; this
script only records whatever the kernel actually gives us.

Usage:
    python3 measure_migration.py --model <file.m> --tokenizer <file.t> \
        --stage-size 7 --steps 128 --out results/

Layout is fixed at four stages, one per T4, because that is the P0 testbed.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict

DLLAMA = pathlib.Path("/home/byh/B01/EdgeVisor-wansongcc/EdgeVisor/dllama")
DEFAULT_MODEL = "/home/byh/B01/models/llama3.2_3b_instruct_q40/dllama_model_llama3.2-3b-instruct_q40.m"
DEFAULT_TOKENIZER = "/home/byh/B01/models/llama3.1_instruct_q40/dllama_tokenizer_llama_3_1.t"
PLAN_SOCK = "/tmp/dllama_plan.sock"

BASE_PORT = 9992
N_STAGES = 4


@dataclass
class RunRecord:
    label: str
    stages: int
    stage_size: int
    steps: int
    max_seq_len: int
    prompt: str
    auto: bool
    tokens_per_second: float | None = None
    ms_per_token: float | None = None
    wall_seconds: float | None = None
    topology: str | None = None
    profile_local: str | None = None
    profile_remote: str | None = None
    migration_armed: bool = False
    migration_events: list[dict] = field(default_factory=list)
    stdout_tail: list[str] = field(default_factory=list)
    stderr_tail: list[str] = field(default_factory=list)

    def summary(self) -> dict:
        """Compact view: the three quantities every later experiment reports."""
        stall = sum(float(e.get("stall_time_ms") or 0.0) for e in self.migration_events)
        moved = sum(int(e.get("state_transfer_bytes") or 0) for e in self.migration_events)
        recompute = sum(int(e.get("recompute_tokens_or_layers") or 0) for e in self.migration_events)
        return {
            "label": self.label,
            "tokens_per_second": self.tokens_per_second,
            "ms_per_token": self.ms_per_token,
            "topology": self.topology,
            "migration_count": len(self.migration_events),
            "stall_ms_total": round(stall, 3),
            "transfer_bytes_total": moved,
            "recompute_tokens_total": recompute,
        }


def wait_for_port(port: int, timeout_s: float = 60.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def start_workers(model: str, log_dir: pathlib.Path, nthreads: int) -> list[subprocess.Popen]:
    procs: list[subprocess.Popen] = []
    for stage in range(1, N_STAGES):
        port = BASE_PORT + (stage - 1)
        log = (log_dir / f"worker_gpu{stage}_port{port}.log").open("w")
        proc = subprocess.Popen(
            [
                str(DLLAMA), "worker",
                "--port", str(port),
                "--model", model,
                "--gpu-index", str(stage),
                "--nthreads", str(nthreads),
            ],
            stdout=log, stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        print(f"  worker stage {stage} → GPU {stage} port {port} pid {proc.pid}", flush=True)

    for stage in range(1, N_STAGES):
        port = BASE_PORT + (stage - 1)
        if not wait_for_port(port):
            stop_workers(procs)
            raise RuntimeError(f"worker on port {port} did not start listening")
    return procs


def stop_workers(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=15)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait(timeout=5)


def parse_metrics(stdout: str) -> dict:
    out: dict = {}
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("tokens/s:"):
            # tokens/s: 21.19 (47.19 ms/tok)
            try:
                rest = line.split(":", 1)[1].strip()
                tps = float(rest.split()[0])
                out["tokens_per_second"] = tps
                if "(" in rest and "ms/tok" in rest:
                    out["ms_per_token"] = float(rest.split("(")[1].split()[0])
            except (IndexError, ValueError):
                pass
        elif line.startswith("topology:"):
            out["topology"] = line.split(":", 1)[1].strip()
        elif line.startswith("profile: local"):
            out["profile_local"] = line.split(":", 1)[1].strip()
        elif line.startswith("profile: 127.0.0.1"):
            out["profile_remote"] = line.split(":", 1)[1].strip()
    return out


def read_events(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    events: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        # runtime_config is emitted once at startup; it carries no measurement.
        if ev.get("event_id") in (None, "runtime_config"):
            continue
        events.append(ev)
    return events


def send_plan_command(moves: list[dict], trigger_pos: int, trigger_layer: int,
                      mode: str = "exact", timeout_s: float = 3.0) -> dict:
    """Inject a migration command into the running root over its plan UDS."""
    req = {
        "op": "set_plan",
        "cmd": {
            "seq": 1,
            "mode": mode,
            "stageIndex": 0,
            "moves": moves,
        },
    }
    if mode == "exact":
        req["cmd"]["triggerPos"] = trigger_pos
        req["cmd"]["triggerLayer"] = trigger_layer

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout_s)
        s.connect(PLAN_SOCK)
        s.sendall((json.dumps(req) + "\n").encode())
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    try:
        return json.loads(buf.decode().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return {"raw": buf.decode(errors="replace")}


def run_once(args, label: str, auto: bool, inject: dict | None,
             out_dir: pathlib.Path) -> RunRecord:
    run_dir = out_dir / label
    run_dir.mkdir(parents=True, exist_ok=True)
    ablation = run_dir / "ablation.jsonl"
    if ablation.exists():
        ablation.unlink()

    env = os.environ.copy()
    env["EDGEVISOR_ABLATION_LOG_PATH"] = str(ablation)
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    print(f"\n=== {label} ===", flush=True)
    print("  starting workers…", flush=True)
    log_dir = run_dir
    workers = start_workers(args.model, log_dir, args.worker_threads)
    try:
        stage = args.stage_size
        ratios = "*".join(f"1@{stage}" for _ in range(N_STAGES))
        workers_arg = [f"127.0.0.1:{BASE_PORT + i}" for i in range(N_STAGES - 1)]
        cmd = [
            str(DLLAMA), "inference",
            "--model", args.model,
            "--tokenizer", args.tokenizer,
            "--prompt", args.prompt,
            "--steps", str(args.steps),
            "--max-seq-len", str(args.max_seq_len),
            "--nthreads", str(args.root_threads),
            "--gpu-index", "0",
            "--workers", *workers_arg,
            "--ratios", ratios,
        ]
        if auto:
            cmd.append("--auto")

        print(f"  running: {' '.join(cmd[-6:])}", flush=True)
        started = time.time()
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                              timeout=args.timeout)
        wall = time.time() - started
        stdout, stderr = proc.stdout, proc.stderr
        (run_dir / "root_stdout.log").write_text(stdout)
        (run_dir / "root_stderr.log").write_text(stderr)

        if args.plan_probe and os.path.exists(PLAN_SOCK):
            print("  plan socket present (root already exited before injection)",
                  flush=True)

    finally:
        stop_workers(workers)

    metrics = parse_metrics(stdout)
    rec = RunRecord(
        label=label,
        stages=N_STAGES,
        stage_size=args.stage_size,
        steps=args.steps,
        max_seq_len=args.max_seq_len,
        prompt=args.prompt,
        auto=auto,
        tokens_per_second=metrics.get("tokens_per_second"),
        ms_per_token=metrics.get("ms_per_token"),
        wall_seconds=round(wall, 3),
        topology=metrics.get("topology"),
        profile_local=metrics.get("profile_local"),
        profile_remote=metrics.get("profile_remote"),
        migration_armed="[kv-migrate] armed" in stdout,
        migration_events=read_events(ablation),
        stdout_tail=stdout.splitlines()[-25:],
        stderr_tail=stderr.splitlines()[-10:],
    )
    print(f"  → {rec.tokens_per_second} tok/s, events={len(rec.migration_events)}",
          flush=True)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--stage-size", type=int, default=7,
                    help="layers per stage (3B has 28 layers, 4 stages → 7)")
    ap.add_argument("--root-threads", type=int, default=8)
    ap.add_argument("--worker-threads", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default="results")
    ap.add_argument("--plan-probe", action="store_true",
                    help="write the plan socket state for debugging")
    ap.add_argument("--skip-auto", action="store_true",
                    help="only run the static baseline")
    args = ap.parse_args()

    if not DLLAMA.exists():
        print(f"dllama binary missing: {DLLAMA}", file=sys.stderr)
        return 1
    for path in (args.model, args.tokenizer):
        if not pathlib.Path(path).exists():
            print(f"missing input: {path}", file=sys.stderr)
            return 1
    if 4 * args.stage_size > 28 + 4:
        print("warning: stage sizes exceed the model layer count", file=sys.stderr)

    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in (PLAN_SOCK,):
        if os.path.exists(stale):
            os.unlink(stale)

    records: list[RunRecord] = []

    # 1. Static baseline: uniform split, no dynamic placement.
    records.append(run_once(args, "static_uniform", auto=False, inject=None,
                            out_dir=out_dir))

    # 2. Dynamic placement: speed-pack profile then layer movement.
    if not args.skip_auto:
        records.append(run_once(args, "auto_speedpack", auto=True, inject=None,
                                out_dir=out_dir))

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": socket.gethostname(),
        "gpus": args.gpu_indexes if hasattr(args, "gpu_indexes") else "0-3",
        "runs": [asdict(r) for r in records],
        "summary": [r.summary() for r in records],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print("\n=== summary ===")
    for s in payload["summary"]:
        print(f"  {s['label']:<16} {s['tokens_per_second']} tok/s  "
              f"topology={s['topology']}  migrations={s['migration_count']}")
    print(f"\nwritten: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
