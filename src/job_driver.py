"""Sequential queue runner for executing sharded CPU or GPU runs inside a compute job."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time

import config

CHILD = {"proc": None}
TERM = {"flag": False}


def _on_term(signum, frame):
    TERM["flag"] = True
    proc = CHILD["proc"]
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGTERM)


def write_json(obj: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def read_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def is_done(root: str, run_id: str, target_tokens: int) -> bool:
    done = read_json(os.path.join(root, "runs", run_id, "DONE.json"))
    spec = config.RUNS[run_id]
    required = config.steps_for_tokens(config.target_tokens_for(spec, target_tokens))
    return bool(done) and done.get("step", 0) >= required


def run_child(cmd: list[str], env: dict[str, str], cwd: str) -> int:
    print(">>", " ".join(cmd), flush=True)
    proc = subprocess.Popen(cmd, env=env, cwd=cwd)
    CHILD["proc"] = proc
    while True:
        try:
            rc = proc.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            continue
    CHILD["proc"] = None
    return rc


def run_selftest(root: str, device: str, cwd: str, env: dict[str, str]) -> bool:
    if os.path.exists(os.path.join(root, "selftest", f"{device}_PASS.json")):
        print("selftest already passed", flush=True)
        return True
    test_env = dict(env, MOE_DIST_TIMEOUT_MIN="10")
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=4",
        "selftest.py",
        "--out_root",
        root,
        "--device",
        device,
    ]
    return run_child(cmd, test_env, cwd) == 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--arms", default="")
    parser.add_argument("--runs", default="", help="comma-separated run IDs (overrides --arms)")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--target_tokens", type=int, default=config.DEFAULT_TARGET_TOKENS)
    parser.add_argument("--job_tag", required=True)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--max_retries", type=int, default=1)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    root = args.out_root
    here = os.path.dirname(os.path.abspath(__file__))
    status_path = os.path.join(root, "status", f"{args.job_tag}.json")

    if args.runs:
        queue = [r.strip() for r in args.runs.split(",") if r.strip()]
    else:
        arms = [x.strip() for x in args.arms.split(",") if x.strip()]
        queue = config.make_queue(arms, args.shard, args.num_shards)
    queue = [r for r in queue if config.RUNS[r].device == args.device]

    status = dict(
        job_tag=args.job_tag,
        host=socket.gethostname(),
        device=args.device,
        queue=queue,
        target_tokens=args.target_tokens,
        done=[],
        failed=[],
        blocked=[],
        current=None,
        started=time.strftime("%Y-%m-%d %H:%M:%S"),
    )

    def save(state: str | None = None) -> None:
        if state:
            status["state"] = state
        status["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        write_json(status, status_path)

    if not os.path.exists(os.path.join(root, "data", "DATA_READY.json")):
        save("blocked_no_data")
        print("dataset not prepared; run prep_data.py first", flush=True)
        return 2

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if args.device == "cuda":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    if args.selftest:
        save("selftest")
        if not run_selftest(root, args.device, here, env):
            save("selftest_failed")
            return 1
        if TERM["flag"]:
            save("preempted")
            return 143

    pending = list(queue)
    for _ in range(2):
        deferred = []
        for run_id in pending:
            if TERM["flag"]:
                save("preempted")
                return 143
            spec = config.RUNS[run_id]
            if is_done(root, run_id, args.target_tokens):
                if run_id not in status["done"]:
                    status["done"].append(run_id)
                continue

            target = config.target_tokens_for(spec, args.target_tokens)
            run_env = dict(env)
            if args.device == "cpu":
                threads = spec.threads_per_rank or max(1, (os.cpu_count() or 1) // spec.dp)
                run_env["MOE_THREADS"] = str(threads)
                # Pin MKL and oneDNN to AVX2 so mixed Xeon generations use identical kernels.
                run_env["MKL_CBWR"] = "AVX2"
                run_env["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
                run_env["ONEDNN_MAX_CPU_ISA"] = "AVX2"
            else:
                run_env["MOE_THREADS"] = "1"
            run_env["OMP_NUM_THREADS"] = run_env["MOE_THREADS"]
            run_env["MKL_NUM_THREADS"] = run_env["MOE_THREADS"]

            cmd = [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={spec.dp}",
                "train.py",
                "--run_id",
                run_id,
                "--out_root",
                root,
                "--target_tokens",
                str(target),
            ]
            rc = None
            for attempt in range(args.max_retries + 1):
                status["current"] = dict(run=run_id, attempt=attempt, since=time.strftime("%H:%M:%S"))
                save("running")
                rc = run_child(cmd, run_env, here)
                if rc in (0, 3) or TERM["flag"]:
                    break
            status["current"] = None
            if TERM["flag"]:
                save("preempted")
                return 143
            if rc == 0:
                status["done"].append(run_id)
            elif rc == 3:
                deferred.append(run_id)
            else:
                status["failed"].append(dict(run=run_id, rc=rc))
            save("running")
        pending = deferred
        if not pending:
            break

    status["blocked"] = pending
    save("finished" if not status["failed"] and not pending else "finished_with_issues")
    print(json.dumps(status, indent=1), flush=True)
    return 0 if not status["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
