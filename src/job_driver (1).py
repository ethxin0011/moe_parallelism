"""Self-managing queue runner. Runs INSIDE the Azure ML job on the target compute
(GPU cluster for the fp16 arm, CPU cluster for the rest). No external watcher:
the notebook only submits; the compute instance can be stopped afterwards.

Per run: skip if DONE (for the requested token budget) -> else torchrun train.py, which
resumes from its latest checkpoint. Re-submitting the same job after a preemption
continues exactly where it stopped. Status is written to <out_root>/status/<job_tag>.json.
"""
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

CHILD = {"p": None}
TERM = {"flag": False}


def _on_term(signum, frame):
    TERM["flag"] = True
    p = CHILD["p"]
    if p is not None and p.poll() is None:
        p.send_signal(signal.SIGTERM)   # train.py checkpoints and exits 143


def wjson(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(path + ".tmp", path)


def rjson(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def is_done(root, rid, target_tokens):
    d = rjson(os.path.join(root, "runs", rid, "DONE.json"))
    spec = config.RUNS[rid]
    need = config.steps_for_tokens(config.target_tokens_for(spec, target_tokens))
    return bool(d) and d.get("step", 0) >= need


def run_child(cmd, env, cwd):
    print(">>", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, env=env, cwd=cwd)
    CHILD["p"] = p
    while True:
        try:
            rc = p.wait(timeout=30)
            break
        except subprocess.TimeoutExpired:
            continue
    CHILD["p"] = None
    return rc


def selftest(root, device, here, env):
    if os.path.exists(os.path.join(root, "selftest", f"{device}_PASS.json")):
        print("selftest already passed", flush=True)
        return True
    nproc = 4
    e = dict(env, MOE_DIST_TIMEOUT_MIN="10")
    rc = run_child([sys.executable, "-m", "torch.distributed.run", "--standalone",
                    f"--nproc_per_node={nproc}", "selftest.py", "--out_root", root, "--device", device],
                   e, here)
    return rc == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--device", choices=["cpu", "cuda"], required=True)
    ap.add_argument("--arms", default="")
    ap.add_argument("--runs", default="", help="explicit comma list (overrides --arms)")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--target_tokens", type=int, default=config.DEFAULT_TARGET_TOKENS)
    ap.add_argument("--job_tag", required=True)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--max_retries", type=int, default=1)
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _on_term)
    root, here = a.out_root, os.path.dirname(os.path.abspath(__file__))
    status_path = os.path.join(root, "status", f"{a.job_tag}.json")
    if a.runs:
        queue = [r.strip() for r in a.runs.split(",") if r.strip()]
    else:
        queue = config.make_queue([x.strip() for x in a.arms.split(",") if x.strip()], a.shard, a.num_shards)
    queue = [r for r in queue if config.RUNS[r].device == a.device]
    st = dict(job_tag=a.job_tag, host=socket.gethostname(), device=a.device, queue=queue,
              target_tokens=a.target_tokens, done=[], failed=[], blocked=[], current=None,
              started=time.strftime("%Y-%m-%d %H:%M:%S"))

    def save(state=None):
        if state:
            st["state"] = state
        st["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        wjson(st, status_path)

    if not os.path.exists(os.path.join(root, "data", "DATA_READY.json")):
        save("blocked_no_data")
        print("data not prepared yet - run the prep job first", flush=True)
        return 2

    env = dict(os.environ)
    env.setdefault("PYTHONUNBUFFERED", "1")
    if a.device == "cuda":
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

    if a.selftest:
        save("selftest")
        if not selftest(root, a.device, here, env):
            save("selftest_failed")
            return 1
        if TERM["flag"]:
            save("preempted")
            return 143

    pending = list(queue)
    for _ in range(2):                      # 2nd pass picks up runs blocked on a parent
        deferred = []
        for rid in pending:
            if TERM["flag"]:
                save("preempted")
                return 143
            spec = config.RUNS[rid]
            if is_done(root, rid, a.target_tokens):
                if rid not in st["done"]:
                    st["done"].append(rid)
                continue
            target = config.target_tokens_for(spec, a.target_tokens)
            e = dict(env)
            if a.device == "cpu":
                e["MOE_THREADS"] = str(spec.threads_per_rank or max(1, (os.cpu_count() or 1) // spec.dp))
            else:
                e["MOE_THREADS"] = "1"
            if a.device == "cpu":
                # E16s_v3 nodes can land on different Intel generations (Broadwell/Skylake/
                # Cascade Lake). Pin MKL/oneDNN to the same AVX2 code path on every node so
                # CPU generation is NOT a hidden numerical factor across layouts.
                e["MKL_CBWR"] = "AVX2"
                e["MKL_ENABLE_INSTRUCTIONS"] = "AVX2"
                e["ONEDNN_MAX_CPU_ISA"] = "AVX2"
            e["OMP_NUM_THREADS"] = e["MOE_THREADS"]
            e["MKL_NUM_THREADS"] = e["MOE_THREADS"]
            cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                   f"--nproc_per_node={spec.dp}", "train.py", "--run_id", rid,
                   "--out_root", root, "--target_tokens", str(target)]
            rc = None
            for attempt in range(a.max_retries + 1):
                st["current"] = dict(run=rid, attempt=attempt, since=time.strftime("%H:%M:%S"))
                save("running")
                rc = run_child(cmd, e, here)
                if rc in (0, 3) or TERM["flag"]:
                    break
            st["current"] = None
            if TERM["flag"]:
                save("preempted")
                return 143
            if rc == 0:
                st["done"].append(rid)
            elif rc == 3:
                deferred.append(rid)
            else:
                st["failed"].append(dict(run=rid, rc=rc))
            save("running")
        pending = deferred
        if not pending:
            break
    st["blocked"] = pending
    save("finished" if not st["failed"] and not pending else "finished_with_issues")
    print(json.dumps(st, indent=1), flush=True)
    return 0 if not st["failed"] else 1


if __name__ == "__main__":
    sys.exit(main())
