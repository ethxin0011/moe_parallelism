"""Train ONE run (launched by job_driver.py through torchrun; one process per rank).

Resume contract (rules 2 + 3):
  * all outputs live under <out_root>/runs/<run_id>/ on the blob datastore mount
  * a consolidated checkpoint is written every TRAIN.ckpt_minutes (15) and on SIGTERM
  * on (re)start: DONE.json -> exit; else latest checkpoint -> resume; else parent anchor
    (branch runs) -> resume; else fresh init.
Exit codes: 0 done | 3 blocked (data/parent not ready) | 143 interrupted (checkpoint saved)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import sys
import time

import torch

import config
import dist_utils as du
import engine as E

STOP = {"flag": False}


def _on_term(signum, frame):
    STOP["flag"] = True


def run_paths(out_root, run_id):
    r = os.path.join(out_root, "runs", run_id)
    return dict(root=r, ckpt=os.path.join(r, "ckpt"), snaps=os.path.join(r, "snaps"),
                metrics=os.path.join(r, "metrics"), done=os.path.join(r, "DONE.json"),
                spec=os.path.join(r, "spec.json"), log=os.path.join(r, "train_log.txt"))


def latest_ckpt(paths):
    meta = E.read_json(os.path.join(paths["ckpt"], "latest.json"))
    if meta and os.path.exists(meta["file"]):
        return meta
    # fallback: newest readable ckpt file
    files = sorted(glob.glob(os.path.join(paths["ckpt"], "ckpt_*.pt")))
    for f in reversed(files):
        try:
            step = int(os.path.basename(f)[5:12])
            return {"file": f, "step": step}
        except Exception:
            continue
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--target_tokens", type=int, default=config.DEFAULT_TARGET_TOKENS)
    a = ap.parse_args()

    spec = config.RUNS[a.run_id]
    tc, mcfg = config.TRAIN, config.model_cfg_for(spec)
    target_step = config.steps_for_tokens(config.target_tokens_for(spec, a.target_tokens))
    P = run_paths(a.out_root, a.run_id)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    device = E.device_for(spec)
    ctx = du.make_ctx(spec.ep, device)
    threads = E.set_threads(spec, ctx)
    sdpa_guard = E.determinism_setup(device)

    def log(msg):
        if ctx.is_main:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {a.run_id} {msg}"
            print(line, flush=True)
            try:
                os.makedirs(P["root"], exist_ok=True)
                with open(P["log"], "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    if ctx.world != spec.dp:
        raise SystemExit(f"world {ctx.world} != spec.dp {spec.dp}")

    done = E.read_json(P["done"])
    if done and done.get("step", 0) >= target_step:
        log(f"already DONE at step {done['step']} (target {target_step})")
        du.destroy()
        return 0

    data_dir = os.path.join(a.out_root, "data")
    if not os.path.exists(os.path.join(data_dir, "DATA_READY.json")):
        log("data not ready -> blocked")
        du.destroy()
        return 3
    local_data = E.stage_data_locally(data_dir, ctx)
    data = E.Data(local_data, mcfg.seq_len, tc.global_batch, spec.data_seed, ctx)

    model = E.build(spec, mcfg, ctx)
    opt = E.AdamW(model, tc)
    scaler = {"scale": tc.fp16_init_scale if spec.precision == "fp16" else 1.0, "good": 0}
    step = 0
    source = "fresh"

    meta = latest_ckpt(P)
    if meta:
        full = torch.load(meta["file"], map_location="cpu", weights_only=False)
        E.apply_full_state(model, opt, full, ctx)
        step, scaler = int(full["step"]), dict(full["scaler"])
        source = f"ckpt@{step}"
        del full
    elif spec.parent:
        anchor = os.path.join(a.out_root, "runs", spec.parent, "snaps", f"anchor_{spec.branch_step:07d}.pt")
        if not os.path.exists(anchor):
            log(f"parent anchor missing ({anchor}) -> blocked")
            du.destroy()
            return 3
        full = torch.load(anchor, map_location="cpu", weights_only=False)
        if spec.ulp_perturb:
            from model import ulp_perturb_
            ulp_perturb_(full["model"], spec.ulp_perturb)
            full.pop("bias_per_rank", None)
        E.apply_full_state(model, opt, full, ctx)
        step, scaler = int(full["step"]), dict(full["scaler"])
        source = f"branch {spec.parent}@{step}"
        del full
    else:
        full = E.initial_full_state(spec, mcfg)
        E.apply_full_state(model, None, full, ctx)
        if full.get("ulp"):
            log(f"ULP perturbation applied at {full['ulp']}")
        del full

    if ctx.is_main:
        os.makedirs(P["root"], exist_ok=True)
        if not os.path.exists(P["spec"]):
            E.write_json_atomic(dict(spec=spec.to_dict(), model=mcfg.__dict__, train=tc.__dict__), P["spec"])
    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu_model = next(l.split(":", 1)[1].strip() for l in f if l.startswith("model name"))
    except Exception:
        pass
    if ctx.is_main:
        E.write_json_atomic(dict(cpu_model=cpu_model, step=step, source=source,
                                 isa=os.environ.get("ONEDNN_MAX_CPU_ISA", ""),
                                 time=time.strftime("%Y-%m-%d %H:%M:%S")),
                            os.path.join(P["root"], "hosts", f"start_{step:07d}_{int(time.time())}.json"))
    log(f"start source={source} step={step} target={target_step} world={ctx.world} ep={ctx.ep} "
        f"device={device} threads/rank={threads} cpu='{cpu_model}'")

    snap_steps = set(s for s in config.SNAP_STEPS if s <= target_step) | {target_step}
    anchor_steps = set(config.ANCHOR_STEPS) if a.run_id in config.ANCHOR_PARENTS else set()
    ckpt_sec = tc.ckpt_minutes * 60.0
    last_ckpt = time.time()
    mbuf, chunk_start = [], step
    t_prev, tok_prev = time.time(), step

    def save_ckpt(cur_step):
        nonlocal mbuf, chunk_start, last_ckpt
        full = E.gather_full_state(model, opt, ctx, with_opt=True,
                                   extra=dict(step=cur_step, scaler=scaler, run_id=a.run_id))
        if ctx.is_main:
            if mbuf:
                path = os.path.join(P["metrics"], f"m_{chunk_start:07d}_{cur_step:07d}.jsonl")
                os.makedirs(P["metrics"], exist_ok=True)
                with open(path + ".tmp", "w") as f:
                    for r in mbuf:
                        f.write(json.dumps(r) + "\n")
                os.replace(path + ".tmp", path)
            f = os.path.join(P["ckpt"], f"ckpt_{cur_step:07d}.pt")
            E.save_atomic(full, f)
            E.write_json_atomic({"file": f, "step": cur_step, "time": time.time()},
                                os.path.join(P["ckpt"], "latest.json"))
            for old in sorted(glob.glob(os.path.join(P["ckpt"], "ckpt_*.pt")))[:-2]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        mbuf, chunk_start, last_ckpt = [], cur_step, time.time()
        du.barrier(ctx)
        log(f"checkpoint saved @ step {cur_step}")

    def save_snapshot(cur_step, anchor=False):
        full = E.gather_full_state(model, opt if anchor else None, ctx, with_opt=anchor,
                                   extra=dict(step=cur_step, scaler=scaler, run_id=a.run_id))
        if ctx.is_main:
            name = f"anchor_{cur_step:07d}.pt" if anchor else f"snap_{cur_step:07d}.pt"
            E.save_atomic(full, os.path.join(P["snaps"], name))
        du.barrier(ctx)

    with sdpa_guard:
        if step == 0:
            v0 = E.evaluate(model, data, spec, ctx, tc.val_seqs)
            mbuf.append(dict(step=0, val_ce=v0))
        while step < target_step:
            x, y = data.batch(step)
            x, y = x.to(ctx.device), y.to(ctx.device)
            st = E.train_step(model, opt, scaler, x, y, spec, tc, ctx, E.lr_at(step, tc))
            step += 1
            rec = dict(step=step, **st)
            if step % tc.val_every == 0 or step == target_step:
                rec["val_ce"] = E.evaluate(model, data, spec, ctx, tc.val_seqs)
            if step % tc.log_every == 0 or step == target_step:
                now = time.time()
                rec["tok_per_s"] = (step - tok_prev) * config.TOKENS_PER_STEP / max(1e-6, now - t_prev)
                t_prev, tok_prev = now, step
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            mbuf.append(rec)
            if step in snap_steps:
                save_snapshot(step)
            if step in anchor_steps:
                save_snapshot(step, anchor=True)
            want = ctx.is_main and (time.time() - last_ckpt >= ckpt_sec)
            if du.any_flag(want or STOP["flag"], ctx) and step < target_step:
                save_ckpt(step)
                if du.any_flag(STOP["flag"], ctx):
                    log("SIGTERM received -> checkpointed, exiting 143")
                    du.destroy()
                    return 143
        save_ckpt(step)
        if ctx.is_main:
            E.write_json_atomic(dict(step=step, tokens=step * config.TOKENS_PER_STEP,
                                     finished=time.strftime("%Y-%m-%d %H:%M:%S")), P["done"])
        log(f"DONE step={step}")
    du.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
