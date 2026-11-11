"""Distributed training entry point for a single RunSpec."""
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
import engine
from model import ulp_perturb_

STOP = {"flag": False}


def _on_term(signum, frame):
    STOP["flag"] = True


def run_paths(out_root: str, run_id: str) -> dict[str, str]:
    root = os.path.join(out_root, "runs", run_id)
    return dict(
        root=root,
        ckpt=os.path.join(root, "ckpt"),
        snaps=os.path.join(root, "snaps"),
        metrics=os.path.join(root, "metrics"),
        done=os.path.join(root, "DONE.json"),
        spec=os.path.join(root, "spec.json"),
        log=os.path.join(root, "train_log.txt"),
    )


def latest_ckpt(paths: dict[str, str]) -> dict | None:
    meta = engine.read_json(os.path.join(paths["ckpt"], "latest.json"))
    if meta and os.path.exists(meta["file"]):
        return meta
    files = sorted(glob.glob(os.path.join(paths["ckpt"], "ckpt_*.pt")))
    for path in reversed(files):
        try:
            step = int(os.path.basename(path)[5:12])
            return {"file": path, "step": step}
        except Exception:
            continue
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--target_tokens", type=int, default=config.DEFAULT_TARGET_TOKENS)
    args = parser.parse_args()

    spec = config.RUNS[args.run_id]
    tc = config.TRAIN
    mcfg = config.model_cfg_for(spec)
    target_step = config.steps_for_tokens(config.target_tokens_for(spec, args.target_tokens))
    paths = run_paths(args.out_root, args.run_id)

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    device = engine.device_for(spec)
    ctx = du.make_ctx(spec.ep, device)
    threads = engine.set_threads(spec, ctx)
    sdpa_guard = engine.determinism_setup(device)

    def log(msg: str) -> None:
        if ctx.is_main:
            line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {args.run_id} {msg}"
            print(line, flush=True)
            try:
                os.makedirs(paths["root"], exist_ok=True)
                with open(paths["log"], "a") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    if ctx.world != spec.dp:
        raise SystemExit(f"world {ctx.world} != spec.dp {spec.dp}")

    done = engine.read_json(paths["done"])
    if done and done.get("step", 0) >= target_step:
        log(f"already finished at step {done['step']} (target {target_step})")
        du.destroy()
        return 0

    data_dir = os.path.join(args.out_root, "data")
    if not os.path.exists(os.path.join(data_dir, "DATA_READY.json")):
        log("dataset not prepared yet; exiting with code 3")
        du.destroy()
        return 3
    local_data = engine.stage_data_locally(data_dir, ctx)
    data = engine.Data(local_data, mcfg.seq_len, tc.global_batch, spec.data_seed, ctx)

    model = engine.build(spec, mcfg, ctx)
    opt = engine.AdamW(model, tc)
    scaler = {"scale": tc.fp16_init_scale if spec.precision == "fp16" else 1.0, "good": 0}
    step = 0
    source = "fresh"

    meta = latest_ckpt(paths)
    if meta:
        full = torch.load(meta["file"], map_location="cpu", weights_only=False)
        engine.apply_full_state(model, opt, full, ctx)
        step = int(full["step"])
        scaler = dict(full["scaler"])
        source = f"ckpt@{step}"
        del full
    elif spec.parent:
        anchor = os.path.join(args.out_root, "runs", spec.parent, "snaps", f"anchor_{spec.branch_step:07d}.pt")
        if not os.path.exists(anchor):
            log(f"parent anchor missing ({anchor}); exiting with code 3")
            du.destroy()
            return 3
        full = torch.load(anchor, map_location="cpu", weights_only=False)
        if spec.ulp_perturb:
            ulp_perturb_(full["model"], spec.ulp_perturb)
            full.pop("bias_per_rank", None)
        engine.apply_full_state(model, opt, full, ctx)
        step = int(full["step"])
        scaler = dict(full["scaler"])
        source = f"branch {spec.parent}@{step}"
        del full
    else:
        full = engine.initial_full_state(spec, mcfg)
        engine.apply_full_state(model, None, full, ctx)
        if full.get("ulp"):
            log(f"applied 1-ULP perturbation at {full['ulp']}")
        del full

    if ctx.is_main:
        os.makedirs(paths["root"], exist_ok=True)
        if not os.path.exists(paths["spec"]):
            engine.write_json_atomic(
                dict(spec=spec.to_dict(), model=mcfg.__dict__, train=tc.__dict__),
                paths["spec"],
            )

    cpu_model = ""
    try:
        with open("/proc/cpuinfo") as f:
            cpu_model = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
    except Exception:
        pass

    if ctx.is_main:
        engine.write_json_atomic(
            dict(
                cpu_model=cpu_model,
                step=step,
                source=source,
                isa=os.environ.get("ONEDNN_MAX_CPU_ISA", ""),
                time=time.strftime("%Y-%m-%d %H:%M:%S"),
            ),
            os.path.join(paths["root"], "hosts", f"start_{step:07d}_{int(time.time())}.json"),
        )
    log(
        f"start source={source} step={step} target={target_step} world={ctx.world} ep={ctx.ep} "
        f"device={device} threads/rank={threads} cpu='{cpu_model}'"
    )

    snap_steps = {s for s in config.SNAP_STEPS if s <= target_step} | {target_step}
    anchor_steps = set(config.ANCHOR_STEPS) if args.run_id in config.ANCHOR_PARENTS else set()
    ckpt_sec = tc.ckpt_minutes * 60.0
    last_ckpt = time.time()
    metrics_buf: list[dict] = []
    chunk_start = step
    t_prev = time.time()
    step_prev = step

    def save_ckpt(cur_step: int) -> None:
        nonlocal metrics_buf, chunk_start, last_ckpt
        full = engine.gather_full_state(
            model,
            opt,
            ctx,
            with_opt=True,
            extra=dict(step=cur_step, scaler=scaler, run_id=args.run_id),
        )
        if ctx.is_main:
            if metrics_buf:
                mpath = os.path.join(paths["metrics"], f"m_{chunk_start:07d}_{cur_step:07d}.jsonl")
                os.makedirs(paths["metrics"], exist_ok=True)
                with open(mpath + ".tmp", "w") as f:
                    for row in metrics_buf:
                        f.write(json.dumps(row) + "\n")
                os.replace(mpath + ".tmp", mpath)
            ckpt_file = os.path.join(paths["ckpt"], f"ckpt_{cur_step:07d}.pt")
            engine.save_atomic(full, ckpt_file)
            engine.write_json_atomic(
                {"file": ckpt_file, "step": cur_step, "time": time.time()},
                os.path.join(paths["ckpt"], "latest.json"),
            )
            for old in sorted(glob.glob(os.path.join(paths["ckpt"], "ckpt_*.pt")))[:-2]:
                try:
                    os.remove(old)
                except OSError:
                    pass
        metrics_buf = []
        chunk_start = cur_step
        last_ckpt = time.time()
        du.barrier(ctx)
        log(f"checkpoint saved at step {cur_step}")

    def save_snapshot(cur_step: int, anchor: bool = False) -> None:
        full = engine.gather_full_state(
            model,
            opt if anchor else None,
            ctx,
            with_opt=anchor,
            extra=dict(step=cur_step, scaler=scaler, run_id=args.run_id),
        )
        if ctx.is_main:
            fname = f"anchor_{cur_step:07d}.pt" if anchor else f"snap_{cur_step:07d}.pt"
            engine.save_atomic(full, os.path.join(paths["snaps"], fname))
        du.barrier(ctx)

    with sdpa_guard:
        if step == 0:
            val0 = engine.evaluate(model, data, spec, ctx, tc.val_seqs)
            metrics_buf.append(dict(step=0, val_ce=val0))
        while step < target_step:
            x, y = data.batch(step)
            x, y = x.to(ctx.device), y.to(ctx.device)
            stats = engine.train_step(model, opt, scaler, x, y, spec, tc, ctx, engine.lr_at(step, tc))
            step += 1
            rec = dict(step=step, **stats)
            if step % tc.val_every == 0 or step == target_step:
                rec["val_ce"] = engine.evaluate(model, data, spec, ctx, tc.val_seqs)
            if step % tc.log_every == 0 or step == target_step:
                now = time.time()
                rec["tok_per_s"] = (step - step_prev) * config.TOKENS_PER_STEP / max(1e-6, now - t_prev)
                t_prev = now
                step_prev = step
                log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in rec.items()))
            metrics_buf.append(rec)
            if step in snap_steps:
                save_snapshot(step)
            if step in anchor_steps:
                save_snapshot(step, anchor=True)
            due = ctx.is_main and (time.time() - last_ckpt >= ckpt_sec)
            if du.any_flag(due or STOP["flag"], ctx) and step < target_step:
                save_ckpt(step)
                if du.any_flag(STOP["flag"], ctx):
                    log("received SIGTERM; checkpoint saved, exiting 143")
                    du.destroy()
                    return 143
        save_ckpt(step)
        if ctx.is_main:
            engine.write_json_atomic(
                dict(
                    step=step,
                    tokens=step * config.TOKENS_PER_STEP,
                    finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                ),
                paths["done"],
            )
        log(f"finished at step={step}")
    du.destroy()
    return 0


if __name__ == "__main__":
    sys.exit(main())
