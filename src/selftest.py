"""Distributed numerical, determinism, and checkpoint round-trip verification suite."""
from __future__ import annotations

import argparse
import os
import sys
import traceback
from dataclasses import replace

import torch
import torch.nn.functional as F

import config
import dist_utils as du
import engine
from model import is_expert_param

TINY = config.ModelCfg(
    vocab=512,
    d=64,
    n_layers=2,
    n_heads=2,
    seq_len=32,
    n_experts=4,
    top_k=2,
    d_expert=64,
)
TC = replace(config.TRAIN, global_batch=8, warmup=1, val_seqs=8)


def make_batch(world: int, rank: int, seed: int = 0, bsz: int = 8, seq: int = 32, vocab: int = 512):
    gen = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, vocab, (bsz, seq + 1), generator=gen)
    per_rank = bsz // world
    local = tokens[rank * per_rank:(rank + 1) * per_rank]
    return local[:, :-1].contiguous(), local[:, 1:].contiguous()


def grads_full(model, ctx: du.Ctx) -> dict[str, torch.Tensor] | None:
    sd = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    exp = {n: t for n, t in sd.items() if is_expert_param(n)}
    shared = {n: t for n, t in sd.items() if n not in exp}
    parts = du.gather_objects_to0({"exp": exp if ctx.ep_group_id == 0 else {}}, ctx)
    if ctx.rank:
        return None
    for part in parts:
        shared.update(part["exp"])
    return shared


def loss_and_grads(spec, ctx: du.Ctx, device: str):
    model = engine.build(spec, TINY, ctx)
    engine.apply_full_state(model, None, engine.initial_full_state(spec, TINY), ctx)
    x, y = make_batch(ctx.world, ctx.rank)
    x, y = x.to(device), y.to(device)
    with engine.autocast_ctx(spec, device):
        logits, aux = model(x)
    ce = F.cross_entropy(logits.float().reshape(-1, TINY.vocab), y.reshape(-1))
    loss = ce + (TC.lbl_coef * aux if spec.balance == "aux" else 0.0)
    loss.backward()
    engine.sync_grads(model, ctx, scale=1.0, clip=None)
    val = loss.detach().float().reshape(1).cpu().contiguous()
    du.ar_(val, ctx.world_group, ctx.world)
    return float(val[0]) / ctx.world, grads_full(model, ctx)


def run_steps(spec, ctx: du.Ctx, device: str, steps: int = 3):
    model = engine.build(spec, TINY, ctx)
    opt = engine.AdamW(model, TC)
    engine.apply_full_state(model, None, engine.initial_full_state(spec, TINY), ctx)
    scaler = {"scale": 1024.0 if spec.precision == "fp16" else 1.0, "good": 0}
    for s in range(steps):
        x, y = make_batch(ctx.world, ctx.rank, seed=100 + s)
        engine.train_step(model, opt, scaler, x.to(device), y.to(device), spec, TC, ctx, 1e-3)
    return model, opt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    dev = "cuda" if args.device == "cuda" else "cpu"
    if dev == "cuda":
        torch.cuda.set_device(0)
    prec = "fp16" if dev == "cuda" else "fp32"
    guard = engine.determinism_setup(dev)
    guard.__enter__()
    torch.set_num_threads(2)

    results = {}
    ok = True
    base_ctx = du.make_ctx(1, dev)
    world = base_ctx.world
    try:
        tol = 1e-4 if dev == "cpu" else 2e-3
        ref_spec = config.RunSpec("selftest", "T", device=dev, precision="fp32", dp=1, ep=1)
        if base_ctx.rank == 0:
            ref_loss, ref_grads = loss_and_grads(ref_spec, du.local_ctx(dev), dev)

        for ep in (1, 2, 4):
            if world % ep:
                continue
            for bal in ("aux", "auxfree"):
                spec = replace(ref_spec, dp=world, ep=ep, balance=bal)
                ctx = du.make_ctx(ep, dev)
                loss, grads = loss_and_grads(spec, ctx, dev)
                if ctx.rank == 0:
                    if bal != "aux":
                        r_loss, r_grads = loss_and_grads(replace(ref_spec, balance=bal), du.local_ctx(dev), dev)
                    else:
                        r_loss, r_grads = ref_loss, ref_grads
                    worst = max(
                        float((grads[n] - r_grads[n]).abs().max() / (r_grads[n].abs().max() + 1e-12))
                        for n in r_grads
                    )
                    rel_loss = abs(loss - r_loss) / abs(r_loss)
                    passed = worst < tol and rel_loss < 1e-5 * (1 if dev == "cpu" else 100)
                    results[f"T1_ep{ep}_{bal}"] = dict(rel_grad_err=worst, rel_loss_err=rel_loss, passed=passed)
                    ok &= passed
                du.barrier(ctx)

        spec = config.RunSpec("selftest", "T", device=dev, precision=prec, dp=world, ep=min(2, world))
        ctx = du.make_ctx(spec.ep, dev)
        m1, _ = run_steps(spec, ctx, dev)
        f1 = engine.gather_full_state(m1, None, ctx, with_opt=False)
        m2, o2 = run_steps(spec, ctx, dev)
        f2 = engine.gather_full_state(m2, o2, ctx, with_opt=True)
        if ctx.rank == 0:
            same = all(torch.equal(f1["model"][n], f2["model"][n]) for n in f1["model"])
            results["T2_bitwise_determinism"] = dict(passed=same)
            ok &= same

        f2 = du.broadcast_object(f2, ctx)
        if world >= 4:
            spec4 = replace(spec, ep=4)
            ctx4 = du.make_ctx(4, dev)
            m4 = engine.build(spec4, TINY, ctx4)
            o4 = engine.AdamW(m4, TC)
            engine.apply_full_state(m4, o4, f2, ctx4)
            f4 = engine.gather_full_state(m4, o4, ctx4, with_opt=True)
            if ctx4.rank == 0:
                same = all(torch.equal(f2["model"][n], f4["model"][n]) for n in f2["model"]) and all(
                    torch.equal(f2["opt"]["m"][n], f4["opt"]["m"][n]) for n in f2["opt"]["m"]
                )
                results["T3_roundtrip"] = dict(passed=same)
                ok &= same

        for kw in (
            dict(lbl_scope="local"),
            dict(cap_scope="local"),
            dict(balance="auxfree", bias_scope="local"),
        ):
            local_spec = replace(spec, **kw)
            local_ctx = du.make_ctx(local_spec.ep, dev)
            run_steps(local_spec, local_ctx, dev, steps=2)
            if local_ctx.rank == 0:
                label = "T4_" + "_".join(f"{k}={v}" for k, v in kw.items())
                results[label] = dict(passed=True)

        dense_spec = replace(spec, moe=False, ep=1)
        dense_ctx = du.make_ctx(1, dev)
        engine.build(dense_spec, replace(TINY, moe=False), dense_ctx)
        results["T4_dense_built"] = dict(passed=True)
    except Exception:
        ok = False
        results["exception"] = traceback.format_exc()
        print(results["exception"], flush=True)

    ok = not du.any_flag(not ok, base_ctx)
    if base_ctx.rank == 0:
        tag = "PASS" if ok else "FAIL"
        out = os.path.join(args.out_root, "selftest", f"{args.device}_{tag}.json")
        engine.write_json_atomic(dict(world=world, device=args.device, results=results, ok=ok), out)
        print(f"SELFTEST {tag}", results, flush=True)
    du.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
