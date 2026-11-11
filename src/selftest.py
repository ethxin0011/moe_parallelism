"""Correctness gate - run BEFORE any real compute (torchrun --nproc_per_node=4).

T1  EP/DP gradient exactness: for ep in {1,2,4} with all-global semantics the distributed
    loss and consolidated gradients must match a single-process reference (fp32 CPU: rel<1e-4).
T2  Bitwise determinism: 3 optimizer steps run twice from the same init -> identical params.
T3  Consolidation round-trip: state from layout (4,2) -> load into (4,4) -> identical.
T4  Local-scope variants execute (lbl/cap/bias local) without hangs.
Writes <out_root>/selftest/<device>_PASS.json or _FAIL.json; exit code 0/1.
"""
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
import engine as E
from model import is_expert_param

TINY = config.ModelCfg(vocab=512, d=64, n_layers=2, n_heads=2, seq_len=32, n_experts=4, top_k=2,
                       d_expert=64)
TC = replace(config.TRAIN, global_batch=8, warmup=1, val_seqs=8)


def batch(world, rank, seed=0, B=8, S=32, V=512):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, V, (B, S + 1), generator=g)
    bl = B // world
    x = x[rank * bl:(rank + 1) * bl]
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def grads_full(model, ctx):
    sd = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}
    exp = {n: t for n, t in sd.items() if is_expert_param(n)}
    sh = {n: t for n, t in sd.items() if n not in exp}
    parts = du.gather_objects_to0({"exp": exp if ctx.ep_group_id == 0 else {}}, ctx)
    if ctx.rank:
        return None
    for p in parts:
        sh.update(p["exp"])
    return sh


def loss_and_grads(spec, ctx, device):
    model = E.build(spec, TINY, ctx)
    E.apply_full_state(model, None, E.initial_full_state(spec, TINY), ctx)
    x, y = batch(ctx.world, ctx.rank)
    x, y = x.to(device), y.to(device)
    with E.autocast_ctx(spec, device):
        logits, aux = model(x)
    ce = F.cross_entropy(logits.float().reshape(-1, TINY.vocab), y.reshape(-1))
    loss = ce + (TC.lbl_coef * aux if spec.balance == "aux" else 0.0)
    loss.backward()
    E.sync_grads(model, ctx, scale=1.0, clip=None)
    t = loss.detach().float().reshape(1).cpu().contiguous()
    du.ar_(t, ctx.world_group, ctx.world)
    return float(t[0]) / ctx.world, grads_full(model, ctx)


def run_steps(spec, ctx, device, n=3):
    model = E.build(spec, TINY, ctx)
    opt = E.AdamW(model, TC)
    E.apply_full_state(model, None, E.initial_full_state(spec, TINY), ctx)
    sc = {"scale": 1024.0 if spec.precision == "fp16" else 1.0, "good": 0}
    for s in range(n):
        x, y = batch(ctx.world, ctx.rank, seed=100 + s)
        E.train_step(model, opt, sc, x.to(device), y.to(device), spec, TC, ctx, 1e-3)
    return model, opt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    dev = "cuda" if a.device == "cuda" else "cpu"
    if dev == "cuda":
        torch.cuda.set_device(0)
    prec = "fp16" if dev == "cuda" else "fp32"
    E.determinism_setup(dev).__enter__()
    torch.set_num_threads(2)
    res, ok = {}, True
    base_ctx = du.make_ctx(1, dev)
    W = base_ctx.world
    try:
        # ---------------- T1 (fp32 exactness, run on the requested device)
        tol = 1e-4 if dev == "cpu" else 2e-3
        ref_spec = config.RunSpec("selftest", "T", device=dev, precision="fp32", dp=1, ep=1)
        if base_ctx.rank == 0:
            ref_loss, ref_g = loss_and_grads(ref_spec, du.local_ctx(dev), dev)
        for ep in (1, 2, 4):
            if W % ep:
                continue
            for bal in ("aux", "auxfree"):
                spec = replace(ref_spec, dp=W, ep=ep, balance=bal)
                ctx = du.make_ctx(ep, dev)
                loss, g = loss_and_grads(spec, ctx, dev)
                if ctx.rank == 0:
                    if bal != "aux":
                        r_loss, r_g = loss_and_grads(replace(ref_spec, balance=bal), du.local_ctx(dev), dev)
                    else:
                        r_loss, r_g = ref_loss, ref_g
                    worst = max(float((g[n] - r_g[n]).abs().max() / (r_g[n].abs().max() + 1e-12))
                                for n in r_g)
                    dl = abs(loss - r_loss) / abs(r_loss)
                    passed = worst < tol and dl < 1e-5 * (1 if dev == "cpu" else 100)
                    res[f"T1_ep{ep}_{bal}"] = dict(rel_grad_err=worst, rel_loss_err=dl, passed=passed)
                    ok &= passed
                du.barrier(ctx)
        # ---------------- T2 determinism (in the arm's precision)
        spec = config.RunSpec("selftest", "T", device=dev, precision=prec, dp=W, ep=min(2, W))
        ctx = du.make_ctx(spec.ep, dev)
        m1, _ = run_steps(spec, ctx, dev)
        f1 = E.gather_full_state(m1, None, ctx, with_opt=False)
        m2, o2 = run_steps(spec, ctx, dev)
        f2 = E.gather_full_state(m2, o2, ctx, with_opt=True)
        if ctx.rank == 0:
            same = all(torch.equal(f1["model"][n], f2["model"][n]) for n in f1["model"])
            res["T2_bitwise_determinism"] = dict(passed=same)
            ok &= same
        # ---------------- T3 consolidation round trip into another layout
        f2 = du.broadcast_object(f2, ctx)
        if W >= 4:
            spec4 = replace(spec, ep=4)
            ctx4 = du.make_ctx(4, dev)
            m4 = E.build(spec4, TINY, ctx4)
            o4 = E.AdamW(m4, TC)
            E.apply_full_state(m4, o4, f2, ctx4)
            f4 = E.gather_full_state(m4, o4, ctx4, with_opt=True)
            if ctx4.rank == 0:
                same = all(torch.equal(f2["model"][n], f4["model"][n]) for n in f2["model"]) and \
                    all(torch.equal(f2["opt"]["m"][n], f4["opt"]["m"][n]) for n in f2["opt"]["m"])
                res["T3_roundtrip"] = dict(passed=same)
                ok &= same
        # ---------------- T4 local-scope variants run
        for kw in (dict(lbl_scope="local"), dict(cap_scope="local"),
                   dict(balance="auxfree", bias_scope="local")):
            s = replace(spec, **kw)
            c = du.make_ctx(s.ep, dev)
            run_steps(s, c, dev, n=2)
            if c.rank == 0:
                res["T4_" + "_".join(f"{k}={v}" for k, v in kw.items())] = dict(passed=True)
        # dense model also syncs
        s = replace(spec, moe=False, ep=1)
        c = du.make_ctx(1, dev)
        model = E.build(s, replace(TINY, moe=False), c)
        res["T4_dense_built"] = dict(passed=True)
    except Exception:
        ok = False
        res["exception"] = traceback.format_exc()
        print(res["exception"], flush=True)
    ok = not du.any_flag(not ok, base_ctx)
    if base_ctx.rank == 0:
        tag = "PASS" if ok else "FAIL"
        out = os.path.join(a.out_root, "selftest", f"{a.device}_{tag}.json")
        E.write_json_atomic(dict(world=W, device=a.device, results=res, ok=ok), out)
        print(f"SELFTEST {tag}", res, flush=True)
    du.destroy()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
