"""Post-hoc analysis (CPU). Reads snapshots/metrics from <out_root>/runs, writes
<out_root>/analysis/{divergence_long.csv, run_summary.csv, summary.json, verdict.md, figures/}.

Safe to run on partial results (only finished snapshot pairs are compared).
Divergence metrics for a pair (ref, other) at a common snapshot step, on a fixed val set:
  kl            mean_t KL(p_ref || p_other)                      (function space)
  top1_agree    argmax agreement
  route_flip    fraction of (token, layer) whose top-k expert SET differs (MoE only)
  usage_l1      mean_layer L1 distance between expert-usage distributions
  param_rel     ||theta_o - theta_r|| / ||theta_r|| (also experts-only / shared-only)
  dce           val CE(other) - val CE(ref)
  barrier       max_a CE(interp_a) - interp of endpoint CEs, a in {.25,.5,.75} (final step)
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import dist_utils as du
import engine as E
from model import LM, is_expert_param

torch.set_grad_enabled(False)


def snaps_of(root, rid):
    out = {}
    for f in glob.glob(os.path.join(root, "runs", rid, "snaps", "snap_*.pt")):
        out[int(os.path.basename(f)[5:12])] = f
    return out


def load_val(root, n_seqs, S):
    d = os.path.join(root, "data")
    v = np.memmap(os.path.join(d, "val.bin"), dtype=np.uint16, mode="r")
    # use sequences AFTER the training-time val slice (held-out for analysis)
    off = 64
    x = np.stack([v[(off + i) * S:(off + i) * S + S + 1] for i in range(n_seqs)]).astype(np.int64)
    t = torch.from_numpy(x)
    return t[:, :-1].contiguous(), t[:, 1:].contiguous()


class Evaluator:
    def __init__(self, root, n_seqs):
        self.root = root
        self.x, self.y = load_val(root, n_seqs, config.MODEL.seq_len)
        self.models = {}

    def model_for(self, spec):
        key = (spec.moe, spec.balance, getattr(spec, "size", "base"))
        if key not in self.models:
            s = config.RunSpec("eval", "eval", moe=spec.moe, balance=spec.balance, capacity_factor=0.0,
                               size=key[2])
            self.models[key] = LM(config.model_cfg_for(s), s, du.local_ctx("cpu")).eval()
        return self.models[key]

    def forward_state(self, spec, state):
        m = self.model_for(spec)
        m.load_state_dict(state, strict=True)
        logits, _ = m(self.x)
        lp = logits.float().log_softmax(-1).reshape(-1, logits.shape[-1])
        ce = float(F.nll_loss(lp, self.y.reshape(-1)))
        routes, usage = [], []
        for layer in m.moe_layers():
            idx = layer.last_idx.sort(-1).values
            routes.append(idx.clone())
            usage.append(torch.bincount(idx.reshape(-1), minlength=layer.E).float() / idx.numel())
        return dict(lp=lp, ce=ce, top1=lp.argmax(-1), routes=routes, usage=usage)

    def out(self, rid, step):
        f = snaps_of(self.root, rid).get(step)
        full = torch.load(f, map_location="cpu", weights_only=False)
        o = self.forward_state(config.RUNS[rid], full["model"])
        o["state"] = full["model"]
        return o


def flat_params(state):
    sh = [v.float().reshape(-1) for k, v in sorted(state.items()) if not is_expert_param(k) and not k.endswith("route_bias")]
    ex = [v.float().reshape(-1) for k, v in sorted(state.items()) if is_expert_param(k)]
    return torch.cat(sh), (torch.cat(ex) if ex else torch.zeros(0))


def pair_metrics(r, o):
    pr = r["lp"].exp()
    kl = float((pr * (r["lp"] - o["lp"])).sum(-1).mean())
    m = dict(kl=max(kl, 0.0), top1_agree=float((r["top1"] == o["top1"]).float().mean()),
             ce_ref=r["ce"], ce_other=o["ce"], dce=o["ce"] - r["ce"])
    if r["routes"] and o["routes"]:
        flips = [float((a != b).any(-1).float().mean()) for a, b in zip(r["routes"], o["routes"])]
        m["route_flip"] = float(np.mean(flips))
        m["usage_l1"] = float(np.mean([float((a - b).abs().sum()) for a, b in zip(r["usage"], o["usage"])]))
    sr, er = flat_params(r["state"])
    so, eo = flat_params(o["state"])
    allr, allo = torch.cat([sr, er]), torch.cat([so, eo])
    m["param_rel"] = float((allo - allr).norm() / allr.norm())
    m["param_rel_shared"] = float((so - sr).norm() / sr.norm())
    if er.numel():
        m["param_rel_experts"] = float((eo - er).norm() / er.norm())
    return m


def barrier(ev, spec, sa, sb):
    ces = []
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for a in alphas:
        st = {k: ((1 - a) * sa[k].float() + a * sb[k].float()).to(sa[k].dtype) if sa[k].is_floating_point()
              else sa[k] for k in sa}
        ces.append(ev.forward_state(spec, st)["ce"])
    lin = [(1 - a) * ces[0] + a * ces[-1] for a in alphas]
    return float(max(c - l for c, l in zip(ces, lin)))


def run_summary(root):
    rows = []
    for rid, spec in config.RUNS.items():
        mfiles = sorted(glob.glob(os.path.join(root, "runs", rid, "metrics", "m_*.jsonl")))
        if not mfiles:
            continue
        recs = {}
        for f in mfiles:
            with open(f) as fh:
                for line in fh:
                    r = json.loads(line)
                    recs[r["step"]] = r                     # dedup: keep last
        df = pd.DataFrame([recs[k] for k in sorted(recs)])
        done = E.read_json(os.path.join(root, "runs", rid, "DONE.json"))
        last_val = df["val_ce"].dropna().iloc[-1] if "val_ce" in df and df["val_ce"].notna().any() else np.nan
        rows.append(dict(run_id=rid, arm=spec.arm, dp=spec.dp, ep=spec.ep, device=spec.device,
                         precision=spec.precision, balance=spec.balance, lbl_scope=spec.lbl_scope,
                         cap_scope=spec.cap_scope, bias_scope=spec.bias_scope, moe=spec.moe,
                         last_step=int(df["step"].max()), done=bool(done), final_val_ce=last_val,
                         mean_tok_per_s=float(df["tok_per_s"].dropna().mean()) if "tok_per_s" in df else np.nan,
                         mean_maxvio=float(df["maxvio"].dropna().mean()) if "maxvio" in df else np.nan,
                         mean_drop=float(df["drop_frac"].dropna().mean()) if "drop_frac" in df else np.nan,
                         skipped_steps=int((~df["finite"].astype(bool)).sum()) if "finite" in df else 0))
        df.to_csv(os.path.join(root, "analysis", "curves", f"{rid}.csv"), index=False)
    return pd.DataFrame(rows)


def divergence(root, n_seqs):
    ev = Evaluator(root, n_seqs)
    by_ref = defaultdict(list)
    for c in config.COMPARISONS:
        by_ref[c["ref"]].append(c)
    rows = []
    for ref, comps in by_ref.items():
        ref_snaps = snaps_of(root, ref)
        steps = sorted(ref_snaps)
        for step in steps:
            todo = [c for c in comps if step >= c["min_step"] and step in snaps_of(root, c["other"])]
            if not todo:
                continue
            r = ev.out(ref, step)
            for c in todo:
                o = ev.out(c["other"], step)
                m = pair_metrics(r, o)
                final_common = step == max(set(ref_snaps) & set(snaps_of(root, c["other"])))
                if final_common and c["category"] in config.BARRIER_CATEGORIES:
                    m["barrier"] = barrier(ev, config.RUNS[ref], r["state"], o["state"])
                rows.append(dict(comparison=c["name"], category=c["category"], ref=ref,
                                 other=c["other"], step=step, final=final_common, **m))
                print(f"{c['name']} @ {step}: kl={m['kl']:.3g} flip={m.get('route_flip', float('nan')):.3g}",
                      flush=True)
                del o
            del r
    return pd.DataFrame(rows)


def growth_slope(df, cat, max_step=256, metric="param_rel"):
    d = df[(df.category == cat) & (df.step <= max_step) & (df[metric] > 0)]
    if len(d) < 3:
        return None
    x, y = d.step.values.astype(float), np.log10(d[metric].values)
    return float(np.polyfit(x, y, 1)[0])


def summarize(df, rs):
    S = {}
    if df.empty:
        return S, "No comparable snapshots yet."
    fin = df[df.final]
    lines = ["# Auto-generated verdict (check numbers before quoting)\n"]

    def band(cat, metric):
        d = fin[fin.category == cat][metric].dropna()
        return (float(d.mean()), float(d.max()), int(len(d))) if len(d) else None

    for metric in ("kl", "route_flip", "param_rel", "dce"):
        S[f"seed_band_{metric}"] = band("seed", metric)
        S[f"ulp_band_{metric}"] = band("ulp", metric)
    sb = S.get("seed_band_kl")
    ub = S.get("ulp_band_kl")
    lines.append(f"Seed band (final KL): {sb}  |  1-ULP band: {ub}\n")
    tab = []
    for cat in ("layout_numeric", "thread", "semantic_vs_samelayout", "semantic_total", "branch_layout",
                "branch_ulp", "auxfree_layout_bglobal", "auxfree_layout_blocal", "auxfree_blocal_vs_bglobal",
                "dense_layout", "fp16_layout", "precision_fp16_vs_fp32"):
        for _, r in fin[fin.category == cat].iterrows():
            ratio = r.kl / sb[0] if sb and sb[0] > 0 else np.nan
            within = bool(sb and r.kl <= sb[1])
            tab.append(dict(category=cat, other=r.other, step=int(r.step), kl=r.kl,
                            kl_over_seed_mean=ratio, within_seed_band=within,
                            route_flip=r.get("route_flip", np.nan), dce=r.dce,
                            barrier=r.get("barrier", np.nan)))
    S["final_table"] = tab
    num = [t for t in tab if t["category"] == "layout_numeric"]
    if num:
        n_in = sum(t["within_seed_band"] for t in num)
        lines.append(f"Numerical-only layouts within seed band at the end: {n_in}/{len(num)}.")
        lines.append("=> Outcome 1 (layouts interchangeable, numerics alone)." if n_in == len(num) else
                     "=> Outcome 2 (numerical layout effect exceeds seed noise for some layouts; report it).")
    sem = [t for t in tab if t["category"] == "semantic_vs_samelayout"]
    if sem:
        big = [t["other"] for t in sem if not t["within_seed_band"]]
        lines.append(f"Semantic toggles beyond seed band vs same-layout numeric run: {big or 'none'}")
    sm, sd = growth_slope(df, "ulp"), growth_slope(df, "dense_ulp")
    S["early_growth_slope_moe"], S["early_growth_slope_dense"] = sm, sd
    if sm is not None and sd is not None and sd != 0:
        S["moe_over_dense_growth"] = sm / sd
        lines.append(f"Early divergence growth (log10 param_rel per step): MoE {sm:.3g} vs dense {sd:.3g} "
                     f"(ratio {sm / sd:.2f}).")
    br = [t for t in tab if t["category"] in ("branch_layout", "branch_ulp")]
    if br:
        lines.append("Elastic-resume branches (final KL): " +
                     ", ".join(f"{t['other']}={t['kl']:.3g}" for t in br))
    for sz, sc, lc, lo in (("base", "fp16_seed", "fp16_layout", "G_dp4_ep2"),
                           ("m", "scale_m_seed", "scale_m_layout", "H_m_dp4_ep2"),
                           ("l", "scale_l_seed", "scale_l_layout", "H_l_dp4_ep2")):
        sd_ = fin[fin.category == sc].kl
        ld_ = fin[(fin.category == lc) & (fin.other == lo)].kl
        if len(sd_) and len(ld_) and sd_.mean() > 0:
            ratio = float(ld_.mean() / sd_.mean())
            S[f"scale_{sz}_layout_over_seed"] = ratio
            lines.append(f"Size {sz}: DP4/EP2 layout KL / seed KL = {ratio:.3g}")
    for sz, sc in (("base", "fp16_seed"), ("m", "scale_m_seed"), ("l", "scale_l_seed")):
        sd_ = fin[fin.category == sc].kl
        ld_ = fin[fin.category == f"scale_{sz}_layout_dp2ep2"].kl
        if len(sd_) and len(ld_) and sd_.mean() > 0:
            ratio = float(ld_.mean() / sd_.mean())
            S[f"scale_{sz}_dp2ep2_over_seed"] = ratio
            lines.append(f"Size {sz}: DP2/EP2 layout KL / seed KL = {ratio:.3g}")
    if not rs.empty:
        S["runs_done"] = int(rs.done.sum())
        S["runs_seen"] = int(len(rs))
    return S, "\n".join(lines) + "\n"


def figures(df, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    if df.empty:
        return

    def band_fill(ax, cat, metric, color, label):
        d = df[df.category == cat].groupby("step")[metric]
        if d.ngroups:
            lo, hi = d.min(), d.max()
            ax.fill_between(lo.index, lo.values, hi.values, color=color, alpha=0.25, label=label)

    for metric, ylab in (("kl", "KL(ref || run) on held-out val"), ("route_flip", "router top-k flip rate"),
                         ("param_rel", "relative parameter distance")):
        if metric not in df:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        band_fill(ax, "seed", metric, "grey", "seed band")
        band_fill(ax, "ulp", metric, "tab:blue", "1-ULP band")
        for oth, d in df[df.category == "layout_numeric"].groupby("other"):
            ax.plot(d.step, d[metric], marker=".", lw=1, label=oth)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("step")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig_{metric}_layout_vs_noise.png"), dpi=150)
        plt.close(fig)
    fin = df[df.final]
    if not fin.empty:
        sb = fin[fin.category == "seed"].kl.mean()
        d = fin[fin.category.isin(["layout_numeric", "semantic_vs_samelayout", "semantic_total", "thread",
                                   "auxfree_blocal_vs_bglobal", "dense_layout"])]
        if not d.empty and sb and sb > 0:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(d))))
            ax.barh(d.other + " (" + d.category + ")", d.kl / sb)
            ax.axvline(1.0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("final KL / mean seed KL")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig_final_ratio.png"), dpi=150)
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for cat, col in (("ulp", "tab:red"), ("dense_ulp", "tab:green")):
        d = df[df.category == cat].groupby("step").param_rel.mean()
        if len(d):
            ax.plot(d.index, d.values, marker="o", color=col, label=f"{cat} (mean)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("relative parameter distance")
    ax.set_title("MoE vs dense: growth of a 1-ULP perturbation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_moe_vs_dense_ulp.png"), dpi=150)
    plt.close(fig)
    for cats, name in ((("branch_layout", "branch_ulp"), "fig_branch.png"),
                       (("fp16_layout", "fp16_seed", "fp16_ulp"), "fig_fp16.png")):
        d = df[df.category.isin(cats)]
        if d.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for oth, g in d.groupby("other"):
            ax.plot(g.step, g.kl, marker=".", label=oth)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("step")
        ax.set_ylabel("KL vs reference")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, name), dpi=150)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--val_seqs", type=int, default=16)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    torch.set_num_threads(a.threads or (os.cpu_count() or 1))
    out = os.path.join(a.out_root, "analysis")
    os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    rs = run_summary(a.out_root)
    rs.to_csv(os.path.join(out, "run_summary.csv"), index=False)
    df = divergence(a.out_root, a.val_seqs)
    df.to_csv(os.path.join(out, "divergence_long.csv"), index=False)
    S, verdict = summarize(df, rs)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(S, f, indent=1, default=float)
    with open(os.path.join(out, "verdict.md"), "w") as f:
        f.write(verdict)
    figures(df, os.path.join(out, "figures"))
    print(verdict)


if __name__ == "__main__":
    main()
