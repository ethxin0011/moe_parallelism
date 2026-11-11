"""Pairwise divergence evaluation, linear mode-connectivity barriers, and run summaries."""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import dist_utils as du
import engine
from model import LM, is_expert_param

torch.set_grad_enabled(False)


def snaps_of(root: str, run_id: str) -> dict[int, str]:
    out = {}
    for path in glob.glob(os.path.join(root, "runs", run_id, "snaps", "snap_*.pt")):
        out[int(os.path.basename(path)[5:12])] = path
    return out


def load_val(root: str, n_seqs: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    val = np.memmap(os.path.join(root, "data", "val.bin"), dtype=np.uint16, mode="r")
    offset = 64
    chunks = np.stack([val[(offset + i) * seq_len:(offset + i) * seq_len + seq_len + 1] for i in range(n_seqs)]).astype(np.int64)
    toks = torch.from_numpy(chunks)
    return toks[:, :-1].contiguous(), toks[:, 1:].contiguous()


class Evaluator:
    def __init__(self, root: str, n_seqs: int):
        self.root = root
        self.x, self.y = load_val(root, n_seqs, config.MODEL.seq_len)
        self.models: dict[tuple, LM] = {}

    def model_for(self, spec) -> LM:
        key = (spec.moe, spec.balance, getattr(spec, "size", "base"))
        if key not in self.models:
            eval_spec = config.RunSpec(
                "eval",
                "eval",
                moe=spec.moe,
                balance=spec.balance,
                capacity_factor=0.0,
                size=key[2],
            )
            self.models[key] = LM(config.model_cfg_for(eval_spec), eval_spec, du.local_ctx("cpu")).eval()
        return self.models[key]

    def forward_state(self, spec, state: dict[str, torch.Tensor]) -> dict:
        model = self.model_for(spec)
        model.load_state_dict(state, strict=True)
        logits, _ = model(self.x)
        log_probs = logits.float().log_softmax(-1).reshape(-1, logits.shape[-1])
        ce = float(F.nll_loss(log_probs, self.y.reshape(-1)))
        routes = []
        usage = []
        for layer in model.moe_layers():
            idx = layer.last_idx.sort(-1).values
            routes.append(idx.clone())
            usage.append(torch.bincount(idx.reshape(-1), minlength=layer.E).float() / idx.numel())
        return dict(lp=log_probs, ce=ce, top1=log_probs.argmax(-1), routes=routes, usage=usage)

    def out(self, run_id: str, step: int) -> dict:
        snap_file = snaps_of(self.root, run_id).get(step)
        full = torch.load(snap_file, map_location="cpu", weights_only=False)
        res = self.forward_state(config.RUNS[run_id], full["model"])
        res["state"] = full["model"]
        return res


def flat_params(state: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    shared = [
        v.float().reshape(-1)
        for k, v in sorted(state.items())
        if not is_expert_param(k) and not k.endswith("route_bias")
    ]
    experts = [v.float().reshape(-1) for k, v in sorted(state.items()) if is_expert_param(k)]
    return torch.cat(shared), (torch.cat(experts) if experts else torch.zeros(0))


def pair_metrics(ref_out: dict, other_out: dict) -> dict:
    p_ref = ref_out["lp"].exp()
    kl = float((p_ref * (ref_out["lp"] - other_out["lp"])).sum(-1).mean())
    metrics = dict(
        kl=max(kl, 0.0),
        top1_agree=float((ref_out["top1"] == other_out["top1"]).float().mean()),
        ce_ref=ref_out["ce"],
        ce_other=other_out["ce"],
        dce=other_out["ce"] - ref_out["ce"],
    )
    if ref_out["routes"] and other_out["routes"]:
        flips = [float((a != b).any(-1).float().mean()) for a, b in zip(ref_out["routes"], other_out["routes"])]
        metrics["route_flip"] = float(np.mean(flips))
        metrics["usage_l1"] = float(
            np.mean([float((a - b).abs().sum()) for a, b in zip(ref_out["usage"], other_out["usage"])])
        )
    sr, er = flat_params(ref_out["state"])
    so, eo = flat_params(other_out["state"])
    all_r = torch.cat([sr, er])
    all_o = torch.cat([so, eo])
    metrics["param_rel"] = float((all_o - all_r).norm() / all_r.norm())
    metrics["param_rel_shared"] = float((so - sr).norm() / sr.norm())
    if er.numel():
        metrics["param_rel_experts"] = float((eo - er).norm() / er.norm())
    return metrics


def loss_barrier(ev: Evaluator, spec, state_a: dict, state_b: dict) -> float:
    ces = []
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    for alpha in alphas:
        interp = {
            k: ((1 - alpha) * state_a[k].float() + alpha * state_b[k].float()).to(state_a[k].dtype)
            if state_a[k].is_floating_point()
            else state_a[k]
            for k in state_a
        }
        ces.append(ev.forward_state(spec, interp)["ce"])
    linear = [(1 - alpha) * ces[0] + alpha * ces[-1] for alpha in alphas]
    return float(max(c - lin for c, lin in zip(ces, linear)))


def run_summary(root: str) -> pd.DataFrame:
    rows = []
    for run_id, spec in config.RUNS.items():
        mfiles = sorted(glob.glob(os.path.join(root, "runs", run_id, "metrics", "m_*.jsonl")))
        if not mfiles:
            continue
        records = {}
        for path in mfiles:
            with open(path) as f:
                for line in f:
                    rec = json.loads(line)
                    records[rec["step"]] = rec
        df = pd.DataFrame([records[k] for k in sorted(records)])
        done = engine.read_json(os.path.join(root, "runs", run_id, "DONE.json"))
        last_val = df["val_ce"].dropna().iloc[-1] if "val_ce" in df and df["val_ce"].notna().any() else np.nan
        rows.append(
            dict(
                run_id=run_id,
                arm=spec.arm,
                dp=spec.dp,
                ep=spec.ep,
                device=spec.device,
                precision=spec.precision,
                balance=spec.balance,
                lbl_scope=spec.lbl_scope,
                cap_scope=spec.cap_scope,
                bias_scope=spec.bias_scope,
                moe=spec.moe,
                last_step=int(df["step"].max()),
                done=bool(done),
                final_val_ce=last_val,
                mean_tok_per_s=float(df["tok_per_s"].dropna().mean()) if "tok_per_s" in df else np.nan,
                mean_maxvio=float(df["maxvio"].dropna().mean()) if "maxvio" in df else np.nan,
                mean_drop=float(df["drop_frac"].dropna().mean()) if "drop_frac" in df else np.nan,
                skipped_steps=int((~df["finite"].astype(bool)).sum()) if "finite" in df else 0,
            )
        )
        df.to_csv(os.path.join(root, "analysis", "curves", f"{run_id}.csv"), index=False)
    return pd.DataFrame(rows)


def divergence(root: str, n_seqs: int) -> pd.DataFrame:
    ev = Evaluator(root, n_seqs)
    by_ref = defaultdict(list)
    for comp in config.COMPARISONS:
        by_ref[comp["ref"]].append(comp)
    rows = []
    for ref, comps in by_ref.items():
        ref_snaps = snaps_of(root, ref)
        for step in sorted(ref_snaps):
            todo = [c for c in comps if step >= c["min_step"] and step in snaps_of(root, c["other"])]
            if not todo:
                continue
            ref_out = ev.out(ref, step)
            for comp in todo:
                other_out = ev.out(comp["other"], step)
                metrics = pair_metrics(ref_out, other_out)
                final_common = step == max(set(ref_snaps) & set(snaps_of(root, comp["other"])))
                if final_common and comp["category"] in config.BARRIER_CATEGORIES:
                    metrics["barrier"] = loss_barrier(ev, config.RUNS[ref], ref_out["state"], other_out["state"])
                rows.append(
                    dict(
                        comparison=comp["name"],
                        category=comp["category"],
                        ref=ref,
                        other=comp["other"],
                        step=step,
                        final=final_common,
                        **metrics,
                    )
                )
                print(
                    f"{comp['name']} @ {step}: kl={metrics['kl']:.3g} "
                    f"flip={metrics.get('route_flip', float('nan')):.3g}",
                    flush=True,
                )
                del other_out
            del ref_out
    return pd.DataFrame(rows)


def growth_slope(df: pd.DataFrame, cat: str, max_step: int = 256, metric: str = "param_rel") -> float | None:
    sub = df[(df.category == cat) & (df.step <= max_step) & (df[metric] > 0)]
    if len(sub) < 3:
        return None
    x = sub.step.values.astype(float)
    y = np.log10(sub[metric].values)
    return float(np.polyfit(x, y, 1)[0])


def summarize(df: pd.DataFrame, rs: pd.DataFrame) -> tuple[dict, str]:
    summary: dict = {}
    if df.empty:
        return summary, "No comparable snapshots found.\n"
    fin = df[df.final]
    lines = ["# Divergence Summary\n"]

    def band(cat: str, metric: str):
        vals = fin[fin.category == cat][metric].dropna()
        return (float(vals.mean()), float(vals.max()), int(len(vals))) if len(vals) else None

    for metric in ("kl", "route_flip", "param_rel", "dce"):
        summary[f"seed_band_{metric}"] = band("seed", metric)
        summary[f"ulp_band_{metric}"] = band("ulp", metric)
    sb = summary.get("seed_band_kl")
    ub = summary.get("ulp_band_kl")
    lines.append(f"Seed band (final KL): {sb}  |  1-ULP band: {ub}\n")

    table = []
    categories = (
        "layout_numeric",
        "thread",
        "semantic_vs_samelayout",
        "semantic_total",
        "branch_layout",
        "branch_ulp",
        "auxfree_layout_bglobal",
        "auxfree_layout_blocal",
        "auxfree_blocal_vs_bglobal",
        "dense_layout",
        "fp16_layout",
        "precision_fp16_vs_fp32",
    )
    for cat in categories:
        for _, row in fin[fin.category == cat].iterrows():
            ratio = row.kl / sb[0] if sb and sb[0] > 0 else np.nan
            within = bool(sb and row.kl <= sb[1])
            table.append(
                dict(
                    category=cat,
                    other=row.other,
                    step=int(row.step),
                    kl=row.kl,
                    kl_over_seed_mean=ratio,
                    within_seed_band=within,
                    route_flip=row.get("route_flip", np.nan),
                    dce=row.dce,
                    barrier=row.get("barrier", np.nan),
                )
            )
    summary["final_table"] = table

    num = [t for t in table if t["category"] == "layout_numeric"]
    if num:
        n_in = sum(t["within_seed_band"] for t in num)
        mean_ratio = float(np.mean([t["kl_over_seed_mean"] for t in num]))
        lines.append(
            f"Numerical-only layouts within seed band at final step: {n_in}/{len(num)} "
            f"(mean KL / seed KL = {mean_ratio:.3f})."
        )

    sem = [t for t in table if t["category"] == "semantic_vs_samelayout"]
    if sem:
        exceeding = [t["other"] for t in sem if not t["within_seed_band"]]
        lines.append(f"Per-rank rule runs exceeding seed band vs same-layout baseline: {exceeding or 'none'}.")

    sm = growth_slope(df, "ulp")
    sd = growth_slope(df, "dense_ulp")
    summary["early_growth_slope_moe"] = sm
    summary["early_growth_slope_dense"] = sd
    if sm is not None and sd is not None and sd != 0:
        summary["moe_over_dense_growth"] = sm / sd
        lines.append(
            f"Early divergence slope (log10 param_rel per step): MoE {sm:.3g} vs dense {sd:.3g} "
            f"(ratio {sm / sd:.2f})."
        )

    branches = [t for t in table if t["category"] in ("branch_layout", "branch_ulp")]
    if branches:
        lines.append(
            "Resume branches (final KL): " + ", ".join(f"{t['other']}={t['kl']:.3g}" for t in branches)
        )

    for sz, seed_cat, lay_cat, lay_run in (
        ("base", "fp16_seed", "fp16_layout", "G_dp4_ep2"),
        ("m", "scale_m_seed", "scale_m_layout", "H_m_dp4_ep2"),
        ("l", "scale_l_seed", "scale_l_layout", "H_l_dp4_ep2"),
    ):
        seed_kl = fin[fin.category == seed_cat].kl
        lay_kl = fin[(fin.category == lay_cat) & (fin.other == lay_run)].kl
        if len(seed_kl) and len(lay_kl) and seed_kl.mean() > 0:
            ratio = float(lay_kl.mean() / seed_kl.mean())
            summary[f"scale_{sz}_layout_over_seed"] = ratio
            lines.append(f"Size {sz}: DP4/EP2 layout KL / seed KL = {ratio:.3g}")

    for sz, seed_cat in (("base", "fp16_seed"), ("m", "scale_m_seed"), ("l", "scale_l_seed")):
        seed_kl = fin[fin.category == seed_cat].kl
        lay_kl = fin[fin.category == f"scale_{sz}_layout_dp2ep2"].kl
        if len(seed_kl) and len(lay_kl) and seed_kl.mean() > 0:
            ratio = float(lay_kl.mean() / seed_kl.mean())
            summary[f"scale_{sz}_dp2ep2_over_seed"] = ratio
            lines.append(f"Size {sz}: DP2/EP2 layout KL / seed KL = {ratio:.3g}")

    if not rs.empty:
        summary["runs_done"] = int(rs.done.sum())
        summary["runs_seen"] = int(len(rs))
    return summary, "\n".join(lines) + "\n"


def save_diagnostic_plots(df: pd.DataFrame, outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    if df.empty:
        return

    def band_fill(ax, cat: str, metric: str, color: str, label: str):
        grouped = df[df.category == cat].groupby("step")[metric]
        if grouped.ngroups:
            lo, hi = grouped.min(), grouped.max()
            ax.fill_between(lo.index, lo.values, hi.values, color=color, alpha=0.25, label=label)

    for metric, ylabel in (
        ("kl", "KL(ref || run) on held-out val"),
        ("route_flip", "router top-k flip rate"),
        ("param_rel", "relative parameter distance"),
    ):
        if metric not in df:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        band_fill(ax, "seed", metric, "grey", "seed band")
        band_fill(ax, "ulp", metric, "tab:blue", "1-ULP band")
        for other, sub in df[df.category == "layout_numeric"].groupby("other"):
            ax.plot(sub.step, sub[metric], marker=".", lw=1, label=other)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"fig_{metric}_layout_vs_noise.png"), dpi=150)
        plt.close(fig)

    fin = df[df.final]
    if not fin.empty:
        seed_mean = fin[fin.category == "seed"].kl.mean()
        sub = fin[
            fin.category.isin(
                [
                    "layout_numeric",
                    "semantic_vs_samelayout",
                    "semantic_total",
                    "thread",
                    "auxfree_blocal_vs_bglobal",
                    "dense_layout",
                ]
            )
        ]
        if not sub.empty and seed_mean and seed_mean > 0:
            fig, ax = plt.subplots(figsize=(8, max(3, 0.25 * len(sub))))
            ax.barh(sub.other + " (" + sub.category + ")", sub.kl / seed_mean)
            ax.axvline(1.0, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("final KL / mean seed KL")
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig_final_ratio.png"), dpi=150)
            plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for cat, color in (("ulp", "tab:red"), ("dense_ulp", "tab:green")):
        series = df[df.category == cat].groupby("step").param_rel.mean()
        if len(series):
            ax.plot(series.index, series.values, marker="o", color=color, label=f"{cat} (mean)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("relative parameter distance")
    ax.set_title("MoE vs dense: 1-ULP parameter distance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig_moe_vs_dense_ulp.png"), dpi=150)
    plt.close(fig)

    for cats, fname in (
        (("branch_layout", "branch_ulp"), "fig_branch.png"),
        (("fp16_layout", "fp16_seed", "fp16_ulp"), "fig_fp16.png"),
    ):
        sub = df[df.category.isin(cats)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4))
        for other, grp in sub.groupby("other"):
            ax.plot(grp.step, grp.kl, marker=".", label=other)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("step")
        ax.set_ylabel("KL vs reference")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, fname), dpi=150)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--val_seqs", type=int, default=32)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(args.threads or (os.cpu_count() or 1))
    out = os.path.join(args.out_root, "analysis")
    os.makedirs(os.path.join(out, "curves"), exist_ok=True)
    rs = run_summary(args.out_root)
    rs.to_csv(os.path.join(out, "run_summary.csv"), index=False)
    df = divergence(args.out_root, args.val_seqs)
    df.to_csv(os.path.join(out, "divergence_long.csv"), index=False)
    summary, report = summarize(df, rs)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    with open(os.path.join(out, "verdict.md"), "w") as f:
        f.write(report)
    save_diagnostic_plots(df, os.path.join(out, "figures"))
    print(report)


if __name__ == "__main__":
    main()
