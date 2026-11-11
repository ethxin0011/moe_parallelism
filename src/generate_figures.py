"""Generate vector PDF figures (fig1..fig6) from measured CSV outputs."""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR = 1e-10

COLORS = {
    "seed": "#555555",
    "data": "#8c564b",
    "layout": "#1f77b4",
    "ulp": "#ff7f0e",
    "thread": "#9467bd",
    "semantic": "#d62728",
    "dense": "#2ca02c",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
        }
    )


def save(fig, out_dir: str, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name))
    plt.close(fig)
    print("wrote", name)


def plot_band(ax, df: pd.DataFrame, cat: str, color: str, label: str, metric: str = "kl", median: bool = True) -> None:
    grouped = df[df.category == cat].groupby("step")[metric]
    if not grouped.ngroups:
        return
    lo = grouped.min().clip(lower=FLOOR)
    hi = grouped.max().clip(lower=FLOOR)
    md = grouped.median().clip(lower=FLOOR)
    if grouped.size().max() > 1:
        ax.fill_between(lo.index, lo.values, hi.values, color=color, alpha=0.18, lw=0)
    if median:
        ax.plot(md.index, md.values, color=color, marker="o", ms=2.8, lw=1.4, label=label)


def set_log_axes(ax, ylabel: str = "KL divergence from reference") -> None:
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_ylim(FLOOR / 2, 1.5)
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "resultraw"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    configure_style()

    div = pd.read_csv(os.path.join(args.data_dir, "divergence_long.csv"))
    gran = pd.read_csv(os.path.join(args.data_dir, "gran_divergence.csv"))
    flip = pd.read_csv(os.path.join(args.data_dir, "flip_vs_margin.csv"))
    traj = pd.read_csv(os.path.join(args.data_dir, "margin_traj.csv"))

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    plot_band(ax, div, "seed", COLORS["seed"], "different random seed")
    plot_band(ax, div, "data", COLORS["data"], "different data order")
    plot_band(ax, div, "semantic_vs_samelayout", COLORS["semantic"], "per-rank definition switched on (9 runs)")
    plot_band(ax, div, "layout_numeric", COLORS["layout"], "different parallel layout (8 layouts)")
    plot_band(ax, div, "ulp", COLORS["ulp"], "one weight nudged by 1 ULP (3 runs)")
    plot_band(ax, div, "thread", COLORS["thread"], "different CPU thread count")
    set_log_axes(ax)
    ax.set_title("MoE trained in fp32 on CPU")
    ax.legend(loc="lower right", frameon=False)
    save(fig, args.out_dir, "fig1_divergence_curves.pdf")

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 3.0), sharey=True)
    ax = axs[0]
    plot_band(ax, div, "layout_numeric", COLORS["layout"], "MoE: layout change")
    plot_band(ax, div, "seed", COLORS["seed"], "MoE: new seed")
    plot_band(ax, div, "dense_layout", COLORS["dense"], "dense: layout change")
    dense_seed = div[div.category == "dense_seed"].groupby("step").kl.median()
    ax.plot(dense_seed.index, dense_seed.values, color=COLORS["dense"], ls="--", lw=1.2, label="dense: new seed")
    set_log_axes(ax)
    ax.set_title("(a) fp32 (CPU)")
    ax.legend(frameon=False, loc="lower right")

    ax = axs[1]
    for (gname, kind), color, ls, label in (
        (("e16", "layout"), COLORS["layout"], "-", "MoE: layout change"),
        (("e16", "seed"), COLORS["seed"], "-", "MoE: new seed"),
        (("dense", "layout"), COLORS["dense"], "-", "dense: layout change"),
        (("dense", "seed"), COLORS["dense"], "--", "dense: new seed"),
    ):
        sub = gran[(gran.gran == gname) & (gran.kind == kind)].sort_values("step")
        ax.plot(
            sub.step,
            sub.kl.clip(lower=FLOOR),
            color=color,
            ls=ls,
            marker="o" if ls == "-" else None,
            ms=2.8,
            lw=1.4 if ls == "-" else 1.2,
            label=label,
        )
    set_log_axes(ax)
    ax.set_ylabel("")
    ax.set_title("(b) fp16 (T4 GPU)")
    save(fig, args.out_dir, "fig2_moe_vs_dense.pdf")

    numeric_pairs = [
        ("A_dp2_ep1", "layout DP2"),
        ("A_dp4_ep1", "layout DP4"),
        ("A_dp8_ep4", "layout DP8/EP4"),
        ("B_ulp1", "1-ULP nudge"),
        ("X_dp1_thr8", "thread count"),
        ("H_base_dp2_ep2", "fp16, 16 experts"),
        ("I_e8_dp2_ep2", "fp16, 8 experts"),
        ("I_e32_dp2_ep2", "fp16, 32 experts"),
        ("I_e64_dp2_ep2", "fp16, 64 experts"),
    ]
    fig, axs = plt.subplots(1, 2, figsize=(6.8, 3.4), gridspec_kw={"width_ratios": [1, 1.1]})
    ax = axs[0]
    cmap = plt.get_cmap("tab10")
    for idx, (other, label) in enumerate(numeric_pairs):
        sub = flip[(flip.other == other) & (flip.step <= 256)].sort_values("step")
        ax.plot(sub.step, sub.auc_flip_has_smaller_margin, marker="o", ms=2.5, lw=1.1, color=cmap(idx), label=label)
    ax.axhline(0.5, color="k", ls=":", lw=0.8)
    ax.set_xscale("log", base=2)
    ax.set_ylim(0.45, 1.02)
    ax.set_xlabel("training step")
    ax.set_ylabel("AUC (flipped had smaller margin)")
    ax.set_title("(a) flips start at near-ties")
    ax.legend(frameon=False, fontsize=6.2, loc="lower left")

    ax = axs[1]
    pair_ids = [p for p, _ in numeric_pairs]
    labels = dict(numeric_pairs)
    step16 = flip[(flip.step == 16) & flip.other.isin(pair_ids)]
    step16 = step16.set_index("other").loc[[p for p in pair_ids if p in set(step16.other)]]
    ypos = np.arange(len(step16))
    ax.barh(
        ypos + 0.2,
        step16.flip_rate_lowest10pct_margin * 100,
        height=0.4,
        color="#d62728",
        label="10% of choices with smallest margin",
    )
    ax.barh(ypos - 0.2, step16.flip_rate_rest * 100, height=0.4, color="#7f7f7f", label="all other choices")
    ax.set_yticks(ypos)
    ax.set_yticklabels([labels[o] for o in step16.index], fontsize=7)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("choices flipped at step 16 (%, log scale)")
    ax.set_title("(b) near-ties flip far more often")
    ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    save(fig, args.out_dir, "fig3_mechanism.pdf")

    refs = {8: "I_e8_dp1_ep1", 16: "G_dp1_ep1", 32: "I_e32_dp1_ep1", 64: "I_e64_dp1_ep1"}
    gran_names = {8: "e8", 16: "e16", 32: "e32", 64: "e64"}
    expert_counts = sorted(refs)
    near = [
        traj[traj.run == refs[e]].sort_values("step").head(3)["near_tie_frac_0.01"].mean() * 100
        for e in expert_counts
    ]
    final = gran.sort_values("step").groupby(["gran", "kind"]).tail(1).set_index(["gran", "kind"])
    flip_final = [final.loc[(gran_names[e], "layout"), "route_flip"] * 100 for e in expert_counts]
    ratio = [
        final.loc[(gran_names[e], "layout"), "kl"] / final.loc[(gran_names[e], "seed"), "kl"] * 100
        for e in expert_counts
    ]
    dense_ratio = final.loc[("dense", "layout"), "kl"] / final.loc[("dense", "seed"), "kl"] * 100

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axs[0]
    ax.plot(expert_counts, near, marker="o", color="#d62728")
    for e, val in zip(expert_counts, near):
        ax.annotate(f"{val:.1f}%", (e, val), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7.5)
    ax.set_xscale("log", base=2)
    ax.set_xticks(expert_counts)
    ax.set_xticklabels([str(e) for e in expert_counts])
    ax.set_ylim(0, max(near) * 1.25)
    ax.set_xlabel("number of experts (total and active size fixed)")
    ax.set_ylabel("near-tie routing choices (%)")
    ax.set_title("(a) more experts, more near-ties")

    ax = axs[1]
    ax.plot(expert_counts, flip_final, marker="o", color=COLORS["layout"], label="routing choices that differ")
    ax.plot(expert_counts, ratio, marker="s", color=COLORS["semantic"], label="output difference vs. new seed")
    ax.axhline(dense_ratio, color=COLORS["dense"], ls="--", lw=1.1, label="dense model, output difference")
    ax.set_xscale("log", base=2)
    ax.set_xticks(expert_counts)
    ax.set_xticklabels([str(e) for e in expert_counts])
    ax.set_ylim(0, 105)
    ax.set_xlabel("number of experts")
    ax.set_ylabel("% at the end of training")
    ax.set_title("(b) routing diverges more, outputs do not")
    ax.legend(frameon=False, fontsize=6.8, loc="lower right")
    save(fig, args.out_dir, "fig4_granularity.pdf")

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    full = div[div.category == "layout_numeric"].groupby("step").kl.median().clip(lower=FLOOR)
    ax.plot(
        full.index,
        full.values,
        color=COLORS["layout"],
        lw=1.6,
        marker="o",
        ms=3,
        label="layout different from step 0 (median of 8)",
    )
    seed_med = div[div.category == "seed"].groupby("step").kl.median()
    ax.plot(seed_med.index, seed_med.values, color=COLORS["seed"], lw=1.2, ls="--", label="different random seed")
    styles = {
        "F_b128_dp8_ep4": ("#17becf", "-", "s", "step 128 (10%): switch to DP8/EP4"),
        "F_b128_dp4_ep1": ("#17becf", "--", "^", "step 128 (10%): switch to DP4"),
        "F_b128_ulp1": ("#17becf", ":", "x", "step 128 (10%): 1-ULP nudge"),
        "F_b1024_dp8_ep4": ("#e377c2", "-", "s", "step 1024 (84%): switch to DP8/EP4"),
        "F_b1024_dp4_ep1": ("#e377c2", "--", "^", "step 1024 (84%): switch to DP4"),
        "F_b1024_ulp1": ("#e377c2", ":", "x", "step 1024 (84%): 1-ULP nudge"),
    }
    for other, (col, ls, mk, lab) in styles.items():
        sub = div[div.other == other].sort_values("step")
        ax.plot(sub.step, sub.kl.clip(lower=FLOOR), color=col, ls=ls, marker=mk, ms=4, lw=1.2, label=lab)
    ax.axvline(128, color="#17becf", lw=0.6, alpha=0.6)
    ax.axvline(1024, color="#e377c2", lw=0.6, alpha=0.6)
    ax.set_yscale("log")
    ax.set_xlim(0, 1260)
    ax.set_ylim(3e-3, 0.6)
    ax.set_xlabel("pretraining step (of 1,221; vertical lines = switch points)")
    ax.set_ylabel("KL divergence from reference")
    ax.set_title("Resuming pretraining on a new layout")
    ax.legend(frameon=False, fontsize=6.4, loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, args.out_dir, "fig5_resume.pdf")

    fig, axs = plt.subplots(1, 2, figsize=(6.6, 2.9))
    ax = axs[0]
    sizes = [
        ("scale_base_layout_dp2ep2", "base (~25M)", "#1f77b4"),
        ("scale_m_layout_dp2ep2", "m (~68M)", "#ff7f0e"),
        ("scale_l_layout_dp2ep2", "l (~117M)", "#2ca02c"),
    ]
    for cat, lab, col in sizes:
        sub = div[div.category == cat].sort_values("step")
        ax.plot(sub.step, sub.kl.clip(lower=FLOOR), marker="o", ms=2.5, color=col, label=lab)
    set_log_axes(ax)
    ax.set_title("(a) layout difference by model size (fp16)")
    ax.legend(frameon=False, loc="lower right")

    ax = axs[1]
    for run_id, lab, col in (
        ("G_dp1_ep1", "base", "#1f77b4"),
        ("H_m_dp1_ep1", "m", "#ff7f0e"),
        ("H_l_dp1_ep1", "l", "#2ca02c"),
    ):
        sub = traj[traj.run == run_id].sort_values("step")
        ax.plot(sub.step, sub["near_tie_frac_0.01"] * 100, marker="o", ms=2.5, color=col, label=lab)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("training step")
    ax.set_ylabel("near-tie routing choices (%)")
    ax.set_title("(b) near-ties by model size")
    ax.legend(frameon=False)
    save(fig, args.out_dir, "fig6_model_size.pdf")


if __name__ == "__main__":
    main()
