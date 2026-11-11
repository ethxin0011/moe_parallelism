"""Router score-margin flip analysis, near-tie trajectories, and expert-granularity evaluation."""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import config
import dist_utils as du
from model import LM

torch.set_grad_enabled(False)

TAUS = (0.01, 0.03, 0.1)

MECH_PAIRS = [
    ("A_dp1_ep1", "A_dp2_ep1"),
    ("A_dp1_ep1", "A_dp4_ep1"),
    ("A_dp1_ep1", "A_dp8_ep4"),
    ("A_dp1_ep1", "B_ulp1"),
    ("A_dp1_ep1", "X_dp1_thr8"),
    ("A_dp1_ep1", "C_dp8_ep1_cap"),
    ("A_dp1_ep1", "C_dp8_ep1_lbl"),
    ("D_dp1_ep1", "D_dp8_ep4_bglobal"),
    ("D_dp8_ep4_bglobal", "D_dp8_ep4_blocal"),
    ("G_dp1_ep1", "H_base_dp2_ep2"),
    ("I_e8_dp1_ep1", "I_e8_dp2_ep2"),
    ("I_e32_dp1_ep1", "I_e32_dp2_ep2"),
    ("I_e64_dp1_ep1", "I_e64_dp2_ep2"),
]

TRAJ_RUNS = [
    "A_dp1_ep1",
    "D_dp1_ep1",
    "G_dp1_ep1",
    "H_m_dp1_ep1",
    "H_l_dp1_ep1",
    "I_e8_dp1_ep1",
    "I_e32_dp1_ep1",
    "I_e64_dp1_ep1",
]

GRAN = {
    "e8": ("I_e8_dp1_ep1", "I_e8_seed1", "I_e8_dp2_ep2", 8),
    "e16": ("G_dp1_ep1", "G_seed1", "H_base_dp2_ep2", 16),
    "e32": ("I_e32_dp1_ep1", "I_e32_seed1", "I_e32_dp2_ep2", 32),
    "e64": ("I_e64_dp1_ep1", "I_e64_seed1", "I_e64_dp2_ep2", 64),
    "dense": ("I_dense_dp1", "I_dense_seed1", "I_dense_dp2", 0),
}


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


class SnapshotEvaluator:
    def __init__(self, root: str, n_seqs: int, device: str):
        self.root = root
        self.device = device
        x, y = load_val(root, n_seqs, config.MODEL.seq_len)
        self.x = x.to(device)
        self.y = y.to(device)
        self.models: dict[tuple, LM] = {}

    def model(self, spec) -> LM:
        cfg = config.model_cfg_for(spec)
        key = (dataclasses.astuple(cfg), spec.balance)
        if key not in self.models:
            eval_spec = config.RunSpec("eval", "eval", moe=spec.moe, balance=spec.balance, capacity_factor=0.0)
            self.models[key] = LM(cfg, eval_spec, du.local_ctx(self.device)).to(self.device).eval()
        return self.models[key]

    def evaluate(self, run_id: str, step: int, margins: bool = False) -> dict:
        spec = config.RUNS[run_id]
        full = torch.load(snaps_of(self.root, run_id)[step], map_location="cpu", weights_only=False)
        model = self.model(spec)
        model.load_state_dict(full["model"], strict=True)

        captured = []
        hooks = []
        if margins:
            hooks = [
                layer.register_forward_pre_hook(lambda mod, inp: captured.append(inp[0].detach()))
                for layer in model.moe_layers()
            ]
        logits, _ = model(self.x)
        for hook in hooks:
            hook.remove()

        log_probs = logits.float().log_softmax(-1).reshape(-1, logits.shape[-1])
        out = dict(
            lp=log_probs,
            ce=float(F.nll_loss(log_probs, self.y.reshape(-1))),
            top1=log_probs.argmax(-1),
            routes=[layer.last_idx.sort(-1).values.clone() for layer in model.moe_layers()],
            params=torch.cat(
                [v.float().reshape(-1) for k, v in sorted(full["model"].items()) if not k.endswith("route_bias")]
            ),
        )
        if margins:
            layer_margins = []
            for layer, xin in zip(model.moe_layers(), captured):
                xt = xin.reshape(-1, xin.shape[-1]).float()
                router_logits = xt @ layer.router.weight.float().t()
                scores = router_logits if spec.balance == "aux" else torch.sigmoid(router_logits) + layer.route_bias
                top_vals = scores.topk(layer.k + 1, dim=-1).values
                norm_gap = (top_vals[:, layer.k - 1] - top_vals[:, layer.k]) / (scores.std(-1) + 1e-12)
                layer_margins.append(norm_gap.cpu())
            out["margins"] = layer_margins
        return out


def auc_smaller(margin: np.ndarray, flipped: np.ndarray) -> float:
    """Probability that a randomly chosen flipped decision has a smaller margin than a kept one."""
    mask = flipped.astype(bool)
    n_pos = int(mask.sum())
    n_neg = int((~mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from scipy.stats import rankdata

        ranks = rankdata(margin)
    except Exception:
        ranks = np.empty(len(margin))
        ranks[np.argsort(margin, kind="mergesort")] = np.arange(1, len(margin) + 1)
    return float((ranks[~mask].sum() - n_neg * (n_neg + 1) / 2) / (n_pos * n_neg))


def pair_divergence(ref_out: dict, other_out: dict) -> dict:
    p_ref = ref_out["lp"].exp()
    kl = float((p_ref * (ref_out["lp"] - other_out["lp"])).sum(-1).mean())
    res = dict(
        kl=max(kl, 0.0),
        top1_agree=float((ref_out["top1"] == other_out["top1"]).float().mean()),
        dce=other_out["ce"] - ref_out["ce"],
        param_rel=float((other_out["params"] - ref_out["params"]).norm() / ref_out["params"].norm()),
    )
    if ref_out["routes"] and other_out["routes"]:
        res["route_flip"] = float(
            np.mean([float((a != b).any(-1).float().mean()) for a, b in zip(ref_out["routes"], other_out["routes"])])
        )
    return res


def analyze_margin_flips(ev: SnapshotEvaluator, root: str) -> pd.DataFrame:
    rows = []
    for ref, other in MECH_PAIRS:
        if ref not in config.RUNS or other not in config.RUNS:
            continue
        common = sorted(set(snaps_of(root, ref)) & set(snaps_of(root, other)))
        for step in common:
            ref_out = ev.evaluate(ref, step, margins=True)
            other_out = ev.evaluate(other, step)
            margins = torch.cat(ref_out["margins"]).numpy()
            flipped = torch.cat([(a != b).any(-1) for a, b in zip(ref_out["routes"], other_out["routes"])]).cpu().numpy()
            q10 = np.quantile(margins, 0.10)
            low = margins <= q10
            row = dict(
                ref=ref,
                other=other,
                step=step,
                flip_rate=float(flipped.mean()),
                auc_flip_has_smaller_margin=auc_smaller(margins, flipped),
                median_margin_flipped=float(np.median(margins[flipped])) if flipped.any() else np.nan,
                median_margin_kept=float(np.median(margins[~flipped])) if (~flipped).any() else np.nan,
                flip_rate_lowest10pct_margin=float(flipped[low].mean()),
                flip_rate_rest=float(flipped[~low].mean()) if (~low).any() else np.nan,
                kl=pair_divergence(ref_out, other_out)["kl"],
            )
            rows.append(row)
            print(
                f"[mech] {other} vs {ref} @{step}: flip={row['flip_rate']:.3g} "
                f"AUC={row['auc_flip_has_smaller_margin']:.3f}",
                flush=True,
            )
            del ref_out, other_out
    return pd.DataFrame(rows)


def analyze_margin_trajectories(ev: SnapshotEvaluator, root: str) -> pd.DataFrame:
    rows = []
    for run_id in TRAJ_RUNS:
        if run_id not in config.RUNS:
            continue
        cfg = config.model_cfg_for(config.RUNS[run_id])
        for step in sorted(snaps_of(root, run_id)):
            out = ev.evaluate(run_id, step, margins=True)
            margins = torch.cat(out["margins"]).numpy()
            row = dict(
                run=run_id,
                experts=cfg.n_experts,
                top_k=cfg.top_k,
                d=cfg.d,
                step=step,
                median_norm_margin=float(np.median(margins)),
            )
            for tau in TAUS:
                row[f"near_tie_frac_{tau}"] = float((margins < tau).mean())
            rows.append(row)
            del out
        print(f"[traj] {run_id} done", flush=True)
    return pd.DataFrame(rows)


def analyze_granularity(ev: SnapshotEvaluator, root: str) -> pd.DataFrame:
    rows = []
    for gran, (ref, seed_run, layout_run, n_experts) in GRAN.items():
        for kind, other in (("seed", seed_run), ("layout", layout_run)):
            if ref not in config.RUNS or other not in config.RUNS:
                continue
            for step in sorted(set(snaps_of(root, ref)) & set(snaps_of(root, other))):
                metrics = pair_divergence(ev.evaluate(ref, step), ev.evaluate(other, step))
                rows.append(dict(gran=gran, experts=n_experts, kind=kind, ref=ref, other=other, step=step, **metrics))
            print(f"[gran] {gran} {kind} done", flush=True)
    return pd.DataFrame(rows)


def summarize(m1: pd.DataFrame, m2: pd.DataFrame, m3: pd.DataFrame) -> tuple[dict, str]:
    summary: dict = {}
    lines = ["# Router Margin and Expert Granularity Summary\n"]
    if not m1.empty:
        early = m1[(m1.flip_rate > 0) & (m1.flip_rate < 0.5)]
        summary["mech_auc_median_early"] = float(early.auc_flip_has_smaller_margin.median()) if len(early) else None
        summary["mech_flip_low10_vs_rest"] = (
            (float(early.flip_rate_lowest10pct_margin.mean()), float(early.flip_rate_rest.mean()))
            if len(early)
            else None
        )
        lines.append("## 1. Router margin vs. routing flips (0 < flip rate < 0.5)")
        lines.append(f"Median AUC that flipped decisions have smaller margins: {summary['mech_auc_median_early']}")
        lines.append(f"Flip rate in lowest-10% margins vs rest: {summary['mech_flip_low10_vs_rest']}\n")
        lines.append(
            early.groupby("other")[["auc_flip_has_smaller_margin", "flip_rate"]].median().round(4).to_string() + "\n"
        )
    if not m2.empty:
        first = (
            m2.sort_values("step")
            .groupby("run")
            .head(3)
            .groupby("run")[[f"near_tie_frac_{t}" for t in TAUS]]
            .mean()
        )
        summary["near_tie_early"] = first.round(5).to_dict(orient="index")
        lines.append("## 2. Near-tie fraction (mean of first 3 snapshots)")
        lines.append(first.round(5).to_string() + "\n")
    if not m3.empty:
        table = []
        for gran, sub in m3.groupby("gran"):
            seed = sub[sub.kind == "seed"].sort_values("step")
            lay = sub[sub.kind == "layout"].sort_values("step")
            if seed.empty or lay.empty:
                continue
            fin_seed = seed.kl.iloc[-1]
            fin_lay = lay.kl.iloc[-1]

            def first_step(threshold: float):
                hits = lay[lay.kl >= threshold]
                return int(hits.step.iloc[0]) if len(hits) else None

            row = dict(
                gran=gran,
                experts=int(sub.experts.iloc[0]),
                final_layout_kl=fin_lay,
                final_seed_kl=fin_seed,
                layout_over_seed=fin_lay / fin_seed if fin_seed > 0 else np.nan,
                layout_kl_at_32=float(lay[lay.step == 32].kl.iloc[0]) if (lay.step == 32).any() else np.nan,
                layout_kl_at_128=float(lay[lay.step == 128].kl.iloc[0]) if (lay.step == 128).any() else np.nan,
                step_kl_ge_1e3=first_step(1e-3),
                step_kl_ge_1e2=first_step(1e-2),
            )
            table.append(row)
        table_df = pd.DataFrame(table).sort_values("experts")
        summary["granularity"] = table_df.to_dict(orient="records")
        lines.append("## 3. Expert granularity (fixed total and active parameters, fp16 T4)")
        lines.append(table_df.to_string(index=False) + "\n")
    return summary, "\n".join(lines)


def save_plots(m1: pd.DataFrame, m2: pd.DataFrame, m3: pd.DataFrame, outdir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not m3.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for gran, sub in m3[m3.kind == "layout"].groupby("gran"):
            sub = sub.sort_values("step")
            ax.plot(sub.step, sub.kl, marker="o", label=gran)
        ax.set_xscale("log")
        ax.set_yscale("symlog", linthresh=1e-9)
        ax.set_xlabel("step")
        ax.set_ylabel("KL(ref || DP2/EP2 layout)")
        ax.set_title("Layout divergence vs expert granularity")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_gran_layout_kl.png"), dpi=150)
        plt.close(fig)

    if not m2.empty:
        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        for run_id, sub in m2.groupby("run"):
            sub = sub.sort_values("step")
            ax.plot(sub.step, sub["near_tie_frac_0.03"], marker=".", label=run_id)
        ax.set_xscale("log")
        ax.set_xlabel("step")
        ax.set_ylabel("near-tie fraction (margin < 0.03 sd)")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig_near_tie.png"), dpi=150)
        plt.close(fig)

    if not m1.empty:
        early = m1[(m1.flip_rate > 0) & (m1.flip_rate < 0.5)]
        if len(early):
            fig, ax = plt.subplots(figsize=(6.5, 4.2))
            for other, sub in early.groupby("other"):
                ax.plot(sub.step, sub.auc_flip_has_smaller_margin, marker="o", label=other)
            ax.axhline(0.5, color="k", ls="--", lw=1)
            ax.set_xscale("log")
            ax.set_xlabel("step")
            ax.set_ylabel("AUC: flipped have smaller margin")
            ax.legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "fig_flip_margin_auc.png"), dpi=150)
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--val_seqs", type=int, default=32)
    args = parser.parse_args()

    dev = args.device
    if dev == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.set_num_threads(os.cpu_count() or 1)

    out = os.path.join(args.out_root, "analysis_ext")
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    ev = SnapshotEvaluator(args.out_root, args.val_seqs, dev)

    m1 = analyze_margin_flips(ev, args.out_root)
    m1.to_csv(os.path.join(out, "flip_vs_margin.csv"), index=False)

    m2 = analyze_margin_trajectories(ev, args.out_root)
    m2.to_csv(os.path.join(out, "margin_traj.csv"), index=False)

    m3 = analyze_granularity(ev, args.out_root)
    m3.to_csv(os.path.join(out, "gran_divergence.csv"), index=False)

    summary, report = summarize(m1, m2, m3)
    summary["minutes"] = (time.time() - t0) / 60
    with open(os.path.join(out, "summary_ext.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    with open(os.path.join(out, "ext_verdict.md"), "w") as f:
        f.write(report)
    save_plots(m1, m2, m3, out)
    with open(os.path.join(out, "EXT_DONE.json"), "w") as f:
        json.dump({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": summary["minutes"]}, f)
    print(report, flush=True)


if __name__ == "__main__":
    main()
