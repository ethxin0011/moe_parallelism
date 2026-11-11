"""Render paper tables (tab1..tab7) as cropped vector PDFs from measured CSV outputs."""
from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import numpy as np
import pandas as pd

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[paperwidth=9in,paperheight=12in,margin=0.4in]{geometry}
\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{booktabs}\usepackage{amsmath}
\usepackage{array}\pagestyle{empty}\begin{document}\small
"""


def build_table(out_dir: str, name: str, body: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tex_path = os.path.join(tmp, "t.tex")
        with open(tex_path, "w") as f:
            f.write(PREAMBLE + body + "\n\\end{document}\n")
        res = subprocess.run(["pdflatex", "-interaction=nonstopmode", "t.tex"], cwd=tmp, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout[-3000:])
            raise SystemExit(f"pdflatex failed for {name}")
        doc = pymupdf.open(os.path.join(tmp, "t.pdf"))
        page = doc[0]
        rect = pymupdf.Rect()
        for blk in page.get_text("blocks"):
            rect |= pymupdf.Rect(blk[:4])
        for drw in page.get_drawings():
            rect |= drw["rect"]
        rect = pymupdf.Rect(rect.x0 - 3, rect.y0 - 3, rect.x1 + 3, rect.y1 + 3) & page.rect
        out = pymupdf.open()
        new_page = out.new_page(width=rect.width, height=rect.height)
        new_page.show_pdf_page(new_page.rect, doc, 0, clip=rect)
        out.save(os.path.join(out_dir, name), garbage=3, deflate=True)
    print("wrote", name)


def fmt_sci(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    if x == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / 10 ** exp
    return f"${mant:.1f}\\times10^{{{exp}}}$"


def fmt_f3(x) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def fmt_pct(x, digits: int = 1) -> str:
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{digits}f}\\%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=os.path.join(REPO_ROOT, "resultraw"))
    parser.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "figures"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    div = pd.read_csv(os.path.join(args.data_dir, "divergence_long.csv"))
    gran = pd.read_csv(os.path.join(args.data_dir, "gran_divergence.csv"))
    flip = pd.read_csv(os.path.join(args.data_dir, "flip_vs_margin.csv"))
    traj = pd.read_csv(os.path.join(args.data_dir, "margin_traj.csv"))
    runs = pd.read_csv(os.path.join(args.data_dir, "run_summary.csv"))

    build_table(
        args.out_dir,
        "tab1_setup.pdf",
        r"""
\begin{tabular}{@{}p{0.30\linewidth}p{0.62\linewidth}@{}}
\toprule
Item & Setting \\
\midrule
\multicolumn{2}{@{}l}{\textit{Model (decoder-only, trained from scratch)}}\\
Base MoE & 6 layers, width 256, 4 heads, RoPE, RMSNorm, QK-Norm, no QKV bias, tied embeddings \\
Experts & 16 SwiGLU experts (width 256), top-4 routing, no shared expert; $\sim$25M total / $\sim$10M active parameters \\
Dense control & same model, one SwiGLU block of width 1024 (same active size as the MoE) \\
Larger sizes (GPU) & m: width 384, 8 layers ($\sim$68M); l: width 512, 8 layers ($\sim$117M) \\
Expert-count sweep (GPU) & 8 / 16 / 32 / 64 experts with top-2 / 4 / 8 / 16 and expert width 512 / 256 / 128 / 64, so total and active size stay fixed \\
Load balancing & auxiliary loss (coefficient 0.01, softmax router) or loss-free bias (sigmoid router, bias step $10^{-3}$) \\
Capacity & factor 1.25 during training (top-1 choices kept first); no dropping at evaluation \\
\addlinespace
\multicolumn{2}{@{}l}{\textit{Data and training}}\\
Data & FineWeb-Edu (sample-10BT); 16{,}384-token byte-level BPE; 120M training tokens, 2M validation tokens \\
Optimizer & AdamW, peak LR $10^{-3}$, 100 warm-up steps then constant, weight decay 0.1, $\beta=(0.9,0.95)$, clip 1.0 \\
Batch and length & 32 sequences $\times$ 512 tokens per step (fixed for every layout); 1{,}221 steps = 20M tokens \\
Same in every run & initial weights (unless the seed is the variable), data order, global batch, hyperparameters \\
\addlinespace
\multicolumn{2}{@{}l}{\textit{Hardware and numerics}}\\
CPU runs (fp32) & Azure Standard\_E16s\_v3 nodes, one run per node; real multi-process PyTorch + gloo; up to 8 ranks; math libraries pinned to one AVX2 code path \\
GPU runs (fp16) & one NVIDIA T4; up to 4 ranks share the GPU; fp16 autocast with loss scaling; collectives through gloo \\
Parallelism & data parallel (DP) $\in\{1,2,4,8\}$, expert parallel (EP) $\in\{1,2,4\}$, real expert exchange between ranks \\
\bottomrule
\end{tabular}""",
    )

    build_table(
        args.out_dir,
        "tab2_experiments.pdf",
        r"""
\begin{tabular}{@{}lp{0.50\linewidth}rl@{}}
\toprule
Group & What changes (everything else held fixed) & Runs & Where \\
\midrule
Layout & 9 DP/EP layouts: (1,1) reference, (2,1), (2,2), (4,1), (4,2), (4,4), (8,1), (8,2), (8,4) & 9 & CPU fp32 \\
Noise yardsticks & 4 new seeds, 3 new data orders, 3 one-ULP weight nudges & 10 & CPU fp32 \\
Thread count & same layout, different CPU thread count & 1 & CPU fp32 \\
Per-rank definitions & load-balancing statistics, capacity unit, or both, switched from global to per-rank at three layouts & 9 & CPU fp32 \\
Loss-free balancing & bias controller fed by global or per-rank expert counts, two layouts, plus 2 seeds & 7 & CPU fp32 \\
Dense control & 2 seeds, 2 one-ULP nudges, 2 layouts & 7 & CPU fp32 \\
Resume on new layout & reference pretraining run continued from step 128 (10\%) or step 1{,}024 (84\%) on a new layout or with a 1-ULP nudge & 6 & CPU fp32 \\
Low precision & 3 layouts, 2 seeds, 1 one-ULP nudge & 7 & GPU fp16 \\
Model size & sizes m and l: reference, 2 seeds, 1-ULP, 1--2 layouts; base: 1 extra layout & 12 & GPU fp16 \\
Expert count & 8, 32, 64 experts: reference, seed, layout; dense: reference, seed, layout & 12 & GPU fp16 \\
\midrule
Total & & 80 & \\
\bottomrule
\end{tabular}""",
    )

    fin = div[div.final]
    step8 = div[div.step == 8]
    seed_ref = {
        "moe": fin[fin.category == "seed"].kl.mean(),
        "auxfree": fin[fin.category == "auxfree_seed"].kl.mean(),
        "dense": fin[fin.category == "dense_seed"].kl.mean(),
    }
    rows = [
        ("MoE: different random seed", "seed", "moe", False),
        ("MoE: different data order", "data", "moe", True),
        ("MoE: different parallel layout (8)", "layout_numeric", "moe", True),
        ("MoE: one weight nudged by 1 ULP (3)", "ulp", "moe", True),
        ("MoE: different CPU thread count", "thread", "moe", True),
        ("MoE: per-rank definition on (9)", "semantic_vs_samelayout", "moe", True),
        ("MoE, loss-free: new layout (2)", "auxfree_layout_bglobal", "auxfree", True),
        ("MoE, loss-free: per-rank bias (2)", "auxfree_blocal_vs_bglobal", "auxfree", True),
        ("MoE: fp16 GPU vs.\\ fp32 CPU", "precision_fp16_vs_fp32", "moe", True),
        ("Dense: different random seed", "dense_seed", "dense", False),
        ("Dense: different parallel layout (2)", "dense_layout", "dense", False),
        ("Dense: one weight nudged by 1 ULP (2)", "dense_ulp", "dense", False),
    ]
    lines = []
    for lab, cat, ref, show_rf in rows:
        sub = fin[fin.category == cat]
        early = step8[step8.category == cat]
        lines.append(
            " & ".join(
                [
                    lab,
                    fmt_sci(early.kl.median()) if len(early) else "--",
                    fmt_f3(sub.kl.mean()),
                    f"{sub.kl.mean() / seed_ref[ref]:.2f}",
                    fmt_f3(sub.top1_agree.mean()),
                    fmt_pct(sub.route_flip.mean(), 0) if show_rf else "--",
                    fmt_f3(sub.barrier.mean()) if sub.barrier.notna().any() else "--",
                    f"{sub.dce.mean():+.3f}",
                ]
            )
            + r" \\"
        )
        if cat == "precision_fp16_vs_fp32":
            lines.append(r"\midrule")
    build_table(
        args.out_dir,
        "tab3_final_divergence.pdf",
        r"""
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & KL at & KL at & vs.\ own & Top-1 & Routing & Loss & Loss \\
Comparison with the reference run & step 8 & end & seed & agree & differs & barrier & change \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}""",
    )

    refs = {
        8: ("e8", "I_e8_dp1_ep1", 2, 512),
        16: ("e16", "G_dp1_ep1", 4, 256),
        32: ("e32", "I_e32_dp1_ep1", 8, 128),
        64: ("e64", "I_e64_dp1_ep1", 16, 64),
    }
    lines = []
    for n_exp, (gname, ref_id, top_k, width) in refs.items():
        lay = gran[(gran.gran == gname) & (gran.kind == "layout")].set_index("step")
        sd = gran[(gran.gran == gname) & (gran.kind == "seed")].set_index("step")
        near = traj[traj.run == ref_id].sort_values("step").head(3)["near_tie_frac_0.01"].mean()
        last = lay.index.max()
        lines.append(
            " & ".join(
                [
                    f"{n_exp} (top-{top_k}, width {width})",
                    fmt_pct(near),
                    fmt_sci(lay.loc[32, "kl"]),
                    fmt_sci(lay.loc[64, "kl"]),
                    fmt_f3(lay.loc[last, "kl"]),
                    f"{lay.loc[last, 'kl'] / sd.loc[last, 'kl']:.2f}",
                    fmt_pct(lay.loc[last, "route_flip"], 0),
                ]
            )
            + r" \\"
        )
    lay = gran[(gran.gran == "dense") & (gran.kind == "layout")].set_index("step")
    sd = gran[(gran.gran == "dense") & (gran.kind == "seed")].set_index("step")
    last = lay.index.max()
    lines.append(
        " & ".join(
            [
                "dense (width 1024)",
                "--",
                fmt_sci(lay.loc[32, "kl"]),
                fmt_sci(lay.loc[64, "kl"]),
                fmt_f3(lay.loc[last, "kl"]),
                f"{lay.loc[last, 'kl'] / sd.loc[last, 'kl']:.2f}",
                "--",
            ]
        )
        + r" \\"
    )
    build_table(
        args.out_dir,
        "tab4_expert_count.pdf",
        r"""
\begin{tabular}{@{}lrrrrrr@{}}
\toprule
 & Near-tie & \multicolumn{3}{c}{KL after a layout change (DP2/EP2)} & vs.\ own & Routing \\
\cmidrule(lr){3-5}
Experts & choices & step 32 & step 64 & end & seed & differs \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}""",
    )

    pairs = [
        ("A_dp2_ep1", "layout DP2 (fp32)"),
        ("A_dp4_ep1", "layout DP4 (fp32)"),
        ("A_dp8_ep4", "layout DP8/EP4 (fp32)"),
        ("B_ulp1", "1-ULP nudge (fp32)"),
        ("X_dp1_thr8", "thread count (fp32)"),
        ("C_dp8_ep1_cap", "per-rank capacity (fp32)"),
        ("C_dp8_ep1_lbl", "per-rank balancing loss (fp32)"),
        ("D_dp8_ep4_bglobal", "loss-free, layout DP8/EP4 (fp32)"),
        ("H_base_dp2_ep2", "16 experts, DP2/EP2 (fp16)"),
        ("I_e8_dp2_ep2", "8 experts, DP2/EP2 (fp16)"),
        ("I_e32_dp2_ep2", "32 experts, DP2/EP2 (fp16)"),
        ("I_e64_dp2_ep2", "64 experts, DP2/EP2 (fp16)"),
    ]
    lines = []
    for other, lab in pairs:
        r8 = flip[(flip.other == other) & (flip.step == 8)].iloc[0]
        r16 = flip[(flip.other == other) & (flip.step == 16)].iloc[0]
        lines.append(
            " & ".join(
                [
                    lab,
                    f"{r8.auc_flip_has_smaller_margin:.3f}",
                    fmt_pct(r16.flip_rate, 2),
                    f"{r16.auc_flip_has_smaller_margin:.3f}",
                    f"{r16.median_margin_flipped:.4f}",
                    f"{r16.median_margin_kept:.3f}",
                    fmt_pct(r16.flip_rate_lowest10pct_margin, 2),
                    fmt_pct(r16.flip_rate_rest, 3),
                ]
            )
            + r" \\"
        )
    build_table(
        args.out_dir,
        "tab5_mechanism.pdf",
        r"""
\begin{tabular}{@{}lrrrrrrr@{}}
\toprule
 & AUC & \multicolumn{6}{c}{at step 16} \\
\cmidrule(lr){3-8}
 & at & Choices & & \multicolumn{2}{c}{Median margin} & \multicolumn{2}{c}{Flip rate} \\
\cmidrule(lr){5-6}\cmidrule(lr){7-8}
Pair (reference vs.\ changed run) & step 8 & flipped & AUC & flipped & kept & lowest 10\% & other 90\% \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}""",
    )

    groups = [
        ("MoE reference + 8 layouts", runs.run_id.str.match(r"^A_")),
        ("MoE new seeds", runs.run_id.str.match(r"^B_seed")),
        ("MoE new data orders", runs.run_id.str.match(r"^B_data")),
        ("MoE 1-ULP nudges", runs.run_id.str.match(r"^B_ulp")),
        ("MoE per-rank definitions", runs.run_id.str.match(r"^C_")),
        ("MoE loss-free balancing", runs.run_id.str.match(r"^D_")),
        ("Dense control", runs.run_id.str.match(r"^E_")),
        ("Resume on new layout", runs.run_id.str.match(r"^F_")),
        ("fp16 GPU, base size", runs.run_id.str.match(r"^G_") | (runs.run_id == "H_base_dp2_ep2")),
        ("fp16 GPU, size m", runs.run_id.str.match(r"^H_m_")),
        ("fp16 GPU, size l", runs.run_id.str.match(r"^H_l_")),
    ]
    lines = []
    for lab, mask in groups:
        sub = runs[mask]
        mv = sub.mean_maxvio.dropna()
        dr = sub.mean_drop.dropna()
        lines.append(
            " & ".join(
                [
                    lab,
                    str(len(sub)),
                    f"{sub.final_val_ce.mean():.3f}",
                    f"{sub.final_val_ce.min():.3f}--{sub.final_val_ce.max():.3f}",
                    f"{mv.mean():.2f}" if len(mv) else "--",
                    fmt_pct(dr.mean(), 2) if len(dr) else "--",
                ]
            )
            + r" \\"
        )
    build_table(
        args.out_dir,
        "tab6_quality.pdf",
        r"""
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & & \multicolumn{2}{c}{Final validation loss} & Expert & Tokens \\
\cmidrule(lr){3-4}
Run group & Runs & mean & range & imbalance & dropped \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}""",
    )

    sizes = [
        ("base ($\\sim$25M)", "G_dp1_ep1", "scale_base_layout_dp2ep2", "fp16_seed"),
        ("m ($\\sim$68M)", "H_m_dp1_ep1", "scale_m_layout_dp2ep2", "scale_m_seed"),
        ("l ($\\sim$117M)", "H_l_dp1_ep1", "scale_l_layout_dp2ep2", "scale_l_seed"),
    ]
    lines = []
    for lab, ref_id, lay_cat, seed_cat in sizes:
        near = traj[traj.run == ref_id].sort_values("step").head(3)["near_tie_frac_0.01"].mean()
        lay = div[div.category == lay_cat].set_index("step")
        seed_kl = fin[fin.category == seed_cat].kl.mean()
        last = lay.index.max()
        lines.append(
            " & ".join(
                [
                    lab,
                    fmt_pct(near),
                    fmt_sci(lay.loc[32, "kl"]),
                    fmt_f3(lay.loc[last, "kl"]),
                    f"{lay.loc[last, 'kl'] / seed_kl:.2f}",
                    fmt_pct(lay.loc[last, "route_flip"], 0),
                ]
            )
            + r" \\"
        )
    build_table(
        args.out_dir,
        "tab7_model_size.pdf",
        r"""
\begin{tabular}{@{}lrrrrr@{}}
\toprule
 & Near-tie & \multicolumn{2}{c}{KL after layout change} & vs.\ own & Routing \\
\cmidrule(lr){3-4}
Model size (16 experts, fp16) & choices & step 32 & end & seed & differs \\
\midrule
"""
        + "\n".join(lines)
        + r"""
\bottomrule
\end{tabular}""",
    )


if __name__ == "__main__":
    main()
