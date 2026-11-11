# In MoE Pretraining, Parallelism Layout Changes the Model, Not Its Quality

Code and results for a controlled study of how the **parallelism layout** used in pretraining (data-parallel and expert-parallel sizes) affects a **Mixture-of-Experts (MoE)** model.

> **TL;DR:** Train the same MoE with the same seed, data and batch, changing only the layout. Loss stays the same, but the trained model doesn't: its predictions end about **80% as different as a new random seed** would make them, and over 90% of routing decisions differ. The cause is **router near-ties**: routing decisions where two experts score almost the same flip under the smallest rounding difference, and the runs then drift apart. Finer-grained experts make routing much less reproducible, but they don't make the outputs drift further. A dense model of the same active size is far more stable.

---

## The question

When an MoE is trained on many devices, the work is split by **data parallelism (DP)** and **expert parallelism (EP)**. Jobs are often resumed on a different number of GPUs, clusters get resized, and ablations run on different setups. In exact math this shouldn't matter. On real hardware, though, a new layout:

1. **changes the order in which numbers are added** (a *numerical* effect), and
2. may change rules that are computed **per device**, such as load-balancing statistics, the expert-capacity limit and loss-free bias counts (a *definitional* effect).

**Does this change the trained model? If so, why, and does it get worse with more experts?**

## Key findings

| # | Finding | Evidence |
|---|---|---|
| 1 | **Same quality, different model.** Every layout reaches the same validation loss (within seed noise), but the final predictions differ by **0.80× the seed level**, and **92%** of routing decisions differ. All 8 layouts, a 1-ULP change to a single weight, and a different CPU thread count are indistinguishable. | fp32 CPU, 8 layouts, 4 seeds, 3 ULP runs |
| 2 | **Per-device rules start large but end the same.** Computing load balancing or capacity per device makes the runs differ **~400×** more at step 8, but they converge to the same final difference. | 9 definitional runs |
| 3 | **The cause is router near-ties.** Flipped routing decisions are almost exactly the ones with the smallest score margin (**AUC 0.97–1.00** in the first 16 steps). The 10% of decisions with the smallest margins flip **65–130×** more often. | token-level flip analysis |
| 4 | **More experts give less reproducible routing but the same output drift.** With total and active size fixed, going from 8 to 64 experts raises near-ties from **2.7% to 23.3%** and routing differences from **68% to 100%**. The output difference stays at **~0.8× seed** throughout. | 8/16/32/64-expert sweep (fp16) |
| 5 | **MoE vs. dense, and precision.** A dense model of the same active size ends at **0.12× seed** in fp32 and **0.58×** in fp16. The MoE ends at **~0.8×** in both, and reaches the 0.01-KL level at step 64 vs. step 128 for dense. | dense control, CPU + GPU |
| 6 | **When a pretraining run switches layout matters.** Resuming on a new layout at **84% of pretraining** (step 1,024 of 1,221) barely changes the final model (KL 0.007). Resuming at **10%** (step 128, just after warm-up) changes it substantially (KL 0.12–0.13). | resume-from-checkpoint runs |

**Practical takeaways**
- Don't expect the same MoE back from a different layout. For exact reproducibility, keep the layout fixed **and** use bitwise-deterministic training. Higher precision alone helps dense models, not MoEs.
- Compare methods on the same layout, with several seeds.
- If a pretraining job must change cluster size, do it after the loss has flattened, not right after warm-up.
- Log the router: the share of near-ties and the routing flip rate show how reproducible a run is, and loss doesn't.

**Scope:** every result is from **pretraining from scratch** (25M–117M parameters, 20M tokens, constant learning rate after warm-up). Mid-training, learning-rate decay and post-training (fine-tuning, RL) are not tested.

---

## How "different" is measured

Two runs are compared at the same step on the same 16,384 held-out tokens:

| Metric | Meaning |
|---|---|
| **KL divergence** | How different the next-token predictions are (0 = identical) |
| **vs. own seed** | KL divided by the KL between two runs with different random seeds (1.0 = as different as a new seed) |
| **Top-1 agreement** | How often both runs predict the same top token |
| **Routing differs** | Share of (token, layer) decisions where the chosen expert set differs |
| **Loss barrier** | Loss bump along the straight line between the two weight sets (≈0 means same "valley") |
| **Router margin** | Gap between the k-th and (k+1)-th expert score, divided by that token's score spread. Below 0.01 counts as a near-tie |

---

## Setup

- **Model:** Qwen3-MoE-style decoder trained from scratch: QK-Norm, no QKV bias, RoPE, RMSNorm, tied embeddings, 16 SwiGLU experts, top-4 routing, no shared expert. ~25M total / ~10M active parameters. Larger sizes are ~68M and ~117M.
- **Dense control:** same model with one SwiGLU block of matching active width.
- **Data:** FineWeb-Edu (sample-10BT), 16k BPE vocabulary, 32 × 512 tokens per step, 1,221 steps (20M tokens).
- **Optimizer:** AdamW, LR 1e-3, 100-step warm-up then constant, weight decay 0.1.
- **Parallelism:** real multi-process PyTorch (gloo) with DP ∈ {1, 2, 4, 8} and EP ∈ {1, 2, 4}, and real expert exchange between processes.
- **Hardware:** fp32 on Azure CPU nodes (math libraries pinned to one AVX2 path), and fp16 on a single NVIDIA T4 shared by up to 4 processes.
- **80 training runs** in total. Each one changes exactly one thing relative to its reference.

| Group | What changes | Runs | Where |
|---|---|---|---|
| A: Layout | 9 DP/EP layouts | 9 | CPU fp32 |
| B: Noise yardsticks | new seed / data order / 1-ULP nudge | 10 | CPU fp32 |
| X: Threads | CPU thread count | 1 | CPU fp32 |
| C: Per-device rules | load-balancing stats, capacity, or both | 9 | CPU fp32 |
| D: Loss-free balancing | global vs. per-device bias counts | 7 | CPU fp32 |
| E: Dense control | seeds, 1-ULP, layouts | 7 | CPU fp32 |
| F: Resume | switch layout at step 128 or 1,024 | 6 | CPU fp32 |
| G: Low precision | layouts, seeds, 1-ULP | 7 | GPU fp16 |
| H: Model size | sizes m and l | 12 | GPU fp16 |
| I: Expert count | 8 / 32 / 64 experts + dense | 12 | GPU fp16 |

---

## Repository structure

```
moe_layout/
├── config.py            # model/training config, full run matrix, comparisons, queue sharding
├── dist_utils.py        # gloo process groups; exact all-gather via all-reduce; EP gather/combine
├── model.py             # MoE LM with real expert parallelism and global/per-rank switches
├── engine.py            # data order, AdamW, grad sync, fp16 loss scaling, bias controller, checkpoints
├── train.py             # trains one run (torchrun); 15-min checkpoints, auto-resume
├── job_driver.py        # in-job queue runner: skips finished runs, resumes the rest
├── selftest.py          # correctness gate (gradient match, determinism, checkpoint round trip)
├── prep_data.py         # FineWeb-Edu -> 16k BPE -> train.bin / val.bin
├── analysis.py          # divergence metrics, noise bands, figures, verdict
├── ext_analysis.py      # router-margin mechanism + expert-count analysis
├── orchestrator.ipynb   # Azure ML notebook that submits jobs only
└── environment/conda.yaml


### Correctness guarantees
Before any training, `selftest.py` checks that:
- every layout gives the **same gradients as a single process** (relative error < 1e-4 in fp32);
- two runs of the same layout are **bit-identical**;
- checkpoints **move between layouts** with no change.

Expert-parallel gathers are built from sum-reductions into zero-filled buffers, which adds no rounding. Every per-device rule has a "global" version that's identical in every layout, so numerical-only runs change nothing but the addition order.

---

## Running the code

### Local, CPU only
```bash
conda env create -f moe_layout/environment/conda.yaml && conda activate moe-layout
cd moe_layout
OUT=./out

# 1) data (one time)
python prep_data.py --out_root $OUT

# 2) correctness gate
torchrun --standalone --nproc_per_node=4 selftest.py --out_root $OUT --device cpu

# 3) train a few runs (finished runs are skipped; interrupted runs resume)
python job_driver.py --out_root $OUT --device cpu --runs A_dp1_ep1,A_dp8_ep4,B_seed1,B_ulp1 --job_tag local

# 4) analysis
python analysis.py --out_root $OUT
python ext_analysis.py --out_root $OUT --device cpu
```

### Azure ML
Open `orchestrator.ipynb` and run the cells in order. The notebook only submits jobs. Queueing, retries, 15-minute checkpoints and resume all run **inside** each job, so the notebook's compute instance can be stopped after submitting. GPU jobs manage their own queue on the GPU node.

> Keep credentials (Azure keys, Hugging Face tokens) in environment variables or Key Vault, never in the notebook or the repository.

### Reproducing the paper figures and tables
```bash
cd paper
python figures/generate_figures.py   # charts  -> figures/fig*.pdf
python figures/make_tables.py        # tables  -> figures/tab*.pdf (needs pdflatex + PyMuPDF)
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

---

## Limitations

- Small models (25M–117M parameters) trained on 20M tokens, so exact ratios will shift at production scale.
- Communication uses gloo on one node or one shared T4; NCCL, tensor parallelism and pipeline parallelism aren't tested, and neither are bf16/FP8.
- One seed per setting in the expert-count and model-size sweeps.
- Pretraining only, with a constant learning rate after warm-up and two resume points (10% and 84%).
- No quality cost from per-device rules at this scale. Large-scale results show global load balancing can improve quality, so this shouldn't be read as saying per-device rules are harmless in general.

---

