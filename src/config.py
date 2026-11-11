"""Model hyperparameters, run specifications, queue sharding, and comparison pairs."""
from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, replace

PROJECT = "moe_layout_v1"


@dataclass(frozen=True)
class ModelCfg:
    vocab: int = 16384
    d: int = 256
    n_layers: int = 6
    n_heads: int = 4
    seq_len: int = 512
    moe: bool = True
    n_experts: int = 16
    top_k: int = 4
    d_expert: int = 256
    qk_norm: bool = True
    rope_theta: float = 10000.0


@dataclass(frozen=True)
class TrainCfg:
    global_batch: int = 32
    lr: float = 1e-3
    warmup: int = 100
    wd: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    clip: float = 1.0
    lbl_coef: float = 0.01
    bias_rate: float = 1e-3
    val_every: int = 100
    val_seqs: int = 32
    ckpt_minutes: float = 15.0
    log_every: int = 25
    fp16_init_scale: float = 2.0 ** 15


TRAIN = TrainCfg()
MODEL = ModelCfg()
TOKENS_PER_STEP = TRAIN.global_batch * MODEL.seq_len
DEFAULT_TARGET_TOKENS = 20_000_000

SNAP_STEPS = [8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072, 4096]
ANCHOR_STEPS = [128, 1024]

SIZE_D = {"base": None, "m": (384, 8), "l": (512, 8)}
GRAN_CFG = {"e8": (8, 2, 512), "e32": (32, 8, 128), "e64": (64, 16, 64)}


def steps_for_tokens(tokens: int) -> int:
    return int(math.ceil(int(tokens) / TOKENS_PER_STEP))


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    arm: str
    device: str = "cpu"
    precision: str = "fp32"
    dp: int = 1
    ep: int = 1
    init_seed: int = 0
    data_seed: int = 0
    ulp_perturb: int = 0
    moe: bool = True
    balance: str = "aux"
    lbl_scope: str = "global"
    cap_scope: str = "global"
    bias_scope: str = "global"
    capacity_factor: float = 1.25
    threads_per_rank: int = 0
    parent: str = ""
    branch_step: int = 0
    target_tokens_override: int = 0
    priority: int = 50
    size: str = "base"

    def to_dict(self):
        return asdict(self)


def _gran_of(spec: RunSpec) -> str:
    rid = getattr(spec, "run_id", "")
    return rid.split("_")[1] if rid.startswith("I_e") else ""


def model_cfg_for(spec: RunSpec) -> ModelCfg:
    sz = SIZE_D.get(getattr(spec, "size", "base"))
    if not sz:
        cfg = replace(MODEL, moe=spec.moe)
    else:
        d, n_layers = sz
        cfg = replace(
            MODEL,
            moe=spec.moe,
            d=d,
            n_layers=n_layers,
            n_heads=d // 64,
            d_expert=MODEL.d_expert * d // MODEL.d,
        )
    gran = GRAN_CFG.get(_gran_of(spec))
    if gran:
        cfg = replace(cfg, n_experts=gran[0], top_k=gran[1], d_expert=gran[2])
    return cfg


def target_tokens_for(spec: RunSpec, default_tokens: int) -> int:
    return spec.target_tokens_override or int(default_tokens)


LAYOUTS = [(1, 1), (2, 1), (2, 2), (4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4)]
REF = "A_dp1_ep1"


def _build_runs() -> OrderedDict[str, RunSpec]:
    runs: OrderedDict[str, RunSpec] = OrderedDict()

    def add(spec: RunSpec):
        assert spec.run_id not in runs, spec.run_id
        cfg = model_cfg_for(spec)
        assert spec.dp % spec.ep == 0
        assert not cfg.moe or cfg.n_experts % spec.ep == 0
        assert TRAIN.global_batch % spec.dp == 0 and TRAIN.val_seqs % spec.dp == 0
        runs[spec.run_id] = spec

    for dp, ep in LAYOUTS:
        prio = 0 if (dp, ep) == (1, 1) else 20
        add(RunSpec(f"A_dp{dp}_ep{ep}", "A_numeric", dp=dp, ep=ep, priority=prio))

    for s in (1, 2, 3, 4):
        add(RunSpec(f"B_seed{s}", "B_baseline", init_seed=s, priority=10))
    for s in (1, 2, 3):
        add(RunSpec(f"B_data{s}", "B_baseline", data_seed=s, priority=15))
    for s in (1, 2, 3):
        add(RunSpec(f"B_ulp{s}", "B_baseline", ulp_perturb=s, priority=12))

    add(RunSpec("X_dp1_thr8", "X_thread", threads_per_rank=8, priority=40))

    for dp, ep in ((2, 1), (8, 1), (8, 4)):
        add(RunSpec(f"C_dp{dp}_ep{ep}_lbl", "C_semantic", dp=dp, ep=ep, lbl_scope="local", priority=25))
        add(RunSpec(f"C_dp{dp}_ep{ep}_cap", "C_semantic", dp=dp, ep=ep, cap_scope="local", priority=25))
        add(RunSpec(f"C_dp{dp}_ep{ep}_both", "C_semantic", dp=dp, ep=ep, lbl_scope="local", cap_scope="local", priority=30))

    add(RunSpec("D_dp1_ep1", "D_auxfree", balance="auxfree", priority=5))
    for s in (1, 2):
        add(RunSpec(f"D_seed{s}", "D_auxfree", balance="auxfree", init_seed=s, priority=18))
    for dp, ep in ((8, 1), (8, 4)):
        for bs in ("global", "local"):
            add(RunSpec(f"D_dp{dp}_ep{ep}_b{bs}", "D_auxfree", dp=dp, ep=ep, balance="auxfree", bias_scope=bs, priority=28))

    add(RunSpec("E_dp1", "E_dense", moe=False, priority=6))
    for s in (1, 2):
        add(RunSpec(f"E_seed{s}", "E_dense", moe=False, init_seed=s, priority=22))
    for s in (1, 2):
        add(RunSpec(f"E_ulp{s}", "E_dense", moe=False, ulp_perturb=s, priority=22))
    for dp in (2, 8):
        add(RunSpec(f"E_dp{dp}", "E_dense", moe=False, dp=dp, priority=26))

    for b in ANCHOR_STEPS:
        for dp, ep in ((8, 4), (4, 1)):
            add(RunSpec(f"F_b{b}_dp{dp}_ep{ep}", "F_branch", dp=dp, ep=ep, parent=REF, branch_step=b, priority=60))
        add(RunSpec(f"F_b{b}_ulp1", "F_branch", ulp_perturb=1, parent=REF, branch_step=b, priority=60))

    gpu = dict(device="cuda", precision="fp16", arm="G_fp16")
    add(RunSpec("G_dp1_ep1", priority=0, **gpu))
    for dp, ep in ((2, 1), (4, 1), (4, 2)):
        add(RunSpec(f"G_dp{dp}_ep{ep}", dp=dp, ep=ep, priority=20, **gpu))
    for s in (1, 2):
        add(RunSpec(f"G_seed{s}", init_seed=s, priority=10, **gpu))
    add(RunSpec("G_ulp1", ulp_perturb=1, priority=12, **gpu))

    for sz in ("m", "l"):
        hg = dict(device="cuda", precision="fp16", arm="H_scale", size=sz)
        add(RunSpec(f"H_{sz}_dp1_ep1", priority=0, **hg))
        add(RunSpec(f"H_{sz}_seed1", init_seed=1, priority=5, **hg))
        add(RunSpec(f"H_{sz}_seed2", init_seed=2, priority=6, **hg))
        add(RunSpec(f"H_{sz}_ulp1", ulp_perturb=1, priority=8, **hg))
        if sz != "l":
            add(RunSpec(f"H_{sz}_dp4_ep2", dp=4, ep=2, priority=10, **hg))
        add(RunSpec(f"H_{sz}_dp2_ep2", dp=2, ep=2, priority=12, **hg))
    add(RunSpec("H_base_dp2_ep2", "H_scale", device="cuda", precision="fp16", dp=2, ep=2, priority=12))

    add(RunSpec("S_cpu_dp1", "S_smoke", target_tokens_override=1_000_000))
    add(RunSpec("S_cpu_dp8_ep4", "S_smoke", dp=8, ep=4, target_tokens_override=1_000_000))
    add(RunSpec("S_gpu_dp1", "S_smoke", device="cuda", precision="fp16", target_tokens_override=1_000_000))
    add(RunSpec("S_gpu_dp4_ep2", "S_smoke", device="cuda", precision="fp16", dp=4, ep=2, target_tokens_override=1_000_000))

    gran_gpu = dict(device="cuda", precision="fp16", arm="I_gran")
    for g, prio in (("e8", 0), ("e32", 3), ("e64", 9)):
        add(RunSpec(f"I_{g}_dp1_ep1", priority=prio, **gran_gpu))
        add(RunSpec(f"I_{g}_seed1", init_seed=1, priority=prio + 1, **gran_gpu))
        add(RunSpec(f"I_{g}_dp2_ep2", dp=2, ep=2, priority=prio + 2, **gran_gpu))
    add(RunSpec("I_dense_dp1", moe=False, priority=6, **gran_gpu))
    add(RunSpec("I_dense_seed1", moe=False, init_seed=1, priority=7, **gran_gpu))
    add(RunSpec("I_dense_dp2", moe=False, dp=2, priority=8, **gran_gpu))

    return runs


RUNS = _build_runs()
ANCHOR_PARENTS = sorted({s.parent for s in RUNS.values() if s.parent})

CPU_ARMS = ["A_numeric", "B_baseline", "X_thread", "C_semantic", "D_auxfree", "E_dense", "F_branch"]
GPU_ARMS = ["G_fp16", "H_scale", "I_gran"]


def _cost(spec: RunSpec) -> float:
    c = 1.0 + (0.15 if spec.dp > 1 else 0.0) + (0.25 if spec.ep > 1 else 0.0)
    if spec.threads_per_rank:
        c *= 2.0
    if spec.parent:
        c *= 0.8
    return c


def make_queue(arms, shard: int, num_shards: int) -> list[str]:
    """Partition runs across shards while keeping branch runs ordered after their parent."""
    runs = [s for s in RUNS.values() if s.arm in arms]
    ids = {s.run_id for s in runs}
    groups: OrderedDict[str, list[RunSpec]] = OrderedDict()
    for s in sorted(runs, key=lambda x: (x.priority, x.run_id)):
        key = s.parent if (s.parent and s.parent in ids) else s.run_id
        groups.setdefault(key, []).append(s)
    for key, grp in groups.items():
        grp.sort(key=lambda x: (0 if x.run_id == key else 1, x.priority, x.run_id))

    loads = [0.0] * num_shards
    assign: list[list[RunSpec]] = [[] for _ in range(num_shards)]
    for _, grp in sorted(groups.items(), key=lambda kv: -sum(_cost(x) for x in kv[1])):
        idx = min(range(num_shards), key=lambda j: loads[j])
        assign[idx].extend(grp)
        loads[idx] += sum(_cost(x) for x in grp)

    shard_specs = assign[shard]
    shard_ids = {x.run_id for x in shard_specs}
    pending = sorted(shard_specs, key=lambda x: (x.priority, x.run_id))
    ordered: list[str] = []
    placed: set[str] = set()
    while pending:
        progressed = False
        for s in list(pending):
            if not s.parent or s.parent in placed or s.parent not in shard_ids:
                ordered.append(s.run_id)
                placed.add(s.run_id)
                pending.remove(s)
                progressed = True
        if not progressed:
            ordered.extend(x.run_id for x in pending)
            break
    return ordered


def _comparisons() -> list[dict]:
    pairs: list[dict] = []

    def add(cat: str, ref: str, other: str, min_step: int = 0):
        if ref in RUNS and other in RUNS:
            pairs.append(
                dict(
                    name=f"{cat}:{other}_vs_{ref}",
                    category=cat,
                    ref=ref,
                    other=other,
                    min_step=min_step,
                )
            )

    for s in (1, 2, 3, 4):
        add("seed", REF, f"B_seed{s}")
    for s in (1, 2, 3):
        add("data", REF, f"B_data{s}")
        add("ulp", REF, f"B_ulp{s}")
    add("thread", REF, "X_dp1_thr8")

    for dp, ep in LAYOUTS:
        if (dp, ep) != (1, 1):
            add("layout_numeric", REF, f"A_dp{dp}_ep{ep}")

    for dp, ep in ((2, 1), (8, 1), (8, 4)):
        for variant in ("lbl", "cap", "both"):
            add("semantic_vs_samelayout", f"A_dp{dp}_ep{ep}", f"C_dp{dp}_ep{ep}_{variant}")
            add("semantic_total", REF, f"C_dp{dp}_ep{ep}_{variant}")

    for s in (1, 2):
        add("auxfree_seed", "D_dp1_ep1", f"D_seed{s}")
    for dp, ep in ((8, 1), (8, 4)):
        add("auxfree_layout_bglobal", "D_dp1_ep1", f"D_dp{dp}_ep{ep}_bglobal")
        add("auxfree_layout_blocal", "D_dp1_ep1", f"D_dp{dp}_ep{ep}_blocal")
        add("auxfree_blocal_vs_bglobal", f"D_dp{dp}_ep{ep}_bglobal", f"D_dp{dp}_ep{ep}_blocal")

    for s in (1, 2):
        add("dense_seed", "E_dp1", f"E_seed{s}")
        add("dense_ulp", "E_dp1", f"E_ulp{s}")
    for dp in (2, 8):
        add("dense_layout", "E_dp1", f"E_dp{dp}")

    for b in ANCHOR_STEPS:
        for dp, ep in ((8, 4), (4, 1)):
            add("branch_layout", REF, f"F_b{b}_dp{dp}_ep{ep}", min_step=b)
        add("branch_ulp", REF, f"F_b{b}_ulp1", min_step=b)

    for s in (1, 2):
        add("fp16_seed", "G_dp1_ep1", f"G_seed{s}")
    add("fp16_ulp", "G_dp1_ep1", "G_ulp1")
    for dp, ep in ((2, 1), (4, 1), (4, 2)):
        add("fp16_layout", "G_dp1_ep1", f"G_dp{dp}_ep{ep}")
    add("precision_fp16_vs_fp32", REF, "G_dp1_ep1")

    for sz in ("m", "l"):
        ref_id = f"H_{sz}_dp1_ep1"
        for s in (1, 2):
            add(f"scale_{sz}_seed", ref_id, f"H_{sz}_seed{s}")
        add(f"scale_{sz}_ulp", ref_id, f"H_{sz}_ulp1")
        add(f"scale_{sz}_layout", ref_id, f"H_{sz}_dp4_ep2")
        add(f"scale_{sz}_layout_dp2ep2", ref_id, f"H_{sz}_dp2_ep2")
    add("scale_base_layout_dp2ep2", "G_dp1_ep1", "H_base_dp2_ep2")

    return pairs


COMPARISONS = _comparisons()
BARRIER_CATEGORIES = {
    "seed",
    "ulp",
    "layout_numeric",
    "semantic_vs_samelayout",
    "dense_seed",
    "dense_ulp",
    "branch_layout",
    "branch_ulp",
    "auxfree_blocal_vs_bglobal",
    "fp16_layout",
    "fp16_seed",
}


if __name__ == "__main__":
    from collections import Counter

    counts = Counter(s.arm for s in RUNS.values())
    print("runs per arm:", dict(counts), "total:", len(RUNS))
    print("comparisons:", len(COMPARISONS))
    for idx in range(6):
        print(f"cpu shard {idx}:", make_queue(CPU_ARMS, idx, 6))
    for arm in GPU_ARMS:
        print(f"gpu {arm}:", make_queue([arm], 0, 1))
