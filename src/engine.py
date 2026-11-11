"""Data loading, optimizer, gradient synchronization, loss scaling, and checkpoint state."""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

import dist_utils as du
from model import LM, init_full_state, is_expert_param, ulp_perturb_


def save_atomic(obj, path: str) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_json_atomic(obj, path: str) -> None:
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    os.replace(tmp, path)


def read_json(path: str, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


class Data:
    """Deterministic global batch ordering sliced contiguously across data-parallel ranks."""

    def __init__(self, data_dir: str, seq_len: int, global_batch: int, data_seed: int, ctx: du.Ctx):
        self.seq_len = seq_len
        self.global_batch = global_batch
        self.ctx = ctx
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        self.num_seqs = (len(self.train) - 1) // seq_len
        self.perm = np.random.default_rng(int(data_seed)).permutation(self.num_seqs)

    def _chunks(self, arr: np.memmap, ids: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.seq_len
        x = np.stack([arr[i * s:i * s + s + 1] for i in ids]).astype(np.int64)
        toks = torch.from_numpy(x)
        return toks[:, :-1].contiguous(), toks[:, 1:].contiguous()

    def batch(self, step: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_rank = self.global_batch // self.ctx.world
        global_ids = [(step * self.global_batch + j) % self.num_seqs for j in range(self.global_batch)]
        local_ids = global_ids[self.ctx.rank * per_rank:(self.ctx.rank + 1) * per_rank]
        return self._chunks(self.train, [int(self.perm[i]) for i in local_ids])

    def val_batch(self, n_seqs: int) -> tuple[torch.Tensor, torch.Tensor]:
        per_rank = n_seqs // self.ctx.world
        ids = list(range(self.ctx.rank * per_rank, (self.ctx.rank + 1) * per_rank))
        return self._chunks(self.val, ids)


def stage_data_locally(store_data_dir: str, ctx: du.Ctx, local_dir: str | None = None) -> str:
    default_cache = os.path.join(tempfile.gettempdir(), "moe_layout_data")
    local_dir = local_dir or os.environ.get("MOE_LOCAL_DATA", default_cache)
    if ctx.local_rank == 0:
        os.makedirs(local_dir, exist_ok=True)
        for fn in ("train.bin", "val.bin"):
            src = os.path.join(store_data_dir, fn)
            dst = os.path.join(local_dir, fn)
            if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
                shutil.copyfile(src, dst + ".part")
                os.replace(dst + ".part", dst)
    du.barrier(ctx)
    return local_dir


class AdamW:
    """AdamW keyed by parameter name so optimizer state is layout-independent."""

    def __init__(self, model: LM, tc):
        self.tc = tc
        self.params = dict(sorted(model.named_parameters(), key=lambda kv: kv[0]))
        self.m = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.v = {n: torch.zeros_like(p) for n, p in self.params.items()}
        self.t = 0

    @staticmethod
    def decays(name: str, param: torch.Tensor) -> bool:
        return param.dim() >= 2 and not name.startswith("emb.")

    @torch.no_grad()
    def step(self, lr: float) -> None:
        tc = self.tc
        self.t += 1
        bc1 = 1 - tc.beta1 ** self.t
        bc2 = 1 - tc.beta2 ** self.t
        for name, param in self.params.items():
            grad = param.grad
            if self.decays(name, param):
                param.mul_(1 - lr * tc.wd)
            m = self.m[name]
            v = self.v[name]
            m.mul_(tc.beta1).add_(grad, alpha=1 - tc.beta1)
            v.mul_(tc.beta2).addcmul_(grad, grad, value=1 - tc.beta2)
            param.addcdiv_(m / bc1, (v / bc2).sqrt_().add_(tc.eps), value=-lr)


def lr_at(step: int, tc) -> float:
    return tc.lr * min(1.0, (step + 1) / tc.warmup)


def autocast_ctx(spec, device: str):
    if spec.precision == "fp16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def _split_params(model: LM) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    shared, expert = [], []
    for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        (expert if is_expert_param(name) else shared).append(param)
    return shared, expert


def _flat_sync(params: list[torch.Tensor], group, size: int, world: int, scale: float):
    if not params:
        return torch.zeros(0), torch.zeros(())
    flat = torch.cat([p.grad.reshape(-1).float() for p in params])
    du.ar_(flat, group, size)
    flat.div_(world * scale)
    offset = 0
    for param in params:
        numel = param.numel()
        param.grad.copy_(flat[offset:offset + numel].view_as(param.grad))
        offset += numel
    return flat, flat.double().pow(2).sum()


@torch.no_grad()
def sync_grads(model: LM, ctx: du.Ctx, scale: float = 1.0, clip: float | None = None) -> tuple[float, bool]:
    shared, expert = _split_params(model)
    _, sq_shared = _flat_sync(shared, ctx.world_group, ctx.world, ctx.world, scale)
    _, sq_expert = _flat_sync(expert, ctx.edp_group, ctx.edp, ctx.world, scale)
    sq_expert = sq_expert.reshape(1).to(torch.float64)
    if ctx.ep > 1:
        sq_expert = du.ar_(sq_expert, ctx.ep_group, ctx.ep)
    total_norm = float(np.sqrt(float(sq_shared) + float(sq_expert.sum())))
    finite = bool(np.isfinite(total_norm))
    if finite and clip:
        coef = min(1.0, clip / (total_norm + 1e-6))
        if coef < 1.0:
            for param in shared + expert:
                param.grad.mul_(coef)
    return total_norm, finite


@torch.no_grad()
def update_route_bias(model: LM, spec, ctx: du.Ctx, rate: float) -> None:
    if spec.balance != "auxfree":
        return
    for layer in model.moe_layers():
        counts = layer.last_counts.to(torch.float32).clone()
        if spec.bias_scope == "global":
            du.ar_(counts, ctx.world_group, ctx.world)
        layer.route_bias.add_(rate * torch.sign(counts.mean() - counts))


def zero_grads(model: LM) -> None:
    for param in model.parameters():
        param.grad = None


def train_step(model: LM, opt: AdamW, scaler: dict, x: torch.Tensor, y: torch.Tensor, spec, tc, ctx: du.Ctx, lr: float) -> dict:
    model.train()
    vocab = model.cfg.vocab
    with autocast_ctx(spec, ctx.device.type):
        logits, aux = model(x)
    ce = F.cross_entropy(logits.float().reshape(-1, vocab), y.reshape(-1))
    loss = ce + (tc.lbl_coef * aux if (spec.balance == "aux" and model.cfg.moe) else 0.0)
    scale = scaler["scale"]
    (loss * scale).backward()
    gnorm, finite = sync_grads(model, ctx, scale=scale, clip=tc.clip)
    if finite:
        opt.step(lr)
    if spec.precision == "fp16":
        if not finite:
            scaler["scale"] = max(1.0, scale / 2)
            scaler["good"] = 0
        else:
            scaler["good"] += 1
            if scaler["good"] % 1000 == 0:
                scaler["scale"] = min(2.0 ** 24, scale * 2)
    update_route_bias(model, spec, ctx, tc.bias_rate)
    zero_grads(model)

    with torch.no_grad():
        moes = model.moe_layers()
        parts = [ce.detach().float().reshape(1).cpu(), aux.detach().float().reshape(1).cpu()]
        if moes:
            drops = torch.stack([m.last_drop.float().cpu() for m in moes]).sum().reshape(1)
            counts = torch.stack([m.last_counts.float().cpu() for m in moes])
            parts.extend([drops, counts.reshape(-1)])
        packed = torch.cat(parts).contiguous()
        du.ar_(packed, ctx.world_group, ctx.world)
        stats = dict(
            ce=float(packed[0]) / ctx.world,
            aux=float(packed[1]) / ctx.world,
            gnorm=gnorm,
            finite=finite,
            scale=scale,
            lr=lr,
        )
        if moes:
            n_layers, n_experts = len(moes), moes[0].E
            cnt = packed[3:].view(n_layers, n_experts)
            stats["drop_frac"] = float(packed[2]) / float(cnt.sum())
            stats["maxvio"] = float((cnt.max(-1).values / cnt.mean(-1) - 1).mean())
    return stats


@torch.no_grad()
def evaluate(model: LM, data: Data, spec, ctx: du.Ctx, n_seqs: int) -> float:
    model.eval()
    x, y = data.val_batch(n_seqs)
    x, y = x.to(ctx.device), y.to(ctx.device)
    with autocast_ctx(spec, ctx.device.type):
        logits, _ = model(x)
    ce = F.cross_entropy(logits.float().reshape(-1, model.cfg.vocab), y.reshape(-1))
    val = ce.detach().float().reshape(1).cpu().contiguous()
    du.ar_(val, ctx.world_group, ctx.world)
    model.train()
    return float(val[0]) / ctx.world


def build(spec, mcfg, ctx: du.Ctx) -> LM:
    return LM(mcfg, spec, ctx).to(ctx.device)


def initial_full_state(spec, mcfg) -> dict:
    full = {"model": init_full_state(mcfg, spec, spec.init_seed)}
    full["ulp"] = ulp_perturb_(full["model"], spec.ulp_perturb)
    return full


@torch.no_grad()
def gather_full_state(model: LM, opt: AdamW | None, ctx: du.Ctx, with_opt: bool = True, extra: dict | None = None):
    sd = {n: t.detach().to("cpu").clone() for n, t in model.state_dict().items()}
    bias = {n: t for n, t in sd.items() if n.endswith("route_bias")}
    exp = {n: t for n, t in sd.items() if is_expert_param(n)}
    shared = {n: t for n, t in sd.items() if n not in exp and n not in bias}
    send_exp = ctx.ep_group_id == 0
    payload = {"exp": exp if send_exp else {}, "bias": bias}
    if with_opt and opt is not None:
        payload["m"] = {n: opt.m[n].to("cpu").clone() for n in exp} if send_exp else {}
        payload["v"] = {n: opt.v[n].to("cpu").clone() for n in exp} if send_exp else {}
    parts = du.gather_objects_to0(payload, ctx)
    if ctx.rank != 0:
        return None
    model_full = dict(shared)
    for part in parts:
        model_full.update(part["exp"])
    bias_per_rank = [part["bias"] for part in parts]
    for name in bias:
        model_full[name] = torch.stack([b[name] for b in bias_per_rank]).mean(0)
    full = {"model": model_full, "bias_per_rank": bias_per_rank, "world": ctx.world, "ep": ctx.ep}
    if with_opt and opt is not None:
        m = {n: opt.m[n].to("cpu").clone() for n in opt.m if not is_expert_param(n)}
        v = {n: opt.v[n].to("cpu").clone() for n in opt.v if not is_expert_param(n)}
        for part in parts:
            m.update(part["m"])
            v.update(part["v"])
        full["opt"] = {"m": m, "v": v, "t": opt.t}
    if extra:
        full.update(extra)
    return full


@torch.no_grad()
def apply_full_state(model: LM, opt: AdamW | None, full: dict, ctx: du.Ctx) -> None:
    msd = model.state_dict()
    same_layout = full.get("world") == ctx.world and full.get("ep") == ctx.ep
    loaded = {}
    for name in msd:
        if name.endswith("route_bias") and same_layout and full.get("bias_per_rank"):
            loaded[name] = full["bias_per_rank"][ctx.rank][name]
        else:
            loaded[name] = full["model"][name]
    model.load_state_dict(loaded, strict=True)
    if opt is not None and "opt" in full:
        for name in opt.m:
            opt.m[name].copy_(full["opt"]["m"][name])
            opt.v[name].copy_(full["opt"]["v"][name])
        opt.t = int(full["opt"]["t"])


def set_threads(spec, ctx: du.Ctx) -> int:
    if ctx.device.type == "cpu":
        n = (
            spec.threads_per_rank
            or int(os.environ.get("MOE_THREADS", "0"))
            or max(1, (os.cpu_count() or 1) // max(1, ctx.world))
        )
    else:
        n = int(os.environ.get("MOE_THREADS", "1"))
    torch.set_num_threads(max(1, n))
    return n


def determinism_setup(device: str):
    import warnings

    warnings.filterwarnings("ignore", message=".*deterministic.*")
    torch.use_deterministic_algorithms(True, warn_only=True)
    if device == "cuda":
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel

            return sdpa_kernel([SDPBackend.MATH])
        except Exception:
            return torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_mem_efficient=False,
                enable_math=True,
            )
    return contextlib.nullcontext()


def device_for(spec, ctx_local_rank: int = 0) -> str:
    if spec.device == "cuda":
        torch.cuda.set_device(0)
        return "cuda"
    return "cpu"
