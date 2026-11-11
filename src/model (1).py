"""Decoder-only LM with a top-k MoE FFN that supports real expert parallelism.

Layout-coupled ("semantic") definitions, each switchable between a layout-invariant
"global" form and the common per-rank "local" form:
  * lbl_scope : expert-frequency statistic f_i of the aux load-balancing loss
  * cap_scope : unit over which expert capacity / token dropping is decided
                (global = whole global batch in DP=1 token order -> layout invariant;
                 local  = the dispatch unit = EP group (= the rank itself when ep == 1))
  * bias_scope: load statistics used by the aux-loss-free bias controller (engine.py)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dist_utils as du


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.type_as(x) * self.weight


def rope_cache(seq, hd, theta, device):
    inv = 1.0 / (theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd))
    t = torch.arange(seq, device=device, dtype=torch.float32)
    f = torch.outer(t, inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    xf = x.float()
    x1, x2 = xf[..., ::2], xf[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return out.type_as(x)


class Attention(nn.Module):
    def __init__(self, d, n_heads, qk_norm=True):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.qn = RMSNorm(d // n_heads)
            self.kn = RMSNorm(d // n_heads)

    def forward(self, x, cos, sin):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q, k = self.qn(q).to(v.dtype), self.kn(k).to(v.dtype)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, S, D))


class SwiGLU(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.w1 = nn.Linear(d, h, bias=False)
        self.w3 = nn.Linear(d, h, bias=False)
        self.w2 = nn.Linear(h, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def capacity_keep(idx_g: torch.Tensor, n_experts: int, cf: float) -> torch.Tensor:
    """Switch/GShard priority: all top-1 choices first, then top-2, in token order.
    idx_g: [N, k] expert ids for every token of the capacity unit. Returns bool [N, k]."""
    N, k = idx_g.shape
    cap = int(math.ceil(cf * N * k / n_experts))
    flat = idx_g.t().reshape(-1)                                   # slot-major order
    oh = F.one_hot(flat, n_experts).to(torch.int64)
    pos = (oh.cumsum(0) * oh).sum(-1) - 1
    return (pos < cap).view(k, N).t().contiguous()


class MoE(nn.Module):
    def __init__(self, cfg, spec, ctx: du.Ctx):
        super().__init__()
        self.E, self.k = cfg.n_experts, cfg.top_k
        self.spec, self.ctx = spec, ctx
        self.router = nn.Linear(cfg.d, self.E, bias=False)
        per = self.E // ctx.ep
        self.local_ids = list(range(ctx.ep_rank * per, (ctx.ep_rank + 1) * per))
        # keys are GLOBAL expert ids so state_dict names are layout independent
        self.experts = nn.ModuleDict({str(e): SwiGLU(cfg.d, cfg.d_expert) for e in self.local_ids})
        self.register_buffer("route_bias", torch.zeros(self.E), persistent=True)
        self.last_counts = None
        self.last_idx = None
        self.last_drop = None

    def forward(self, x):
        c, spec = self.ctx, self.spec
        B, S, D = x.shape
        xt = x.reshape(-1, D)
        T = xt.shape[0]
        with torch.autocast(device_type=xt.device.type, enabled=False):
            logits = F.linear(xt.float(), self.router.weight.float())
        if spec.balance == "aux":
            probs = logits.softmax(-1)
            sel = probs
        else:
            probs = torch.sigmoid(logits)
            sel = probs + self.route_bias
        idx = sel.detach().topk(self.k, dim=-1).indices                     # [T, k]
        gates = probs.gather(1, idx)
        gates = gates / gates.sum(-1, keepdim=True)
        counts = torch.bincount(idx.reshape(-1), minlength=self.E)
        self.last_counts = counts.detach()
        self.last_idx = idx.detach()

        # ---- capacity (training only; eval is dropless)
        cf = spec.capacity_factor if self.training else 0.0
        if cf and cf > 0:
            if spec.cap_scope == "global":
                grp, size, gi = c.world_group, c.world, c.rank
            else:
                grp, size, gi = c.ep_group, c.ep, c.ep_rank
            idx_g = du.gather_rows(idx, grp, size, gi)
            keep = capacity_keep(idx_g, self.E, cf)[gi * T:(gi + 1) * T]
        else:
            keep = torch.ones_like(idx, dtype=torch.bool)
        self.last_drop = (~keep).sum().detach()

        # ---- aux load-balancing loss (Switch form: E * sum_i f_i * P_i)
        aux = probs.new_zeros(())
        if spec.balance == "aux" and self.training:
            f = counts.to(torch.float32)
            n = T
            if spec.lbl_scope == "global":
                f = du.ar_(f.clone(), c.world_group, c.world)
                n = T * c.world
            f = f / (n * self.k)
            aux = self.E * (f * probs.mean(0)).sum()

        # ---- dispatch: one differentiable gather of [x | gates], one int gather
        if c.ep > 1:
            XG = du.gather_ep(torch.cat([xt.float(), gates], dim=-1), c)
            X, G = XG[:, :D], XG[:, D:]
            IK = du.gather_rows(torch.cat([idx, keep.to(idx.dtype)], dim=-1), c.ep_group, c.ep, c.ep_rank)
            IDX, KEEP = IK[:, :self.k], IK[:, self.k:].bool()
        else:
            X, G, IDX, KEEP = xt, gates, idx, keep
            XG = None
        out = torch.zeros(X.shape[0], D, device=X.device, dtype=torch.float32)
        for e in self.local_ids:
            m = (IDX == e) & KEEP
            tok, slot = m.nonzero(as_tuple=True)          # row-major -> tok ascending, unique
            if tok.numel() == 0:
                continue
            h = self.experts[str(e)](X[tok])   # autocast (fp16 arm) casts inside Linear
            out = out.index_add(0, tok, h.float() * G[tok, slot].unsqueeze(-1))
        if XG is not None:
            # keep the gather in the graph on every rank (prevents collective mismatch
            # when a rank's local experts receive zero tokens)
            out = out + XG[:0].sum()
            out = du.reduce_ep(out, c)[c.ep_rank * T:(c.ep_rank + 1) * T]
        return out.view(B, S, D).to(x.dtype), aux


class Block(nn.Module):
    def __init__(self, cfg, spec, ctx):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.attn = Attention(cfg.d, cfg.n_heads, getattr(cfg, "qk_norm", True))
        self.n2 = RMSNorm(cfg.d)
        self.is_moe = cfg.moe
        self.ffn = MoE(cfg, spec, ctx) if cfg.moe else SwiGLU(cfg.d, cfg.top_k * cfg.d_expert)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        if self.is_moe:
            y, aux = self.ffn(self.n2(x))
        else:
            y, aux = self.ffn(self.n2(x)), None
        return x + y, aux


class LM(nn.Module):
    def __init__(self, cfg, spec, ctx):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg, spec, ctx) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d)
        self._rope = None

    def moe_layers(self):
        return [b.ffn for b in self.blocks if b.is_moe]

    def forward(self, idx):
        B, S = idx.shape
        if self._rope is None or self._rope[0].shape[0] < S or self._rope[0].device != idx.device:
            self._rope = rope_cache(self.cfg.seq_len, self.cfg.d // self.cfg.n_heads,
                                    self.cfg.rope_theta, idx.device)
        cos, sin = self._rope[0][:S], self._rope[1][:S]
        x = self.emb(idx)
        aux_total = x.new_zeros((), dtype=torch.float32)
        for b in self.blocks:
            x, aux = b(x, cos, sin)
            if aux is not None:
                aux_total = aux_total + aux
        h = self.norm(x)
        logits = F.linear(h, self.emb.weight)                      # tied head
        return logits, aux_total


def is_expert_param(name: str) -> bool:
    return ".experts." in name


def init_full_state(cfg, spec, seed: int):
    """Canonical full (unsharded) init, identical for every layout."""
    model = LM(cfg, spec, du.local_ctx("cpu"))
    g = torch.Generator().manual_seed(int(seed))
    n_out_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
    with torch.no_grad():
        for name, p in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            if name.endswith(("n1.weight", "n2.weight", "qn.weight", "kn.weight")) or name == "norm.weight":
                p.fill_(1.0)
                continue
            std = 0.02
            if name.endswith("proj.weight") or name.endswith("w2.weight"):
                std = 0.02 * n_out_scale
            p.copy_(torch.randn(p.shape, generator=g) * std)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def ulp_perturb_(state: dict, seed: int):
    """Move ONE weight element by one ULP (towards +inf). Deterministic choice."""
    if not seed:
        return None
    names = sorted(k for k, v in state.items() if v.is_floating_point() and v.dim() >= 2)
    g = torch.Generator().manual_seed(1000 + int(seed))
    name = names[int(torch.randint(len(names), (1,), generator=g))]
    t = state[name].view(-1)
    i = int(torch.randint(t.numel(), (1,), generator=g))
    t[i] = torch.nextafter(t[i], torch.tensor(float("inf"), dtype=t.dtype))
    return name, i
