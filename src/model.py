"""Decoder-only language model with optional top-k MoE feed-forward layers and expert parallelism."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import dist_utils as du


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return normed.type_as(x) * self.weight


def rope_cache(seq: int, head_dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    pos = torch.arange(seq, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    xf = x.float()
    x1, x2 = xf[..., ::2], xf[..., 1::2]
    rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)
    return rotated.type_as(x)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, qk_norm: bool = True):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.qk_norm = qk_norm
        if qk_norm:
            self.qn = RMSNorm(d // n_heads)
            self.kn = RMSNorm(d // n_heads)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seq, dim = x.shape
        q, k, v = self.qkv(x).view(bsz, seq, 3, self.h, dim // self.h).permute(2, 0, 3, 1, 4)
        if self.qk_norm:
            q = self.qn(q).to(v.dtype)
            k = self.kn(k).to(v.dtype)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(bsz, seq, dim))


class SwiGLU(nn.Module):
    def __init__(self, d: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def capacity_keep(idx_g: torch.Tensor, n_experts: int, capacity_factor: float) -> torch.Tensor:
    """Slot-priority token mask: top-1 assignments are admitted before top-2 and lower slots."""
    num_tokens, top_k = idx_g.shape
    cap = int(math.ceil(capacity_factor * num_tokens * top_k / n_experts))
    flat = idx_g.t().reshape(-1)
    one_hot = F.one_hot(flat, n_experts).to(torch.int64)
    pos = (one_hot.cumsum(0) * one_hot).sum(-1) - 1
    return (pos < cap).view(top_k, num_tokens).t().contiguous()


class MoE(nn.Module):
    def __init__(self, cfg, spec, ctx: du.Ctx):
        super().__init__()
        self.E = cfg.n_experts
        self.k = cfg.top_k
        self.spec = spec
        self.ctx = ctx
        self.router = nn.Linear(cfg.d, self.E, bias=False)
        per_rank = self.E // ctx.ep
        self.local_ids = list(range(ctx.ep_rank * per_rank, (ctx.ep_rank + 1) * per_rank))
        # Keyed by global expert index so state_dict names do not depend on EP degree.
        self.experts = nn.ModuleDict({str(e): SwiGLU(cfg.d, cfg.d_expert) for e in self.local_ids})
        self.register_buffer("route_bias", torch.zeros(self.E), persistent=True)
        self.last_counts = None
        self.last_idx = None
        self.last_drop = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ctx, spec = self.ctx, self.spec
        bsz, seq, dim = x.shape
        xt = x.reshape(-1, dim)
        num_tokens = xt.shape[0]

        with torch.autocast(device_type=xt.device.type, enabled=False):
            logits = F.linear(xt.float(), self.router.weight.float())
        if spec.balance == "aux":
            probs = logits.softmax(-1)
            scores = probs
        else:
            probs = torch.sigmoid(logits)
            scores = probs + self.route_bias

        idx = scores.detach().topk(self.k, dim=-1).indices
        gates = probs.gather(1, idx)
        gates = gates / gates.sum(-1, keepdim=True)
        counts = torch.bincount(idx.reshape(-1), minlength=self.E)
        self.last_counts = counts.detach()
        self.last_idx = idx.detach()

        cf = spec.capacity_factor if self.training else 0.0
        if cf and cf > 0:
            if spec.cap_scope == "global":
                grp, size, rank_idx = ctx.world_group, ctx.world, ctx.rank
            else:
                grp, size, rank_idx = ctx.ep_group, ctx.ep, ctx.ep_rank
            idx_g = du.gather_rows(idx, grp, size, rank_idx)
            keep = capacity_keep(idx_g, self.E, cf)[rank_idx * num_tokens:(rank_idx + 1) * num_tokens]
        else:
            keep = torch.ones_like(idx, dtype=torch.bool)
        self.last_drop = (~keep).sum().detach()

        aux = probs.new_zeros(())
        if spec.balance == "aux" and self.training:
            freq = counts.to(torch.float32)
            denom = num_tokens
            if spec.lbl_scope == "global":
                freq = du.ar_(freq.clone(), ctx.world_group, ctx.world)
                denom = num_tokens * ctx.world
            freq = freq / (denom * self.k)
            aux = self.E * (freq * probs.mean(0)).sum()

        if ctx.ep > 1:
            xg = du.gather_ep(torch.cat([xt.float(), gates], dim=-1), ctx)
            disp_x, disp_gates = xg[:, :dim], xg[:, dim:]
            ik = du.gather_rows(
                torch.cat([idx, keep.to(idx.dtype)], dim=-1),
                ctx.ep_group,
                ctx.ep,
                ctx.ep_rank,
            )
            disp_idx, disp_keep = ik[:, :self.k], ik[:, self.k:].bool()
        else:
            disp_x, disp_gates, disp_idx, disp_keep = xt, gates, idx, keep
            xg = None

        out = torch.zeros(disp_x.shape[0], dim, device=disp_x.device, dtype=torch.float32)
        for e in self.local_ids:
            mask = (disp_idx == e) & disp_keep
            tok, slot = mask.nonzero(as_tuple=True)
            if tok.numel() == 0:
                continue
            hidden = self.experts[str(e)](disp_x[tok])
            out = out.index_add(0, tok, hidden.float() * disp_gates[tok, slot].unsqueeze(-1))

        if xg is not None:
            # Anchor xg in the autograd graph even if local experts receive zero tokens.
            out = out + xg[:0].sum()
            out = du.reduce_ep(out, ctx)[ctx.ep_rank * num_tokens:(ctx.ep_rank + 1) * num_tokens]
        return out.view(bsz, seq, dim).to(x.dtype), aux


class Block(nn.Module):
    def __init__(self, cfg, spec, ctx: du.Ctx):
        super().__init__()
        self.n1 = RMSNorm(cfg.d)
        self.attn = Attention(cfg.d, cfg.n_heads, getattr(cfg, "qk_norm", True))
        self.n2 = RMSNorm(cfg.d)
        self.is_moe = cfg.moe
        self.ffn = MoE(cfg, spec, ctx) if cfg.moe else SwiGLU(cfg.d, cfg.top_k * cfg.d_expert)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        x = x + self.attn(self.n1(x), cos, sin)
        if self.is_moe:
            y, aux = self.ffn(self.n2(x))
        else:
            y, aux = self.ffn(self.n2(x)), None
        return x + y, aux


class LM(nn.Module):
    def __init__(self, cfg, spec, ctx: du.Ctx):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList([Block(cfg, spec, ctx) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.d)
        self._rope = None

    def moe_layers(self) -> list[MoE]:
        return [b.ffn for b in self.blocks if b.is_moe]

    def forward(self, idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq = idx.shape
        if self._rope is None or self._rope[0].shape[0] < seq or self._rope[0].device != idx.device:
            self._rope = rope_cache(
                self.cfg.seq_len,
                self.cfg.d // self.cfg.n_heads,
                self.cfg.rope_theta,
                idx.device,
            )
        cos, sin = self._rope[0][:seq], self._rope[1][:seq]
        x = self.emb(idx)
        aux_total = x.new_zeros((), dtype=torch.float32)
        for block in self.blocks:
            x, aux = block(x, cos, sin)
            if aux is not None:
                aux_total = aux_total + aux
        logits = F.linear(self.norm(x), self.emb.weight)
        return logits, aux_total


def is_expert_param(name: str) -> bool:
    return ".experts." in name


def init_full_state(cfg, spec, seed: int) -> dict[str, torch.Tensor]:
    """Layout-independent parameter initialization on CPU."""
    model = LM(cfg, spec, du.local_ctx("cpu"))
    gen = torch.Generator().manual_seed(int(seed))
    residual_scale = 1.0 / math.sqrt(2 * cfg.n_layers)
    with torch.no_grad():
        for name, param in sorted(model.named_parameters(), key=lambda kv: kv[0]):
            if name.endswith(("n1.weight", "n2.weight", "qn.weight", "kn.weight")) or name == "norm.weight":
                param.fill_(1.0)
                continue
            std = 0.02 * residual_scale if name.endswith(("proj.weight", "w2.weight")) else 0.02
            param.copy_(torch.randn(param.shape, generator=gen) * std)
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def ulp_perturb_(state: dict[str, torch.Tensor], seed: int) -> tuple[str, int] | None:
    """Shift a single 2D+ weight entry by 1 ULP toward +inf."""
    if not seed:
        return None
    names = sorted(k for k, v in state.items() if v.is_floating_point() and v.dim() >= 2)
    gen = torch.Generator().manual_seed(1000 + int(seed))
    name = names[int(torch.randint(len(names), (1,), generator=gen))]
    flat = state[name].view(-1)
    idx = int(torch.randint(flat.numel(), (1,), generator=gen))
    flat[idx] = torch.nextafter(flat[idx], torch.tensor(float("inf"), dtype=flat.dtype))
    return name, idx
