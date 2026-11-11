"""Process groups + exact, gloo-only collectives.

Design notes
* Every collective is an all_reduce (the only op gloo supports for every dtype/device
  combination we need). An all-gather is done as "zero buffer + own slot + all_reduce":
  x + 0 == x exactly, so the gather itself adds NO rounding.
* CUDA tensors are reduced on a CPU copy in fp32 (works on a single T4 with several
  ranks sharing the GPU; reduction order is then gloo's, which is fixed per world size).
* Layout: ranks are split into EP groups of consecutive ranks
  [g*ep, ..., g*ep+ep-1]; expert shard j lives on every rank with ep_rank == j and is
  data-parallel across the "expert-DP" group {j, j+ep, j+2ep, ...}.
"""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


class Ctx:
    def __init__(self, rank=0, world=1, ep=1, device="cpu"):
        assert world % ep == 0
        self.rank, self.world, self.ep = rank, world, ep
        self.ep_rank = rank % ep
        self.ep_group_id = rank // ep
        self.edp = world // ep
        self.device = torch.device(device)
        self.world_group = None
        self.ep_group = None
        self.edp_group = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    @property
    def is_main(self):
        return self.rank == 0


def local_ctx(device="cpu") -> Ctx:
    """Single-process context (analysis / reference computations)."""
    c = Ctx(0, 1, 1, device)
    c.local_rank = 0
    return c


def init_dist(timeout_min: int = 120):
    timeout_min = int(os.environ.get("MOE_DIST_TIMEOUT_MIN", timeout_min))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=timeout_min))
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def make_ctx(ep: int, device: str) -> Ctx:
    rank, world = init_dist()
    if world % ep != 0:
        raise ValueError(f"world={world} not divisible by ep={ep}")
    c = Ctx(rank, world, ep, device)
    if world > 1:
        c.world_group = dist.group.WORLD
        # all ranks must create all groups in the same order
        for g in range(world // ep):
            ranks = list(range(g * ep, (g + 1) * ep))
            grp = dist.new_group(ranks)
            if rank in ranks:
                c.ep_group = grp
        for j in range(ep):
            ranks = list(range(j, world, ep))
            grp = dist.new_group(ranks)
            if rank in ranks:
                c.edp_group = grp
    return c


# ----------------------------------------------------------------------------- primitives
def ar_(t: torch.Tensor, group, size: int) -> torch.Tensor:
    """In-place SUM all_reduce. No-op when the group has one member."""
    if size <= 1:
        return t
    if t.device.type != "cpu":
        c = t.detach().to("cpu", torch.float32 if t.is_floating_point() else t.dtype).contiguous()
        dist.all_reduce(c, group=group)
        t.copy_(c.to(t.dtype))
    else:
        if not t.is_contiguous():
            raise ValueError("ar_ needs a contiguous tensor")
        dist.all_reduce(t, group=group)
    return t


def gather_rows(x: torch.Tensor, group, size: int, idx: int) -> torch.Tensor:
    """Exact all-gather along dim 0 (no autograd)."""
    if size <= 1:
        return x
    T = x.shape[0]
    buf = x.new_zeros((size * T,) + tuple(x.shape[1:]))
    buf[idx * T:(idx + 1) * T] = x
    return ar_(buf, group, size)


class GatherRows(torch.autograd.Function):
    """Differentiable exact all-gather. Backward = all_reduce(grad) then take own rows."""

    @staticmethod
    def forward(ctx, x, group, size, idx):
        ctx.meta = (group, size, idx, x.shape[0])
        with torch.no_grad():
            return gather_rows(x.detach(), group, size, idx)

    @staticmethod
    def backward(ctx, g):
        group, size, idx, T = ctx.meta
        g = g.contiguous().clone()
        ar_(g, group, size)
        return g[idx * T:(idx + 1) * T], None, None, None


class AllReduceSum(torch.autograd.Function):
    """y = sum over group of x. Every rank consumes y, so dL/dx = all_reduce(dL/dy)."""

    @staticmethod
    def forward(ctx, x, group, size):
        ctx.meta = (group, size)
        y = x.detach().clone().contiguous()
        ar_(y, group, size)
        return y

    @staticmethod
    def backward(ctx, g):
        group, size = ctx.meta
        g = g.contiguous().clone()
        ar_(g, group, size)
        return g, None, None


def gather_ep(x, c: Ctx):
    if c.ep <= 1:
        return x
    return GatherRows.apply(x, c.ep_group, c.ep, c.ep_rank)


def reduce_ep(x, c: Ctx):
    if c.ep <= 1:
        return x
    return AllReduceSum.apply(x, c.ep_group, c.ep)


def allreduce_world_(t, c: Ctx):
    return ar_(t, c.world_group, c.world)


def any_flag(flag: bool, c: Ctx) -> bool:
    if c.world <= 1:
        return bool(flag)
    t = torch.tensor([1.0 if flag else 0.0])
    dist.all_reduce(t, group=c.world_group)
    return bool(t.item() > 0)


def barrier(c: Ctx):
    if c.world > 1:
        dist.barrier(group=c.world_group)


def gather_objects_to0(obj, c: Ctx):
    if c.world <= 1:
        return [obj]
    out = [None] * c.world if c.rank == 0 else None
    dist.gather_object(obj, out, dst=0, group=c.world_group)
    return out


def broadcast_object(obj, c: Ctx):
    if c.world <= 1:
        return obj
    lst = [obj]
    dist.broadcast_object_list(lst, src=0, group=c.world_group)
    return lst[0]


def destroy():
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
