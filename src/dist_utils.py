"""Distributed process groups and gloo collectives for data and expert parallelism."""
from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


class Ctx:
    def __init__(self, rank: int = 0, world: int = 1, ep: int = 1, device: str = "cpu"):
        assert world % ep == 0
        self.rank = rank
        self.world = world
        self.ep = ep
        self.ep_rank = rank % ep
        self.ep_group_id = rank // ep
        self.edp = world // ep
        self.device = torch.device(device)
        self.world_group = None
        self.ep_group = None
        self.edp_group = None
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def local_ctx(device: str = "cpu") -> Ctx:
    ctx = Ctx(0, 1, 1, device)
    ctx.local_rank = 0
    return ctx


def init_dist(timeout_min: int = 120) -> tuple[int, int]:
    timeout_min = int(os.environ.get("MOE_DIST_TIMEOUT_MIN", timeout_min))
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ and not dist.is_initialized():
        dist.init_process_group("gloo", timeout=datetime.timedelta(minutes=timeout_min))
    if dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def make_ctx(ep: int, device: str) -> Ctx:
    rank, world = init_dist()
    if world % ep != 0:
        raise ValueError(f"world={world} is not divisible by ep={ep}")
    ctx = Ctx(rank, world, ep, device)
    if world > 1:
        ctx.world_group = dist.group.WORLD
        for g in range(world // ep):
            ranks = list(range(g * ep, (g + 1) * ep))
            grp = dist.new_group(ranks)
            if rank in ranks:
                ctx.ep_group = grp
        for j in range(ep):
            ranks = list(range(j, world, ep))
            grp = dist.new_group(ranks)
            if rank in ranks:
                ctx.edp_group = grp
    return ctx


def ar_(tensor: torch.Tensor, group, size: int) -> torch.Tensor:
    """In-place sum reduction across group; stages CUDA tensors through fp32 CPU buffers for gloo."""
    if size <= 1:
        return tensor
    if tensor.device.type != "cpu":
        dtype = torch.float32 if tensor.is_floating_point() else tensor.dtype
        buf = tensor.detach().to("cpu", dtype).contiguous()
        dist.all_reduce(buf, group=group)
        tensor.copy_(buf.to(tensor.dtype))
    else:
        if not tensor.is_contiguous():
            raise ValueError("ar_ requires a contiguous tensor")
        dist.all_reduce(tensor, group=group)
    return tensor


def gather_rows(x: torch.Tensor, group, size: int, idx: int) -> torch.Tensor:
    """Exact row gather along dim 0 via sum-reduction into a zero-initialized buffer."""
    if size <= 1:
        return x
    rows = x.shape[0]
    buf = x.new_zeros((size * rows,) + tuple(x.shape[1:]))
    buf[idx * rows:(idx + 1) * rows] = x
    return ar_(buf, group, size)


class GatherRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size, idx):
        ctx.meta = (group, size, idx, x.shape[0])
        with torch.no_grad():
            return gather_rows(x.detach(), group, size, idx)

    @staticmethod
    def backward(ctx, grad_out):
        group, size, idx, rows = ctx.meta
        grad = grad_out.contiguous().clone()
        ar_(grad, group, size)
        return grad[idx * rows:(idx + 1) * rows], None, None, None


class AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, group, size):
        ctx.meta = (group, size)
        out = x.detach().clone().contiguous()
        ar_(out, group, size)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        group, size = ctx.meta
        grad = grad_out.contiguous().clone()
        ar_(grad, group, size)
        return grad, None, None


def gather_ep(x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    if ctx.ep <= 1:
        return x
    return GatherRows.apply(x, ctx.ep_group, ctx.ep, ctx.ep_rank)


def reduce_ep(x: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    if ctx.ep <= 1:
        return x
    return AllReduceSum.apply(x, ctx.ep_group, ctx.ep)


def allreduce_world_(tensor: torch.Tensor, ctx: Ctx) -> torch.Tensor:
    return ar_(tensor, ctx.world_group, ctx.world)


def any_flag(flag: bool, ctx: Ctx) -> bool:
    if ctx.world <= 1:
        return bool(flag)
    val = torch.tensor([1.0 if flag else 0.0])
    dist.all_reduce(val, group=ctx.world_group)
    return bool(val.item() > 0)


def barrier(ctx: Ctx) -> None:
    if ctx.world > 1:
        dist.barrier(group=ctx.world_group)


def gather_objects_to0(obj, ctx: Ctx):
    if ctx.world <= 1:
        return [obj]
    out = [None] * ctx.world if ctx.rank == 0 else None
    dist.gather_object(obj, out, dst=0, group=ctx.world_group)
    return out


def broadcast_object(obj, ctx: Ctx):
    if ctx.world <= 1:
        return obj
    items = [obj]
    dist.broadcast_object_list(items, src=0, group=ctx.world_group)
    return items[0]


def destroy() -> None:
    if dist.is_initialized():
        try:
            dist.barrier()
        except Exception:
            pass
        dist.destroy_process_group()
