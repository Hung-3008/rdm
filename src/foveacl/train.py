"""FoveaCL training: ResNet-18 + SimCLR (NT-Xent) with GPU view generation (docs/propose/propose04.md).

Single GPU:
    python -m src.foveacl.train --data data/images/foveacl/in100_256 --out outputs/foveacl/R2 --mode fix --a_rel 0.238
Two nodes (one GPU each), run on both with --node_rank 0 / 1:
    torchrun --nnodes 2 --nproc_per_node 1 --node_rank R --master_addr IP --master_port 29500 -m src.foveacl.train ...

--batch is the per-GPU batch size. An online linear probe on detached features reports top-1 on the validation split.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from src.foveacl.views import ViewMaker


class Batches(torch.utils.data.Dataset):
    """Index with a sorted array of image indices -> (uint8 images, labels). Opens the memmap lazily per worker."""

    def __init__(self, root, split):
        meta = json.loads((Path(root) / "meta.json").read_text())
        self.path, self.n, self.s = Path(root) / f"{split}_images.u8", meta[f"n_{split}"], meta["size"]
        self.labels = torch.from_numpy(np.load(Path(root) / f"{split}_labels.npy").astype(np.int64))
        self.mm = None

    def __getitem__(self, idx):
        if self.mm is None:
            self.mm = np.memmap(self.path, np.uint8, "r", shape=(self.n, self.s, self.s, 3))
        return torch.from_numpy(self.mm[idx]), self.labels[idx]


class EpochSampler:
    """Yields sorted index batches; the permutation depends on (seed, epoch) and is split across ranks."""

    def __init__(self, n, batch, rank, world, seed, shuffle=True, drop_last=True):
        self.n, self.batch, self.rank, self.world, self.seed, self.shuffle, self.drop_last = n, batch, rank, world, seed, shuffle, drop_last
        self.epoch = 0

    def __len__(self):
        per = self.n // self.world
        return per // self.batch if self.drop_last else math.ceil(per / self.batch)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        perm = rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        perm = perm[: self.n // self.world * self.world][self.rank::self.world]
        for i in range(len(self)):
            yield np.sort(perm[i * self.batch:(i + 1) * self.batch])


def loader(ds, sampler, workers):
    return torch.utils.data.DataLoader(ds, batch_size=None, sampler=sampler, num_workers=workers, pin_memory=True,
                                       persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


class Encoder(nn.Module):
    def __init__(self, proj_hidden=2048, proj_dim=128):
        super().__init__()
        self.backbone = torchvision.models.resnet18(zero_init_residual=True)
        self.backbone.fc = nn.Identity()
        self.proj = nn.Sequential(nn.Linear(512, proj_hidden), nn.BatchNorm1d(proj_hidden), nn.ReLU(inplace=True),
                                  nn.Linear(proj_hidden, proj_dim))

    def forward(self, x):
        h = self.backbone(x)
        return h, self.proj(h)


class GatherLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        out = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(out, x)
        return tuple(out)

    @staticmethod
    def backward(ctx, *grads):
        g = torch.stack(grads)
        dist.all_reduce(g)
        return g[dist.get_rank()]


def nt_xent(z1, z2, tau):
    """z1, z2 (B, d). Negatives are all other views in the global batch."""
    z = F.normalize(torch.cat([z1, z2]), dim=1).float()
    B = z1.shape[0]
    if dist.is_initialized() and dist.get_world_size() > 1:
        za = torch.cat(GatherLayer.apply(z))
        W, r = dist.get_world_size(), dist.get_rank()
    else:
        za, W, r = z, 1, 0
    sim = z @ za.T / tau                                                   # (2B, 2BW)
    idx = torch.arange(2 * B, device=z.device)
    sim[idx, r * 2 * B + idx] = float("-inf")
    target = r * 2 * B + torch.cat([idx[B:], idx[:B]])
    return F.cross_entropy(sim, target)


class LARS(torch.optim.Optimizer):
    """SGD with momentum and layer-wise trust ratio; 1-d parameters (bias, norm) skip weight decay and adaptation."""

    def __init__(self, params, lr, weight_decay=1e-4, momentum=0.9, eta=0.02):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum, eta=eta))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                d = p.grad
                if p.ndim > 1:
                    d = d.add(p, alpha=g["weight_decay"])
                    pn, dn = p.norm(), d.norm()
                    q = torch.where((pn > 0) & (dn > 0), g["eta"] * pn / dn, torch.ones_like(pn))
                    d = d * q
                buf = self.state[p].setdefault("mu", torch.zeros_like(p))
                buf.mul_(g["momentum"]).add_(d)
                p.add_(buf, alpha=-g["lr"])


def acuity_sigma(epoch, epochs, on, sigma0=4.0, frac=0.3, steps=5):
    """Blur schedule of CATDiet: sigma0 -> 0 in `steps` equal stages over the first `frac` of training."""
    if not on or epoch >= frac * epochs:
        return 0.0
    stage = int(epoch / (frac * epochs) * steps)
    return sigma0 * (1 - stage / (steps - 1)) if steps > 1 else 0.0


@torch.no_grad()
def evaluate(model, probe, views, dl, dev):
    model.eval()
    correct = total = 0
    for img, y in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            h, _ = model(views.test_view(img.to(dev, non_blocking=True)))
        correct += (probe(h.float()).argmax(1).cpu() == y).sum().item()
        total += len(y)
    model.train()
    return correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=["crop", "fix"], default="fix")
    p.add_argument("--a_rel", type=float, default=math.inf, help="a / R; inf = uniform sampling")
    p.add_argument("--sigma_c", type=float, default=0.3)
    p.add_argument("--f_min", type=float, default=0.5)
    p.add_argument("--n", type=int, default=112)
    p.add_argument("--acuity", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=0.15, help="per 256 images of the global batch")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.15)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=0, help="benchmark: stop after this many steps")
    p.add_argument("--fp16_comm", action="store_true")
    a = p.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group(os.environ.get("FOVEACL_BACKEND", "nccl"))
    rank, world = (dist.get_rank(), dist.get_world_size()) if ddp else (0, 1)
    dev = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(dev)
    torch.manual_seed(a.seed * 1000 + rank)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(a.out)
    if rank == 0:
        out.mkdir(parents=True, exist_ok=True)
        (out / "args.json").write_text(json.dumps(vars(a)))

    train_ds, val_ds = Batches(a.data, "train"), Batches(a.data, "val")
    n_cls = int(train_ds.labels.max()) + 1
    sampler = EpochSampler(train_ds.n, a.batch, rank, world, a.seed)
    dl = loader(train_ds, sampler, a.workers)
    val_dl = loader(val_ds, EpochSampler(val_ds.n, 500, 0, 1, 0, shuffle=False, drop_last=False), 2)
    views = ViewMaker(a.n, a.mode, a.a_rel, a.sigma_c, a.f_min)

    model = Encoder().to(dev).to(memory_format=torch.channels_last)
    probe = nn.Linear(512, n_cls).to(dev)
    if ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model) if os.environ.get("FOVEACL_SYNCBN") else model
        model = nn.parallel.DistributedDataParallel(model, device_ids=[dev.index], gradient_as_bucket_view=True)
        if a.fp16_comm:
            from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
            model.register_comm_hook(None, default_hooks.fp16_compress_hook)
    opt = LARS(model.parameters(), lr=0.0, weight_decay=a.wd)
    probe_opt = torch.optim.SGD(probe.parameters(), lr=0.1, momentum=0.9)
    base_lr, steps_per_epoch = a.lr * a.batch * world / 256, len(sampler)
    total, warm = a.epochs * steps_per_epoch, a.warmup * steps_per_epoch

    start = 0
    if (out / "last.pt").exists():
        ck = torch.load(out / "last.pt", map_location=dev)
        (model.module if ddp else model).load_state_dict(ck["model"])
        probe.load_state_dict(ck["probe"]); opt.load_state_dict(ck["opt"]); probe_opt.load_state_dict(ck["probe_opt"])
        start = ck["epoch"] + 1

    step = start * steps_per_epoch
    for epoch in range(start, a.epochs):
        sampler.epoch = epoch
        sig = acuity_sigma(epoch, a.epochs, a.acuity)
        t0, seen, nb = time.time(), 0, 0
        run_loss, run_acc = torch.zeros((), device=dev), torch.zeros((), device=dev)
        for img, y in dl:
            img, y = img.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            lr = base_lr * step / warm if step < warm else base_lr * 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr
            v1, v2 = views.train_view(img, sig), views.train_view(img, sig)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h, z = model(torch.cat([v1, v2]))
            loss = nt_xent(*z.chunk(2), a.tau)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            logits = probe(h[: len(y)].detach().float())
            ploss = F.cross_entropy(logits, y)
            probe_opt.zero_grad(set_to_none=True)
            ploss.backward()
            probe_opt.step()
            step += 1
            seen += len(y)
            nb += 1
            run_loss += loss.detach()
            run_acc += (logits.argmax(1) == y).float().mean()
            if a.max_steps and step >= a.max_steps:
                break
        dt = time.time() - t0
        if rank == 0:
            rec = dict(epoch=epoch, loss=run_loss.item() / nb, probe_train_acc=run_acc.item() / nb, lr=lr, acuity_sigma=sig,
                       img_per_s=seen * world / dt, epoch_s=dt)
            if epoch % 10 == 9 or epoch == a.epochs - 1:
                rec["probe_val_acc"] = evaluate(model.module if ddp else model, probe, views, val_dl, dev)
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(rec, flush=True)
            ck = dict(model=(model.module if ddp else model).state_dict(), probe=probe.state_dict(), opt=opt.state_dict(),
                      probe_opt=probe_opt.state_dict(), epoch=epoch, args=vars(a))
            torch.save(ck, out / "last.tmp")
            os.replace(out / "last.tmp", out / "last.pt")
        if a.max_steps and step >= a.max_steps:
            break
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
