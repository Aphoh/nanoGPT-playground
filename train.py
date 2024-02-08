"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python train.py --batch_size=32 --compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import os
import wandb
from dataclasses import dataclass

import time
import math
from contextlib import nullcontext
from omegaconf import OmegaConf

from tqdm import tqdm
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from model import GPT
from data import get_dataset_iters
from config import Config, get_config


@dataclass
class WorldInfo:
    ddp: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    seed_offset: int = 0  # each process gets a different seed


@dataclass
class RunState:
    winfo: WorldInfo
    gradient_accumulation_steps: int
    wandb_log: bool  # rank 0 and cfg.wandb_log
    device: torch.device
    ctx: any  # torch context
    iter_num: int = 0
    local_iter_num: int = 0

    def print(self, *args, **kwargs):
        print(f"[rank {self.winfo.rank}]", *args, **kwargs)

    def r0_print(self, *args, **kwargs):
        if self.winfo.rank == 0:
            self.print(*args, **kwargs)


def save_ckpt(rs: RunState, model: GPT, optimizer: torch.optim.Optimizer, cfg: Config):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iter_num": rs.iter_num,
        "config": cfg,
    }
    t0 = time.time()
    ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
    torch.save(checkpoint, ckpt_path)
    rs.r0_print(f"saved checkpoint to {ckpt_path} in {time.time()-t0:.1f}s")
    if cfg.wandb_log and rs.local_iter_num == cfg.eval_interval:
        # if this is the first eval after the initial one, watch the checkpoint file
        rs.r0_print("watching checkpoint file for changes")
        wandb.save(ckpt_path, policy="live")


def get_word_info(cfg) -> WorldInfo:
    winfo = WorldInfo(ddp=int(os.environ.get("RANK", -1)) != -1)
    if winfo.ddp:
        dist.init_process_group(backend=cfg.backend)
        winfo.rank = int(os.environ["RANK"])
        winfo.local_rank = int(os.environ["LOCAL_RANK"])
        winfo.world_size = int(os.environ["WORLD_SIZE"])
        # this process will do logging, checkpointing etc.
        winfo.seed_offset = winfo.rank  # each process gets a different seed

    return winfo


def init_state(cfg: Config, winfo: WorldInfo) -> RunState:
    torch.manual_seed(1337 + winfo.seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    assert (
        gradient_accumulation_steps % winfo.world_size == 0
    ), f"world_size {winfo.world_size} must divide gradient_accumulation_steps {gradient_accumulation_steps} evenly"
    gradient_accumulation_steps //= winfo.world_size

    wandb_log = cfg.wandb_log and winfo.rank == 0
    if wandb_log:
        wandb.init(
            project=cfg.wandb_project, config=OmegaConf.to_container(cfg)
        )  # do this here so we can log everything

    if winfo.rank == 0:
        os.makedirs(cfg.out_dir, exist_ok=True)

    if winfo.ddp:
        device = torch.device(f"cuda:{winfo.local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device(cfg.device)
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[cfg.dtype]
    ctx = nullcontext()
    if device.type == "cuda":
        ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype)

    rs = RunState(
        winfo=winfo,
        gradient_accumulation_steps=gradient_accumulation_steps,
        wandb_log=wandb_log,
        device=device,
        ctx=ctx,
    )
    rs.print(f"initialized run state {rs}")
    return rs


def build_model(rs: RunState, cfg: Config) -> tuple[GPT, torch.optim.Optimizer]:
    model = GPT(cfg.gpt, print_fn=rs.r0_print).to(rs.device)
    checkpoint = None
    if cfg.init_from == "scratch":
        rs.r0_print("Training from scratch")
        # When we wrap with DDP, it will sync the model weights across all processes
    elif cfg.init_from == "resume" and rs.winfo.rank == 0:  # only load on rank 0
        rs.r0_print(f"Resuming training from {cfg.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(cfg.out_dir, "ckpt.pt")
        if os.path.isfile(ckpt_path):
            rs.r0_print(f"Found checkpoint {ckpt_path}")
            checkpoint = torch.load(ckpt_path, map_location=rs.device)
            read_cfg: Config = checkpoint["config"]
            if read_cfg != cfg:
                raise ValueError(
                    f"""checkpoint config does not match command line config: 
                    Checkpoint Config:
                    {OmegaConf.to_yaml(read_cfg)}\n
                    Command Line Config:
                    {OmegaConf.to_yaml(cfg)}"""
                )
            state_dict = checkpoint["model"]
            model.load_state_dict(state_dict, strict=True)
            rs.iter_num = checkpoint["iter_num"]
            rs.r0_print(f"loaded checkpoint at iter_num={rs.iter_num}")

    optimizer = model.configure_optimizers(
        cfg.weight_decay, cfg.learning_rate, (cfg.beta1, cfg.beta2), rs.device.type
    )
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None

    return model, optimizer


@torch.no_grad()
def estimate_loss(cfg: Config, rs: RunState, model: GPT, train_iter, val_iter):
    eval_iters = rs.gradient_accumulation_steps * cfg.eval_batches
    out = {}
    model.eval()
    for split, loader in [
        ("train", train_iter),
        ("val", val_iter),
    ]:
        t0 = time.time()
        losses = torch.zeros(eval_iters, device=rs.device)
        for k, (X, Y) in tqdm(
            zip(range(eval_iters), iter(loader)),
            desc=f"Evaluating {split}",
            disable=rs.winfo.rank != 0,
            leave=False,
        ):
            with rs.ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        rank_mean = losses.mean()
        if rs.winfo.ddp:
            dist.reduce(rank_mean, dst=0, op=dist.ReduceOp.SUM, async_op=False)
        out[split] = rank_mean.item() / winfo.world_size
        rs.print(f"completed {split} eval in {time.time()-t0:.1f}s")
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(cfg: Config, it):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


if __name__ == "__main__":
    cfg: Config = get_config()

    # various inits, derived attributes, I/O setup
    winfo = get_word_info(cfg)
    rs = init_state(cfg, winfo)
    rs.r0_print(f"CONFIG:\n{OmegaConf.to_yaml(cfg)}")

    train_iter, val_iter = get_dataset_iters(cfg, rs.device)

    model, optimizer = build_model(rs, cfg)

    rs.r0_print("------Training Model------")
    rs.r0_print(model)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.dtype == "float16"))

    # compile the model
    raw_model = model
    if cfg.compile:
        rs.r0_print("compiling the model... (takes a ~minute)")
        model = torch.compile(model)  # requires PyTorch 2.0
        rs.r0_print("compilation finished")

    # wrap model into DDP container
    if rs.winfo.ddp:
        rs.print("wrapping model")
        model = DDP(model, device_ids=[rs.winfo.local_rank])
        rs.print("model wrapped")

    # logging
    if rs.wandb_log and rs.iter_num == 0:
        wandb.watch(raw_model, log="all", log_freq=cfg.eval_interval)
        wandb.log(
            {
                "flops_per_token": raw_model.estimate_flops_per_token(),
                "flops_per_token_per_weight": raw_model.estimate_flops_per_token_per_weight(),
                "num_parameters": raw_model.get_num_params(),
                "num_parameters_non_lmhead": raw_model.get_num_params(non_lm_head=True),
            },
            step=0,
        )

    # training loop
    rs.print("Ready to train!")
    X, Y = next(train_iter)  # fetch the very first batch
    t0 = time.time()
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(cfg, rs.iter_num) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if rs.iter_num % cfg.eval_interval == 0:
            losses = estimate_loss(cfg, rs, model, train_iter, val_iter)
            rs.r0_print(
                f"step {rs.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            if rs.iter_num == 0 and cfg.eval_only:
                break
            if rs.wandb_log:
                log_dict = {
                    "iter": rs.iter_num,
                    "train/loss": losses["train"],
                    "val/loss": losses["val"],
                }
                if rs.local_iter_num > 5:  # let the training loop settle a bit
                    log_dict["lr"] = lr
                    log_dict["mfu"] = running_mfu * 100
                wandb.log(log_dict, step=rs.iter_num)
            if cfg.save_checkpoints and winfo.rank == 0 and rs.local_iter_num > 0:
                save_ckpt(rs, raw_model, optimizer, cfg)

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(rs.gradient_accumulation_steps):
            if winfo.ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (
                    micro_step == rs.gradient_accumulation_steps - 1
                )
            with rs.ctx:
                logits, loss = model(X, Y)
                loss = (
                    loss / rs.gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = next(train_iter)
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if rs.iter_num % cfg.log_interval == 0 and winfo.rank == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = loss.item() * rs.gradient_accumulation_steps
            if rs.local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    cfg.micro_batch_size * rs.gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            rs.r0_print(
                f"iter {rs.iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%"
            )
            if rs.wandb_log:
                wandb.log(
                    {
                        "train/loss_iter": lossf,
                    },
                    step=rs.iter_num,
                )
        rs.iter_num += 1
        rs.local_iter_num += 1

        # termination conditions
        if rs.iter_num > cfg.max_iters:
            break

    if rs.ddp:
        dist.destroy_process_group()
