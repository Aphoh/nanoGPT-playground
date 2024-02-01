# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import time
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from lightning.fabric.utilities import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader
from .config import Config, GPTConfig, get_config
from data import get_owt_dataset
from .model import GPT, Block
from .utils import chunked_cross_entropy


def setup(
    config: Config,
    precision: Optional[str] = None,
    resume: Union[bool, Path] = False,
) -> None:
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size
    assert gradient_accumulation_steps > 0

    precision = "bf16-mixed" if config.dtype == "bfloat16" else "16-mixed"

    if config.devices > 1:
        strategy = FSDPStrategy(
            auto_wrap_policy={Block},
            activation_checkpointing_policy={Block},
            state_dict_type="full",
            limit_all_gathers=True,
            cpu_offload=False,
        )
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=config.devices, strategy=strategy, precision=precision)
    fabric.print(config)
    fabric.launch(main, config=config, resume=resume)


def main(fabric: L.Fabric, config: Config, resume: Union[bool, Path]) -> None:
    out_dir = Path(config.out_dir)
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(
        1337, workers=True
    )  # same seed for every process to init model (FSDP)

    t0 = time.perf_counter()
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    model.apply(model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {model.get_num_params(non_embedding=False):,}")

    device = fabric.device
    optimizer = model.configure_optimizers(
        config.weight_decay,
        config.learning_rate,
        (config.beta1, config.beta2),
        device.type,
    )
    model = fabric.setup(model)
    optimizer = fabric.setup_optimizers(optimizer)
    if config.compile:
        fabric.print("Compile isn't totally supported in fabric yet")

    train_data = get_owt_dataset(
        "train", config.micro_batch_size, config.gpt.block_size, shuffle=True
    )
    val_data = get_owt_dataset(
        "val", config.micro_batch_size, config.gpt.block_size, shuffle=True
    )
    train_dataloader = DataLoader(
        train_data, batch_size=config.micro_batch_size, num_workers=2
    )
    val_dataloader = DataLoader(
        val_data, batch_size=config.micro_batch_size, num_workers=2
    )
    train_dataloader, val_dataloader = fabric.setup_dataloaders(
        train_dataloader, val_dataloader
    )

    state = {
        "model": model,
        "optimizer": optimizer,
        "config": config,
        "iter_num": 0,
        "step_count": 0,
    }

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=lambda p: int(p.name.split("-")[1]))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(
    fabric: L.Fabric,
    state: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
) -> None:
    model = state["model"]
    optimizer = state["optimizer"]
    config: Config = state["config"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `flops_per_batch=estimated_flops` instead
        estimated_flops = (
            meta_model.estimate_flops_per_token()
            * config.gpt.block_size
            * config.micro_batch_size
        )
        fabric.print(
            f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}"
        )
        x = torch.randint(0, 1, (config.micro_batch_size, model.max_seq_length))
        forward_fn = lambda: meta_model(x)
        loss_fn = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, forward_fn, loss_fn)
        fabric.print(
            f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}"
        )
        del meta_model, x

    throughput = ThroughputMonitor(fabric, window_size=50)
    total_t0 = time.perf_counter()

    train_iter = iter(train_dataloader)
    gradient_accumulation_steps = config.batch_size // config.micro_batch_size

    for state["iter_num"] in range(state["iter_num"], config.max_iters):
        # determine and set the learning rate for this iteration
        lr = (
            get_lr(state["iter_num"], config.warmup_iters, config.max_iters)
            if config.decay_lr
            else config.learning_rate
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        iter_num = state["iter_num"] + 1
        iter_t0 = time.perf_counter()

        input_ids, targets = next(train_iter)

        is_accumulating = iter_num % gradient_accumulation_steps != 0
        with fabric.no_backward_sync(model, enabled=is_accumulating):
            logits = model(input_ids)
            loss = chunked_cross_entropy(logits, targets, chunk_size=0)
            fabric.backward(loss / gradient_accumulation_steps)

        if not is_accumulating:
            fabric.clip_gradients(model, optimizer, max_norm=config.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            state["step_count"] += 1

        if iter_num % config.log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * config.micro_batch_size,
                lengths=iter_num * config.micro_batch_size * model.max_seq_length,
                flops=measured_flops * config.log_interval,
            )
            throughput.compute_and_log(step=iter_num)
            fabric.print(
                f"iter {iter_num} step {state['step_count']}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms{' (optimizer.step)' if not is_accumulating else ''}"
            )

        if not is_accumulating and state["step_count"] % config.eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(
                fabric, model, val_dataloader, max_iters=config.eval_iters
            )
            t1 = time.perf_counter() - t0
            fabric.print(
                f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms"
            )
            fabric.barrier()
        if not is_accumulating and state["step_count"] % config.save_interval == 0:
            checkpoint_path = config.out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(
    fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, max_iters: int
) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    val_iter = iter(val_dataloader)

    losses = torch.zeros(max_iters, device=fabric.device)
    for k in range(max_iters):
        input_ids, targets = next(val_iter)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits, targets, chunk_size=0)
    out = losses.mean()

    model.train()
    return out


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int, cfg: Config) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return cfg.learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    config: Config = get_config()
    setup(config)
