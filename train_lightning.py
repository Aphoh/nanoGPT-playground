# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import time
from typing import Any, Dict, Mapping, Optional
import wandb

from pathlib import Path
import lightning as L
import torch
from lightning.fabric.utilities import measure_flops
from lightning.pytorch.callbacks import ModelCheckpoint, ThroughputMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.plugins.environments import SLURMEnvironment
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from config import Config, get_config
from model import GPT
from data import get_owt_dataset


class LightningGPTModule(L.LightningModule):
    module: GPT

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.config = config
        self.module: Optional[torch.nn.Module] = None
        self.flops_per_batch: Optional[int] = None

    def configure_model(self) -> None:
        self.module = GPT(self.config.gpt)
        self.module.apply(self.module._init_weights)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.module is None:
            raise RuntimeError("You forgot to call `model.configure_model()`")

        return self.module.configure_optimizers(
            self.config.weight_decay,
            self.config.learning_rate,
            (self.config.beta1, self.config.beta2),
            self.device.type,
        )

    def on_fit_start(self) -> None:
        if self.module is None:
            raise RuntimeError("You forgot to call `model.configure_model()`")

        trainer = self.trainer
        with torch.device("meta"):
            meta_model = GPT(self.module.config)
            # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
            # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
            # consider setting `self.flops_per_batch = estimated_flops` instead
            estimated_flops = (
                meta_model.estimate_flops_per_token()
                * self.config.gpt.block_size
                * self.config.micro_batch_size
            )
            self.print(
                f"Estimated TFLOPs: {estimated_flops * trainer.world_size / 1e12:.2f}"
            )
            x = torch.randint(
                0, 1, (self.config.micro_batch_size, self.config.gpt.block_size)
            )
            forward_fn = lambda: meta_model(x, targets=x)
            loss_fn = lambda y: y[1]
            self.flops_per_batch = measure_flops(meta_model, forward_fn, loss_fn)
            self.print(
                f"Measured TFLOPs: {self.flops_per_batch * trainer.world_size / 1e12:.2f}"
            )
            num_weights = self.module.get_num_params(non_embedding=True)
            flops_per_token = self.flops_per_batch / (
                self.config.gpt.block_size * self.config.micro_batch_size
            )
            flops_per_token_per_weight = flops_per_token / num_weights
            if self.config.wandb_log:
                wandb.config.update(
                    {
                        "num_weights": num_weights,
                        "flops_per_token": flops_per_token,
                        "flops_per_token_per_weight": flops_per_token_per_weight,
                    }
                )

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if not self.config.decay_lr:
            return
        # determine and set the learning rate for this iteration
        lr = get_lr(
            self.config,
            self.trainer.fit_loop.total_batch_idx,
        )
        for optimizer in self.trainer.strategy.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        input_ids, targets = batch
        logits, loss = self.module(input_ids, targets=targets)
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, targets = batch
        logits, loss = self.module(input_ids, targets=targets)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if self.module is None:
            raise RuntimeError("You forgot to call `model.configure_model()`")
        return self.module.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        if self.module is None:
            raise RuntimeError("You forgot to call `model.configure_model()`")
        return self.module.load_state_dict(state_dict, *args, **kwargs)


def main(config: Config) -> None:
    precision = "bf16-mixed" if config.dtype == "bfloat16" else "16-mixed"

    logger = WandbLogger(
        offline=not config.wandb_log,
        project=config.wandb_project,
        log_model=config.wandb_log,
    )

    if config.wandb_log:
        wandb.config.update(OmegaConf.to_container(cfg))

    throughput = ThroughputMonitor(
        batch_size_fn=lambda batch: batch[0].size(0),
    )
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=1,
        dirpath=config.out_dir,
        verbose=True,
    )
    gradient_accumulation_steps = config.batch_size // (
        config.micro_batch_size * config.devices * config.nodes
    )
    log_every_n_steps = config.log_interval
    if gradient_accumulation_steps % config.log_interval != 0:
        log_every_n_steps = (
            gradient_accumulation_steps  # weird thing in throughput monitor
        )

    trainer = L.Trainer(
        accelerator="cpu" if config.device == "cpu" else "auto",
        devices=config.devices,
        num_nodes=config.nodes,
        strategy="ddp" if config.devices > 1 else "auto",
        precision=precision,
        logger=logger,
        callbacks=[throughput, model_checkpoint],
        max_steps=config.max_iters,
        max_epochs=1,
        limit_val_batches=config.eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=log_every_n_steps,
        val_check_interval=config.eval_interval,
        plugins=[SLURMEnvironment()] if config.nodes > 1 else [],
    )

    L.seed_everything(
        1337, workers=True
    )  # same seed for every process to init model (FSDP)

    trainer.print(config)

    if trainer.global_rank == 0:
        Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    model = LightningGPTModule(config)
    trainer.print(
        f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds."
    )

    train_data = get_owt_dataset(
        "train", config.micro_batch_size, config.gpt.block_size, shuffle=True
    )
    val_data = get_owt_dataset(
        "val", config.micro_batch_size, config.gpt.block_size, shuffle=True
    )
    args = {
        "batch_size": None,
        "num_workers": config.num_workers,
    }
    train_dataloader = DataLoader(train_data, **args)
    val_dataloader = DataLoader(val_data, **args)

    t0 = time.perf_counter()
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path="last")
    trainer.print(f"Training time: {(time.perf_counter()-t0):.2f}s")
    if trainer.strategy.root_device.type == "cuda":
        trainer.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(config: Config, it: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > config.max_iters:
        return config.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    cfg: Config = get_config()
    main(cfg)
