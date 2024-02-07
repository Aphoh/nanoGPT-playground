# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import time
from typing import Any, Dict, Mapping, Optional
import os

from pathlib import Path
import lightning as L
import wandb
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
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
        self.module: Optional[GPT] = None

    def configure_model(self) -> None:
        self.module = GPT(self.config.gpt)
        self.module.apply(self.module._init_weights)
        if self.config.compile:
            self.print("Compiling model...")
            self.orig_mod = self.module
            self.module = torch.compile(self.module)
            self.print("Model compiled!")

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

        with torch.device("meta"):
            meta_model = GPT(self.module.config)
            fpt = meta_model.estimate_flops_per_token()
            self.print(f"Estimated GFLOPs per token: {fpt / 1e9:.2f}")
            num_weights = self.module.get_num_params(non_embedding=True)
            num_non_lm_weights = self.module.get_num_params(
                non_embedding=True, non_lm_head=True
            )
            fpt_pw = fpt / num_weights
            fpt_pnlmw = fpt / num_non_lm_weights
            if self.global_rank == 0 and self.config.wandb_log:
                wandb.config.update(
                    {
                        "num_weights": num_weights,
                        "num_nonlm_weights": num_weights,
                        "flops_per_token": fpt,
                    }
                )
                wandb.config.update(
                    {
                        "flops_per_token_per_weight": fpt_pw,
                        "flops_per_token_per_nonlm_weight": fpt_pnlmw,
                    },
                    allow_val_change=True,  # this is fine since these are rough floating point numbers
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
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=False,  # do not use in monitor callbacks!!!
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        input_ids, targets = batch
        logits, loss = self.module(input_ids, targets=targets)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=self.config.devices > 1,
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if self.module is None:
            raise RuntimeError("You forgot to call `model.configure_model()`")

        if self.config.compile:
            return self.orig_mod.state_dict()
        else:
            return self.module.state_dict()

    def load_state_dict(self, state_dict: Mapping[str, Any], *args, **kwargs):
        if self.module is None:
            raise RuntimeError("You forgot to call `model.configure_model()`")

        if self.config.compile:
            return self.orig_mod.load_state_dict(state_dict, *args, **kwargs)
        else:
            return self.module.load_state_dict(state_dict, *args, **kwargs)


def main(config: Config) -> None:
    precision = "bf16-mixed" if config.dtype == "bfloat16" else "16-mixed"

    logger = WandbLogger(
        offline=not config.wandb_log,
        project=config.wandb_project,
        log_model=config.wandb_log,
    )

    gradient_accumulation_steps = config.batch_size // (
        config.micro_batch_size * config.devices * config.nodes
    )

    # For some reason monitoring here with ddp bricks the training loop
    # it's probably because we disable sync_dist due to nccl bugs
    model_checkpoint = ModelCheckpoint(
        save_last=True,
        every_n_train_steps=gradient_accumulation_steps * config.eval_interval,
        dirpath=config.out_dir,
        verbose=True,
    )

    trainer = L.Trainer(
        accelerator="cpu" if config.device == "cpu" else "auto",
        devices=config.devices,
        num_nodes=config.nodes,
        strategy="ddp" if config.devices > 1 else "auto",
        precision=precision,
        logger=logger,
        callbacks=[model_checkpoint],
        max_steps=config.max_iters,
        max_epochs=1,
        limit_val_batches=config.eval_iters,
        accumulate_grad_batches=gradient_accumulation_steps,
        log_every_n_steps=config.log_interval * gradient_accumulation_steps,
        val_check_interval=config.eval_interval * gradient_accumulation_steps,
        use_distributed_sampler=False,  # this shouldn't matter since we use an iterable dataset, but just in case
        plugins=(
            [SLURMEnvironment(auto_requeue=True)]
            if "SLURM_JOB_ID" in os.environ
            else []
        ),
    )

    if trainer.global_rank == 0 and config.wandb_log:
        logger.experiment.config.update(OmegaConf.to_container(cfg))

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
    train_dataloader = DataLoader(train_data, shuffle=False, **args)
    val_dataloader = DataLoader(val_data, shuffle=False, **args)

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
