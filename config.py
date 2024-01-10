from dataclasses import dataclass, field
import torch
from omegaconf import OmegaConf
import sys
import os


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    block_linear: bool = (
        False  # False: use Linear layers, like GPT-2. False: use BlockLinear layer
    )
    block_m: int = 8  # m dimension for block linear layer
    block_k: int = 16  # k dimension for block linear layer
    block_n: int = 32  # n dimension for block linear layer
    mlp_ratio: int = 4  # ratio of mlp middle hidden dimension to embedding dimension
    mlp_init_std: float = 0.02  # std dev of gaussian initialization for mlp weights


@dataclass
class Config:
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    gpt: GPTConfig = field(default_factory=GPTConfig)

    out_dir: str = "out"
    data_dir: str = "data"
    eval_interval: int = 2000
    log_interval: int = 1
    eval_iters: int = 200
    eval_only: bool = False  # if True, script exits right after the first eval
    always_save_checkpoint: bool = True  # always save a checkpoint after each eval
    init_from: str = "scratch"  # 'scratch' or 'resume' or 'gpt2*'
    wandb_log: bool = False  # disabled by default
    wandb_project: str = "nanoGPT-playground"
    dataset: str = "openwebtext"
    num_workers: int = 0

    gradient_accumulation_steps: int = 5 * 8  # used to simulate larger batch sizes
    # if gradient_accumulation_steps > 1, this is the micro-batch size
    batch_size: int = 12
    learning_rate: float = 6e-4  # max learning rate
    max_iters: int = 600000  # total number of training iterations
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr: bool = True  # whether to decay the learning rate
    warmup_iters: int = 2000  # how many steps to warm up for
    lr_decay_iters: int = 600000  # should be ~= max_iters per Chinchilla
    # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    min_lr: float = 6e-5
    # DDP settings
    backend: str = "nccl"  # 'nccl', 'gloo', etc.
    # system
    device: str = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'mps'
    dtype: str = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    compile: bool = True  # use PyTorch 2.0 to compile the model to be faster


def get_config() -> Config:
    cfg: Config = OmegaConf.structured(Config)
    config_file = sys.argv[1]
    if config_file.endswith(".yml") and "--" not in config_file:
        cfg: Config = OmegaConf.merge(cfg, OmegaConf.load(config_file))
        sys.argv.pop(1)  # remove the config file
    cfg: Config = OmegaConf.merge(cfg, OmegaConf.from_cli())
    cfg.out_dir = os.environ.get("OUT_DIR", cfg.out_dir)  # override from env var
    cfg.data_dir = os.environ.get("DATA_DIR", cfg.data_dir)  # override from env var
    OmegaConf.set_readonly(cfg, True)  # make read-only to avoid accidental overwrites
    return cfg


def test_get_config_yaml():
    sys.argv = [
        "python",
        "config/train_gpt2_mini_block.yml",
        "eval_iters=101",
        "gpt.block_size=63",
    ]
    cfg = get_config()
    assert cfg.eval_iters == 101
    assert cfg.gpt.block_size == 63
