"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect

import torch
import torch.nn as nn
from torch.nn import functional as F
from config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


def block_linear_stats(m: int, k: int, n: int, in_features: int, out_features: int):
    """Computes various stats for the block linear layer

    Args:
        m (int): the m dimension the input token is reshaped into
        k (int): the k dimension the input token is reshaped into
        n (int): the n dimension of the output token
        in_features (int): the number of features in the input token, must be divisible by (m*k)
        out_features (int): the number of features in the output token, must be divisible by (m*n)

    Returns:
        int: output flops per token
    """
    flops_per_token = (2 * in_features * out_features) // m + (in_features * n) // k
    n_weights = (in_features * out_features) // int(m**2)
    flops_per_token_per_weight = flops_per_token / n_weights
    return {
        "flops_per_token": flops_per_token,
        "flops_per_token_per_weight": flops_per_token_per_weight,
        "n_weights": n_weights,
    }


class BatchPemuteLinear(nn.Module):
    """
    Linear layer that takes input vectors of size n=mb and applies transformations/permutations
    to them to produce another vector of size n=mb
    """

    n: int
    m: int
    b: int
    weights: list[torch.Tensor]

    def __init__(
        self, n: int, b: int, device: torch.device = None, dtype: torch.dtype = None
    ) -> None:
        super().__init__()
        assert n % b == 0, "b must divide n"
        assert math.sqrt(n) == int(math.sqrt(n)), "n must be a perfect square"
        assert b**2 / math.sqrt(n) >= 1, "b^2 must be >= sqrt(n)"
        assert int(b**2) % int(math.sqrt(n)) == 0, "b^2 must be divisible by sqrt(n)"
        self.n = n
        self.b = b
        self.m = n // b
        self.num_stages = 2 * int(b**2) // int(math.sqrt(n))
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.m, self.m, device=device, dtype=dtype))
                for i in range(self.num_stages)
            ]
        )

    def reset_parameters(self):
        for i in range(self.num_stages):
            nn.init.kaiming_uniform_(self.weights[i], a=math.sqrt(5))

    def forward(self, x):
        # input is (B, T, n)
        B, T, n = x.shape
        assert n == self.n
        x = x.view(B * T * self.b, self.m)  # (B*T*b, m) get the m-size vectors
        for i in range(self.num_stages):
            x = x.matmul(
                self.weights[i]
            )  # (B*T*b, m) I think pytorch likes square matrices more
            x = (
                x.view(B, T, self.b, self.m)
                .transpose(-1, -2)
                .reshape(B * T * self.b, self.m)
            )
        x = x.view(B, T, self.n)
        return x

    def __repr__(self):
        return f"BatchPemuteLinear(n={self.n}, b={self.b}, m={self.m}, num_stages={self.num_stages})"


def test_batch_permute_linear():
    n = 1024
    b = 8
    a = BatchPemuteLinear(n, b)
    m = n // b
    print(list(a.parameters()))
    assert sum(p.numel() for p in a.parameters()) == int(
        2 * n ** (3 / 2)
    ), "Correct number of parameters"
    for i in range(a.num_stages):
        a.weights[i] = torch.eye(n // b, n // b)
    B, T = 4, 16
    inp = torch.randn(B, T, n)
    out = a(inp)
    print(inp.shape, out.shape)
    assert torch.allclose(inp, out), "Identity weights yield identity"

    mat = torch.randn(m, m)
    res1 = (inp.view(B, T * b, m) @ mat).view(B, T, m, b)
    res2 = (inp.view(B, T, b, m) @ mat).view(B, T, m, b)
    assert torch.allclose(res1, res2), "Matmul is the same as batch matmul"


class BlockLinear(nn.Module):
    """
    Linear layer that takes input data as a sequence of m x k matrices and
    transforms it into another sequence of m x n matrices
    """

    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        m: int = 8,
        k: int = 16,
        n: int = 32,
        bias=True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert (
            in_features % m * k == 0 and out_features % n * m == 0
        ), "in_features and out_features must be multiples of m*k and n*m"
        self.in_features = in_features
        self.out_features = out_features
        self.m = m
        self.n = n
        self.k = k
        self.in_mats = in_features // (m * k)
        self.out_mats = out_features // (m * n)
        self.weight = nn.Parameter(
            torch.empty(self.out_mats, self.in_mats, k, n, device=device, dtype=dtype)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.out_mats, m, n, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def weight_per_input_ratio(self):
        return self.weight.numel() / self.in_features

    def flops_per_token(self):
        return block_linear_stats(
            self.m, self.k, self.n, self.in_features, self.out_features
        )["flops_per_token"]

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            # bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            # nn.init.uniform_(self.bias, -bound, bound)
            nn.init.zeros_(self.bias)

    def forward(self, input):
        # input is (B, T, in_features)
        B, T, _ = input.shape
        input = input.view(B * T, self.in_mats, self.m, self.k)
        res = block_linear(input, self.weight)
        if self.bias is not None:
            res += self.bias
        return res.view(B, T, self.out_features)

    def __repr__(self):
        return f"BlockLinear(in_features={self.in_features}, out_features={self.out_features}, m={self.m}, k={self.k}, n={self.n}, bias={self.bias is not None})"


@torch.jit.script
def block_linear(x, W):
    """
    x: [B, in_mat, m, k]
    W: [out_mat, in_mat, k, n]
    output: [B, out_mat, m, n]
    """
    B, in_mat, m, k = x.shape
    out_mat, in_mat, k, n = W.shape
    out = torch.empty(B, out_mat, m, n, device=x.device, dtype=x.dtype)
    for i in range(out_mat):
        Wi = W[i].unsqueeze(0)
        out[:, i] = torch.matmul(x, Wi).sum(1)
    return out


def test_block_linear():
    with torch.no_grad():
        in_f = 512
        out_f = 1024
        m, n, k = 16, 16, 16
        in_mat = in_f // (m * k)
        out_mat = out_f // (m * n)
        B = 4
        a = torch.randn(B, in_mat, m, k)
        b = torch.randn(out_mat, in_mat, k, n)
        res2 = block_linear(a, b).squeeze()
        for j in range(B):
            xj = a[j]
            for i in range(b.size(0)):
                Wi = b[i]
                res = torch.matmul(xj, Wi).sum(0)
                assert torch.allclose(res, res2[j, i])

        # test the flops per token calculation
        stats = block_linear_stats(m, k, n, in_f, out_f)
        assert stats["flops_per_token"] == 2 * in_f * out_f // m + in_f * n // k
        assert stats["flops_per_token_per_weight"] >= 2 * m


def test_model_flops_per_token():
    cfg = GPTConfig(n_layer=4, n_embd=512, bias=False, block_linear=True, n_head=4)
    model = GPT(cfg)
    flops_per_token = model.estimate_flops_per_token()
    # block_m - 1 because it should be a little less
    assert flops_per_token >= 2 * (cfg.block_m - 1) * model.get_num_params()


def t_block_attn_mask(t, N):
    # Create a matrix where each block of size t along the diagonal is 0 and everything else is 1
    block_mask = torch.eye(N).repeat_interleave(t, dim=0).repeat_interleave(t, dim=1)
    # Create a cumulative mask that allows attending to all previous tokens (lower triangular)
    cum_mask = torch.tril(torch.ones(t * N, t * N), diagonal=0)
    boolean_mask = (block_mask + cum_mask) > 0
    return boolean_mask


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        if config.t_expand_attn_mask and config.t_expand > 1:
            self.register_buffer(
                "attn_mask", t_block_attn_mask(config.t_expand, config.block_size)
            )
        else:
            self.attn_mask = None
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.dao_flash = False
        if (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability()[0] >= 8
            and not (
                config.t_expand_attn_mask and config.t_expand > 1
            )  # can't use dao attention with t_block_attn_mask
        ):
            try:
                import flash_attn

                self.flash = flash_attn.flash_attn_qkvpacked_func
                print("Using dao flash attention")
                self.dao_flash = True
            except Exception:
                print("Unable to load flash-attn.")
        else:
            print(
                "Device does not support DAO flash-attn or t_block_attn_mask is enabled"
            )

    def forward(self, x):
        # batch, sequence, n_embd
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)  # shape (B, T, 3 * n_embd)

        if self.dao_flash:
            y = self.flash(
                qkv.view(B, T, 3, self.n_head, C // self.n_head),
                causal=True,
                dropout_p=self.dropout if self.training else 0.0,
            )
            y = y.view(B, T, C)
        else:
            q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
            # (B, nh, T, hs)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
            # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=self.attn_mask[:T, :T]
                if self.attn_mask is not None
                else None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=self.attn_mask is None,  # if not set, just do causal
            )
            y = (
                y.transpose(1, 2).contiguous().view(B, T, C)
            )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.dropout = nn.Dropout(config.dropout)
        self.gelu = nn.GELU()
        if config.bpl:
            assert config.mlp_ratio == 1, "bpl only works with mlp_ratio=1"
            assert not config.bias, "bpl only works with bias=False"
            self.c_fc = BatchPemuteLinear(config.n_embd, config.bpl_b)
            self.c_proj = BatchPemuteLinear(config.n_embd, config.bpl_b)
        else:
            LinCls = BlockLinear if config.block_linear else nn.Linear
            args = {"bias": config.bias}
            if config.block_linear:
                args = args | {
                    "m": config.block_m,
                    "n": config.block_n,
                    "k": config.block_k,
                }
            args = {"bias": config.bias}
            # here we adjust the middle size for block linear to maintain the same number of weights
            # and get purely more flops/token from block_m
            middle_size = (
                config.n_embd
                * config.mlp_ratio
                * (config.block_m**2 if config.block_linear else 1)
            )
            self.c_fc = LinCls(config.n_embd, middle_size, **args)
            self.c_proj = LinCls(middle_size, config.n_embd, **args)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd * config.t_expand),
                wpe=nn.Embedding(config.block_size * config.t_expand, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        if config.t_expand == 1 or not config.t_expand_sep_lm_head:
            self.lm_head = nn.Linear(
                config.n_embd * config.t_expand, config.vocab_size, bias=False
            )
            self.transformer.wte.weight = (
                self.lm_head.weight
            )  # https://paperswithcode.com/method/weight-tying
        else:
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
            if self.config.t_expand != 1 and self.config.t_expand_sep_lm_head:
                # we aren't weight sharing so remove the wte embeddings
                n_params -= self.transformer.wte.weight.numel()

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, BlockLinear):  # TODO: figure out how to init these
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.mlp_init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, BatchPemuteLinear):
            for elem in module.weights:
                torch.nn.init.normal_(elem, mean=0.0, std=self.config.mlp_init_std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(
            0, t * self.config.t_expand, dtype=torch.long, device=device
        )  # shape (t * expand)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # shape (b, t, n_embd * expand)
        pos_emb = self.transformer.wpe(pos)  # (t * expand, n_embd)
        tok_expanded = tok_emb.view(b, t * self.config.t_expand, self.config.n_embd)
        x = self.transformer.drop(tok_expanded + pos_emb)
        for block in self.transformer.h:
            x = block(x)  # shape (b, t * expand, n_embd)
        x = self.transformer.ln_f(x)
        if self.config.t_expand == 1 or not self.config.t_expand_sep_lm_head:
            x = x.view(b, t, self.config.n_embd * self.config.t_expand)
        else:
            every = self.config.t_expand
            x = x[:, every - 1 :: every, :]  # (b, t, n_embd)
            assert x.shape[1] == t

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(
                x[:, [-1], :]
            )  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        assert self.config.t_expand == 1, "not implemented for t_expand > 1"
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(
            self.transformer.wpe.weight[:block_size]
        )
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        override_args = override_args or {}  # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k == "dropout" for k in override_args)
        from transformers import GPT2LMHeadModel

        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M params
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M params
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M params
        }[model_type]
        print("forcing vocab_size=50257, block_size=1024, bias=True")
        config_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
        config_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
        config_args["bias"] = True  # always True for GPT model checkpoints
        # we can override the dropout rate, if desired
        if "dropout" in override_args:
            print(f"overriding dropout rate to {override_args['dropout']}")
            config_args["dropout"] = override_args["dropout"]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [
            k for k in sd_keys if not k.endswith(".attn.bias")
        ]  # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")
        ]  # ignore these, just a buffer
        sd_keys_hf = [
            k for k in sd_keys_hf if not k.endswith(".attn.bias")
        ]  # same, just the mask (buffer)
        transposed = [
            "attn.c_attn.weight",
            "attn.c_proj.weight",
            "mlp.c_fc.weight",
            "mlp.c_proj.weight",
        ]
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(
            sd_keys
        ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(
            f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
        )
        print(
            f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
        )
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(
            optim_groups, lr=learning_rate, betas=betas, **extra_args
        )
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        flops_per_token = self.estimate_flops_per_token()
        flops_per_fwdbwd = flops_per_token * self.config.block_size
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    def estimate_flops_per_token(self):
        """
        estimate the flops per token for a forward + backward pass through the model
        """
        cfg = self.config
        flops_per_token = cfg.n_embd  # embedding backprop
        if cfg.block_linear:
            for module in self.modules():
                if isinstance(module, BlockLinear):
                    flops_per_token += (
                        3 * module.flops_per_token()
                    )  # 1 for fwd, 2 for bwd
                elif isinstance(module, nn.Linear):
                    flops_per_token += 6 * module.weight.numel()
        else:
            flops_per_token = 6 * self.get_num_params()  # 2 for fwd, 4 for bwd
        L = cfg.n_layer
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        flops_per_token += 12 * L * H * Q * Q
        return self.config.t_expand * flops_per_token

    def estimate_flops_per_token_per_weight(self):
        """
        estimate the flops per token per weight for a forward + backward pass through the model
        """
        return self.estimate_flops_per_token() / self.get_num_params()

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def test_gpt_flops_per_token_per_weight():
    cfg = GPTConfig(
        n_layer=4,
        n_embd=256,
        bias=False,
        n_head=8,
        block_m=16,
        block_linear=True,
        block_k=16,
        block_n=16,
        mlp_ratio=4,
        block_size=256,
        vocab_size=50304,
    )
    model = GPT(cfg)
    flops_per_token_per_weight = model.estimate_flops_per_token_per_weight()
    num_params = model.get_num_params()
    assert (
        round(flops_per_token_per_weight) == 18
    ), "Should be 18 flops per token per weight"
    cfg.block_linear = False
    model = GPT(cfg)
    flops_per_token_per_weight = model.estimate_flops_per_token_per_weight()
    assert (
        round(flops_per_token_per_weight) == 6
    ), "Should be 6 flops per token per weight"
    assert model.get_num_params() == num_params, "Should have same number of params"

    cfg.n_layer = 6
    cfg.n_embd = 512
    cfg.vocab_size = 512  # super shrink this so we actually see the flop increase
    for t_expand in [2, 4, 8]:
        cfg.t_expand = t_expand
        model = GPT(cfg)
        flops_per_token_per_weight = model.estimate_flops_per_token_per_weight()
        assert round(flops_per_token_per_weight) > 6 * (t_expand - 1)
        print((cfg.n_embd * cfg.t_expand * cfg.vocab_size) / model.get_num_params())
        assert model.get_num_params() <= (num_params * 1.5), "Should a few more params"
