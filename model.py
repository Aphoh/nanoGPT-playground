"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from functools import partial
from config import GPTConfig


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


@torch.jit.script
def block_linear_1_step(x, w):
    """x: [batch, b, m]
    w: [m, k*m]
    """
    out1 = x.matmul(w)
    out1 = out1.transpose(-1, -2).reshape(x.shape[:2] + (w.shape[1],))
    return out1


@torch.jit.script
def block_linear_1(x, w1, w2, b: int = 8):
    """
    x: [..., n]
    w1: [n/b, k/b]
    w2: [k/b, k/b]
    outputs: [..., k]
    """
    n = x.shape[-1]
    assert n % int(b**2) == 0
    xv = x.view(-1, b, n // b)
    out1 = block_linear_1_step(xv, w1)
    out1 = block_linear_1_step(out1, w2)
    return out1.reshape(x.shape[:-1] + (-1,))


@torch.jit.script
def block_linear_2_step(x, w):
    """
    x: [batch, b, m]
    w: [m, k*m]
    """
    batch, b, _ = x.shape
    _, m2 = w.shape
    assert m2 % b == 0, f"w.shape[1]={m2} must be divisible by b={b}"
    out1 = x.matmul(w)
    out1 = out1.view((batch, b, m2 // b, b)).transpose(-3, -1)
    return out1.reshape((batch, b, m2))


@torch.jit.script
def block_linear_2(x, w1, w2, b: int = 8):
    """
    x: [..., n]
    w1: [n/b, k/b]
    w2: [k/b, k/b]
    outputs: [..., k]
    """
    assert x.shape[-1] % int(b**2) == 0
    xv = x.view(-1, b, x.shape[-1] // b)
    out1 = block_linear_2_step(xv, w1)
    out1 = block_linear_2_step(out1, w2)
    return out1.view(x.shape[:-1] + (-1,))


@torch.jit.script
def monarch(x, w1, w2):
    xv = x.view(-1, x.shape[-1])
    batch, n = xv.shape
    nblocks, w1_out, w1_in = w1.shape
    d, w2_out, w2_in = w2.shape
    assert nblocks == d
    assert n % nblocks == 0
    assert n // nblocks == w1_in
    assert w1_out == w2_in
    xv = xv.view(batch, nblocks, n // nblocks)

    # k qp matrices doing a mvm with bk p-length vectors
    out1 = torch.einsum("kqp,bkp->bkq", w1, xv)
    out1 = out1.view(batch, w1_out, nblocks).transpose(-1, -2)
    # same as before, but we output the transpose (qk instead of kq)
    out1 = torch.einsum("kqp,bkp->bqk", w2, out1)
    return out1.reshape(x.shape[:-1] + (w2_out * nblocks,))
    # out1 = rearrange(rearrange(out1, "b k q -> b (k q)"), "b (r l) -> b l r", l=l)


def test_monarch_jit():
    from einops import rearrange

    n_blocks, n, seq, batch = 4, 256, 16, 5
    x = torch.randn(batch, seq, n)
    w1 = torch.randn(n_blocks, 2 * n // n_blocks, n // n_blocks)
    w2 = torch.randn(n_blocks, 4 * n // n_blocks, 2 * n // n_blocks)
    k, q, p = w1.shape
    l, s, r = w2.shape
    assert k * p == n
    assert l * r == k * q
    x_reshaped = x.view(-1, n)
    w1_dense = torch.block_diag(*torch.unbind(w1, dim=0))
    out1 = F.linear(x_reshaped, w1_dense)
    out1 = rearrange(out1, "b (r l) -> b (l r)", l=l)
    w2_dense = torch.block_diag(*torch.unbind(w2, dim=0))
    out2 = F.linear(out1, w2_dense)
    out2 = rearrange(out2, "b (l s) -> b (s l)", l=l)
    out2 = out2.view(batch, seq, -1)

    assert torch.allclose(out2, monarch(x, w1, w2))


class MonarchLinear(nn.Module):
    n_blocks: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_blocks: int = 4,
        bias: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_blocks = n_blocks
        assert (
            in_features % n_blocks == 0
        ), f"in_features={in_features} n_blocks={n_blocks}"
        assert (
            out_features % n_blocks == 0
        ), f"out_features={out_features} n_blocks={n_blocks}"
        ib = in_features // n_blocks
        ob = out_features // n_blocks
        if in_features > out_features:  # we do right matmuls here for some reason
            self.w1 = nn.Parameter(torch.empty(n_blocks, ib, ib))
            self.w2 = nn.Parameter(torch.empty(n_blocks, ob, ib))
        else:
            self.w1 = nn.Parameter(torch.empty(n_blocks, ob, ib))
            self.w2 = nn.Parameter(torch.empty(n_blocks, ob, ob))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def forward(self, x):
        return monarch(x, self.w1, self.w2)

    def reset_parameters(self) -> None:
        for w in [self.w1, self.w2]:
            fan_in = w.shape[-1]
            gain = nn.init.calculate_gain(nonlinearity="leaky_relu", param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = (
                math.sqrt(3.0) * std
            )  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                w.uniform_(-bound, bound)
        if self.bias is not None:
            fan_in = self.bias.shape[-1]
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, n_blocks={self.n_blocks}"


class BlockLinear(nn.Module):
    """
    Linear layer that takes input vectors of size n=mb and applies transformations/permutations
    to them to produce another vector of size n=k*mb
    """

    b: int
    m1: int
    m2: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        b: int,
        version: int = 1,
        bias: bool = False,
        device: torch.device = None,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        assert (
            in_features % int(b**2) == 0
        ), f"(b={b})^2 must divide in_features={in_features}"
        assert (
            out_features % int(b**2) == 0
        ), f"(b={b})^2 must divide out_features={out_features}"
        self.in_features = in_features
        self.out_features = out_features
        self.m1 = in_features // b
        self.m2 = out_features // b
        self.b = b
        kwargs = {"device": device, "dtype": dtype}
        if self.in_features < self.out_features:
            self.w1 = nn.Parameter(torch.empty(self.m1, self.m2, **kwargs))
            self.w2 = nn.Parameter(torch.empty(self.m2, self.m2, **kwargs))
        else:
            self.w1 = nn.Parameter(torch.empty(self.m1, self.m1, **kwargs))
            self.w2 = nn.Parameter(torch.empty(self.m1, self.m2, **kwargs))
        self.block_linear = block_linear_1 if version == 1 else block_linear_2
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for mat in [self.w1, self.w2]:
            torch.nn.init.kaiming_uniform_(mat)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):
        x = self.block_linear(x, self.w1, self.w2, b=self.b)
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, b={self.b}"


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        LinCls = nn.Linear
        if config.block_linear and not config.bl_only_mlp:
            LinCls = partial(BlockLinear, b=config.bl_b, version=config.bl_version)
        elif config.monarch_linear:
            LinCls = partial(MonarchLinear, n_blocks=config.monarch_n_blocks)

        self.c_attn = LinCls(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = LinCls(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0

    def forward(self, x):
        # batch, sequence, n_embd
        B, T, C = x.shape

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
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
            dropout_p=self.dropout if self.training else 0,
            is_causal=True,
        )
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        LinCls = nn.Linear
        if cfg.block_linear:
            LinCls = partial(BlockLinear, b=cfg.bl_b, version=cfg.bl_version)
        elif cfg.monarch_linear:
            LinCls = partial(MonarchLinear, n_blocks=cfg.monarch_n_blocks)

        dim_mlp = cfg.n_embd * cfg.mlp_ratio
        self.mlp = nn.Sequential(
            LinCls(cfg.n_embd, dim_mlp, bias=cfg.bias),
            nn.GELU(),
            LinCls(dim_mlp, cfg.n_embd, bias=cfg.bias),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.mlp(x)


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
    config: GPTConfig

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        if not config.block_linear:
            for pn, p in self.named_parameters():
                if pn.endswith("c_proj.weight"):
                    torch.nn.init.normal_(
                        p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                    )

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True, non_lm_head=False):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        if non_lm_head:
            n_params -= self.lm_head.weight.numel()

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, BlockLinear):
            module.reset_parameters()
        if isinstance(module, MonarchLinear):
            module.reset_parameters()
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert (
            t <= self.config.block_size
        ), f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t * expand)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        tok_expanded = tok_emb.view(b, t, self.config.n_embd)
        x = self.transformer.drop(tok_expanded + pos_emb)
        for block in self.transformer.h:
            x = block(x)  # shape (b, t, n_embd)
        x = self.transformer.ln_f(x)
        x = x.view(b, t, self.config.n_embd)

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

    def configure_optimizers(
        self, weight_decay, learning_rate, betas, device_type
    ) -> torch.optim.Optimizer:
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
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=learning_rate,
            betas=betas,
            fused=device_type == "cuda",
        )
        print(f"using fused AdamW: {device_type=='cuda'}")

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
        flops_per_token = 0
        if cfg.block_linear:
            emb_weights = self.transformer.wte.weight.numel()
            non_emb_weights = self.get_num_params(non_embedding=True) - emb_weights
            flops_per_token += 6 * (cfg.bl_b * non_emb_weights + emb_weights)
        else:
            flops_per_token += 6 * self.get_num_params()  # 2 for fwd, 4 for bwd
        L = cfg.n_layer
        H = cfg.n_head
        Q = cfg.n_embd // cfg.n_head
        flops_per_token += 12 * L * H * Q * Q
        return flops_per_token

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
        block_linear=True,
        bl_b=16,
        mlp_ratio=4,
        block_size=256,
        vocab_size=50304,
    )
    model = GPT(cfg)
    flops_per_token_per_weight = model.estimate_flops_per_token_per_weight()
    assert (
        round(flops_per_token_per_weight) == 13
    ), "Should be 18 flops per token per weight"
    cfg.block_linear = False
    model = GPT(cfg)
    flops_per_token_per_weight = model.estimate_flops_per_token_per_weight()
    assert (
        round(flops_per_token_per_weight) == 6
    ), "Should be 6 flops per token per weight"
