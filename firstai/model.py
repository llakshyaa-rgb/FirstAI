"""FirstGPT decoder-only Transformer.

Architecture v1 remains available for historical checkpoints. New runs use
architecture v2: RMSNorm, rotary position embeddings and SwiGLU feed-forward blocks.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 1024
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.1
    architecture_version: int = 1

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("Model dimensions and vocabulary size must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.architecture_version not in (1, 2):
            raise ValueError("architecture_version must be 1 or 2")
        if self.architecture_version == 2 and (self.n_embd // self.n_head) % 2:
            raise ValueError("Architecture v2 requires an even attention head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return normed.to(dtype=x.dtype) * self.weight


def apply_rotary(q, k):
    length, dim = q.shape[-2], q.shape[-1]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=q.device, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(length, device=q.device, dtype=torch.float32), inv_freq)
    cos = torch.repeat_interleave(angles.cos().to(q.dtype), 2, dim=-1)[None, None, :, :]
    sin = torch.repeat_interleave(angles.sin().to(q.dtype), 2, dim=-1)[None, None, :, :]

    def rotate(x):
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        return x * cos + rotated * sin

    return rotate(q), rotate(k)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.architecture_version = config.architecture_version
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        if self.architecture_version == 1:
            self.register_buffer("causal_mask", torch.tril(
                torch.ones(config.block_size, config.block_size)
            ).view(1, 1, config.block_size, config.block_size), persistent=False)

    def forward(self, x):
        batch, length, channels = x.shape
        q, k, v = self.qkv(x).split(self.n_embd, dim=2)
        head_dim = channels // self.n_head
        q = q.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        k = k.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        v = v.view(batch, length, self.n_head, head_dim).transpose(1, 2)
        if self.architecture_version == 2:
            q, k = apply_rotary(q, k)
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        else:
            causal = self.causal_mask[:, :, :length, :length].bool()
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=causal,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
        out = out.transpose(1, 2).contiguous().view(batch, length, channels)
        return self.resid_dropout(self.proj(out))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = None
        if config.architecture_version == 2:
            hidden = ((8 * config.n_embd // 3 + 63) // 64) * 64
            self.gate = nn.Linear(config.n_embd, hidden, bias=False)
            self.up = nn.Linear(config.n_embd, hidden, bias=False)
            self.down = nn.Linear(hidden, config.n_embd, bias=False)
        else:
            self.net = nn.Sequential(
                nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
                nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout),
            )

    def forward(self, x):
        if self.net is not None:
            return self.net(x)
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        norm = RMSNorm if config.architecture_version == 2 else nn.LayerNorm
        self.ln1, self.attn = norm(config.n_embd), CausalSelfAttention(config)
        self.ln2, self.mlp = norm(config.n_embd), MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class FirstGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = (
            nn.Embedding(config.block_size, config.n_embd)
            if config.architecture_version == 1 else None
        )
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        norm = RMSNorm if config.architecture_version == 2 else nn.LayerNorm
        self.final_norm = norm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, targets=None):
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        _, length = input_ids.shape
        if not 1 <= length <= self.config.block_size:
            raise ValueError("Input length must be between 1 and block_size")
        x = self.token_embedding(input_ids)
        if self.position_embedding is not None:
            x = x + self.position_embedding(torch.arange(length, device=input_ids.device))
        logits = self.lm_head(self.final_norm(self.blocks(self.dropout(x))))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None, top_p=1.0, repetition_penalty=1.0):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("generation prompt must have shape [batch, non-empty sequence]")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be nonnegative and temperature positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        if not 0 < top_p <= 1 or repetition_penalty < 1:
            raise ValueError("top_p must be in (0, 1] and repetition_penalty must be >= 1")
        for _ in range(max_new_tokens):
            logits, _ = self(self._context(input_ids))
            logits = logits[:, -1, :]
            if repetition_penalty > 1:
                for row in range(input_ids.shape[0]):
                    seen = input_ids[row].unique()
                    selected = logits[row, seen]
                    logits[row, seen] = torch.where(selected > 0, selected / repetition_penalty, selected * repetition_penalty)
            logits = logits / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            if top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                remove = cumulative > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                filtered = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.full_like(logits, -float("inf")).scatter(-1, sorted_indices, filtered)
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids

    def _context(self, input_ids):
        return input_ids[:, -self.config.block_size:]
