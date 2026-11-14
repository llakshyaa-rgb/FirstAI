"""Decoder-only GPT model with custom Transformer architecture and PyTorch tensor/autograd."""
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

    def __post_init__(self):
        if min(self.vocab_size, self.block_size, self.n_layer, self.n_head, self.n_embd) < 1:
            raise ValueError("Model dimensions and vocabulary size must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
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
        # PyTorch can select fused/Flash attention kernels on supported GPUs.
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
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd), nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd), nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1, self.attn = nn.LayerNorm(config.n_embd), CausalSelfAttention(config)
        self.ln2, self.mlp = nn.LayerNorm(config.n_embd), MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class FirstGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Tie after initialization so the shared matrix is not initialized twice.
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
        positions = torch.arange(length, device=input_ids.device)
        x = self.dropout(self.token_embedding(input_ids) + self.position_embedding(positions))
        logits = self.lm_head(self.final_norm(self.blocks(x)))
        loss = None
        if targets is not None:
            if targets.shape != input_ids.shape:
                raise ValueError("targets must have the same shape as input_ids")
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("generation prompt must have shape [batch, non-empty sequence]")
        if max_new_tokens < 0 or temperature <= 0:
            raise ValueError("max_new_tokens must be nonnegative and temperature positive")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive")
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < values[:, [-1]], -float("inf"))
            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=1)
        return input_ids
