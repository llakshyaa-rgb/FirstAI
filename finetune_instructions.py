"""Supervised instruction tuning for FirstGPT checkpoints.

JSONL format, one object per line:
{"prompt":"Owner: ...","response":"Orbit: ..."}
This small educational SFT stage is not a substitute for a large curated dataset.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import random
from pathlib import Path
import torch
from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer


def load_examples(path):
    examples = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
        prompt, response = item.get("prompt"), item.get("response")
        if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
            raise ValueError(f"Line {line_no} needs non-empty string fields 'prompt' and 'response'")
        examples.append((prompt.strip(), response.strip()))
    if len(examples) < 4:
        raise ValueError("At least four instruction examples are required for train/validation splitting")
    return examples


def encode_examples(examples, tokenizer, block_size):
    rows = []
    for prompt, response in examples:
        prefix = tokenizer.encode(prompt + "\nOrbit:")
        answer = tokenizer.encode(" " + response + "\n")
        if len(answer) >= block_size:
            answer = answer[-(block_size - 1):]
            prefix = []
        else:
            prefix = prefix[-(block_size - len(answer)):]
        ids = prefix + answer
        if len(ids) >= 2:
            rows.append((ids, len(prefix)))
    if len(rows) < 2:
        raise ValueError("Not enough usable examples after applying block_size")
    return rows


def make_batch(rows, batch_size, block_size, device):
    selected = [rows[random.randrange(len(rows))] for _ in range(batch_size)]
    x = torch.zeros((batch_size, block_size), dtype=torch.long)
    y = torch.full((batch_size, block_size), -100, dtype=torch.long)
    for i, (tokens, prompt_length) in enumerate(selected):
        row = tokens[-(block_size + 1):]
        # If a row is trimmed, adjust the response boundary by the removed prefix.
        removed = max(0, len(tokens) - len(row))
        boundary = max(0, prompt_length - removed)
        inp, target = row[:-1], row[1:]
        x[i, :len(inp)] = torch.tensor(inp)
        # Only response tokens contribute to the supervised objective.
        for target_pos, token in enumerate(target):
            original_pos = target_pos + 1
            if original_pos >= boundary:
                y[i, target_pos] = token
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser(description="Supervised instruction tuning for a FirstGPT checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/firstgpt.best.pt")
    parser.add_argument("--data", default="data/instructions.jsonl")
    parser.add_argument("--out", default="checkpoints/orbit-instruct.pt")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_interval) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, batch-size, eval-interval and lr must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    examples = load_examples(args.data)
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    random.shuffle(examples)
    split = max(2, min(len(examples) - 2, int(len(examples) * 0.9)))
    train_raw, val_raw = examples[:split], examples[split:]

    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    tokenizer = BPETokenizer.from_dict(saved["tokenizer"])
    config = GPTConfig(**saved["config"])
    train_rows = encode_examples(train_raw, tokenizer, config.block_size)
    val_rows = encode_examples(val_raw, tokenizer, config.block_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FirstGPT(config).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()

    @torch.no_grad()
    def evaluate():
        model.eval()
        values = []
        for _ in range(min(30, max(5, len(val_rows)))):
            x, y = make_batch(val_rows, min(args.batch_size, len(val_rows)), config.block_size, device)
            logits, _ = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
            )
            values.append(float(loss.item()))
        model.train()
        return sum(values) / len(values)

    best = float("inf")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    for step in range(1, args.steps + 1):
        x, y = make_batch(train_rows, args.batch_size, config.block_size, device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss; stopping without saving a broken checkpoint")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            val = evaluate()
            print(f"Step {step:>6} | train {loss.item():.4f} | validation {val:.4f}")
            if val < best:
                best = val
                payload = dict(saved)
                payload.update({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": int(saved.get("step", 0)) + step,
                    "best_validation_loss": best,
                    "instruction_data_sha256": digest,
                    "instruction_tuning_steps": step,
                })
                temporary = target.with_suffix(target.suffix + ".tmp")
                torch.save(payload, temporary)
                temporary.replace(target)
    print(f"Instruction-tuned checkpoint: {args.out}")
    print(f"Device: {device}; examples: {len(examples)}; parameters: {sum(p.numel() for p in model.parameters()):,}")
