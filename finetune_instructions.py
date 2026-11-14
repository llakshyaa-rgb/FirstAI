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
    if len(examples) < 10:
        raise ValueError("At least ten instruction examples are required for a stable train/validation split")
    return examples


def encode_examples(examples, tokenizer, block_size):
    rows = []
    for prompt, response in examples:
        prefix = tokenizer.encode(prompt + "\nOrbit:")
        answer_text = response.strip()
        if answer_text.startswith("Orbit:"):
            answer_text = answer_text[len("Orbit:"):].lstrip()
        answer = tokenizer.encode(" " + answer_text + "\n")
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
    parser.add_argument("--resume", action="store_true", help="Resume model and optimizer from --out.")
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_interval) < 1 or not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("steps, batch-size, eval-interval and lr must be positive")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    examples = load_examples(args.data)
    digest = hashlib.sha256(Path(args.data).read_bytes()).hexdigest()
    random.shuffle(examples)
    split = max(4, min(len(examples) - 4, int(len(examples) * 0.9)))
    train_raw, val_raw = examples[:split], examples[split:]

    if args.resume:
        if not Path(args.out).is_file():
            raise FileNotFoundError(f"Cannot resume: {args.out} does not exist")
        saved = torch.load(args.out, map_location="cpu", weights_only=True)
        if saved.get("instruction_data_sha256") != digest:
            raise ValueError("Instruction dataset changed since checkpoint; use a new --out path or start fresh.")
        start_step = int(saved.get("instruction_tuning_steps", 0))
        best = float(saved.get("best_validation_loss", float("inf")))
    else:
        saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        start_step, best = 0, float("inf")
    tokenizer = BPETokenizer.from_dict(saved["tokenizer"])
    config = GPTConfig(**saved["config"])
    train_rows = encode_examples(train_raw, tokenizer, config.block_size)
    val_rows = encode_examples(val_raw, tokenizer, config.block_size)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FirstGPT(config).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.resume and saved.get("optimizer"):
        optimizer.load_state_dict(saved["optimizer"])
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

    target = Path(args.out)
    best_path = target.with_name(target.stem + ".best" + target.suffix)
    print(f"Device: {device}; examples: {len(examples):,} (train {len(train_raw):,}, validation {len(val_raw):,}); starting step: {start_step}")
    target.parent.mkdir(parents=True, exist_ok=True)
    for step in range(start_step, start_step + args.steps):
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
        completed = step + 1
        if completed % args.eval_interval == 0 or completed == start_step + args.steps:
            val = evaluate()
            print(f"Step {completed:>6} | train {loss.item():.4f} | validation {val:.4f}")
            payload = dict(saved)
            payload.update({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": int(saved.get("step", 0)) + (completed - start_step),
                "best_validation_loss": min(best, val),
                "instruction_data_sha256": digest,
                "instruction_tuning_steps": completed,
                "instruction_examples": len(examples),
            })
            temporary = target.with_suffix(target.suffix + ".tmp")
            torch.save(payload, temporary)
            temporary.replace(target)
            if val < best:
                best = val
                payload["best_validation_loss"] = best
                best_tmp = best_path.with_suffix(best_path.suffix + ".tmp")
                torch.save(payload, best_tmp)
                best_tmp.replace(best_path)
                print(f"  saved best checkpoint: {best_path}")
            saved = payload
    print(f"Latest resumable SFT checkpoint: {args.out}")
    print(f"Best validation checkpoint: {best_path}")
    print(f"Device: {device}; examples: {len(examples)}; parameters: {sum(p.numel() for p in model.parameters()):,}")
