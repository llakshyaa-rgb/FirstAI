"""Epoch-based, resumable supervised fine-tuning for FirstGPT / Orbit.

Every training example is visited exactly once per epoch. Validation is evaluated
over the complete held-out split, with token-weighted loss. Checkpoints are
written atomically and can resume at the exact next row of the current epoch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer


def load_examples(path):
    examples = []
    with Path(path).open("r", encoding="utf-8", newline="") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            prompt, response = item.get("prompt"), item.get("response")
            if (
                not isinstance(prompt, str) or not prompt.strip()
                or not isinstance(response, str) or not response.strip()
            ):
                raise ValueError(
                    f"Line {line_no} needs non-empty string fields 'prompt' and 'response'"
                )
            examples.append((prompt.strip(), response.strip()))
    if len(examples) < 20:
        raise ValueError("At least 20 examples are required for a stable train/validation split.")
    return examples


def encode_examples(examples, tokenizer, block_size):
    """Encode prompts and complete answers; never truncate an answer."""
    import re

    rows = []
    stats = {
        "unchanged": 0,
        "history_turns_removed": 0,
        "long_current_prompt_compacted": 0,
    }

    for row_number, (prompt, response) in enumerate(examples, 1):
        normalized_prompt = prompt.strip()
        if not normalized_prompt.startswith(("Owner:", "Tool planner:")):
            normalized_prompt = "Owner: " + normalized_prompt

        is_tool = normalized_prompt.startswith("Tool planner:")
        marker = "\nJSON:" if is_tool else "\nOrbit:"
        answer_text = response.strip()
        if answer_text.startswith("Orbit:"):
            answer_text = answer_text[len("Orbit:"):].lstrip()

        answer_text = " " + answer_text + "\n"
        try:
            answer_ids = tokenizer.encode(answer_text, add_eos=True)
        except TypeError as exc:
            if "add_eos" not in str(exc):
                raise
            answer_ids = tokenizer.encode(answer_text)

        if not answer_ids:
            raise ValueError(f"Example {row_number} has an empty encoded answer.")
        if len(answer_ids) >= block_size:
            raise ValueError(
                f"Example {row_number} answer is {len(answer_ids)} tokens; "
                f"maximum is {block_size - 1}. Filter/rebuild the dataset; "
                "answers are never silently truncated."
            )

        available = block_size - len(answer_ids)
        full_prefix = tokenizer.encode(normalized_prompt + marker)
        if len(full_prefix) <= available:
            prefix_ids = full_prefix
            stats["unchanged"] += 1
        else:
            owner_starts = [
                match.start()
                for match in re.finditer(r"(?m)^Owner:\s*", normalized_prompt)
            ]
            selected = None
            for start in owner_starts:
                candidate = tokenizer.encode(normalized_prompt[start:] + marker)
                if len(candidate) <= available:
                    selected = candidate
                    break

            if selected is not None:
                prefix_ids = selected
                stats["history_turns_removed"] += 1
            else:
                role = "Tool planner:" if is_tool else "Owner:"
                role_ids = tokenizer.encode(role + " ")
                marker_ids = tokenizer.encode(marker)
                body_ids = tokenizer.encode(normalized_prompt[len(role):].lstrip())
                body_budget = available - len(role_ids) - len(marker_ids)
                if body_budget < 2:
                    raise ValueError(
                        f"Example {row_number} leaves no room for prompt role and marker."
                    )
                if len(body_ids) <= body_budget:
                    kept_body = body_ids
                else:
                    opening = max(1, body_budget // 4)
                    closing = body_budget - opening
                    kept_body = body_ids[:opening] + body_ids[-closing:]
                prefix_ids = role_ids + kept_body + marker_ids
                stats["long_current_prompt_compacted"] += 1

        ids = prefix_ids + answer_ids
        if len(ids) > block_size:
            raise AssertionError(f"Encoded row {row_number} exceeds block_size.")
        rows.append((ids, len(prefix_ids)))

    if not rows:
        raise ValueError("No usable examples after encoding.")
    encode_examples.last_stats = stats
    return rows


def collate(rows, device):
    """Dynamic right-padding; only answer tokens receive labels."""
    max_len = max(len(tokens) - 1 for tokens, _ in rows)
    x = torch.zeros((len(rows), max_len), dtype=torch.long)
    y = torch.full((len(rows), max_len), -100, dtype=torch.long)
    for index, (tokens, prompt_length) in enumerate(rows):
        input_ids = tokens[:-1]
        x[index, :len(input_ids)] = torch.tensor(input_ids, dtype=torch.long)
        first_answer_target = prompt_length - 1
        y[index, first_answer_target:len(input_ids)] = torch.tensor(
            tokens[prompt_length:], dtype=torch.long
        )
    return x.to(device), y.to(device)


def answer_token_count(rows):
    return sum(len(tokens) - prompt_length for tokens, prompt_length in rows)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Epoch-based, fully covered, resumable Orbit instruction tuning."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--save-interval", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--min-lr", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if min(
        args.epochs, args.batch_size, args.grad_accum_steps,
        args.eval_batch_size, args.save_interval, args.patience
    ) < 1:
        parser.error("epochs, batch sizes, accumulation, intervals and patience must be positive")
    if not (math.isfinite(args.lr) and math.isfinite(args.min_lr)
            and 0 < args.min_lr <= args.lr):
        parser.error("Learning rates must satisfy 0 < min-lr <= lr.")
    if not 0 <= args.warmup_ratio < 1:
        parser.error("warmup-ratio must be in [0, 1).")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    data_path = Path(args.data)
    base_path = Path(args.checkpoint)
    out_path = Path(args.out)
    best_path = out_path.with_name(out_path.stem + ".best" + out_path.suffix)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    if not base_path.is_file() and not (args.resume and out_path.is_file()):
        raise FileNotFoundError(base_path)

    data_bytes = data_path.read_bytes()
    data_hash = hashlib.sha256(data_bytes).hexdigest()
    base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest() if base_path.exists() else None
    examples = load_examples(data_path)

    split_rng = random.Random(args.seed)
    split_rng.shuffle(examples)
    split = max(4, min(len(examples) - 4, int(len(examples) * 0.9)))
    train_raw, val_raw = examples[:split], examples[split:]

    if args.resume:
        if not out_path.is_file():
            raise FileNotFoundError(f"Cannot resume; latest checkpoint missing: {out_path}")
        saved = torch.load(out_path, map_location="cpu", weights_only=True)
        if saved.get("instruction_data_sha256") != data_hash:
            raise ValueError("Dataset hash changed. Refusing unsafe resume.")
        if saved.get("instruction_seed") != args.seed:
            raise ValueError("Resume seed differs from saved run.")
        if saved.get("instruction_epochs_target") != args.epochs:
            raise ValueError("Resume epoch target differs from saved run.")
        if saved.get("instruction_batch_size") != args.batch_size:
            raise ValueError("Resume batch size differs from saved run.")
        if saved.get("instruction_grad_accum_steps") != args.grad_accum_steps:
            raise ValueError("Resume gradient accumulation differs from saved run.")
        if saved.get("instruction_base_sha256") != base_hash:
            raise ValueError("Base checkpoint changed since this SFT run.")
        start_epoch = int(saved.get("instruction_epoch", 0))
        next_row = int(saved.get("instruction_next_row", 0))
        completed_steps = int(saved.get("instruction_tuning_steps", 0))
        best = float(saved.get("best_validation_loss", float("inf")))
        stale_epochs = int(saved.get("instruction_stale_epochs", 0))
        print(
            f"RESUMING epoch {start_epoch + 1}/{args.epochs}, "
            f"next row {next_row:,}, optimizer steps {completed_steps:,}",
            flush=True,
        )
    else:
        if out_path.exists() or best_path.exists():
            raise FileExistsError(
                f"{out_path} or {best_path} already exists. Use --resume or choose a new output."
            )
        saved = torch.load(base_path, map_location="cpu", weights_only=True)
        start_epoch, next_row, completed_steps = 0, 0, 0
        best, stale_epochs = float("inf"), 0

    tokenizer = BPETokenizer.from_dict(saved["tokenizer"])
    config = GPTConfig(**saved["config"])
    train_rows = encode_examples(train_raw, tokenizer, config.block_size)
    train_stats = dict(encode_examples.last_stats)
    val_rows = encode_examples(val_raw, tokenizer, config.block_size)
    val_stats = dict(encode_examples.last_stats)

    if len(train_rows) != len(train_raw) or len(val_rows) != len(val_raw):
        raise RuntimeError("Some examples were dropped during encoding.")
    print(
        f"DATASET: {len(examples):,} total | {len(train_rows):,} train | "
        f"{len(val_rows):,} validation | SHA256 {data_hash}",
        flush=True,
    )
    print(f"ENCODER train={train_stats} validation={val_stats}", flush=True)
    print(
        f"ANSWER TOKENS: train={answer_token_count(train_rows):,} "
        f"validation={answer_token_count(val_rows):,}",
        flush=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    elif args.amp:
        print("AMP requested but CUDA is unavailable; using CPU float32.", flush=True)

    model = FirstGPT(config).to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    batches_per_epoch = math.ceil(len(train_rows) / args.batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / args.grad_accum_steps)
    total_updates = updates_per_epoch * args.epochs
    warmup_updates = int(total_updates * args.warmup_ratio)
    lr_floor = args.min_lr / args.lr

    def lr_scale(step):
        if warmup_updates and step < warmup_updates:
            return max(1e-8, (step + 1) / warmup_updates)
        progress = min(
            1.0,
            max(0.0, (step - warmup_updates) / max(1, total_updates - warmup_updates)),
        )
        return lr_floor + (1.0 - lr_floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    use_amp = bool(args.amp and device == "cuda")
    amp_dtype = (
        torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    if args.resume:
        model.load_state_dict(saved["model"])
        if saved.get("optimizer"):
            optimizer.load_state_dict(saved["optimizer"])
        if saved.get("scheduler"):
            scheduler.load_state_dict(saved["scheduler"])
        if saved.get("scaler"):
            scaler.load_state_dict(saved["scaler"])

    base_step = int(saved.get("base_checkpoint_step", saved.get("step", 0)))
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no EOS token; refusing to train without end-of-answer markers.")

    def make_payload(epoch, row_cursor):
        payload = dict(saved)
        payload.update({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": vars(config),
            "tokenizer": tokenizer.to_dict(),
            "step": base_step + completed_steps,
            "base_checkpoint_step": base_step,
            "best_validation_loss": best,
            "instruction_data_sha256": data_hash,
            "instruction_base_sha256": base_hash,
            "instruction_tuning_steps": completed_steps,
            "instruction_examples": len(examples),
            "instruction_train_examples": len(train_rows),
            "instruction_validation_examples": len(val_rows),
            "instruction_epoch": epoch,
            "instruction_next_row": row_cursor,
            "instruction_completed_epochs": epoch if row_cursor == 0 else epoch,
            "instruction_seed": args.seed,
            "instruction_epochs_target": args.epochs,
            "instruction_batch_size": args.batch_size,
            "instruction_grad_accum_steps": args.grad_accum_steps,
            "instruction_sampling": "deterministic shuffled permutation; every train row exactly once per epoch",
            "instruction_stale_epochs": stale_epochs,
        })
        return payload

    def evaluate_all():
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        batches = [
            val_rows[index:index + args.eval_batch_size]
            for index in range(0, len(val_rows), args.eval_batch_size)
        ]
        with torch.inference_mode():
            for batch_index, rows in enumerate(batches, 1):
                x, y = collate(rows, device)
                with torch.autocast(
                    device_type=device, dtype=amp_dtype, enabled=use_amp
                ):
                    logits, _ = model(x)
                loss_sum = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    y.reshape(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                total_loss += float(loss_sum.item())
                total_tokens += int((y != -100).sum().item())
                if batch_index % 100 == 0 or batch_index == len(batches):
                    print(
                        f"  validation {batch_index}/{len(batches)} batches",
                        flush=True,
                    )
        model.train()
        if total_tokens == 0:
            raise RuntimeError("Validation produced zero answer tokens.")
        return total_loss / total_tokens, total_tokens

    print(
        f"TRAINING PLAN: {args.epochs} full epochs | {len(train_rows):,} rows/epoch | "
        f"{updates_per_epoch:,} optimizer updates/epoch | "
        f"effective batch up to {args.batch_size * args.grad_accum_steps}",
        flush=True,
    )
    print(
        "Every training row is visited once per epoch; final partial batches are retained.",
        flush=True,
    )

    model.train()
    stopped_early = False

    for epoch in range(start_epoch, args.epochs):
        order_rng = random.Random(args.seed + 1009 * (epoch + 1))
        order = list(range(len(train_rows)))
        order_rng.shuffle(order)

        group_capacity = args.batch_size * args.grad_accum_steps
        start_at = next_row if epoch == start_epoch else 0
        if start_at < 0 or start_at > len(order) or start_at % group_capacity != 0:
            if start_at != len(order):
                raise ValueError("Saved resume cursor is invalid for the current batch settings.")

        epoch_loss_sum = 0.0
        epoch_token_count = 0
        groups = [
            order[position:position + group_capacity]
            for position in range(0, len(order), group_capacity)
        ]
        first_group = len(groups) if start_at == len(order) else start_at // group_capacity

        print(
            f"\nEPOCH {epoch + 1}/{args.epochs}: "
            f"{len(order):,} rows; resuming at row {start_at:,}",
            flush=True,
        )

        for group_number in range(first_group, len(groups)):
            selected_indices = groups[group_number]
            selected_rows = [train_rows[index] for index in selected_indices]
            micro_batches = [
                selected_rows[index:index + args.batch_size]
                for index in range(0, len(selected_rows), args.batch_size)
            ]
            group_tokens = answer_token_count(selected_rows)
            if group_tokens <= 0:
                raise RuntimeError("Training group contains no answer tokens.")

            optimizer.zero_grad(set_to_none=True)
            for rows in micro_batches:
                x, y = collate(rows, device)
                with torch.autocast(
                    device_type=device, dtype=amp_dtype, enabled=use_amp
                ):
                    logits, _ = model(x)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        y.reshape(-1),
                        ignore_index=-100,
                        reduction="sum",
                    )
                    scaled_loss = loss_sum / group_tokens
                if not torch.isfinite(scaled_loss):
                    raise RuntimeError("Non-finite loss; stopping without saving a broken checkpoint.")
                if use_scaler:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
                epoch_loss_sum += float(loss_sum.detach().item())
                epoch_token_count += int((y != -100).sum().item())

            if use_scaler:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            completed_steps += 1

            next_cursor = min(
                len(order), (group_number + 1) * group_capacity
            )

            if completed_steps % args.save_interval == 0:
                saved = make_payload(epoch, next_cursor)
                atomic_save(saved, out_path)
                print(
                    f"  checkpoint saved | step {completed_steps:,} | "
                    f"coverage {next_cursor:,}/{len(order):,}",
                    flush=True,
                )

            if group_number % 100 == 0 or group_number == len(groups) - 1:
                coverage = min(len(order), (group_number + 1) * group_capacity)
                print(
                    f"  epoch {epoch + 1} | update {group_number + 1}/{len(groups)} "
                    f"| rows {coverage:,}/{len(order):,} "
                    f"| loss {epoch_loss_sum / max(1, epoch_token_count):.4f} "
                    f"| lr {optimizer.param_groups[0]['lr']:.2e}",
                    flush=True,
                )

        # Full held-out evaluation at each completed epoch.
        val_loss, val_tokens = evaluate_all()
        train_loss = epoch_loss_sum / max(1, epoch_token_count)
        print(
            f"EPOCH {epoch + 1} COMPLETE | train token loss {train_loss:.5f} "
            f"| full validation token loss {val_loss:.5f} "
            f"| validation tokens {val_tokens:,}",
            flush=True,
        )

        improved = val_loss < best
        if improved:
            best = val_loss
            stale_epochs = 0
        else:
            stale_epochs += 1

        # Mark the epoch complete before saving, so a resume never replays it.
        saved = make_payload(epoch + 1, 0)
        atomic_save(saved, out_path)
        if improved:
            atomic_save(saved, best_path)
            print(f"  BEST checkpoint updated: {best_path}", flush=True)
        else:
            print(f"  No validation improvement ({stale_epochs}/{args.patience} stale epochs).", flush=True)

        next_row = 0
        if stale_epochs >= args.patience and epoch + 1 < args.epochs:
            stopped_early = True
            print("EARLY STOP: validation did not improve; best checkpoint preserved.", flush=True)
            break

    print("\n" + "=" * 72, flush=True)
    print("SFT FINISHED" if not stopped_early else "SFT EARLY-STOPPED", flush=True)
    print(f"Optimizer updates: {completed_steps:,}", flush=True)
    print(f"Best full-validation loss: {best:.6f}", flush=True)
    print(f"Latest resumable checkpoint: {out_path}", flush=True)
    print(f"Best checkpoint: {best_path}", flush=True)
    print("Original base checkpoint and source dataset were not modified.", flush=True)


if __name__ == "__main__":
    main()
