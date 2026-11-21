"""FirstGPT training entry point with verbose progress and resumable total-step targets."""
import argparse
import hashlib
import math
import time
from pathlib import Path

import torch

print("[train] PyTorch imported; loading FirstGPT modules...", flush=True)
from firstai.data import get_batch, make_splits
from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer
print("[train] FirstGPT modules imported.", flush=True)


def log(message):
    print(f"[train] {message}", flush=True)


def save_checkpoint(path, model, optimizer, scaler, config, tokenizer,
                    step, best_val, corpus_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    log(f"Checkpoint save started: step={step}, path={path}")
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "config": vars(config),
        "tokenizer": tokenizer.to_dict(),
        "step": step,
        "best_validation_loss": best_val,
        "corpus_sha256": corpus_hash,
        "rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }, temporary)
    temporary.replace(path)
    log(f"Checkpoint save complete: {path} ({path.stat().st_size:,} bytes)")


def main():
    started = time.perf_counter()
    log("Parsing command-line arguments.")

    parser = argparse.ArgumentParser(description="Train FirstGPT from random initialization.")
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--out", default="checkpoints/firstgpt.pt")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Total target optimizer steps, including steps already in a resumed checkpoint.")
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--architecture-version", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tokenizer-train-chars", type=int, default=200000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--amp", action="store_true")
    args = parser.parse_args()

    log("Arguments parsed; validating settings.")
    positive = (args.steps, args.eval_interval, args.checkpoint_interval,
                args.eval_batches, args.batch_size, args.grad_accum_steps,
                args.block_size, args.layers, args.heads, args.width,
                args.min_frequency)
    if min(positive) < 1:
        raise ValueError("Steps, intervals, dimensions, batch and accumulation values must be positive")
    if args.vocab_size < 256:
        raise ValueError("vocab-size must be at least 256 for byte-level BPE")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be a finite positive number")
    log("Argument validation passed.")
    log(f"Requested total target: {args.steps} optimizer steps.")

    log("Checking compute device.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        log(f"CUDA ready: {torch.cuda.get_device_name(0)}")
        log(f"CUDA runtime: {torch.version.cuda}")
    else:
        log("CUDA unavailable; running on CPU.")

    data_path = Path(args.data)
    out_path = Path(args.out)
    log(f"Checking dataset path: {data_path.resolve()}")
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    log(f"Dataset found ({data_path.stat().st_size:,} bytes).")
    log(f"Checking checkpoint path: {out_path.resolve()}")
    if args.resume and not out_path.is_file():
        raise FileNotFoundError(f"--resume requested but checkpoint is missing: {out_path}")
    if out_path.is_file():
        log(f"Checkpoint found ({out_path.stat().st_size:,} bytes).")
    else:
        log("No checkpoint found; fresh training mode.")

    log("Reading dataset text.")
    stage = time.perf_counter()
    text = data_path.read_text(encoding="utf-8")
    log(f"Dataset read complete: {len(text):,} characters in {time.perf_counter() - stage:.2f}s.")

    log("Computing dataset SHA-256.")
    stage = time.perf_counter()
    corpus_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    log(f"Dataset hash ready in {time.perf_counter() - stage:.2f}s: {corpus_hash}")

    log("Setting initial random seeds.")
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    checkpoint = None
    start_step = 0
    best_val = float("inf")

    if args.resume:
        log("Loading checkpoint onto selected device; this can take a little time.")
        stage = time.perf_counter()
        checkpoint = torch.load(out_path, map_location=device, weights_only=True)
        log(f"Checkpoint deserialized in {time.perf_counter() - stage:.2f}s.")
        for key in ("model", "optimizer", "config", "tokenizer", "step", "corpus_sha256"):
            if key not in checkpoint:
                raise ValueError(f"Checkpoint missing required field: {key}")
            log(f"Checkpoint field verified: {key}")
        if checkpoint["corpus_sha256"] != corpus_hash:
            raise ValueError("Dataset differs from checkpoint; resume with the exact original dataset.")
        log("Dataset hash matches checkpoint.")

        log("Restoring tokenizer from checkpoint (no tokenizer retraining).")
        tokenizer = BPETokenizer.from_dict(checkpoint["tokenizer"])
        log(f"Tokenizer restored: vocabulary={tokenizer.vocab_size}.")

        log("Restoring model configuration.")
        config = GPTConfig(**checkpoint["config"])
        log(f"Model config restored: layers={config.n_layer}, heads={config.n_head}, width={config.n_embd}, block={config.block_size}, vocab={config.vocab_size}.")
    else:
        log("Training tokenizer from configured corpus sample.")
        stage = time.perf_counter()
        tokenizer_text = text[:min(len(text), args.tokenizer_train_chars,
                                   max(1, int(len(text) * 0.9)))]
        tokenizer = BPETokenizer.train(tokenizer_text, args.vocab_size, args.min_frequency)
        log(f"Tokenizer training complete in {time.perf_counter() - stage:.2f}s; vocabulary={tokenizer.vocab_size}.")
        log("Building model configuration.")
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.layers,
            n_head=args.heads,
            n_embd=args.width,
            architecture_version=args.architecture_version,
        )

    log("Encoding corpus and building train/validation splits.")
    stage = time.perf_counter()
    train_data, val_data = make_splits(text, tokenizer)
    log(f"Tokenization and split complete in {time.perf_counter() - stage:.2f}s.")
    log(f"Training tokens={len(train_data):,}; validation tokens={len(val_data):,}.")
    if len(train_data) <= config.block_size or len(val_data) <= config.block_size:
        raise ValueError(f"Corpus too small for block_size={config.block_size}; add text or lower --block-size")
    log("Dataset length checks passed.")

    log("Constructing model.")
    stage = time.perf_counter()
    model = FirstGPT(config).to(device)
    log(f"Model constructed in {time.perf_counter() - stage:.2f}s.")
    log("Constructing AdamW optimizer.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.resume:
        log("Loading model weights.")
        model.load_state_dict(checkpoint["model"])
        log("Model weights restored.")
        log("Loading optimizer state.")
        optimizer.load_state_dict(checkpoint["optimizer"])
        log("Optimizer state restored.")
        start_step = int(checkpoint["step"])
        if start_step < 0:
            raise ValueError(f"Invalid checkpoint step: {start_step}")
        best_val = float(checkpoint.get("best_validation_loss", float("inf")))
        log(f"Checkpoint reports completed optimizer step {start_step}.")
        log(f"Restored best validation loss: {best_val}.")
    else:
        log("Fresh model and optimizer initialized.")

    log("Configuring mixed precision and gradient scaler.")
    use_amp = args.amp and device == "cuda"
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    use_scaler = use_amp and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    if args.resume and checkpoint.get("scaler"):
        log("Restoring gradient scaler state.")
        scaler.load_state_dict(checkpoint["scaler"])
        log("Gradient scaler restored.")
    if "rng_state" in (checkpoint or {}):
        log("Restoring CPU RNG state.")
        torch.set_rng_state(checkpoint["rng_state"].cpu())
    if device == "cuda" and (checkpoint or {}).get("cuda_rng_state"):
        log("Restoring CUDA RNG state.")
        torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng_state"]])
    log(f"AMP={'enabled' if use_amp else 'disabled'}; dtype={amp_dtype}; scaler={use_scaler}.")

    if start_step > args.steps:
        raise ValueError(f"Checkpoint is at step {start_step}, beyond requested total target {args.steps}.")
    remaining = args.steps - start_step
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    log(f"Model parameters: {parameter_count:,}.")
    log(f"Effective batch size: {args.batch_size * args.grad_accum_steps}.")
    log(f"RESUME STATUS: {start_step}/{args.steps} completed; {remaining} optimizer steps remain.")

    @torch.no_grad()
    def estimate():
        log("Evaluation started.")
        model.eval()
        results = {}
        for name, data in (("train", train_data), ("validation", val_data)):
            losses = []
            for index in range(args.eval_batches):
                log(f"Evaluation {name}: batch {index + 1}/{args.eval_batches}.")
                x, y = get_batch(data, args.batch_size, config.block_size, device)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                    _, loss = model(x, y)
                losses.append(loss.item())
            results[name] = sum(losses) / len(losses)
            log(f"Evaluation {name} complete: loss={results[name]:.6f}.")
        model.train()
        log("Evaluation finished.")
        return results

    best_path = str(out_path.with_name(out_path.stem + ".best.pt"))
    model.train()

    if remaining == 0:
        log("Target already reached; no optimizer steps are needed.")

    for step in range(start_step, args.steps):
        step_started = time.perf_counter()
        log(f"STEP {step + 1}/{args.steps}: started.")

        if step % args.eval_interval == 0:
            scores = estimate()
            perplexity = math.exp(min(scores["validation"], 20))
            log(f"Metrics at step {step}: train={scores['train']:.4f}, validation={scores['validation']:.4f}, perplexity={perplexity:.2f}.")
            if scores["validation"] < best_val:
                best_val = scores["validation"]
                log("Validation improved; saving best checkpoint.")
                save_checkpoint(best_path, model, optimizer, scaler, config, tokenizer, step, best_val, corpus_hash)

        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for microstep in range(args.grad_accum_steps):
            log(f"Step {step + 1}: fetching microbatch {microstep + 1}/{args.grad_accum_steps}.")
            x, y = get_batch(train_data, args.batch_size, config.block_size, device)
            log(f"Step {step + 1}: running forward pass for microbatch {microstep + 1}.")
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=use_amp):
                _, loss = model(x, y)
                scaled_loss = loss / args.grad_accum_steps
            log(f"Step {step + 1}: backward pass for microbatch {microstep + 1}.")
            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()
            accumulated_loss += loss.detach().item()
            log(f"Step {step + 1}: microbatch {microstep + 1} complete; loss={loss.detach().item():.5f}.")

        log(f"Step {step + 1}: clipping gradients.")
        if use_scaler:
            scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        log(f"Step {step + 1}: optimizer update.")
        if use_scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        completed = step + 1
        average_loss = accumulated_loss / args.grad_accum_steps
        log(f"STEP COMPLETE: {completed}/{args.steps}; loss={average_loss:.5f}; elapsed={time.perf_counter() - step_started:.2f}s.")

        if completed % args.checkpoint_interval == 0:
            save_checkpoint(out_path, model, optimizer, scaler, config, tokenizer, completed, best_val, corpus_hash)

    log("Training loop finished; writing final resumable checkpoint.")
    save_checkpoint(out_path, model, optimizer, scaler, config, tokenizer, args.steps, best_val, corpus_hash)
    log(f"TRAINING COMPLETE: {args.steps}/{args.steps}.")
    log(f"Final checkpoint: {out_path}.")
    log(f"Best validation loss: {best_val}.")
    log(f"Total elapsed time: {time.perf_counter() - started:.2f}s.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FATAL: {type(error).__name__}: {error}")
        raise
