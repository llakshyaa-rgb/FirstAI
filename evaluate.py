"""Evaluate a saved FirstGPT checkpoint against its original corpus split."""
import argparse
import hashlib
import math
from pathlib import Path
import torch

from firstai.bpe import BPETokenizer
from firstai.model import FirstGPT, GPTConfig


def main():
    parser = argparse.ArgumentParser(description="Evaluate FirstGPT on train and held-out validation text.")
    parser.add_argument("--checkpoint", default="checkpoints/firstgpt.best.pt")
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--batches", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if min(args.batches, args.batch_size) < 1:
        raise ValueError("batches and batch-size must be positive")

    text = Path(args.data).read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if saved.get("corpus_sha256") != digest:
        raise ValueError("The supplied corpus does not match the checkpoint's training corpus")
    tokenizer = BPETokenizer.from_dict(saved["tokenizer"])
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split = max(1, int(0.9 * len(ids)))
    train_data, val_data = ids[:split], ids[split:]
    config = GPTConfig(**saved["config"])
    model = FirstGPT(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    generator = torch.Generator().manual_seed(args.seed)
    print(f"Device: {device} | checkpoint step: {saved.get('step', 'unknown')}")
    with torch.no_grad():
        for name, data in (("train", train_data), ("validation", val_data)):
            if len(data) <= config.block_size:
                raise ValueError(f"{name} split is too short for block_size={config.block_size}")
            losses = []
            high = len(data) - config.block_size
            for _ in range(args.batches):
                starts = torch.randint(high, (args.batch_size,), generator=generator)
                x = torch.stack([data[i:i + config.block_size] for i in starts]).to(device)
                y = torch.stack([data[i + 1:i + config.block_size + 1] for i in starts]).to(device)
                _, loss = model(x, y)
                losses.append(loss.item())
            mean_loss = sum(losses) / len(losses)
            print(f"{name:10s} loss: {mean_loss:.4f} | perplexity: {math.exp(min(mean_loss, 20)):.2f}")


if __name__ == "__main__":
    main()
