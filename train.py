"""Train FirstGPT from random initialization on a plain-text corpus."""
import argparse
import math
from pathlib import Path
import torch

from firstai.data import get_batch, make_splits
from firstai.model import FirstGPT, GPTConfig
from firstai.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--out", default="checkpoints/firstgpt.pt")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    text = Path(args.data).read_text(encoding="utf-8")
    tokenizer = CharTokenizer(text)
    train_data, val_data = make_splits(text, tokenizer)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        block_size=args.block_size,
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.width,
    )
    model = FirstGPT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Corpus characters: {len(text):,} | Vocabulary: {tokenizer.vocab_size}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    @torch.no_grad()
    def estimate():
        model.eval()
        result = {}
        for name, data in (("train", train_data), ("validation", val_data)):
            losses = []
            for _ in range(20):
                x, y = get_batch(data, args.batch_size, args.block_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
        model.train()
        return result

    model.train()
    for step in range(args.steps):
        if step % args.eval_interval == 0 or step == args.steps - 1:
            scores = estimate()
            print(
                f"step {step:5d} | train {scores['train']:.4f} "
                f"| validation {scores['validation']:.4f} "
                f"| ppl {math.exp(min(scores['validation'], 20)):.2f}"
            )
        x, y = get_batch(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "config": vars(config),
         "tokenizer": tokenizer.to_dict(), "step": args.steps},
        output,
    )
    print(f"Saved checkpoint: {output}")


if __name__ == "__main__":
    main()
