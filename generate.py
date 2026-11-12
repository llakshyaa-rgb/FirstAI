"""Generate text from a trained FirstGPT checkpoint."""
import argparse
import torch
from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text from a FirstGPT checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/firstgpt.best.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()
    if args.tokens < 0 or args.temperature <= 0 or args.top_k < 1:
        raise ValueError("tokens must be >= 0; temperature and top-k must be positive")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tokenizer = BPETokenizer.from_dict(saved["tokenizer"])
    config = GPTConfig(**saved["config"])
    model = FirstGPT(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt must contain at least one encodable character")
    ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    result = model.generate(ids, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
