"""Generate text from a FirstGPT checkpoint."""
import argparse
import torch
from firstai.model import FirstGPT, GPTConfig
from firstai.tokenizer import CharTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/firstgpt.pt")
    parser.add_argument("--prompt", default="The ")
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
    tokenizer = CharTokenizer.from_dict(saved["tokenizer"])
    config = GPTConfig(**saved["config"])
    model = FirstGPT(config).to(device)
    model.load_state_dict(saved["model"])
    model.eval()

    encoded = tokenizer.encode(args.prompt)
    ids = torch.tensor([encoded], dtype=torch.long, device=device)
    result = model.generate(ids, args.tokens, args.temperature, args.top_k)
    print(tokenizer.decode(result[0].tolist()))


if __name__ == "__main__":
    main()
