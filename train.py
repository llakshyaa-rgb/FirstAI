import argparse
import hashlib
import math
from pathlib import Path
import torch

from firstai.data import get_batch, make_splits
from firstai.model import FirstGPT, GPTConfig
from firstai.bpe import BPETokenizer


def save_checkpoint(path, model, optimizer, config, tokenizer, step, best_val, corpus_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save({
        'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
        'config': vars(config), 'tokenizer': tokenizer.to_dict(),
        'step': step, 'best_validation_loss': best_val,
        'corpus_sha256': corpus_hash,
    }, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description='Train FirstGPT from random initialization.')
    parser.add_argument('--data', default='data/input.txt')
    parser.add_argument('--out', default='checkpoints/firstgpt.pt')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--steps', type=int, default=2000, help='Additional optimizer steps.')
    parser.add_argument('--eval-interval', type=int, default=200)
    parser.add_argument('--checkpoint-interval', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--layers', type=int, default=4)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--vocab-size', type=int, default=512)
    parser.add_argument('--min-frequency', type=int, default=2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=1337)
    args = parser.parse_args()
    if min(args.steps, args.eval_interval, args.checkpoint_interval, args.batch_size) < 1:
        raise ValueError('steps, intervals and batch-size must be positive')
    if args.block_size < 1:
        raise ValueError('block-size must be positive')

    torch.manual_seed(args.seed)
    text = Path(args.data).read_text(encoding='utf-8')
    corpus_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
    tokenizer_text = text[:max(1, int(len(text) * 0.9))]
    tokenizer = BPETokenizer.train(tokenizer_text, args.vocab_size, args.min_frequency)
    train_data, val_data = make_splits(text, tokenizer)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    start_step, best_val = 0, float('inf')

    if args.resume:
        checkpoint = torch.load(args.out, map_location=device, weights_only=True)
        if checkpoint.get('corpus_sha256') != corpus_hash:
            raise ValueError('Corpus differs from checkpoint; resume with the original corpus')
        config = GPTConfig(**checkpoint['config'])
        saved_tokenizer = BPETokenizer.from_dict(checkpoint['tokenizer'])
        if saved_tokenizer.to_dict() != tokenizer.to_dict():
            raise ValueError('Tokenizer differs from checkpoint; cannot resume safely')
        if config.block_size != args.block_size:
            raise ValueError('block_size differs from checkpoint')
        model = FirstGPT(config).to(device)
        model.load_state_dict(checkpoint['model'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_step = int(checkpoint['step'])
        best_val = float(checkpoint.get('best_validation_loss', float('inf')))
        print(f'Resuming at optimizer step {start_step}; using saved architecture and optimizer settings')
    else:
        config = GPTConfig(vocab_size=tokenizer.vocab_size, block_size=args.block_size,
                           n_layer=args.layers, n_head=args.heads, n_embd=args.width)
        model = FirstGPT(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if len(train_data) <= config.block_size + 1 or len(val_data) <= config.block_size + 1:
        raise ValueError(f'Corpus too small for block_size={config.block_size}; add text or lower --block-size')
    print(f'Device: {device}' + (f' | GPU: {torch.cuda.get_device_name(0)}' if device == 'cuda' else ''))
    print(f'Characters: {len(text):,} | tokens: {len(train_data)+len(val_data):,} | vocabulary: {tokenizer.vocab_size}')
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,} | architecture: {config.n_layer} layers, {config.n_head} heads, width {config.n_embd}')

    @torch.no_grad()
    def estimate():
        model.eval()
        result = {}
        for name, data in (('train', train_data), ('validation', val_data)):
            losses = []
            for _ in range(20):
                x, y = get_batch(data, args.batch_size, config.block_size, device)
                _, loss = model(x, y)
                losses.append(loss.item())
            result[name] = sum(losses) / len(losses)
        model.train()
        return result

    best_path = str(Path(args.out).with_name(Path(args.out).stem + '.best.pt'))
    model.train()
    for step in range(start_step, start_step + args.steps):
        if step % args.eval_interval == 0 or step == start_step + args.steps - 1:
            scores = estimate()
            print(f"step {step:5d} | train {scores['train']:.4f} | validation {scores['validation']:.4f} | perplexity {math.exp(min(scores['validation'], 20)):.2f}")
            if scores['validation'] < best_val:
                best_val = scores['validation']
                save_checkpoint(best_path, model, optimizer, config, tokenizer, step, best_val, corpus_hash)
                print(f'  saved best checkpoint: {best_path}')
        x, y = get_batch(train_data, args.batch_size, config.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        completed = step + 1
        if completed % args.checkpoint_interval == 0:
            save_checkpoint(args.out, model, optimizer, config, tokenizer, completed, best_val, corpus_hash)
            print(f'  checkpointed at step {completed}: {args.out}')
    save_checkpoint(args.out, model, optimizer, config, tokenizer,
                    start_step + args.steps, best_val, corpus_hash)
    print(f'Saved resumable checkpoint: {args.out}')


if __name__ == '__main__':
    main()
