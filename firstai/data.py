"""Tokenize a corpus and produce shuffled next-token training batches."""
import torch


def make_splits(text, tokenizer, train_fraction=0.9):
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(ids) < 20:
        raise ValueError("Corpus is too short; provide at least 20 tokens")
    split = max(1, int(train_fraction * len(ids)))
    train_ids = ids[:split]
    val_ids = ids[split:]
    if len(val_ids) < 2:
        raise ValueError("Corpus is too short to create a validation split")
    return train_ids, val_ids


def get_batch(data, batch_size, block_size, device):
    if len(data) <= block_size + 1:
        raise ValueError("Dataset must contain more tokens than block_size + 1")
    starts = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
