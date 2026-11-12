import torch


def make_splits(text, tokenizer, train_fraction=0.9):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError('train_fraction must be between 0 and 1')
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(ids) < 20:
        raise ValueError('Corpus is too short; provide at least 20 tokens')
    split = max(1, int(train_fraction * len(ids)))
    train_ids, val_ids = ids[:split], ids[split:]
    if len(val_ids) < 2:
        raise ValueError('Corpus is too short to create a validation split')
    return train_ids, val_ids


def get_batch(data, batch_size, block_size, device):
    if batch_size < 1 or block_size < 1:
        raise ValueError('batch_size and block_size must be positive')
    high = len(data) - block_size
    if high < 1:
        raise ValueError(f'Dataset has {len(data)} tokens; need at least {block_size + 1}')
    starts = torch.randint(high, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)
