"""Byte-level BPE tokenizer implemented without external tokenization packages.

Base vocabulary is all 256 byte values. Learned merges add IDs 256 onward.
It can encode any UTF-8 text, including characters absent from training data.
"""
from collections import Counter


class BPETokenizer:
    def __init__(self, merges=None):
        self.merges = [tuple(map(int, pair)) for pair in (merges or [])]
        self.ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size=512, min_frequency=2):
        if vocab_size < 256:
            raise ValueError("BPE vocab_size must be at least 256")
        tokens = list(text.encode("utf-8"))
        merges = []
        target_merges = vocab_size - 256

        for _ in range(target_merges):
            pairs = Counter(zip(tokens, tokens[1:]))
            if not pairs:
                break
            pair, count = pairs.most_common(1)[0]
            if count < min_frequency:
                break
            new_id = 256 + len(merges)
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
            merges.append(pair)
        return cls(merges)

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) > 1:
            best_rank = None
            best_pair = None
            for pair in zip(tokens, tokens[1:]):
                rank = self.ranks.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank, best_pair = rank, pair
            if best_pair is None:
                break
            new_id = 256 + best_rank
            merged = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                    merged.append(new_id)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def decode(self, ids):
        pieces = [bytes([i]) if i < 256 else self._expand(i) for i in ids]
        return b"".join(pieces).decode("utf-8", errors="replace")

    def _expand(self, token_id):
        if token_id < 256:
            return bytes([token_id])
        left, right = self.merges[token_id - 256]
        return self._expand(left) + self._expand(right)

    def to_dict(self):
        return {"type": "byte_bpe", "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, data):
        if data.get("type") != "byte_bpe":
            raise ValueError("Checkpoint does not contain a byte BPE tokenizer")
        return cls(data["merges"])
