"""Byte-level BPE tokenizer implemented without external tokenization packages.

Base vocabulary is all 256 byte values. Learned merges add IDs 256 onward.
It can encode any UTF-8 text, including characters absent from training data.
"""
from collections import Counter
import heapq


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
        """Apply learned merges with a priority queue instead of rescanning the full sequence."""
        tokens = list(text.encode("utf-8"))
        if len(tokens) < 2 or not self.merges:
            return tokens

        size = len(tokens)
        previous = [i - 1 for i in range(size)]
        following = [i + 1 if i + 1 < size else -1 for i in range(size)]
        alive = [True] * size
        heap = []

        def push_pair(left):
            if left < 0 or not alive[left]:
                return
            right = following[left]
            if right < 0 or not alive[right]:
                return
            rank = self.ranks.get((tokens[left], tokens[right]))
            if rank is not None:
                heapq.heappush(heap, (rank, left, right))

        for index in range(size - 1):
            push_pair(index)

        while heap:
            rank, left, right = heapq.heappop(heap)
            if not alive[left] or not alive[right] or following[left] != right:
                continue
            if self.ranks.get((tokens[left], tokens[right])) != rank:
                continue

            tokens[left] = 256 + rank
            alive[right] = False
            after = following[right]
            following[left] = after
            if after >= 0:
                previous[after] = left

            push_pair(previous[left])
            push_pair(left)

        result = []
        index = next((i for i, is_alive in enumerate(alive) if is_alive and previous[i] == -1), -1)
        while index >= 0:
            result.append(tokens[index])
            index = following[index]
        return result

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
