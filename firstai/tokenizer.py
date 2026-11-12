"""A transparent character tokenizer for the first training milestone."""
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(set(text))
        if not self.chars:
            raise ValueError("Tokenizer needs a non-empty training corpus")
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}

    @property
    def vocab_size(self):
        return len(self.chars)

    def encode(self, text):
        unknown = sorted(set(text) - set(self.stoi))
        if unknown:
            raise ValueError(f"Text contains unseen characters: {unknown!r}")
        return [self.stoi[ch] for ch in text]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self):
        return {"type": "character", "chars": self.chars}

    @classmethod
    def from_dict(cls, data):
        obj = cls("".join(data["chars"]))
        return obj
