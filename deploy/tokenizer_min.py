import json
from typing import Dict, List


class SimpleTokenizer:
    """Minimal char-level tokenizer for deployment inference."""

    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"

    def __init__(self, vocab: Dict[str, int]):
        self.token2id = vocab
        self.id2token = {v: k for k, v in vocab.items()}

    @classmethod
    def load(cls, path: str) -> "SimpleTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        return cls(vocab=vocab)

    @property
    def pad_token_id(self) -> int:
        return int(self.token2id.get(self.PAD_TOKEN, 0))

    def encode(self, text: str) -> List[int]:
        unk_id = int(self.token2id.get(self.UNK_TOKEN, 1))
        return [int(self.token2id.get(ch, unk_id)) for ch in text]
