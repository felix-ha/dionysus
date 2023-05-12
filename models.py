import torch
import torch.nn as nn
from torch.nn import functional as F

from data import get_distinct_characters


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss


if __name__ == "__main__":
    with open('data/text.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    vocab = get_distinct_characters(text)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BigramLanguageModel(vocab_size=len(vocab))
    m = model.to(device)
