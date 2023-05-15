import torch
import torch.nn as nn
from torch.nn import functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx):
        logits = self.token_embedding_table(idx) 
        return logits

    
class BigramLanguageModelV2(nn.Module):
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        tok_emb = self.token_embedding_table(idx) 
        logits = self.lm_head(tok_emb)
        return logits
    

class BigramLanguageModelV3(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.device = device

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        positinal_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + positinal_emb
        logits = self.lm_head(x)
        return logits
    

class BigramLanguageModelV4(nn.Module):
    """
    Model with single self-attention head
    """
    def __init__(self, vocab_size, n_embd, head_size, block_size, device):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(head_size, n_embd, block_size)
        self.lm_head = nn.Linear(head_size, vocab_size)
        self.device = device

    def forward(self, idx):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) 
        positinal_emb = self.position_embedding_table(torch.arange(T, device=self.device))
        x = tok_emb + positinal_emb
        x = self.sa_head(x)
        logits = self.lm_head(x)
        return logits


class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size, n_embd, block_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)   
        q = self.query(x) 
        wei = q @ k.transpose(-2,-1) * C**-0.5 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        v = self.value(x) 
        out = wei @ v 
        return out
   
    
def generate(model, idx, max_new_tokens):
    for _ in range(max_new_tokens):
        logits = model(idx)
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1) 
        idx_next = torch.multinomial(probs, num_samples=1) 
        idx = torch.cat((idx, idx_next), dim=1) 
    return idx
