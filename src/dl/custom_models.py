import torch
import torch.nn as nn
from . import models


class SimpleTransformerClassifier(nn.Module):

    def __init__(self, vocab_size, dim_model, max_context_len, num_heads, num_layers, number_classes, padding_idx=None):
        super(SimpleTransformerClassifier, self).__init__()
        assert dim_model % num_heads == 0, f"{dim_model=} must be a multiple of {num_heads=}"
        self.max_context_len = max_context_len
        self.padding_idx = padding_idx
        self.token_embedding = nn.Embedding(vocab_size, dim_model, padding_idx=padding_idx)
        self.position_embedding =  models.PositionalEncoding(dim_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim_model, nhead=num_heads, batch_first=True),num_layers=num_layers)
        self.attention_weighted_sum = models.AttentionAvg(models.DotScore(dim_model))
        self.pred = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(dim_model, dim_model),
            nn.LeakyReLU(),
            nn.BatchNorm1d(dim_model),
            nn.Linear(dim_model, number_classes)
        )
    
    def forward(self, input):
        assert input.shape[-1] <= self.max_context_len

        if self.padding_idx is not None:
            mask = input != self.padding_idx
            src_key_padding_mask = torch.logical_not(mask)
        else:
            mask = input == input 
            src_key_padding_mask = None

        x = self.token_embedding(input)
        x = self.position_embedding(x)
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        context = x.sum(dim=1)/mask.sum(dim=1).unsqueeze(1)
        logits = self.pred(self.attention_weighted_sum(x, context, mask=mask))

        return logits