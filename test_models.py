import unittest
from models import *
import torch
import torch.nn as nn


class TestBigramLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
      self.vocab_size=26

    def test_single_token_input_bigram(self):
        x = torch.tensor([1])

        model = BigramLanguageModel(vocab_size=self.vocab_size)
                  
        logits = model(x)
        self.assertTrue(logits.shape == (1, self.vocab_size))

    def test_single_token_input_bigram2(self):
        x = torch.tensor([[1]])

        model = BigramLanguageModel(vocab_size=self.vocab_size)
                  
        logits = model(x)
        self.assertTrue(logits.shape == (1, 1, self.vocab_size))

    def test_single_token_input_gpt1(self):
        x = torch.tensor([1])

        model =  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=500,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')
        # ToDo: Find better way to implement positional encoding
        with self.assertRaises(ValueError):
            logits = model(x)

    def test_single_token_input_gpt(self):
        x = torch.tensor([[1]])

        model =  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=500,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')


        logits = model(x)
        self.assertTrue(logits.shape == (1, 1, self.vocab_size))
    
    def test_single_batch_loop(self):
        x = torch.tensor([[1,2]])

        models = [BigramLanguageModel(vocab_size=self.vocab_size), 
                  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=2,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')]

        for model in models:
            logits = model(x)
            B, T, C = logits.shape
            self.assertTrue(B == 1) # B_x = 1
            self.assertTrue(T == 2) # T_x = 2
            self.assertTrue(C == self.vocab_size) # Channels are logits that yield into a distribution for the vocabulary

    def test_several_batches(self):
        x = torch.tensor([[1, 2, 3, 4, 5],
                          [4, 23, 1, 0, 25],
                          [22, 2, 4, 12, 19]]) 
        
        models = [BigramLanguageModel(vocab_size=self.vocab_size), 
                  simpleGPT(vocab_size=self.vocab_size,
                                   n_embd=5,
                                   num_heads=1,
                                   block_size=5,
                                   n_layer=1, 
                                   dropout=0.2, 
                                   device='cpu')]

        for model in models:
            logits = model(x)
            B, T, C = logits.shape
            self.assertTrue(B == 3) # B_x = 1
            self.assertTrue(T == 5) # T_x = 5
            self.assertTrue(C == self.vocab_size)

    def test_embedding(self):
        x = torch.tensor([[1, 2, 3, 4, 5],
                        [4, 23, 1, 0, 25],
                        [22, 2, 4, 12, 19]]) 
        
        B, T = x.shape
        self.assertTrue(B == 3)
        self.assertTrue(T == 5)
        
        vocab_size = torch.max(x).item() + 1
        self.assertTrue(vocab_size == 26) 

        n_embd = 50
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)

        out = embedding(x)

        self.assertTrue(out.shape == (B, T, n_embd))

    def test_self_attention_head(self):
        B = 1
        n_embd = 26
        block_size = 5 # This is T
        head_size = 3
        x = torch.rand((B, block_size, n_embd))
              
        head = Head(head_size, n_embd, block_size)
        out = head(x)

        self.assertTrue(out.shape == (B, block_size, head_size))

    def test_self_attention_multi_head(self):
        B = 1
        n_embd = 8
        block_size = 5 # This is T
        head_size = 2
        x = torch.rand((B, block_size, n_embd))
        
        num_heads = 4
        head = MultiHeadAttention(num_heads, head_size, n_embd, block_size)
        out = head(x)

        self.assertTrue(out.shape == (B, block_size, num_heads * head_size))

    def test_embedding_self_attention(self):
        x = torch.tensor([[1, 2, 3, 4, 5],
                        [4, 23, 1, 0, 25],
                        [22, 2, 4, 12, 19]]) 
        
        B, T = x.shape
        self.assertTrue(B == 3)
        self.assertTrue(T == 5)
        
        vocab_size = torch.max(x).item() + 1
        self.assertTrue(vocab_size == 26) 

        n_embd = 50
        head_size = 25
        block_size = T
        embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=n_embd)
        head = Head(head_size, n_embd, block_size)

        emb = embedding(x)
        out = head(emb)

        self.assertTrue(emb.shape == (B, T, n_embd))
        self.assertTrue(out.shape == (B, T, head_size))

    def test_nn_linear_no_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([1, 2], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([1.0, 2.0, 0.0])))

    def test_nn_linear_empty_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([[1, 2]], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([[1.0, 2.0, 0.0]])))
    
    
    def test_nn_linear_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([[1, 2],
                              [3, 4],
                              [5, 6]], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([[1.0, 2.0, 0.0],
                                                         [3.0, 4.0, 0.0],
                                                         [5.0, 6.0, 0.0]])))

    def test_nn_linear_meta_batch(self):
        with torch.no_grad():
            layer = nn.Linear(2, 3, bias=False)
            x = torch.tensor([[[1, 2],
                              [3, 4],
                              [5, 6]], 
                              
                             [[1,8],
                              [3, 9],
                              [5, 7]]], dtype=torch.float32)

            layer.weight[0,0] = 1
            layer.weight[1,1] = 1
            layer.weight[0,1] = 0
            layer.weight[1,0] = 0
            layer.weight[2,0] = 0
            layer.weight[2,1] = 0

            y = layer(x)
            self.assertTrue(torch.equal(y, torch.tensor([[[1.0, 2.0, 0.0],
                                                         [3.0, 4.0, 0.0],
                                                         [5.0, 6.0, 0.0]],
                                                         
                                                         [[1.0, 8.0, 0.0],
                                                         [3.0, 9.0, 0.0],
                                                         [5.0, 7.0, 0.0]]])))

if __name__ == '__main__':
    unittest.main() 
