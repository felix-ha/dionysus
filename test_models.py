import unittest
from models import *

class TestBigramLanguageModel(unittest.TestCase):
    @classmethod
    def setUpClass(self):
      self.vocab_size=26
    
    def test_single_batch_loop(self):
        x = torch.tensor([[1,2]])

        models = [BigramLanguageModel(vocab_size=self.vocab_size), 
                  BigramLanguageModelV2(vocab_size=self.vocab_size, n_embd=5),
                  BigramLanguageModelV3(vocab_size=self.vocab_size, n_embd=5, block_size=2, device='cpu'),
                  BigramLanguageModelV4(vocab_size=self.vocab_size, n_embd=5, head_size=3, block_size=5, device='cpu')]

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
                  BigramLanguageModelV2(vocab_size=self.vocab_size, n_embd=5),
                  BigramLanguageModelV3(vocab_size=self.vocab_size, n_embd=5, block_size=5, device='cpu'),
                  BigramLanguageModelV4(vocab_size=self.vocab_size, n_embd=5, head_size=3, block_size=5, device='cpu')]

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


if __name__ == '__main__':
    unittest.main() 
