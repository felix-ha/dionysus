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
                  BigramLanguageModelV3(vocab_size=self.vocab_size, n_embd=5, block_size=2, device='cpu')]

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
                  BigramLanguageModelV3(vocab_size=self.vocab_size, n_embd=5, block_size=5, device='cpu')]

        for model in models:
            logits = model(x)
            B, T, C = logits.shape
            self.assertTrue(B == 3) # B_x = 1
            self.assertTrue(T == 5) # T_x = 5
            self.assertTrue(C == self.vocab_size)


if __name__ == '__main__':
    unittest.main() 
