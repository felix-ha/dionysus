import unittest
from models import *

class TestBigramLanguageModel(unittest.TestCase):
    def test_single_batch(self):
        vocab_size=26

        x = torch.tensor([[1,2]]) 
        model = BigramLanguageModel(vocab_size=vocab_size)
        logits = model(x)

        B, T, C = logits.shape
        self.assertTrue(B == 1) # B_x = 1
        self.assertTrue(T == 2) # T_x = 2
        self.assertTrue(C == vocab_size) # Channels are logits that yield into a distribution for the vocabulary

    def test_several_batches(self):
        vocab_size=26

        x = torch.tensor([[1, 2, 3, 4, 5],
                          [4, 23, 1, 0, 25],
                          [22, 2, 4, 12, 19]]) 
        model = BigramLanguageModel(vocab_size=vocab_size)
        logits = model(x)

        B, T, C = logits.shape
        self.assertTrue(B == 3) # B_x = 1
        self.assertTrue(T == 5) # T_x = 5
        self.assertTrue(C == vocab_size)

class TestBigramLanguageModelV2(unittest.TestCase):
    def test_single_batch(self):
        vocab_size=26

        x = torch.tensor([[1,2]]) 
        model = BigramLanguageModelV2(vocab_size=vocab_size, n_embd=5)
        logits = model(x)

        B, T, C = logits.shape
        self.assertTrue(B == 1) # B_x = 1
        self.assertTrue(T == 2) # T_x = 2
        self.assertTrue(C == vocab_size) # Channels are logits that yield into a distribution for the vocabulary

    def test_several_batches(self):
        vocab_size=26

        x = torch.tensor([[1, 2, 3, 4, 5],
                          [4, 23, 1, 0, 25],
                          [22, 2, 4, 12, 19]]) 
        model = BigramLanguageModelV2(vocab_size=vocab_size, n_embd=5)
        logits = model(x)

        B, T, C = logits.shape
        self.assertTrue(B == 3) # B_x = 1
        self.assertTrue(T == 5) # T_x = 5
        self.assertTrue(C == vocab_size)


if __name__ == '__main__':
    unittest.main() 
