import unittest
import torch

from src.dl.data import *
from src.dl.custom_models import SimpleTransformerClassifier


class TestTransformer(unittest.TestCase):   
    def test_pad_batch(self):
            batch = [(torch.tensor([ 8, 15, 23,  9,  5, 1, 4, 5]), 4),
                    (torch.tensor([ 5, 12,  9, 15, 16, 15, 21, 12, 15, 19]), 7),
                    (torch.tensor([16,  1, 25, 14,  5, 3, 5]), 4),
                    (torch.tensor([ 2, 18,  1, 21, 14,  5]), 6)]
            
            x_padded, y = pad_batch(batch, padding_value=0)

            x_expected = torch.tensor([[ 8, 15, 23,  9,  5,  1,  4,  5,  0,  0],
                                        [ 5, 12,  9, 15, 16, 15, 21, 12, 15, 19],
                                        [16,  1, 25, 14,  5,  3,  5,  0,  0,  0],
                                        [ 2, 18,  1, 21, 14,  5,  0,  0,  0,  0]])
            y_expected = torch.tensor([4, 7, 4, 6])

            self.assertTrue(torch.equal(x_padded, x_expected))
            self.assertTrue(torch.equal(y, y_expected))

    def test_custom_simple_transformer_classifier(self):
        vocab_size = 4
        embedding_dim = 2
        padding_idx = 0
        number_classes = 3
        batch = torch.tensor([[ 1, 1, 2, 3],
                        [ 1, 1, 2,  padding_idx],
                        [3, 1, padding_idx, padding_idx]])
        
        model = SimpleTransformerClassifier(vocab_size,
                                             embedding_dim, 
                                             max_context_len=4, 
                                             num_heads=2, 
                                             num_layers=1, 
                                             number_classes=number_classes, 
                                             padding_idx=padding_idx)

        out = model(batch)
        
        self.assertTrue(out.shape == (3,  number_classes))



if __name__ == '__main__':
    unittest.main() 
    