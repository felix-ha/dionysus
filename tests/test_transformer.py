import unittest
import os
import sys

sys.path.insert(0, os.getcwd())
from dl.data import *


class TestTransformer(unittest.TestCase):   
    def test_padding_batch(self):
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


if __name__ == '__main__':
    unittest.main() 