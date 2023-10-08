import unittest
import torch
import torch.nn.functional as F
from src.dionysus.architectures import LeNet5


class Test(unittest.TestCase):
    def test_lenet5(self):
        model = LeNet5()
        input_tensor = torch.randn(1, 1, 28, 28)
        output_tensor = model(input_tensor)
        assert output_tensor.shape == (1, 10)
        assert torch.allclose(F.softmax(output_tensor.sum(dim=1)), torch.ones(1))
