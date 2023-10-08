import unittest
import pytest
import torch
import torch.nn.functional as F
from src.dionysus.architectures import LeNet5, AlexNet, VGGNet


class Test(unittest.TestCase):
    def run_inference_model(self, model):
        input_tensor = torch.randn(1, 1, 28, 28)
        output_tensor = model(input_tensor)
        assert output_tensor.shape == (1, 10)
        assert torch.allclose(F.softmax(output_tensor.sum(dim=1)), torch.ones(1))

    def test_lenet5(self):
        self.run_inference_model(LeNet5())

    def test_alexnet(self):
        self.run_inference_model(AlexNet())

    def test_vggnet(self):
        self.run_inference_model(VGGNet())
