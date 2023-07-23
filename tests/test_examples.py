import unittest

import os
import sys

sys.path.insert(0, os.getcwd())
from examples import *


class Test(unittest.TestCase):
    def test_feadforward_moon(self):
        feadforward_moon()
    def test_bigram(self):
        bigram()
    def test_simpleGPT(self):
        run_simpleGPT()
    def test_GPT1(self):
        run_GPT1()
    def test_GPT2(self):
        run_GPT2()
    def test_custom_loss(self):
        custom_loss()

if __name__ == '__main__':
    unittest.main() 
    