import unittest
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
        

if __name__ == '__main__':
    unittest.main() 