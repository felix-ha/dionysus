import unittest
from examples import *


class Test(unittest.TestCase):
    def test_feadforward_moon(self):
        feadforward_moon()
    def test_bigram(self):
        bigram()
    def test_bigramV6(self):
        bigramV6()
        

if __name__ == '__main__':
    unittest.main() 