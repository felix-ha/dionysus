import unittest
from examples import feadforward_moon, bigram


class Test(unittest.TestCase):
    def test_feadforward_moon(self):
        feadforward_moon()
    def test_bigram(self):
        bigram()
        

if __name__ == '__main__':
    unittest.main() 