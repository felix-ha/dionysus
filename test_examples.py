import unittest
from examples import feadforward_moon, bigram, bigramV2, bigramV3


class Test(unittest.TestCase):
    def test_feadforward_moon(self):
        feadforward_moon()
    def test_bigram(self):
        bigram()
    def test_bigramV2(self):
        bigramV2()
    def test_bigramV3(self):
        bigramV3()
        

if __name__ == '__main__':
    unittest.main() 