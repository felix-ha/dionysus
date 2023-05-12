import unittest
from data import *


class Test(unittest.TestCase):   
    @classmethod
    def setUpClass(self):
      #this method is only run once for the entire class rather than being run for each test which is done for setUp()
      self.constant = 3
    
    def test_get_distinct_characters(self):
        corpus_raw = "Hello World.\n"
        vocabulary_actual = get_distinct_characters(corpus_raw)
        vocabulary_expected = ['\n', ' ', '.', 'H', 'W', 'd', 'e', 'l', 'o', 'r']
        self.assertTrue(vocabulary_actual == vocabulary_expected)
    
    def test_create_token_index_mappings(self):
        vocabulary = ['\n', ' ', '.', 'H', 'W', 'd', 'e', 'l', 'o', 'r']
        token_to_index_actual, index_to_token_actual = create_token_index_mappings(vocabulary)
        token_to_index_expected = {'\n': 0, ' ': 1, '.': 2, 'H': 3, 'W': 4, 'd': 5, 'e': 6, 'l': 7, 'o': 8, 'r': 9}
        index_to_token_expected = {0: '\n', 1: ' ', 2: '.', 3: 'H', 4: 'W', 5: 'd', 6: 'e', 7: 'l', 8: 'o', 9: 'r'}
        self.assertTrue(token_to_index_expected == token_to_index_actual)
        self.assertTrue(index_to_token_expected == index_to_token_actual)

    def test_create_encoder(self):
        token_to_index = {'\n': 0, ' ': 1, '.': 2, 'H': 3, 'W': 4, 'd': 5, 'e': 6, 'l': 7, 'o': 8, 'r': 9}
        encoder = create_encoder(token_to_index)
        encoding_actual = encoder("World Hello.\n")
        encoding_expected = [4, 8, 9, 7, 5, 1, 3, 6, 7, 7, 8, 2, 0]
        self.assertTrue(encoding_actual == encoding_expected)

    def test_create_decoder(self):
        index_to_token = {0: '\n', 1: ' ', 2: '.', 3: 'H', 4: 'W', 5: 'd', 6: 'e', 7: 'l', 8: 'o', 9: 'r'}
        decoder = create_decoder(index_to_token)
        decoding_actual = decoder([4, 8, 9, 7, 5, 1, 3, 6, 7, 7, 8, 2, 0])
        decoding_expected = "World Hello.\n"
        self.assertTrue(decoding_expected == decoding_actual)

    def test_create_corpus_index(self):
        corpus_raw = "Hello World.\n"
        corpus_index_actual = create_corpus_index(corpus_raw)
        corpus_index_expected = torch.tensor([3, 6, 7, 7, 8, 1, 4, 8, 9, 7, 5, 2, 0])
        self.assertTrue(torch.equal(corpus_index_actual, corpus_index_expected))

    def test_create_train_val_split(self):
        corpus_raw = "Hello World.\n"
        corpus_index = create_corpus_index(corpus_raw)
        train_corpus_actual, validation_corpus_actual = create_train_val_split(corpus_index, 0.9)
        train_corpus_expected = torch.tensor([3, 6, 7, 7, 8, 1, 4, 8, 9, 7, 5])
        validation_corpus_expected = torch.tensor([2, 0])
        self.assertTrue(torch.equal(train_corpus_actual, train_corpus_expected))
        self.assertTrue(torch.equal(validation_corpus_actual, validation_corpus_expected))

    def test_get_batch(self):
        torch.manual_seed(0)
        data = torch.tensor([3, 6, 7, 7, 8, 1, 4, 8, 9, 7, 5])
        x_actual, y_actual = get_batch(data, block_size=2, batch_size=3)
        x_expected = torch.tensor( [[9, 7],
                                    [3, 6],
                                    [7, 7]])
        y_expected = torch.tensor( [[7, 5],
                                    [6, 7],
                                    [7, 8]])
        self.assertTrue(torch.equal(x_actual, x_expected))
        self.assertTrue(torch.equal(y_actual, y_expected))
        