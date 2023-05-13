import torch
from torch.utils.data import Dataset, DataLoader


def get_distinct_characters(text: str) -> list[str]:
    """
    Returns all unique charachters of the text as a list. 
    """
    return sorted(list(set(text)))

def create_token_index_mappings(vocabulary: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    """
    Creates for a given list of distinct tokens, i. e. the vocabulary, 
    dictionaries for token to index and index to token mappings. 
    The index is given via the position in the list. 
    """
    token_to_index = { token:index for index, token in enumerate(vocabulary) }
    index_to_token = { index:token for index, token in enumerate(vocabulary) }
    return token_to_index, index_to_token

def create_encoder(token_to_index):
    """
    Creates a function that converts a string into a list of integers,
    that represent the indexes of the tokens given the token_to_index dictionary. 
    """
    return lambda string: [token_to_index[char] for char in string]

def create_decoder(index_to_token):
    """
    Creates a function that converts a list of integers that represent the indexes
    of the tokens given the token_to_index dictionary and converts it to a string. 
    """
    return lambda idxs: ''.join([index_to_token[index] for index in idxs])

def create_corpus_index(corpus_raw: str):
    """
    Given a string (corpus_raw), one dimensional tensor with the indexes
    wrt the vocabulary based on this string is created.
    """
    vocabulary = get_distinct_characters(corpus_raw)
    token_to_index, index_to_token = create_token_index_mappings(vocabulary)
    encoder = create_encoder(token_to_index)
    decoder = create_decoder(index_to_token)
    corpus_index = torch.tensor(encoder(corpus_raw), dtype=torch.long)
    return corpus_index, vocabulary, decoder, encoder

def create_train_val_split(corpus_index: torch.Tensor, validation_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Splits the corpus_index into training and validation set.
        The first (1-validation_ratio) % tokens will be the training data and 
        last validation_ratio % tokens the validation data.
        """
        n = int(validation_ratio*len(corpus_index)) 
        corpus_index_training = corpus_index[:n]
        corpus_index_validation = corpus_index[n:]
        return corpus_index_training,corpus_index_validation

def get_batch(data, block_size, batch_size):
    """
    Creates a batch with batch_size sequences, x is a sequence of length block_size
    y is a sequence of length block_size shifted one index to the right. 
    """
    max_index = len(data) - block_size
    ix = torch.randint(high=max_index, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


class LanguageModelDataset(Dataset):
    def __init__(self, text_file, block_size, vocabulary=None, encoder=None, decoder=None):
        self.block_size = block_size
        with open(text_file, 'r', encoding='utf-8') as f:
            self.corpus_raw = f.read()
        if vocabulary is None or decoder is None or encoder is None:
            self.corpus_index, self.vocabulary, self.decoder, self.encoder = create_corpus_index(self.corpus_raw)
        else:
            self.vocabulary = vocabulary
            self.encoder = encoder
            self.decoder = decoder
            self.corpus_index = torch.tensor(encoder(self.corpus_raw), dtype=torch.long)

    def __len__(self):
        # last elements must not be collected, because they have no successor
        return len(self.corpus_index)-self.block_size

    def __getitem__(self, idx):
        return self.corpus_index[idx:idx+self.block_size],self.corpus_index[idx+1:idx+self.block_size+1]

if __name__ == "__main__":

    dataset = LanguageModelDataset('data/text.txt', block_size=10)
    dataloader = DataLoader(dataset, batch_size=1)

    for x, y in dataloader:
        print(f"{x}, {y}")
