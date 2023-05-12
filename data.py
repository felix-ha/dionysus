import torch


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

def create_corpus_index(corpus_raw: str) -> torch.Tensor:
    """
    Given a string (corpus_raw), one dimensional tensor with the indexes
    wrt the vocabulary based on this string is created.
    """
    vocabulary = get_distinct_characters(corpus_raw)
    token_to_index, _ = create_token_index_mappings(vocabulary)
    encoder = create_encoder(token_to_index)
    corpus_index = torch.tensor(encoder(corpus_raw), dtype=torch.long)
    return corpus_index

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